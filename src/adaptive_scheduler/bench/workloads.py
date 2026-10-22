"""Benchmark workloads: task DAG families with controlled shape and cost heterogeneity.

Every workload is generated deterministically from ``(family, instance, master seed)``. Per-task
output length is the task type's prior times a log-normal factor (``heterogeneity`` = its sigma),
so some branches are much longer than others - the situation where critical-path priority and
avoiding level barriers matter.

Families span the parallelism spectrum (``W / S`` from 1 to ~10):

========== ===================================================================================
chain      strictly sequential steps (no parallelism)
fan_out    independent lookups followed by one merge (map-reduce)
fork_join  several stages, each a fan-out followed by a join
tree       binary reduction tree (pairwise combination)
layered    random layered DAG: each task depends on a random subset of the previous layer
random     random DAG (each forward edge present with probability p)
realistic  plans the heuristic planner derives from a fixed suite of natural-language requests
========== ===================================================================================
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from adaptive_scheduler.agent.planning import heuristic_plan
from adaptive_scheduler.cost.profiles import (
    OUTPUT_TOKEN_PRIORS,
    ContextSize,
    TaskProfile,
    build_profiles,
)
from adaptive_scheduler.graph.model import Task, TaskGraph, TaskType

CONTEXT = ContextSize()

REQUESTS: tuple[str, ...] = (
    "Research the EV markets of India, China and Germany, then compare their growth and risks, "
    "then write a short investment summary",
    "Summarize the attached quarterly report",
    "Find the population, GDP and capital of France, Japan, Brazil and Kenya, then rank them",
    "Draft a project plan, then estimate its cost, then list the main risks, then write an email",
    "- review the API design\n- review the database schema\n- review the auth flow\n"
    "- review the deployment scripts",
    "Compare PostgreSQL, MySQL and SQLite for a small analytics app, then recommend one",
    "Translate the abstract to French, Spanish and German",
    "Outline the history of the internet; then explain how TCP works; then write a quiz",
    "Collect pricing for AWS, GCP and Azure GPUs, then compute the cheapest option for 100 hours",
    "Explain the proof of the four colour theorem",
)


@dataclass(frozen=True)
class Workload:
    family: str
    instance: int
    graph: TaskGraph
    profiles: dict[str, TaskProfile]

    @property
    def name(self) -> str:
        return f"{self.family}#{self.instance}"


def stable_seed(*parts: object) -> int:
    """A seed stable across processes and Python runs (unlike ``hash``)."""
    return zlib.crc32("|".join(map(str, parts)).encode())


# ---------------------------------------------------------------------------- shapes
def chain(n: int) -> dict[str, list[str]]:
    return {f"c{i}": [f"c{i - 1}"] if i else [] for i in range(n)}


def fan_out(width: int) -> dict[str, list[str]]:
    deps: dict[str, list[str]] = {f"m{i}": [] for i in range(width)}
    deps["reduce"] = list(deps)
    return deps


def fork_join(stages: int, width: int) -> dict[str, list[str]]:
    deps: dict[str, list[str]] = {"start": []}
    prev = "start"
    for s in range(stages):
        branch = [f"s{s}_b{j}" for j in range(width)]
        for b in branch:
            deps[b] = [prev]
        prev = f"s{s}_join"
        deps[prev] = branch
    return deps


def tree(leaves: int) -> dict[str, list[str]]:
    deps: dict[str, list[str]] = {f"leaf{i}": [] for i in range(leaves)}
    level, layer = 0, list(deps)
    while len(layer) > 1:
        nxt = []
        for i in range(0, len(layer), 2):
            node = f"l{level}_{i // 2}"
            deps[node] = layer[i : i + 2]
            nxt.append(node)
        level, layer = level + 1, nxt
    return deps


def layered(rng: np.random.Generator, depth: int, width: tuple[int, int]) -> dict[str, list[str]]:
    deps: dict[str, list[str]] = {}
    prev: list[str] = []
    for d in range(depth):
        layer = [f"L{d}_{j}" for j in range(int(rng.integers(width[0], width[1] + 1)))]
        for node in layer:
            if prev:
                k = int(rng.integers(1, min(3, len(prev)) + 1))
                deps[node] = sorted(rng.choice(prev, size=k, replace=False).tolist())
            else:
                deps[node] = []
        prev = layer
    return deps


def random_dag(rng: np.random.Generator, n: int, p: float) -> dict[str, list[str]]:
    return {f"r{j}": [f"r{i}" for i in range(j) if rng.random() < p] for j in range(n)}


# ---------------------------------------------------------------------------- generation
def _typed_graph(deps: dict[str, list[str]]) -> TaskGraph:
    """Sources retrieve, sinks synthesize, everything else reasons."""
    succ: dict[str, int] = dict.fromkeys(deps, 0)
    for ds in deps.values():
        for d in ds:
            succ[d] += 1
    tasks = []
    for tid, ds in deps.items():
        if not ds:
            ttype = TaskType.RETRIEVAL
        elif succ[tid] == 0:
            ttype = TaskType.SYNTHESIS
        else:
            ttype = TaskType.REASONING
        tasks.append(
            Task(id=tid, instruction=f"Perform step {tid}.", task_type=ttype, depends_on=tuple(ds))
        )
    return TaskGraph(tasks=tuple(tasks))


def _with_costs(
    graph: TaskGraph, rng: np.random.Generator, heterogeneity: float
) -> dict[str, TaskProfile]:
    sigma = heterogeneity
    outputs = {
        t: OUTPUT_TOKEN_PRIORS[graph.task(t).task_type]
        * float(rng.lognormal(-0.5 * sigma**2, sigma))
        for t in graph.ids
    }
    return build_profiles(graph, CONTEXT, per_task_output=outputs)


ShapeFn = Callable[[np.random.Generator, int], dict[str, list[str]]]

FAMILIES: dict[str, ShapeFn] = {
    "chain": lambda rng, i: chain((6, 8, 10, 12)[i % 4]),
    "fan_out": lambda rng, i: fan_out((4, 8, 12, 16)[i % 4]),
    "fork_join": lambda rng, i: fork_join(stages=(2, 3, 3, 4)[i % 4], width=(3, 4, 6, 4)[i % 4]),
    "tree": lambda rng, i: tree((4, 8, 8, 16)[i % 4]),
    "layered": lambda rng, i: layered(rng, depth=int(rng.integers(3, 6)), width=(2, 6)),
    "random": lambda rng, i: random_dag(rng, n=int(rng.integers(10, 19)), p=0.18),
}


def generate(family: str, instance: int, seed: int = 0, heterogeneity: float = 0.6) -> Workload:
    rng = np.random.default_rng(stable_seed("workload", family, instance, seed))
    if family == "realistic":
        graph = heuristic_plan(REQUESTS[instance % len(REQUESTS)]).to_graph()
    elif family in FAMILIES:
        graph = _typed_graph(FAMILIES[family](rng, instance))
    else:
        raise ValueError(f"unknown workload family {family!r}")
    return Workload(family, instance, graph, _with_costs(graph, rng, heterogeneity))


def suite(
    families: tuple[str, ...] = (*FAMILIES, "realistic"),
    instances: int = 4,
    seed: int = 0,
    heterogeneity: float = 0.6,
) -> list[Workload]:
    out: list[Workload] = []
    for family in families:
        count = len(REQUESTS) if family == "realistic" else instances
        out.extend(generate(family, i, seed, heterogeneity) for i in range(count))
    return out
