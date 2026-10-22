# Adaptive Parallelism and Task Scheduling for LangGraph Agents

[![CI](https://github.com/Thrya29/Adaptive-Parallelism-and-Task-Scheduling-for-LangGraph-Agents/actions/workflows/ci.yml/badge.svg)](https://github.com/Thrya29/Adaptive-Parallelism-and-Task-Scheduling-for-LangGraph-Agents/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> **Research question.** Can an algorithm predict which tasks of an LLM agent workflow should run
> in parallel and which sequentially — choosing the execution architecture that minimizes latency
> and cost without reducing task success?

This project builds a LangGraph workflow that decomposes a request into tasks, represents their
dependencies as a **DAG**, **estimates task costs**, analyses the **critical path**, and uses a
**cost-aware adaptive scheduler** to run independent tasks concurrently only when doing so pays
off. It is evaluated against sequential and static-parallel execution.

## Motivation

Adding more agents is not a free win. A large controlled study of 180 agent configurations found
that centralized multi-agent coordination improved a parallelizable financial-analysis benchmark by
80.9%, yet every multi-agent variant degraded a sequential planning benchmark by 39–70%, and
independent agents amplified errors 17.2× ([Kim et al., 2025][scaling]). LLMCompiler showed that
planning function calls as a DAG and executing them in parallel can cut latency up to 3.7×
([Kim et al., 2024][llmcompiler]) — but it parallelizes eagerly.

This project asks the next question: **when** is parallelism worth it, and **how much**? Instead of
adding agents, it treats agent execution as a classical scheduling problem with LLM-specific costs
(rate limits, duplicated context, merge overhead, per-call latency).

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Project scaffold, configuration, CI, design docs | ✅ done |
| 1 | Core algorithms: DAG model, validation, critical path, list scheduler, task fusion | ✅ done |
| 2 | Cost model, mock LLM, simulator, Monte Carlo strategy selector | ✅ done |
| 3 | LangGraph planner / executor / synthesizer, Anthropic backend, reliability | ✅ done |
| 4 | Benchmark suite and experiment harness (simulation) | ⏳ next |
| 5 | Real-LLM experiments, analysis, report | planned |
| 6 | FastAPI service, observability, Docker, deployment | planned |

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/Thrya29/Adaptive-Parallelism-and-Task-Scheduling-for-LangGraph-Agents.git
cd Adaptive-Parallelism-and-Task-Scheduling-for-LangGraph-Agents
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,anthropic]"
apsched info
```

### With or without an API key — both work

| You have… | What happens |
|---|---|
| **No API key** | Runs on a deterministic **mock LLM** that simulates realistic latency, token usage and failures. Free, offline, reproducible. |
| **An Anthropic API key** | Put it in `.env` (copy from `.env.example`) as `ANTHROPIC_API_KEY=...`; real Claude calls are used automatically. |

Control it explicitly with `APSCHED_LLM_PROVIDER` (`auto` | `mock` | `anthropic`) or per command:

```bash
apsched --provider mock info
```

### Run the agent

```bash
apsched run "Research the EV markets of India, China and Germany, then compare their growth and risks, then write a short investment summary"
apsched run --strategy sequential "..."     # or static_parallel / eager_parallel baselines
apsched run --json "..."                    # full machine-readable result
```

Example output (mock LLM, no key; mock latencies run at 0.1x wall-clock):

```
Plan (7 tasks, attempts=1):
  [retrieval ] s1_1: Research the EV markets of India
  [retrieval ] s1_2: Research the EV markets of China
  [retrieval ] s1_3: Research the EV markets of Germany
  [reasoning ] s2_1 <- s1_1, s1_2, s1_3: compare their growth
  [reasoning ] s2_2 <- s1_1, s1_2, s1_3: compare their risks
  [reasoning ] s3_1 <- s2_1, s2_2: write a short investment summary
  [synthesis ] final <- s3_1: Combine the results into the final answer to the request.

Strategy: adaptive -> adaptive[k=3,fused]
  estimated work W=41.2s, critical path S=25.2s, parallelism W/S=1.63
  candidate                    mean      p95    cost $  calls   succ
  adaptive[k=3]               30.7s    52.4s   0.01689    7.1  1.000
 *adaptive[k=3,fused]         30.7s    53.2s   0.01681    6.2  1.000
  adaptive[k=2]               32.5s    53.8s   0.01689    7.1  1.000
  adaptive[k=1]               41.7s    62.3s   0.01690    7.1  1.000
  ...

Execution timeline:
  s1_1              |#######                                                     |   0.37s completed
  s1_2              |##########                                                  |   0.54s completed
  s1_3              |###############                                             |   0.83s completed
  s2_1              |               ######                                       |   0.34s completed
  s2_2              |               ##############                               |   0.81s completed
  fused:s3_1..final |                             ###############################|   1.77s completed

Metrics: wall-clock 4.42s (plan 0.86s, select 0.13s, execute 3.40s, synthesize 0.00s)
         7 model calls, 4228 in / 3247 out tokens, ~$0.02046, peak concurrency 3, success=yes
```

The selector ran the three lookups concurrently (k = 3, the plan's width) and fused the final
chain `s3_1 → final` into one call, saving a model call at no latency cost.

### Use it as a library

```python
import asyncio
from adaptive_scheduler.agent import AdaptiveAgent, Strategy
from adaptive_scheduler.config import Settings

agent = AdaptiveAgent.from_settings(Settings())       # mock or Anthropic, per your settings
result = asyncio.run(agent.arun("your request", strategy=Strategy.ADAPTIVE))
print(result.answer, result.metrics, result.decision.config_name)
```

API keys are read only from the environment / `.env` (git-ignored) and are never logged.
Note: API keys come from [console.anthropic.com](https://console.anthropic.com) and are separate
from a Claude.ai subscription.

## Design overview

```
Request → Planner → DAG validator → Cost estimator → Critical-path analyzer
        → Strategy selector (sequential | static-parallel | adaptive, concurrency k, fusion)
        → Priority-queue list scheduler → LangGraph executor → Synthesizer → Response
                     ↑                                              │
                     └──────── observed latency / tokens (learning) ┘
```

| Concern | Technique |
|---|---|
| Dependency model | DAG; Kahn's topological sort, cycle detection, transitive reduction |
| Latency bound | Critical path (longest path DP), work/span, `max(S, W/k)` lower bound |
| Ordering | HEFT-style upward rank in a priority queue (list scheduling, Graham bound) |
| Architecture choice | Monte Carlo prediction of mean / P95 latency and cost per strategy |
| Call reduction | Task fusion by dynamic programming on chains |
| Adaptation | Online cost-model updates; contextual bandit (stretch goal) |

Full details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Experiments (planned)

Strategies compared: single-agent sequential, topological sequential, static level-parallel,
LLMCompiler-style eager parallel, and the adaptive scheduler (plus ablations).

Metrics: task success rate, mean / P50 / P95 / P99 wall-clock latency, token cost, number of model
calls, speedup over sequential, and parallel efficiency — reported with bootstrap confidence
intervals over multiple seeds.

## Development

```bash
ruff check . && ruff format --check .   # lint / format
mypy                                    # strict type checking
pytest --cov                            # tests (mock LLM; no key needed)
pre-commit install                      # optional git hooks
```

## Repository layout

```
src/adaptive_scheduler/
  graph/        Task / TaskGraph model, Kahn's sort, cycle detection, transitive reduction,
                critical path, upward rank, slack, graph profile and bounds
  scheduling/   ReadyQueue (shared by simulator and executor), list / sequential /
                level-parallel schedulers, schedule validator, chain-fusion DP
  cost/         provider latency & pricing model, per-task token profiles (information
                flows along edges), online estimator (weighted least squares + EWMA)
  simulation/   stochastic discrete-event engine (client slots, provider cap / rate limits,
                contention, heavy tails, retries, failure isolation) and the Monte Carlo
                strategy selector
  llm/          deterministic MockChatModel (LangChain-compatible) and provider factory
  agent/        LangGraph workflow (plan -> repair -> schedule -> execute -> synthesize),
                LLM and heuristic planners, async TaskExecutor, prompts
  report.py     text rendering of a run (plan, candidate table, Gantt timeline, metrics)
  config.py     settings and LLM provider resolution (auto | mock | anthropic)
  cli.py        `apsched` command
tests/
  unit/         hand-checked examples
  property/     Hypothesis invariants over thousands of random DAGs
  integration/  end-to-end agent runs on the mock LLM; live Anthropic test (`pytest -m live`)
docs/           architecture and research report
```

### What the tests guarantee

Property-based tests (300 random DAGs per property, up to 24 tasks) check that for every graph,
duration vector, worker count `k` and priority policy:

- every schedule is valid: each task runs once, after all of its dependencies, never more than `k`
  at a time (checked by an independent validator);
- `max(S, W/k) <= makespan <= W/k + (1 - 1/k)·S` — the theoretical lower bound and Graham's
  list-scheduling bound;
- with `k >= n` workers the list scheduler achieves the critical path exactly;
- the chain-fusion DP equals brute-force enumeration of all segmentations, and fusion never
  increases total work or the critical path;
- the transitive reduction preserves reachability and is minimal;
- with randomness switched off, the stochastic engine reproduces the deterministic list and
  level-parallel schedulers exactly, and every task is accounted for (completed / failed /
  cancelled) under random failures and rate limits.

### Early simulation check (default provider model, 200 samples per strategy)

| Plan shape | W / S | Sequential | Static parallel | Eager parallel | Adaptive (chosen config) |
|---|---|---|---|---|---|
| chain of 8 | 1.00 | 54.6 s | 54.6 s | 54.6 s | **54.6 s** (k = 1) |
| fan-out 16 → merge | 7.75 | 115.6 s | 33.5 s | 33.5 s | **32.4 s** (k = 8, provider limit) |
| mixed, 11 tasks | 2.20 | 75.4 s | 44.2 s | 40.0 s | **40.0 s** (k = 6) |

The selector matches the best fixed strategy on each shape without being told the shape. These
are sanity checks, not the study results; the full experiment suite (Phase 4) sweeps provider
limits, cost heterogeneity and failure rates, and adds a dependency-blind multi-agent baseline.

## References

- Y. Kim et al. *Towards a Science of Scaling Agent Systems.* arXiv:2512.08296, 2025.
  Blog: [When and why agent systems work][scaling-blog], Google Research, Jan 2026.
- S. Kim et al. *An LLM Compiler for Parallel Function Calling.* ICML 2024. [arXiv:2312.04511][llmcompiler]
- H. Topcuoglu, S. Hariri, M.-Y. Wu. *Performance-Effective and Low-Complexity Task Scheduling for
  Heterogeneous Computing* (HEFT). IEEE TPDS, 2002.
- R. L. Graham. *Bounds on Multiprocessing Timing Anomalies.* SIAM J. Appl. Math., 1969.

## License

[MIT](LICENSE)

[scaling]: https://arxiv.org/abs/2512.08296
[scaling-blog]: https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/
[llmcompiler]: https://arxiv.org/abs/2312.04511
