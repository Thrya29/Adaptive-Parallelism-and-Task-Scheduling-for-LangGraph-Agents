"""Graph algorithms on plain dependency mappings.

Every function takes ``preds``: a mapping from task id to the ids it depends on. Iteration order
of the mapping is the tie-breaking order, which makes every result deterministic.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

Preds = Mapping[str, Sequence[str]]


class CycleError(ValueError):
    """Raised when a dependency graph is not acyclic. ``cycle`` lists the tasks in run order."""

    def __init__(self, cycle: Sequence[str]) -> None:
        self.cycle = tuple(cycle)
        super().__init__("dependency cycle: " + " -> ".join(self.cycle))


def successors_of(preds: Preds) -> dict[str, list[str]]:
    succ: dict[str, list[str]] = {node: [] for node in preds}
    for node, deps in preds.items():
        for dep in deps:
            succ[dep].append(node)
    return succ


def find_cycle(preds: Preds) -> list[str] | None:
    """Return one cycle in run order (first node repeated at the end), or ``None`` if acyclic.

    Iterative three-colour DFS over successor edges, ``O(V + E)``.
    """
    succ = successors_of(preds)
    white, gray, black = 0, 1, 2
    colour = dict.fromkeys(preds, white)

    for root in preds:
        if colour[root] != white:
            continue
        path: list[str] = [root]
        stack = [iter(succ[root])]
        colour[root] = gray
        while stack:
            nxt = next(stack[-1], None)
            if nxt is None:
                colour[path.pop()] = black
                stack.pop()
            elif colour[nxt] == gray:
                start = path.index(nxt)
                return [*path[start:], nxt]
            elif colour[nxt] == white:
                colour[nxt] = gray
                path.append(nxt)
                stack.append(iter(succ[nxt]))
    return None


def topological_order(preds: Preds) -> list[str]:
    """Kahn's algorithm, ``O(V + E)``. Ready nodes are emitted in mapping order (FIFO)."""
    succ = successors_of(preds)
    remaining = {node: len(deps) for node, deps in preds.items()}
    ready = deque(node for node, count in remaining.items() if count == 0)
    order: list[str] = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for nxt in succ[node]:
            remaining[nxt] -= 1
            if remaining[nxt] == 0:
                ready.append(nxt)
    if len(order) != len(preds):
        raise CycleError(find_cycle(preds) or ())
    return order


def levels(preds: Preds) -> dict[str, int]:
    """Level of each node: 0 for sources, else ``1 + max(level of dependencies)``."""
    level: dict[str, int] = {}
    for node in topological_order(preds):
        level[node] = 1 + max((level[d] for d in preds[node]), default=-1)
    return level


def transitive_reduction(preds: Preds) -> dict[str, tuple[str, ...]]:
    """Remove every edge ``u -> v`` implied by a longer path from ``u`` to ``v``.

    Reachability sets are bitsets (Python ints) filled in reverse topological order, so the cost is
    ``O(V * E / wordsize)``. The reduction of a DAG is unique and preserves reachability, hence it
    never changes which schedules are valid.
    """
    order = topological_order(preds)
    bit = {node: 1 << i for i, node in enumerate(order)}
    succ = successors_of(preds)
    reach: dict[str, int] = {}
    for node in reversed(order):
        mask = 0
        for nxt in succ[node]:
            mask |= bit[nxt] | reach[nxt]
        reach[node] = mask

    # Successors of u reachable through some other successor of u are redundant direct edges.
    covered = {}
    for node in order:
        mask = 0
        for nxt in succ[node]:
            mask |= reach[nxt]
        covered[node] = mask

    return {
        node: tuple(dep for dep in deps if not covered[dep] & bit[node])
        for node, deps in preds.items()
    }
