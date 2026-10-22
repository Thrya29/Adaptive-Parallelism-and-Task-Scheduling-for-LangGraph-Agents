"""Named execution strategies: the adaptive selector and the fixed baselines it is compared to."""

from __future__ import annotations

from enum import StrEnum

from adaptive_scheduler.scheduling.simulator import PriorityPolicy
from adaptive_scheduler.simulation.engine import ExecutionConfig, Mode


class Strategy(StrEnum):
    ADAPTIVE = "adaptive"  # Monte Carlo selector picks k and fusion per request
    SEQUENTIAL = "sequential"  # one task at a time, topological order
    STATIC_PARALLEL = "static_parallel"  # level-synchronous fan-out (LangGraph superstep style)
    EAGER_PARALLEL = "eager_parallel"  # unbounded greedy fan-out (LLMCompiler style)


BASELINES: dict[Strategy, ExecutionConfig] = {
    Strategy.SEQUENTIAL: ExecutionConfig(
        Strategy.SEQUENTIAL.value, workers=1, policy=PriorityPolicy.FIFO
    ),
    Strategy.STATIC_PARALLEL: ExecutionConfig(
        Strategy.STATIC_PARALLEL.value, workers=None, mode=Mode.LEVEL_SYNC
    ),
    Strategy.EAGER_PARALLEL: ExecutionConfig(
        Strategy.EAGER_PARALLEL.value, workers=None, policy=PriorityPolicy.FIFO
    ),
}
