"""Checkpoint serialization with an explicit type allowlist.

LangGraph serializes graph state with msgpack. By default it deserializes *any* recorded type
(with a deprecation warning: this will be blocked in a future version). Restricting it to the
exact types this agent stores is both forward-compatible and safer: a tampered checkpoint cannot
make the process instantiate arbitrary classes.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# (module, class) of every custom type that can appear in AgentState.
CHECKPOINT_TYPES: tuple[tuple[str, str], ...] = (
    ("adaptive_scheduler.agent.executor", "ExecutionReport"),
    ("adaptive_scheduler.agent.executor", "TaskRecord"),
    ("adaptive_scheduler.agent.executor", "TaskStatus"),
    ("adaptive_scheduler.agent.planning", "CallUsage"),
    ("adaptive_scheduler.agent.workflow", "CandidateSummary"),
    ("adaptive_scheduler.agent.workflow", "StrategyDecision"),
    ("adaptive_scheduler.cost.profiles", "TaskProfile"),
    ("adaptive_scheduler.graph.model", "Task"),
    ("adaptive_scheduler.graph.model", "TaskGraph"),
    ("adaptive_scheduler.graph.model", "TaskType"),
    ("adaptive_scheduler.scheduling.simulator", "PriorityPolicy"),
    ("adaptive_scheduler.simulation.engine", "Mode"),
    ("adaptive_scheduler.simulation.engine", "PreparedPlan"),
    ("adaptive_scheduler.simulation.strategies", "Strategy"),
)


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


def memory_checkpointer() -> InMemorySaver:
    """In-process checkpointer for development and tests.

    For multi-replica deployments use a shared saver (e.g. ``langgraph-checkpoint-postgres``)
    constructed with ``serde=serializer()``.
    """
    return InMemorySaver(serde=serializer())
