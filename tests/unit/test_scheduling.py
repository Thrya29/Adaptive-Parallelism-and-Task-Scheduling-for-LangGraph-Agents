import pytest

from adaptive_scheduler.graph import TaskGraph
from adaptive_scheduler.scheduling import (
    InvalidScheduleError,
    PriorityPolicy,
    ReadyQueue,
    Schedule,
    ScheduledTask,
    level_parallel_schedule,
    list_schedule,
    sequential_schedule,
    validate_schedule,
)


def test_sequential_makespan_is_total_work() -> None:
    g = TaskGraph.from_dependencies({"a": [], "b": [], "c": ["a"]})
    d = {"a": 1.0, "b": 2.0, "c": 3.0}
    s = sequential_schedule(g, d)
    validate_schedule(s, g, d)
    assert s.makespan == 6.0
    assert s.dispatch_order == ("a", "b", "c")


def test_critical_path_priority_beats_fifo() -> None:
    # Two workers. FIFO starts the short independent tasks first and delays the long chain head.
    g = TaskGraph.from_dependencies({"x": [], "y": [], "a": [], "b": ["a"]})
    d = {"x": 1.0, "y": 1.0, "a": 4.0, "b": 4.0}
    fifo = list_schedule(g, d, 2, PriorityPolicy.FIFO)
    cp = list_schedule(g, d, 2, PriorityPolicy.CRITICAL_PATH)
    for s in (fifo, cp):
        validate_schedule(s, g, d)
    assert fifo.makespan == 9.0
    assert cp.makespan == 8.0  # optimal: equals the critical path a -> b


def test_level_parallel_pays_for_barriers() -> None:
    # Level barrier makes the short branch wait: a(1) b(5) | c(1, after a) d(1, after b).
    g = TaskGraph.from_dependencies({"a": [], "b": [], "c": ["a"], "e": ["c"], "d": ["b"]})
    d = {"a": 1.0, "b": 5.0, "c": 1.0, "e": 5.0, "d": 1.0}
    static = level_parallel_schedule(g, d)
    adaptive = list_schedule(g, d, 2)
    validate_schedule(static, g, d)
    validate_schedule(adaptive, g, d)
    assert static.makespan == 11.0  # 5 + 1 + 5
    assert adaptive.makespan == 7.0  # critical path a -> c -> e


def test_level_parallel_with_limited_workers() -> None:
    g = TaskGraph.from_dependencies({"a": [], "b": [], "c": []})
    d = dict.fromkeys("abc", 1.0)
    s = level_parallel_schedule(g, d, workers=2)
    validate_schedule(s, g, d)
    assert s.makespan == 2.0


def test_zero_duration_tasks() -> None:
    g = TaskGraph.from_dependencies({"a": [], "b": ["a"], "c": ["b"]})
    d = {"a": 0.0, "b": 0.0, "c": 2.0}
    s = list_schedule(g, d, 3)
    validate_schedule(s, g, d)
    assert s.makespan == 2.0


def test_invalid_worker_counts() -> None:
    g = TaskGraph.from_dependencies({"a": []})
    with pytest.raises(ValueError, match="workers"):
        list_schedule(g, {"a": 1.0}, 0)
    with pytest.raises(ValueError, match="workers"):
        level_parallel_schedule(g, {"a": 1.0}, workers=0)


# ---------------------------------------------------------------- validator catches bad schedules
@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ((ScheduledTask("a", 0, 0, 1),), "exactly once"),
        ((ScheduledTask("a", 0, 0, 1), ScheduledTask("b", 0, 0.5, 1.5)), "before dependency"),
        ((ScheduledTask("a", 0, 0, 1), ScheduledTask("b", 0, 1, 3)), "duration"),
        ((ScheduledTask("a", 5, 0, 1), ScheduledTask("b", 0, 1, 2)), "out of range"),
    ],
)
def test_validator_rejects(entries: tuple[ScheduledTask, ...], message: str) -> None:
    g = TaskGraph.from_dependencies({"a": [], "b": ["a"]})
    with pytest.raises(InvalidScheduleError, match=message):
        validate_schedule(Schedule("bad", 1, entries), g, {"a": 1.0, "b": 1.0})


def test_validator_rejects_worker_overlap() -> None:
    g = TaskGraph.from_dependencies({"a": [], "b": []})
    s = Schedule("bad", 2, (ScheduledTask("a", 0, 0, 1), ScheduledTask("b", 0, 0.5, 1.5)))
    with pytest.raises(InvalidScheduleError, match="concurrently"):
        validate_schedule(s, g, {"a": 1.0, "b": 1.0})


# ---------------------------------------------------------------- ready queue
def test_ready_queue_orders_by_priority_and_releases_dependents() -> None:
    g = TaskGraph.from_dependencies({"a": [], "b": [], "c": ["a", "b"]})
    q = ReadyQueue(g, {"a": 1.0, "b": 2.0, "c": 0.0})
    assert len(q) == 2
    assert q.peek() == "b"
    assert q.pop() == "b"
    assert q.complete("b") == []
    assert q.pop() == "a"
    assert q.complete("a") == ["c"]
    assert q.pop() == "c"
    assert not q.finished
    q.complete("c")
    assert q.finished


def test_ready_queue_failure_cancels_only_dependents() -> None:
    g = TaskGraph.from_dependencies({"a": [], "b": ["a"], "c": ["b"], "x": []})
    q = ReadyQueue(g, dict.fromkeys(["a", "b", "c", "x"], 0.0))
    assert q.pop() == "a"
    assert q.fail("a") == ["b", "c"]
    assert q.cancelled == {"b", "c"}
    assert q.pop() == "x"
    q.complete("x")
    assert q.finished
    assert not q


def test_ready_queue_guards_misuse() -> None:
    g = TaskGraph.from_dependencies({"a": [], "b": ["a"]})
    q = ReadyQueue(g, {"a": 0.0, "b": 0.0})
    with pytest.raises(ValueError, match="not dispatched"):
        q.complete("b")
    q.pop()
    q.complete("a")
    with pytest.raises(ValueError, match="already finished"):
        q.complete("a")
    with pytest.raises(ValueError, match="missing priorities"):
        ReadyQueue(g, {"a": 0.0})
    q.pop()
    with pytest.raises(IndexError):
        q.pop()
