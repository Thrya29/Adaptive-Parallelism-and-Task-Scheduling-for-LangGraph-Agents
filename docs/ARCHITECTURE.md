# Architecture

This document describes the target design. Sections marked *(planned)* are not implemented yet;
see the roadmap in the [README](../README.md) for current status.

## 1. Problem statement

Given a user request, an LLM agent system must decide **which sub-tasks to run in parallel and which
to run sequentially** so that wall-clock latency and token cost are minimized without hurting task
success. Parallelism is not free:

| Cost of parallelism | Why it matters |
|---|---|
| Provider concurrency / rate limits | Fan-out beyond the allowed rate queues anyway, adding overhead and retries |
| Duplicated context | Each parallel call re-sends shared context, multiplying input tokens |
| Merge / synthesis step | Parallel branches must be reconciled, adding an extra call |
| Hidden dependencies | Running "independent-looking" tasks blind to each other's outputs can lower accuracy and amplify errors |
| Per-call overhead | Many tiny calls cost more latency (time-to-first-token) than one combined call |

The scheduler's job is to **predict** when the benefits of concurrency outweigh these costs.

## 2. Pipeline

```
Request ─► Input guardrails ─► Planner (LLM, structured output) ─► DAG validator
        ─► Cost estimator ─► Graph analyzer ─► Strategy selector (+ task fusion)
        ─► Scheduler (priority queue, concurrency / token / budget limits)
        ─► LangGraph executor (Send fan-out, workers, retries, checkpoints)
        ─► Synthesizer ─► Output guardrails ─► Response
                ▲                                         │
                └──── run history: observed latency / tokens / outcome ◄─┘
```

## 3. Components

### 3.1 Task graph model
A request is decomposed into a DAG `G = (V, E)`. Each task `v` has a type (e.g. `retrieve`,
`reason`, `tool`, `synthesize`), an instruction, declared inputs (edges), and an estimated cost
`c(v) = (latency distribution, input tokens, output tokens, success probability)`.

### 3.2 Validator
* Cycle detection and topological order with **Kahn's algorithm**, `O(V + E)`; reports the
  offending cycle so the planner can be re-prompted once with a precise error.
* **Transitive reduction** to remove redundant edges (they do not change the schedule but distort
  fan-in statistics).
* Hard limits on node count, depth and fan-out (guardrail against runaway plans).

### 3.3 Cost estimator
Per-task latency is modelled as `TTFT + output_tokens / throughput` with a log-normal noise term,
using priors by task type that are refined online from run history (EWMA of means, streaming
quantiles for tails). Token cost uses provider pricing.

### 3.4 Graph analyzer
* **Critical path** — longest path under expected latency via DP over the topological order.
* **Upward rank** (HEFT): `rank(v) = c(v) + max_{s ∈ succ(v)} rank(s)`.
* **Work / span**: `W = Σ c(v)`, `S = critical path length`; available parallelism `W / S`.
* **Lower bound on makespan** with `k` workers: `max(S, W / k)`.

### 3.5 Strategy selector
Evaluates candidate execution strategies — sequential, static level-parallel, adaptive list
scheduling at several concurrency levels `k`, with or without task fusion — by **Monte Carlo
simulation** over the estimated latency distributions. This predicts both mean and **P95**
latency, plus cost and call count, for each candidate. The selector minimizes

```
J = E[latency] + α · P95(latency) + λ · cost      subject to   P(success) ≥ τ
```

When `W / S ≈ 1` (an essentially sequential graph) the selector picks sequential execution and
avoids paying coordination overhead — the regime where blind multi-agent fan-out is known to hurt.

### 3.6 Task fusion
Merges a chain `a → b` (or small sibling groups) into a single LLM call when the saved per-call
overhead exceeds the lost overlap. Only chain edges `u -> v` with `out(u) = 1` and `in(v) = 1` are fused:
contracting them cannot remove parallelism or create a cycle. Each maximal chain is segmented
optimally by dynamic programming in `O(n · L_max)`, minimizing
`o + Σ max(d_i − o, 0) + μ·(L − 1)²` per fused segment (fixed overhead `o`, convex quality
penalty `μ`) subject to per-call task and work limits. Fusion provably never increases total work
`W` or span `S`. A greedy list schedule's makespan is still not monotone in task durations
(Graham's timing anomalies), so the strategy selector simulates fused and unfused variants
rather than assuming fusion always helps. Sibling fusion (which trades away parallelism) is left
to future work.

### 3.7 Scheduler
Event-driven **list scheduling**: a ready-queue (binary heap) ordered by upward rank, dispatching
whenever a worker slot, rate-limit token (token bucket) and budget are all available. As tasks
finish, observed durations replace estimates and remaining ranks are updated. Graham's bound gives
`makespan ≤ (2 − 1/k) · OPT` for any list schedule, which we verify empirically against the lower
bound.

### 3.8 Executor
LangGraph `StateGraph` with the `Send` API for dynamic fan-out, reducers for merging task results
into shared state, per-task timeouts, retries with exponential backoff + jitter, failure isolation
(dependents of a failed task are skipped, independent branches continue), and checkpointing for
resumability.

### 3.9 LLM backends
A single interface with two implementations:

* **Mock LLM** — deterministic given a seed; simulates realistic latency (TTFT + throughput,
  heavy-tailed noise), token usage and controllable failure rates. Used for tests, CI and
  large-scale experiments; requires no key.
* **Anthropic** — real calls via `langchain-anthropic` when `ANTHROPIC_API_KEY` is present.

Selection is automatic (`APSCHED_LLM_PROVIDER=auto`) or explicit (`mock` / `anthropic`).

### 3.10 Learning loop *(stretch)*
A contextual bandit (Thompson sampling) over graph features (`W/S`, depth, width, cost
heterogeneity) that learns which strategy wins on which workload, compared against the
model-based selector.

## 4. Non-functional requirements

* **Reliability** — time / token / dollar budgets, cancellation, partial results.
* **Security** — secrets only from environment, never logged; plan-size limits; schema validation
  of all LLM outputs.
* **Observability** — OpenTelemetry span per task (rendered as Gantt charts), Prometheus metrics,
  structured logs.
* **Reproducibility** — seeded mock LLM, versioned benchmark suites, raw results stored per run.
* **Quality gates** — ruff, mypy `--strict`, unit + property-based tests (Hypothesis) of scheduler
  invariants, CI on every push.
