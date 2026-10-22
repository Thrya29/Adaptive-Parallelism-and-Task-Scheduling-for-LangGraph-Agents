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

### 3.3 Cost model and estimator
Call latency is `(ttft + n_in/prefill + n_out/decode) · noise · tail · (1 + β·(active − 1))`:
log-normal noise with mean 1, a heavy tail (probability `p`, multiplier `m`), and contention `β`
from concurrent calls. The provider serves at most `C` calls at once; excess calls wait and pay a
rate-limit penalty (HTTP 429 + back-off). A task's prompt includes the system prompt, shared
context, its instruction and **the outputs of its dependencies**, so fan-in tasks are expensive.

The `CostEstimator` calibrates this online: EWMA of output tokens and success rate per task type
(seeded with priors), and a decayed **weighted** least-squares fit of `ttft` and decode speed.
Weights are `1 / expected_latency²` because latency noise is multiplicative; ordinary least
squares let long, noisy generations dominate the intercept (a test caught this: OLS estimated
1.39 s for a true 1.2 s overhead).

### 3.4 Graph analyzer
* **Critical path** — longest path under expected latency via DP over the topological order.
* **Upward rank** (HEFT): `rank(v) = c(v) + max_{s ∈ succ(v)} rank(s)`.
* **Work / span**: `W = Σ c(v)`, `S = critical path length`; available parallelism `W / S`.
* **Lower bound on makespan** with `k` workers: `max(S, W / k)`.

### 3.5 Strategy selector
Implemented in `simulation/selector.py` on top of a stochastic discrete-event engine
(`simulation/engine.py`) that models two nested concurrency limits - client slots `k` (held across
retries, like a semaphore-guarded worker) and the provider cap `C` - plus retries with jittered
exponential back-off and failure isolation. It evaluates candidate execution strategies — sequential, static level-parallel, adaptive list
scheduling at several concurrency levels `k`, with or without task fusion — by **Monte Carlo
simulation** over the estimated latency distributions. This predicts both mean and **P95**
latency, plus cost and call count, for each candidate. The selector minimizes

```
J = E[latency] + α · P95(latency) + λ · cost      subject to   P(success) ≥ τ
```

All candidates share random seeds (common random numbers) to reduce the variance of their
differences. Candidates within a 1 % *parsimony* margin of the best are tie-broken towards the
simplest architecture (fewest workers, then fewest calls), and candidates whose predicted success
rate is more than 1 % below the best are excluded - which is how the selector declines task fusion
when packing steps into one call would make failures (and whole-call retries) too likely.

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

### 3.8 LangGraph workflow and executor

```
START -> plan --(invalid, attempts left)--> plan
              \-(valid, or fallback)-----> schedule -> execute -> synthesize -> END
```

* **plan** — the planner returns a `PlanSpec` (structured output / tool calling with a real model;
  a deterministic heuristic decomposition with the mock). It is validated into a `TaskGraph`,
  transitively reduced and checked against size limits. The exact error (e.g.
  `dependency cycle: a -> b -> a`, `plan has 40 tasks; limit is 32`) is fed back for one repair
  attempt; a still-invalid plan degrades to a single-task plan and the run is marked unsuccessful.
* **schedule** — builds cost profiles from the estimator's current beliefs and runs the selector
  (or applies a fixed baseline), storing every candidate's prediction in the state so the decision
  is explainable.
* **execute** — `TaskExecutor` runs the (possibly fused) graph and its observations update the
  `CostEstimator`.
* **synthesize** — if the plan has a single sink that completed, its output *is* the answer (no
  extra call); otherwise one synthesis call combines the available results and names the missing
  parts.

**Why the executor is not a set of LangGraph `Send` branches.** LangGraph executes graphs in
Pregel-style *supersteps*: branches fanned out with `Send` run in parallel, but the next superstep
starts only after all of them finish. Mapping each plan level onto a superstep therefore yields
exactly the level-synchronous `static_parallel` baseline, with its barrier stalls. To get greedy
list scheduling, the `execute` node runs an asyncio worker pool driven by the shared `ReadyQueue`:
a task starts the moment its own dependencies finish and a worker slot is free.

The executor holds a worker slot across retries, enforces a timeout per attempt, retries any
provider error with jittered exponential back-off, and on final failure cancels only the failed
task's transitive dependents. Real Anthropic clients are created with `max_retries=0` so every
retry is owned, counted and scheduled by the executor.

### 3.9 LLM backends
A single interface with two implementations:

* **Mock LLM** (`MockChatModel`) — a LangChain chat model driven by the same `ProviderProfile`
  as the simulator: sampled latency with contention, a provider concurrency cap with rate-limit
  penalty, `usage_metadata` token counts, and injectable transient failures. Each call's random
  stream derives from `(seed, prompt, n-th call with that prompt)`, so results do not depend on
  the order in which concurrent calls are scheduled. Requires no key.
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
