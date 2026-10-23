# Adaptive Parallelism and Task Scheduling for LLM Agents

*Research report. All numbers are reproducible from the data in `docs/results/` with the commands
in §9.*

## Abstract

Multi-agent LLM systems often parallelize by default. A controlled study of 180 agent
configurations found that this helps on parallelizable tasks but hurts on sequential ones
([Kim et al., 2025][scaling]). Recent systems decide it with a token-cost rule per DAG layer
([SquidAgent][squid]), or parallelize every independent call ([LLMCompiler][llmcompiler]). We
study a complementary, wall-clock-oriented approach. A request is decomposed into a task DAG, each task's latency and cost are estimated, and a Monte Carlo selector
predicts, *per request*, how much concurrency to use and whether to fuse chained calls. The
scheduler executes with critical-path priority, inside a LangGraph workflow.

In **136,000 simulated runs** (7 workload families × 5 provider conditions × 8 strategies), the
adaptive scheduler is within 5% of the best fixed strategy in **100%** of 35 cells, while the best
fixed strategy changes across cells. Under a tight provider rate limit it is **14% faster** (mean
and P95) than LLMCompiler-style eager fan-out. Dependency-blind multi-agent fan-out succeeds in
only 7.5–22% of runs on dependent work. On **real Claude Haiku 5.5 calls** (160 executions, 609
task calls, $0.70), parallel execution is **1.5–1.66× faster** than sequential at equal cost with
100% success. The adaptive scheduler is never slower than the best baseline. After a calibration
fix found by the live run, its latency forecast has **−2.4% aggregate bias on fresh data**.

A simple rule, *concurrency = min(provider limit, plan width) with critical-path priority*,
captures most of the latency benefit. The model-based selector's additional value is choosing
*not* to parallelize, fusing calls, and deriving that rule from a calibrated cost model instead of
hard-coding it.

## 1. Problem

Given a request, decide which of its sub-tasks run concurrently and which wait, minimizing
wall-clock latency (mean and P95) and token cost without reducing task success. Parallelism has
LLM-specific costs:

* provider concurrency and rate limits (excess calls are throttled, then back off);
* contention between concurrent calls;
* information flow: a task needs its dependencies' outputs in its prompt, and skipping that
  makes answers wrong;
* fixed per-call overhead, which makes many tiny calls expensive.

## 2. Related work and positioning

**Whether, and how much, to parallelize.**

* **Scaling agent systems** ([Kim et al., 2025][scaling]; [Google Research, 2026][scaling-blog]).
  Across 180 configurations, centralized coordination improved a parallelizable benchmark by
  80.9%, every multi-agent variant degraded a sequential planning benchmark by 39–70%, and
  independent agents amplified errors 17.2×. A predictive model chose the best architecture for
  87% of held-out configurations. Here the choice is made at the level of agent architecture.
* **SquidAgent** ([Lin et al., NeurIPS 2026][squid]) is the closest work to ours. It decides
  *per topological layer* of the task DAG whether to parallelize. A layer is parallelized only when
  its critical-path cost plus the overheads of parallel workers (re-exploration of context and
  alignment of independent outputs) is below the serial cost, with costs measured in *predicted
  output tokens*. It also reduces those overheads by forking workers from the orchestrator's
  session and pre-generating a shared convention block. The authors report a 2.6× mean wall-time
  speedup over Claude Code.
* **TIPEX** ([Xu, Tian & Jiang, ICML 2026][tipex]) separates *structural* parallelism (concurrency
  within one solution path) from *replica* parallelism (several complete solution paths). It finds
  that "overly aggressive parallel strategies do not necessarily yield better performance".
* **M1-Parallel** ([2025][m1par]) runs several multi-agent teams concurrently and stops at the first
  to finish, trading cost for latency.

**Planning parallelizable task graphs.**

* **LLMCompiler** ([Kim et al., ICML 2024][llmcompiler]) plans function calls as a DAG and
  dispatches every ready task immediately, reporting up to 3.7× latency reduction. Our
  `eager_parallel` baseline re-implements its execution policy. It is not the original system.
* **Plan-over-Graph** ([Zhang et al., 2025][pog]) and **GAP** ([Wu et al., 2025][gap]) train models
  to produce parallelizable task graphs, GAP with reinforcement learning. We use an off-the-shelf
  model with structured output and do not study planning quality.

**Scheduling inside the serving engine.** **Parrot** ([Lin et al., OSDI 2024][parrot]),
**Ayo** ([Tan et al., 2025][ayo]) and **Autellix** ([Luo et al., 2025][autellix]) expose
application dependency graphs to the LLM serving system, so it can prioritize critical requests
and programs across many users. **PASTE / B-PASTE** ([2026][paste], [2026][bpaste]) execute tool
calls speculatively; B-PASTE uses a HEFT-inspired critical-path objective. These systems operate
*inside* the inference engine. We schedule *within one request, on the client side*, against a
hosted API whose engine we do not control.

**Classical scheduling.** List scheduling and HEFT (Graham, 1969; Topcuoglu et al., 2002) supply
the critical-path priority (upward rank), the `W/k + (1 − 1/k)·S` bound we verify, and the timing
anomalies that rule out assuming monotonicity. For LLM agent teams, [Tan, Wilson & Lu (2026)][cpteams]
measured critical-path vs FIFO makespan gains of only 0.51–0.96% on normalized-work schedules.
That is consistent with our modest H3 result (−2.4% mean latency).

**What this work adds.** Prior work establishes three things:
* parallelism must be applied selectively ([Kim et al.][scaling], [SquidAgent][squid], [TIPEX][tipex]);
* how to plan and execute task DAGs ([LLMCompiler][llmcompiler], [Plan-over-Graph][pog], [GAP][gap]);
* how to schedule agent programs inside serving engines ([Parrot][parrot], [Ayo][ayo], [Autellix][autellix]).

Our contribution is narrower and mainly empirical:

1. **A wall-clock, distribution-aware decision under provider limits.** A client-side scheduler
   predicts the mean, P95 and cost of every candidate concurrency level by Monte Carlo simulation.
   The simulation models explicit provider concurrency and rate limits, retries and contention,
   and includes exact DP fusion of chained calls. This complements SquidAgent's token-denominated
   per-layer rule.
2. **Calibration against real API calls.** We report how real LLM latency behaves (close to linear
   in output length; variance dominated by output length) and three estimator failure modes found
   in live data, each fixed and confirmed on fresh calls.
3. **Evaluation methodology and a negative result.** Selection, evaluation and oracle use separate
   random seeds, and forecasts are judged by calibration metrics. A simple rule,
   `k = min(provider limit, plan width)` with critical-path priority, captures most of the latency
   benefit.
4. **An open, tested, production-grade implementation** on LangGraph, with a streaming API,
   dashboard and observability.

We have **not** compared against SquidAgent or the original LLMCompiler implementation on shared
benchmarks. That comparison is the most important next step (§8).

## 3. Method

```
request → planner (LLM, structured output; one repair attempt on invalid plans)
        → DAG validation (Kahn, cycle report, transitive reduction, size limits)
        → cost profiles (tokens flow along edges) → Monte Carlo selector
        → async executor (priority queue by upward rank, k workers, retries, failure isolation)
        → synthesis → answer           observed calls ──► online cost-model calibration
```

* **Graph analysis.** Critical path S, total work W, available parallelism W/S, lower bound
  `max(S, W/k)`.
* **Selector.** For each candidate (k ∈ {1, 2, 4, 8, 16} capped at plan width, × fusion on/off) it
  simulates M = 64 executions with a stochastic discrete-event model. The model covers client slots,
  provider cap with rate-limit penalty, contention, heavy tails, retries and failure cancellation.
  The selector minimizes `J = mean + 0.5·P95 + λ·cost` subject to the plan's success probability.
  Success probability is computed analytically (`Π (1 − (1 − p)^(r+1))`), and near-ties go to the
  simplest configuration.
* **Fusion.** An exact DP segments linear chains into single calls when fixed overhead outweighs a
  convex quality penalty. It provably never increases W or S.
* **Executor.** The same `ReadyQueue` drives the simulator and the live executor, so the code that
  is benchmarked is the code that runs. LangGraph's superstep model synchronizes parallel
  branches, so mapping plan levels onto supersteps would *be* the static baseline. The executor
  therefore runs an asyncio worker pool inside a node.

## 4. Experimental setup

**Simulation** (`apsched bench --profile full`).
* **Workloads:** 34 workloads in 7 families (chain, fan-out, fork-join, tree, layered, random
  DAGs, and plans from 10 natural-language requests), with heterogeneous output lengths.
* **Conditions (5):** baseline; tight limit (2 concurrent calls); heavy tail; unreliable
  (4× failures); misestimated (the provider is slower than believed).
* **Strategies (8):** sequential, static level-parallel, eager unbounded, dependency-blind agents
  plus a decision agent, adaptive, and three ablations.
* **Seeds and statistics:** the selector predicts on its own seeds, strategies are scored on 100
  shared seeds, and hindsight optima are scored on a separate 200-run oracle sample. Statistics are
  paired-bootstrap CIs and Wilcoxon tests with Holm correction, reported with effect sizes.

**Live validation** (`apsched live-bench`).
* **Model:** Claude Haiku 5.5 ($0.10 / $0.50 per million input / output tokens).
* **Plans:** the real LLM planner plans each of 10 requests once, and every strategy then executes
  that same plan.
* **Run 1:** 3 repetitions in shuffled strategy order. Repetition 0 uses the prior cost model;
  later repetitions use the online-calibrated one. Every latency is predicted *before* the run.
* **Run 2 (confirmation):** fresh executions with the estimator warm-started on run 1's calls.

## 5. Results: simulation

| condition | sequential | static | eager (LLMCompiler-style) | **adaptive** |
|---|---|---|---|---|
| baseline | 73.6 s | 42.6 s | 40.2 s | **39.9 s** |
| tight provider limit | 73.4 s | 56.6 s | 55.6 s | **48.0 s** |
| heavy-tailed latency | 96.2 s | 61.7 s | 57.6 s | **56.9 s** |
| unreliable provider | 81.4 s | 48.1 s | 45.1 s | **45.0 s** |
| misestimated cost model | 99.0 s | 60.3 s | 57.0 s | **55.5 s** |

* **H1, adaptive vs the best fixed strategy: supported.** Within 5% in 100% of 35
  family × condition cells, better or equal in 66%. The median is 1.8% better on J; the worst
  cell is +3.8%.
* **Provider limits.** At a 2-call limit, adaptive bounds concurrency and is 14% faster than eager
  fan-out (P95 102 s vs 119 s). Level barriers cost static 7–18% versus adaptive in every condition.
* **H2, dependency-blind fan-out: supported (conditionally).** Success is 7.5–22% on dependent
  families and 69–78% on mostly parallel plans, versus ~100% for DAG execution. This assumes a
  per-task correctness of q = 0.8 when inputs are missing; a sensitivity table over q ∈ [0.6, 1]
  is in `RESULTS.md`.
* **H3, critical-path priority: supported, modest.** Versus FIFO: −2.4% mean and −1.9% P95
  latency, largest under tight limits (−3.6%).
* **H4, fusion: supported, modest.** −2.5% calls and −0.8% cost, with unchanged success.
* **Selector accuracy.** It picked the hindsight-optimal configuration in 70% of units and one
  within 5% in 91% (mean regret 1.0%). With a correct cost model, predicted latency is within
  ±1.1%. With a 30% misestimate, predictions are off by about 30%, but rankings, and therefore
  decisions, hold (regret 1.1%).
* **Negative result.** `k = min(provider limit, width)` with critical-path priority matches
  adaptive latency to within 1%.

## 6. Results: live validation on Claude Haiku 5.5

| strategy | run 1: mean / P95 | speedup [95% CI] | run 2: mean / P95 | speedup [95% CI] |
|---|---|---|---|---|
| sequential | 37.3 s / 74.4 s | 1.00× | 39.3 s / 78.7 s | 1.00× |
| static parallel | 23.5 s / 45.2 s | 1.59× [1.48, 1.71] | 24.2 s / 47.6 s | 1.63× [1.41, 1.86] |
| eager parallel | 23.7 s / 44.9 s | 1.57× [1.45, 1.71] | 26.2 s / 45.7 s | 1.50× [1.25, 1.79] |
| **adaptive** | **22.4 s / 42.9 s** | **1.66× [1.53, 1.80]** | **23.8 s / 38.6 s** | **1.65× [1.37, 1.99]** |

* **Plans.** Of the planner's plans, 8 of 10 were DAGs of 4–5 tasks with 2–4 parallel branches
  and depth ≤ 3; 2 were single tasks.
* **Decisions.** The adaptive scheduler chose k = 1 for every single-task plan, k = plan width
  otherwise, and fused a chain 3 times.
* **Success and cost.** Success was 100% in every execution, and cost per execution was equal
  across strategies (≈ $0.0043).
* **Parallel strategies tie.** Real plans are shallow and narrow, and no run hit a rate limit.
  That is the regime where the simulation also predicts near-ties, and the CIs overlap. The robust
  effect is parallel vs sequential.

**How real LLM latency behaves.** Latency is close to linear in output length
(corr = 0.93; ~190 tokens/s end-to-end), with ~0.7–0.85 s fixed overhead. Contention is small:
+1.4% to +3.4% per additional concurrent call. Haiku 5.5, with adaptive thinking on by default,
wrote **4–6× more tokens than the priors assumed** (mean 1,600–2,600 per task). Output length
varies widely within a task type: retrieval tasks had a median of 1,432 tokens and a p90 of 3,441.

**Forecast quality.** Forecasts are evaluated as distributions, because single-run error is
dominated by intrinsic variance.

| estimator | data | aggregate bias | P50 coverage | P95 coverage |
|---|---|---|---|---|
| v1 (as run) | run 1, calibrated reps | +53.6% | 74% | 98% |
| v2, prequential replay | run 1, calibrated reps | +1.4% | 40% | 89% |
| **v2, fresh data** | **run 2 (confirmation)** | **−2.4%** | **45%** | **88%** |

*Ideal values: aggregate bias 0%, P50 coverage 50%, P95 coverage 95%.*

**Diagnosis of v1.** Re-predicting every run with its *actual* token counts and the fitted latency
model gives r = 0.98 between predicted and measured makespan. The DAG, scheduling and latency
structure was therefore right, and the bias had two separable causes:

1. **Output-length estimation:** a fast moving average on heavy-tailed lengths.
2. **Shrinkage that never released a bad prior:** the decayed effective sample size caps at
   1/(1 − decay) = 200, so a decode-speed prior 2.6× too slow kept ~13% weight indefinitely. The
   simulator also added a heavy-tail factor that real latency, given tokens, does not have.

**v2 changes.** All are principled; none were tuned on the data:
* shrinkage on total evidence;
* per-type decayed means with a prior pseudo-count, plus the measured log-spread of output length;
* residual noise measured instead of assumed.

v2 was evaluated prequentially on run 1 and then confirmed on fresh run 2 data.

## 7. Methodological lessons

Each of these was found by the experiments themselves, fixed, and covered by a test:

* **Constraints must be noise-aware.** Filtering candidates on a Monte Carlo success estimate
  with a 1% tolerance (below its ~2% sampling error) excluded good configurations and cost 10%
  latency under the unreliable condition. Success probability is now analytic.
* **Oracles must be large.** Under heavy tails a 30-run mean has a ±30% CI; apparent "24% regret"
  choices were optimal when re-scored on 400 runs.
* **Evaluate distributions, not draws.** A correct model shows a positive median error on
  right-skewed latency. Use aggregate bias and quantile coverage.
* **Simulated ground truth must not depend on beliefs.** Feeding the estimator's beliefs into the
  mock's "true" output lengths made estimates random-walk.
* **Variability lives in output length.** For real LLM calls, uncertainty is mainly *how long* the
  answer is, not noise given its length, so that is where the model should put its variance.

## 8. Limitations and future work

* **No head-to-head comparison with the closest systems.** We did not run SquidAgent or the
  original LLMCompiler code; `eager_parallel` reproduces only LLMCompiler's execution policy. A
  comparison on shared benchmarks with verifiable answers (for example LLMCompiler's ParallelQA,
  HotpotQA or GAIA) is the first priority.

* **Small live scale.** One model, 10 requests and 160 executions. Real plans were narrow (≤ 4
  branches) and never hit rate limits, so the regimes where adaptivity matters most (tight limits,
  wide plans, heterogeneous branches) are shown in simulation, not yet live.
* **Success means execution success.** Answer quality was not graded. H2's quality penalty is an
  assumption backed by the cited study, not measured here; an LLM-judge or verifiable task suite
  is the natural next step.
* **Tails are slightly too narrow.** P95 coverage is 88% against a 95% target. Sampling output
  lengths from the empirical per-type distribution, rather than a pooled log-normal, should close
  the gap.
* **Selector cost and statefulness.** The selector costs 0.1–0.3 s per request, small against
  20–40 s runs, and could be cut with analytic pruning of dominated candidates. Estimator state
  is per process; sharing it across replicas, and a contextual bandit over graph features, are
  natural extensions.

## 9. Reproducibility

```bash
pip install -e ".[dev,anthropic,experiments,api,observability]"
apsched bench --profile full --jobs 4 --out docs/results          # simulation, no API key, ~3 min
apsched live-bench --reps 3 --budget 2 --out docs/results/live     # needs ANTHROPIC_API_KEY
apsched live-bench --reps 1 --warm-start docs/results/live --out docs/results/live_confirm
```

Raw data: `docs/results/runs.csv.gz` (simulation), `docs/results/live*/runs.csv` and `calls.csv`
(every live execution and call), `plans.json`, and `replay_v2.json` (prequential replay).

[scaling]: https://arxiv.org/abs/2512.08296
[scaling-blog]: https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/
[llmcompiler]: https://arxiv.org/abs/2312.04511
[squid]: https://arxiv.org/abs/2610.08647
[tipex]: https://arxiv.org/abs/2608.05791
[m1par]: https://arxiv.org/abs/2507.08944
[pog]: https://arxiv.org/abs/2502.14563
[gap]: https://arxiv.org/abs/2510.25320
[parrot]: https://www.usenix.org/conference/osdi24/presentation/lin-chaofan
[ayo]: https://www.researchgate.net/publication/390315200_Towards_End-to-End_Optimization_of_LLM-based_Applications_with_Ayo
[autellix]: https://arxiv.org/abs/2502.13965
[paste]: https://arxiv.org/abs/2603.18897
[bpaste]: https://arxiv.org/abs/2604.16469
[cpteams]: https://journal.jci.co.id/jitie/article/view/598
