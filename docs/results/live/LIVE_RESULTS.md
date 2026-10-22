# Live validation results

Model `claude-haiku-5-5`; 10 requests, 120 executions, 455 task calls; estimated spend $0.5200.

## Strategies (measured)

| strategy | runs | success | mean (s) | p95 (s) | cost ($) | calls | speedup vs seq [95% CI] |
|---|---|---|---|---|---|---|---|
| adaptive | 30 | 1.00 | 22.4 | 42.9 | 0.00427 | 3.8 | 1.66x [1.53, 1.80] |
| sequential | 30 | 1.00 | 37.3 | 74.4 | 0.00421 | 3.8 | 1.00x [1.00, 1.00] |
| static_parallel | 30 | 1.00 | 23.5 | 45.2 | 0.00436 | 3.8 | 1.59x [1.48, 1.71] |
| eager_parallel | 30 | 1.00 | 23.7 | 44.9 | 0.00438 | 3.8 | 1.57x [1.45, 1.71] |

## Does the cost model predict real latency?

Latency is noisy and right-skewed, so the forecast is judged as a distribution: aggregate
bias of the predicted mean (ideal 0%) and coverage of the predicted P50 / P95 (ideal 50% / 95%).
Single-run error is shown for reference only; it is dominated by the intrinsic run-to-run
variance of LLM calls rather than by model error.

| model used for the prediction | runs | aggregate bias | P50 coverage | P95 coverage | single-run abs. error |
|---|---|---|---|---|---|
| prior (before any observation) | 40 | +33.9% | 57% | 92% | 128% |
| calibrated online | 80 | +53.6% | 74% | 98% | 164% |

Strategy ranking per request (calibrated reps): mean Spearman rho = 0.45; predicted fastest = actual fastest in 25% of requests.

## Calibration

| parameter | prior | fitted from real calls |
|---|---|---|
| time to first token + fixed overhead (s) | 0.60 | 0.69 |
| decode throughput (tokens/s) | 80 | 184 |
| per-call latency error | 168.5% | 28.8% |

Measured contention: each additional concurrent call changed latency by +3.4% (slope of latency / predicted vs. concurrent calls).

## Plans produced by the planner

| request | tasks | width | depth | planning attempts |
|---|---|---|---|---|
| 0 | 5 | 3 | 3 | 1 |
| 1 | 1 | 1 | 1 | 1 |
| 2 | 5 | 4 | 2 | 1 |
| 3 | 4 | 2 | 3 | 1 |
| 4 | 5 | 4 | 2 | 1 |
| 5 | 4 | 3 | 2 | 1 |
| 6 | 4 | 3 | 2 | 1 |
| 7 | 4 | 3 | 2 | 1 |
| 8 | 5 | 3 | 3 | 1 |
| 9 | 1 | 1 | 1 | 1 |
