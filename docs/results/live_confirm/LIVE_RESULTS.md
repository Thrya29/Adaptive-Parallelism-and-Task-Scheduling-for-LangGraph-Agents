# Live validation results

Model `claude-haiku-5-5`; 10 requests, 40 executions, 154 task calls; estimated spend $0.1784.

## Strategies (measured)

| strategy | runs | success | mean (s) | p95 (s) | cost ($) | calls | speedup vs seq [95% CI] |
|---|---|---|---|---|---|---|---|
| adaptive | 10 | 1.00 | 23.8 | 38.6 | 0.00425 | 3.7 | 1.65x [1.37, 1.99] |
| sequential | 10 | 1.00 | 39.3 | 78.7 | 0.00432 | 3.9 | 1.00x [1.00, 1.00] |
| static_parallel | 10 | 1.00 | 24.2 | 47.6 | 0.00442 | 3.9 | 1.63x [1.41, 1.86] |
| eager_parallel | 10 | 1.00 | 26.2 | 45.7 | 0.00451 | 3.9 | 1.50x [1.25, 1.79] |

## Does the cost model predict real latency?

Latency is noisy and right-skewed, so the forecast is judged as a distribution: aggregate
bias of the predicted mean (ideal 0%) and coverage of the predicted P50 / P95 (ideal 50% / 95%).
Single-run error is shown for reference only; it is dominated by the intrinsic run-to-run
variance of LLM calls rather than by model error.

| model used for the prediction | runs | aggregate bias | P50 coverage | P95 coverage | single-run abs. error |
|---|---|---|---|---|---|
| prior (before any observation) | 0 | n/a | n/a | n/a | n/a |
| calibrated online | 40 | -2.4% | 45% | 88% | 136% |

Strategy ranking per request (calibrated reps): mean Spearman rho = 0.52; predicted fastest = actual fastest in 38% of requests.

## Calibration

| parameter | prior | fitted from real calls |
|---|---|---|
| time to first token + fixed overhead (s) | 0.60 | 0.85 |
| decode throughput (tokens/s) | 80 | 212 |
| per-call latency error | 164.3% | 17.1% |

Measured contention: each additional concurrent call changed latency by +1.4% (slope of latency / predicted vs. concurrent calls).

## Plans produced by the planner

| request | tasks | width | depth | planning attempts |
|---|---|---|---|---|
| 0 | 5 | 3 | 3 | 1 |
| 1 | 1 | 1 | 1 | 1 |
| 2 | 5 | 4 | 2 | 1 |
| 3 | 5 | 4 | 2 | 1 |
| 4 | 5 | 4 | 2 | 1 |
| 5 | 4 | 3 | 2 | 1 |
| 6 | 4 | 3 | 2 | 1 |
| 7 | 4 | 3 | 2 | 1 |
| 8 | 5 | 3 | 3 | 1 |
| 9 | 1 | 1 | 1 | 1 |
