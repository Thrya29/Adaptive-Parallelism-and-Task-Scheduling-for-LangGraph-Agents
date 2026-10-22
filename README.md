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
| 1 | Core algorithms: DAG model, validation, critical path, list scheduler, task fusion | ⏳ next |
| 2 | Cost model, mock LLM, simulator, Monte Carlo strategy selector | planned |
| 3 | LangGraph planner / executor / synthesizer, Anthropic backend, reliability | planned |
| 4 | Benchmark suite and experiment harness (simulation) | planned |
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
src/adaptive_scheduler/   library code (config, CLI; more modules land per phase)
tests/                    unit, property-based and integration tests
docs/                     architecture and research report
.github/workflows/        CI
```

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
