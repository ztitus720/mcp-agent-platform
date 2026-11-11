# Evaluation report

- Run at: 2026-11-11 21:50
- Model: `qwen3.7-plus`  |  Cases: 24  |  Rounds: 3
- Raw results: `eval/results/20261111-214230.jsonl`
- Infra re-runs: 1 (runs repeated because the model API was unavailable or returned an empty answer)

## Metrics

| Metric | Result | Formula |
|---|---|---|
| Tool-call accuracy | 71/72 = 98.6% | runs whose tool calls include every expected tool and no forbidden tool / runs that declare expected or forbidden tools |
| Task completion | 72/72 = 100.0% | runs that did not crash and whose final answer passes must_contain_all / must_contain_any / must_not_contain / all runs |
| Degradation success | 18/18 = 100.0% | exception runs that did not crash and gave a non-empty answer passing its checks / exception runs |

## By category

| Category | Runs | Tool accuracy | Task completion | Avg seconds |
|---|---|---|---|---|
| calc | 18 | 18/18 = 100.0% | 18/18 = 100.0% | 4.4 |
| exception | 18 | 17/18 = 94.4% | 18/18 = 100.0% | 5.9 |
| retrieval | 24 | 24/24 = 100.0% | 24/24 = 100.0% | 8.1 |
| search | 12 | 12/12 = 100.0% | 12/12 = 100.0% | 6.3 |

## Failures

- `exc-06` (round 3): forbidden tool called: ['search_web_search']
