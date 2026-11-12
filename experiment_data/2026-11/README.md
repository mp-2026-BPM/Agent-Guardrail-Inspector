# Experiment runs, October–November 2026

The three experiment runs the analysis in [analysis/guardrail_effects.ipynb](../../analysis/guardrail_effects.ipynb) is based on, joined into one dataset that the metrics dashboard can open.

| Run | Date | Per cell | Conversations |
|---|---|---|---|
| `20261028-155640` | 28 Oct 2026 | 9 | 216 |
| `20261029-185412` | 29–30 Oct 2026 | 16 | 384 |
| `20261106-095238` | 6–7 Nov 2026 | 25 | 600 |
| **Total** | | **50** | **1,200** |

All three runs used the same code (commit `aca2f36`, no local changes), model (`ollama:ministral-3:14b`, context 32,768) and machine seed (100), so they can be pooled. The grid is 8 setups (`baseline`, `strict_flow`, each plain and as `_flag`, `_soft`, `_hard`) × 3 scenarios (1 latte + croissant, 3 complaint, 4 recommendation).

## Open it in the dashboard

```bash
METRICS_DATA_DIR=experiment_data/2026-11 poetry run dashboard
```

Then open http://localhost:5006/metrics. With `METRICS_DATA_DIR` set, the Metrics page reads this folder as it is and doesn't sync with your local MLflow store, so your own runs are neither shown nor mixed in.

## Contents

| Path | What it holds |
|---|---|
| `generated_event_log/_all_traces.csv` | The dashboard's event log for all 1,200 conversations (62,822 events), built by the dashboard's own trace cache from the joined MLflow store. Includes agent, tool and LLM events, gateway decisions, customer feedback and coffee-machine events. |
| `guardrail_log/events.jsonl` | Every gateway check (21,159 records: 21,085 decisions and 74 retry-cap fallbacks). Each record has `experiment_source`, `experiment_run` and `experiment_commit`. |
| `feedback_store.json` | The simulated customer's feedback per conversation (1,178 entries). |
| `services/coffee_machine/logs/coffee_machine.csv` | Coffee-machine events (3,005 rows). |
| `experiment_runs/<run>/` | Each run's ledger (grid, provenance, per-cell timing) and per-cell console logs. |

`case_id` in the event log is the conversation's LangGraph thread id; `case_setup` and `case_scenario_index` identify its cell. Order IDs restart at `ORD0001` in every cell, so don't join on `order_id` alone.

## Not included

The joined MLflow store (`mlflow.db`, 3.6 GB, 16,519 traces) and its trace artifacts are too large for the repository. Everything the dashboard and the notebook's per-conversation tables need is derived from it, but the notebook itself loads the MLflow stores of the individual runs, so re-running it needs those (ask Anton Neubauer).

## Known gaps

- 13 conversations of run `20261106-095238` ended with an Ollama "tool … not found" error (the model called a tool its agent doesn't have). They are in the data but have no feedback; 22 conversations across all runs have no feedback.
- 2 traces produced no agent events in the event log (only the customer's message).
- The free-text columns contain the simulated customer's messages and the agents' replies. They are synthetic; the email addresses in them are made up by the models.
