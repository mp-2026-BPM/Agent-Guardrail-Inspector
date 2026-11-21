# Implementing and Evaluating Process-Centric Guardrails in a Multi-Agent System

This repository accompanies the paper **"Implementing and Evaluating Process-Centric Guardrails in a Multi-Agent System"**, submitted to the GenAI4PM workshop at ICPM 2027.

> Pit Buttchereit, Anton Neubauer, Nikolas Hoellerl, Nina Reznikova, Franka Traupe, Anjo Seidel, Maximilian Körner, Mathias Weske
> Hasso Plattner Institute, University of Potsdam · SAP, Berlin
>
> 📄 Preprint: *TODO link* · 🔖 [Citation](#citation)

## Overview

AI agents can support the execution of business processes, but their autonomous actions may violate business policies or process requirements. This repository implements **process-centric guardrails** for a multi-agent system (MAS) and evaluates how different guardrail configurations affect process execution.

The case study is an **agentic coffee shop** that executes an order-to-pay process, building on Vu et al. (*Agent Behavior Mining*, BPM 2026). Four business agents (Order, Inventory, Barista, Customer Service) collaborate in a LangGraph Swarm and are driven by an LLM-simulated customer. Every tool call, including agent handovers, passes through a **gateway** that evaluates the configured guardrails and returns one of three verdicts: `DENY`, `FLAG` or `ALLOW`:

- **Hard guardrails** are deterministic predicates over tool-call arguments and process state.
- **Soft guardrails** are evaluated by an LLM judge.

Hard guardrails run before soft ones so that a call can be denied before any judge is invoked. A denied call returns an explanation to the agent, which can then recover. The traces are converted into an object-centric event log (OCEL) and analyzed for success, conversation length and token consumption.

![The Coffee Shop Agent Observatory.](assets/main_screen.png)

## Table of Contents

- [Repository Structure](#repository-structure)
- [Configurations](#configurations)
  - [Setups](#setups)
  - [Guardrail and agent definitions](#guardrail-and-agent-definitions)
  - [Scenarios](#scenarios)
  - [Further setups](#further-setups)
- [Results](#results)
- [Setup](#setup)
  - [Reproducing the experiments](#reproducing-the-experiments)
- [Using the System](#using-the-system)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Repository Structure

```
├── config/setups/<setup>/          # One folder per setup (all YAML)
│   ├── agents/                     #   Agent prompts, tools, allowed handovers
│   ├── guardrails/                 #   Guardrail definitions (hard/soft, deny/flag)
│   └── guidelines/                 #   Guidelines given to the agents
├── src/
│   ├── agents/                     # Order, Inventory, Barista, Customer Service, simulated Customer
│   ├── control_plane/              # Gateway and guardrails (see table below)
│   ├── trace_processing/           # MLflow traces → event log CSV / OCEL
│   ├── dashboard/                  # Panel dashboard (Interaction Observatory, Metrics)
│   ├── visualization/              # OC-DFG, OC-PN, object-type mapping
│   ├── simulate.py                 # Headless simulation CLI
│   └── coffee_shop.py              # Main CoffeeShop facade
├── services/coffee_machine/        # Simulated coffee machine (external system)
├── scripts/run_experiment.py       # Experiment sweep: setups × scenarios × conversations
├── analysis/
│   ├── guardrail_effects.ipynb     # Analysis notebook behind the paper's results
│   ├── run_data.py, rate_stats.py  # Data loading and small-sample statistics
│   └── figures/                    # Exported figures
├── experiment_data/2026-11/        # The 1,200 conversations evaluated in the paper
└── tests/                          # unittest suite
```

## Configurations

A **setup** is a self-contained configuration of agents, guardrails and guidelines under `config/setups/<name>/`. Guardrail predicate *logic* lives in [`src/control_plane/predicates.py`](src/control_plane/predicates.py) and is referenced by name from the YAML files.

### Setups

The paper compares two rule sets with four enforcement variants each (8 setups). Within a rule set, agents, prompts, guidelines and the strength of every rule are identical.

| Enforcement variant | Baseline rule set | Strict flow rule set |
|---|---|---|
| **flag**: rules are only logged, never block | [`baseline_flag`](config/setups/baseline_flag/) | [`strict_flow_flag`](config/setups/strict_flow_flag/) |
| **mixed**: mostly hard predicates, some LLM judges | [`baseline`](config/setups/baseline/) | [`strict_flow`](config/setups/strict_flow/) |
| **soft**: every rule enforced by an LLM judge | [`baseline_soft`](config/setups/baseline_soft/) | [`strict_flow_soft`](config/setups/strict_flow_soft/) |
| **hard**: every rule enforced by a deterministic predicate | [`baseline_hard`](config/setups/baseline_hard/) | [`strict_flow_hard`](config/setups/strict_flow_hard/) |

- **Baseline** enforces regular business constraints: order-lifecycle gates on the simulated ERP backend (e.g., no payment or fulfillment before order confirmation), restricted handover targets, and a 30% discount cap.
- **Strict flow** adds per-conversation limits (e.g., a single successful `process_order`), a restricted handover flow after customer service, tighter bounds on order size, total and discount, and checks on price quotes and menu suggestions.

*In the soft variants the hard predicates still run as `flag` shadows. Deterministic call counters stay hard.*

### Where to find the guardrail definitions

| Setup | Guardrails | Agents |
|---|---|---|
| `baseline` | [coffee_shop.yaml](config/setups/baseline/guardrails/coffee_shop.yaml) | [order](config/setups/baseline/agents/order_agent.yaml) · [inventory](config/setups/baseline/agents/inventory_agent.yaml) · [barista](config/setups/baseline/agents/barista_agent.yaml) · [customer service](config/setups/baseline/agents/customer_service_agent.yaml) |
| `baseline_flag` | [coffee_shop.yaml](config/setups/baseline_flag/guardrails/coffee_shop.yaml) | [order](config/setups/baseline_flag/agents/order_agent.yaml) · [inventory](config/setups/baseline_flag/agents/inventory_agent.yaml) · [barista](config/setups/baseline_flag/agents/barista_agent.yaml) · [customer service](config/setups/baseline_flag/agents/customer_service_agent.yaml) |
| `baseline_soft` | [coffee_shop.yaml](config/setups/baseline_soft/guardrails/coffee_shop.yaml) | [order](config/setups/baseline_soft/agents/order_agent.yaml) · [inventory](config/setups/baseline_soft/agents/inventory_agent.yaml) · [barista](config/setups/baseline_soft/agents/barista_agent.yaml) · [customer service](config/setups/baseline_soft/agents/customer_service_agent.yaml) |
| `baseline_hard` | [coffee_shop.yaml](config/setups/baseline_hard/guardrails/coffee_shop.yaml) | [order](config/setups/baseline_hard/agents/order_agent.yaml) · [inventory](config/setups/baseline_hard/agents/inventory_agent.yaml) · [barista](config/setups/baseline_hard/agents/barista_agent.yaml) · [customer service](config/setups/baseline_hard/agents/customer_service_agent.yaml) |
| `strict_flow` | [coffee_shop.yaml](config/setups/strict_flow/guardrails/coffee_shop.yaml) | [order](config/setups/strict_flow/agents/order_agent.yaml) · [inventory](config/setups/strict_flow/agents/inventory_agent.yaml) · [barista](config/setups/strict_flow/agents/barista_agent.yaml) · [customer service](config/setups/strict_flow/agents/customer_service_agent.yaml) |
| `strict_flow_flag` | [coffee_shop.yaml](config/setups/strict_flow_flag/guardrails/coffee_shop.yaml) | [order](config/setups/strict_flow_flag/agents/order_agent.yaml) · [inventory](config/setups/strict_flow_flag/agents/inventory_agent.yaml) · [barista](config/setups/strict_flow_flag/agents/barista_agent.yaml) · [customer service](config/setups/strict_flow_flag/agents/customer_service_agent.yaml) |
| `strict_flow_soft` | [coffee_shop.yaml](config/setups/strict_flow_soft/guardrails/coffee_shop.yaml) | [order](config/setups/strict_flow_soft/agents/order_agent.yaml) · [inventory](config/setups/strict_flow_soft/agents/inventory_agent.yaml) · [barista](config/setups/strict_flow_soft/agents/barista_agent.yaml) · [customer service](config/setups/strict_flow_soft/agents/customer_service_agent.yaml) |
| `strict_flow_hard` | [coffee_shop.yaml](config/setups/strict_flow_hard/guardrails/coffee_shop.yaml) | [order](config/setups/strict_flow_hard/agents/order_agent.yaml) · [inventory](config/setups/strict_flow_hard/agents/inventory_agent.yaml) · [barista](config/setups/strict_flow_hard/agents/barista_agent.yaml) · [customer service](config/setups/strict_flow_hard/agents/customer_service_agent.yaml) |

### Scenarios

The paper names its scenarios A, B and C. In the code they are customer scenarios **1, 3 and 4** in `CUSTOMER_SCENARIO_DEFS` ([`src/agents/customer_agent.py`](src/agents/customer_agent.py)).

| Paper | Code index | Label in code | Milestones (paper, Table 1) |
|---|---|---|---|
| **A** | `1` | Large latte & croissant | order went through · latte on tray · croissant on tray |
| **B** | `3` | Complaint & resolution | remedy started (refund or replacement) · remedy completed |
| **C** | `4` | Ask for recommendation | order went through · first order attempt accepted · item on tray |

### Further setups

The other folders under `config/setups/` were used during development and are **not part of the paper**:

- `unconstrained`: full handover graph, no guardrails, no guidelines.
- `sensible_ranges`: `deny` caps on order size, total, discount and partial refund.
- `narrow_ranges`: much tighter caps.

List all setups with `poetry run simulate --list-setups`.

## Results

All agents use **Ministral 3 14B** (`ollama:ministral-3:14b`). Each of the 8 setups ran 50 conversations per scenario: 150 per setup and **1,200 in total**.

- **Notebook:** [`analysis/guardrail_effects.ipynb`](analysis/guardrail_effects.ipynb). It reads the Parquet tables in `experiment_data/2026-11/analysis/`, so it runs on a plain checkout without the MLflow stores.
- **Data:** [`experiment_data/2026-11/`](experiment_data/2026-11/README.md). This folder holds the three pooled experiment runs, the event log (`_all_traces.csv`), the gateway log, customer feedback, the coffee-machine log and the analysis tables. Its README covers provenance (commit, model, seed), contents and known gaps.

To explore the same data interactively in the Metrics Dashboard:

```bash
METRICS_DATA_DIR=experiment_data/2026-11 poetry run dashboard   # → http://localhost:5006/metrics
```

## Setup

**Requirements**

- Python 3.13
- [Poetry](https://python-poetry.org/) (recommended) or pip
- [Ollama](https://ollama.com/) with `ministral-3:14b`, or another LangChain-supported provider

```bash
git clone -b genai4pm https://github.com/mp-2026-BPM/Agent-Guardrail-Inspector.git
cd Agent-Guardrail-Inspector

poetry install
cp .env.example .env              # LLM_PROVIDER=ollama (default)
ollama pull ministral-3:14b
```

The LLM provider is configured in `.env` (see [`.env.example`](.env.example)).

To run the analysis notebook you also need Jupyter:

```bash
poetry run pip install jupyter
poetry run jupyter lab analysis/guardrail_effects.ipynb
```

### Reproducing the experiments

[`scripts/run_experiment.py`](scripts/run_experiment.py) runs the paper's grid: 8 setups × scenarios 1, 3, 4. It starts one isolated `simulate` process per cell and resets the database and coffee-machine service before every cell (seed 100).

```bash
poetry run python scripts/run_experiment.py --dry-run                                  # print the grid
poetry run python scripts/run_experiment.py --setups baseline --scenarios 1 --count 2  # pilot
poetry run python scripts/run_experiment.py --count 50                                 # full grid, 50 per cell
poetry run python scripts/run_experiment.py --resume experiment_runs/<dir>             # continue after an interrupt
```

The paper's data was collected in three runs of 9, 16 and 25 conversations per cell, all at the same commit (see the [data README](experiment_data/2026-11/README.md)). LLM outputs are stochastic, so a rerun will not reproduce the numbers exactly.

While a sweep runs:

- **Don't run the test suite.** `tests/__init__.py` deletes `coffee_shop.db` at import time.
- **Join on `case_id`, not `order_id`.** Order IDs restart at `ORD0001` in every cell.

## Using the System

### Headless simulation

```bash
poetry run simulate --setup baseline --traces 10 --scenario all --export-logs
poetry run simulate --batches baseline:1:10 baseline_hard:4:10    # setup:scenario:traces
poetry run simulate --help
```

### Dashboard

```bash
poetry run dashboard    # http://localhost:5006
```

- **Interaction Observatory (`/`):** the four business agents live, with tool-call logs, handoff context and the message history each agent sees. You can switch the customer between simulated and manual.
- **Metrics Dashboard (`/metrics`):** KPIs, guardrail decisions, per-agent workload and OCEL visualizations (object-type mapping, OC-DFG, OC-PN).

Reset local traces with `poetry run reset-traces` (`--dry-run` to preview).

### Tests

```bash
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@inproceedings{buttchereit2027guardrails,
  title     = {Implementing and Evaluating Process-Centric Guardrails in a Multi-Agent System},
  author    = {Buttchereit, Pit and Neubauer, Anton and Hoellerl, Nikolas and Reznikova, Nina and
               Traupe, Franka and Seidel, Anjo and K{\"o}rner, Maximilian and Weske, Mathias},
  booktitle = {TODO: GenAI4PM Workshop @ ICPM 2027},
  year      = {2027}
}
```

## Acknowledgements

The coffee shop builds on the agentic coffee shop by Vu et al. (*Agent Behavior Mining: Generative AI Agent Governance in Business Processes*, BPM 2026). We thank Michael Perscheid and Gregor Berg from SAP SE, and Charlotte Maude Balcke and Raban Giessler for their contributions to this research project.
