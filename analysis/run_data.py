"""Load experiment runs into per-conversation and per-tool-call tables.

An experiment run folder is what `scripts/run_experiment.py` leaves behind on
the simulation machine: `mlflow.db` (traces and spans) plus
`experiment_runs/<run_id>/ledger.json` (grid and provenance). Every
conversation is one MLflow session; its traces carry the `setup` and
`scenario_index` tags, and every executed tool call is a TOOL span inside one
of those traces.

Tool calls a guardrail denied never reach the tool, so they have no TOOL span:
the tool table only holds calls that actually ran.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

ENFORCEMENT_SUFFIXES = {"_flag": "flag", "_soft": "soft", "_hard": "hard"}


def split_setup(setup: str) -> tuple[str, str]:
    """`strict_flow_soft` -> (`strict_flow`, `soft`); `baseline` -> (`baseline`, `normal`)."""
    for suffix, enforcement in ENFORCEMENT_SUFFIXES.items():
        if setup.endswith(suffix):
            return setup.removesuffix(suffix), enforcement
    return setup, "normal"


def load_ledger(run_dir: Path) -> dict:
    [ledger_path] = sorted(Path(run_dir).glob("experiment_runs/*/ledger.json"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["run_id"] = ledger_path.parent.name
    return ledger


def _connect(run_dir: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(run_dir) / 'mlflow.db'}?mode=ro", uri=True)


def load_conversations(run_dir: Path, run: str) -> pd.DataFrame:
    """One row per conversation (MLflow session) with its setup and scenario."""
    with _connect(run_dir) as con:
        df = pd.read_sql_query(
            """
            select m.value as thread_id,
                   s.value as setup,
                   cast(c.value as integer) as scenario,
                   min(i.timestamp_ms) as started_ms,
                   count(distinct m.request_id) as traces
            from trace_request_metadata m
            join trace_tags s on s.request_id = m.request_id and s.key = 'setup'
            join trace_tags c on c.request_id = m.request_id and c.key = 'scenario_index'
            join trace_info i on i.request_id = m.request_id
            where m.key = 'mlflow.trace.session'
            group by m.value, s.value, c.value
            """,
            con,
        )
    if df.thread_id.duplicated().any():
        raise ValueError(f"{run}: a session carries more than one setup/scenario tag")
    df[["family", "enforcement"]] = df.setup.apply(lambda s: pd.Series(split_setup(s)))
    df.insert(0, "run", run)
    df["started"] = pd.to_datetime(df.pop("started_ms"), unit="ms", utc=True)
    return df.sort_values("started").reset_index(drop=True)


def _parse_json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def load_tool_calls(run_dir: Path, run: str) -> pd.DataFrame:
    """One row per executed tool call, with parsed inputs and result."""
    with _connect(run_dir) as con:
        df = pd.read_sql_query(
            """
            select m.value as thread_id,
                   s.name as tool,
                   s.start_time_unix_nano as start_ns,
                   json_extract(s.content, '$.attributes."mlflow.spanInputs"') as inputs_json,
                   json_extract(s.content, '$.attributes."mlflow.spanOutputs"') as outputs_json
            from spans s
            join trace_request_metadata m
              on m.request_id = s.trace_id and m.key = 'mlflow.trace.session'
            where s.type = 'TOOL'
            """,
            con,
        )
    df.insert(0, "run", run)
    df["inputs"] = df.pop("inputs_json").map(lambda t: _parse_json(t) or {})
    outputs = df.pop("outputs_json").map(lambda t: _parse_json(t) or {})
    # The span output is the ToolMessage; its `content` is the tool's return value.
    df["result_text"] = outputs.map(
        lambda o: o.get("content") if isinstance(o, dict) else None
    )
    df["result"] = df.result_text.map(_parse_json)
    return df.sort_values(["thread_id", "start_ns"]).reset_index(drop=True)


def load_runs(
    run_dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load several run folders; returns (runs, conversations, tool_calls)."""
    runs, conversations, tool_calls = [], [], []
    for run, run_dir in run_dirs.items():
        ledger = load_ledger(run_dir)
        provenance = ledger["provenance"]
        runs.append(
            {
                "run": run,
                "run_id": ledger["run_id"],
                "commit": provenance["git"]["commit"][:7]
                + ("+dirty" if provenance["git"].get("dirty") else ""),
                "model": f"{provenance['llm_provider']}:{provenance['llm_model']}",
                "machine_seed": provenance.get("machine_seed"),
                "setups": len(ledger["grid"]["setups"]),
                "scenarios": ledger["grid"]["scenarios"],
                "per_cell": ledger["grid"]["count"],
                "cells_done": sum(
                    c["status"] == "done" for c in ledger["cells"].values()
                ),
                "cells": len(ledger["cells"]),
            }
        )
        conversations.append(load_conversations(run_dir, run))
        tool_calls.append(load_tool_calls(run_dir, run))
    return (
        pd.DataFrame(runs),
        pd.concat(conversations, ignore_index=True),
        pd.concat(tool_calls, ignore_index=True),
    )
