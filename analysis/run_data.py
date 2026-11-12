"""Load experiment runs into per-conversation and per-tool-call tables.

An experiment run folder is what `scripts/run_experiment.py` leaves behind on
the simulation machine: `mlflow.db` (traces and spans) plus
`experiment_runs/<run_id>/ledger.json` (grid and provenance). Every
conversation is one MLflow session; its traces carry the `setup` and
`scenario_index` tags, and every executed tool call is a TOOL span inside one
of those traces.

Tool calls a guardrail denied never reach the tool, so they have no TOOL span:
the tool table only holds calls that actually ran. The gateway's own record of
every checked call (allow / flag / deny) is in `guardrail_log/events.jsonl`.

LLM calls are CHAT_MODEL spans in the same traces. Their `langgraph_node`
tells the agents' own calls (`llm`) apart from soft-guardrail judge calls
(`gateway` for tool calls, `response_gateway` for agent messages). The customer
simulator and the feedback call run outside the conversation's traces and are
not included.

The MLflow stores are several GB and stay on the machines that ran the
experiments. `save_snapshot` writes the loaded tables as Parquet files (a few
MB) that are shared in the repository, and `load_snapshot` reads them back
unchanged, so the analysis runs without the stores.
"""

import json
import sqlite3
from pathlib import Path
from typing import NamedTuple

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
                   max(i.timestamp_ms + coalesce(i.execution_time_ms, 0)) as ended_ms,
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
    # One trace per customer message; duration runs from the first message to the
    # end of the last agent reply (customer think time included, feedback excluded).
    df["duration_s"] = (df.ended_ms - df.started_ms) / 1000
    df["started"] = pd.to_datetime(df.pop("started_ms"), unit="ms", utc=True)
    df.pop("ended_ms")
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


LLM_ROLES = {"llm": "agent", "gateway": "judge", "response_gateway": "judge"}


def load_llm_calls(run_dir: Path, run: str) -> pd.DataFrame:
    """One row per LLM call inside a conversation, with its role and token usage."""
    with _connect(run_dir) as con:
        df = pd.read_sql_query(
            """
            select m.value as thread_id,
                   json_extract(json_extract(s.content, '$.attributes.metadata'), '$.langgraph_node') as node,
                   s.start_time_unix_nano as start_ns,
                   (s.end_time_unix_nano - s.start_time_unix_nano) / 1e9 as seconds,
                   json_extract(s.content, '$.attributes."mlflow.chat.tokenUsage"') as usage_json
            from spans s
            join trace_request_metadata m
              on m.request_id = s.trace_id and m.key = 'mlflow.trace.session'
            where s.type = 'CHAT_MODEL'
            """,
            con,
        )
    df.insert(0, "run", run)
    df["role"] = df.node.map(LLM_ROLES).fillna("other")
    usage = df.pop("usage_json").map(lambda t: _parse_json(t) or {})
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        df[key] = usage.map(lambda u: u.get(key, 0)).astype(int)
    return df.sort_values(["thread_id", "start_ns"]).reset_index(drop=True)


def load_gateway_decisions(run_dir: Path, run: str) -> pd.DataFrame:
    """One row per gateway check (tool call or assistant message), with its decision.

    `assistant_message` is the response gateway's pseudo-tool. Fallback events
    (the retry cap was hit) have no decision and are kept with `event_type`.
    """
    records = []
    with (Path(run_dir) / "guardrail_log" / "events.jsonl").open(encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            records.append(
                {
                    "thread_id": event.get("thread_id"),
                    "event_type": event.get("event_type"),
                    "agent_id": event.get("agent_id"),
                    "tool": event.get("tool_name"),
                    "decision": event.get("final_decision"),
                    # Rules that denied or flagged this call, with their type.
                    "violations": [
                        {
                            "guardrail": v["guardrail_name"],
                            "type": v.get("guardrail_type"),
                            "effect": v["effect"],
                        }
                        for v in event.get("verdicts", [])
                        if v.get("effect") in ("deny", "flag")
                    ],
                    "ts": event.get("ts"),
                }
            )
    df = pd.DataFrame(records)
    df.insert(0, "run", run)
    return df


class RunData(NamedTuple):
    runs: pd.DataFrame
    conversations: pd.DataFrame
    tool_calls: pd.DataFrame
    gateway: pd.DataFrame
    llm_calls: pd.DataFrame


def load_runs(run_dirs: dict[str, Path]) -> RunData:
    """Load several run folders into one table per kind of record."""
    runs, conversations, tool_calls, gateway, llm_calls = [], [], [], [], []
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
        gateway.append(load_gateway_decisions(run_dir, run))
        llm_calls.append(load_llm_calls(run_dir, run))
    return RunData(
        pd.DataFrame(runs),
        pd.concat(conversations, ignore_index=True),
        pd.concat(tool_calls, ignore_index=True),
        pd.concat(gateway, ignore_index=True),
        pd.concat(llm_calls, ignore_index=True),
    )


# Columns holding parsed JSON (dicts, lists, None); Parquet stores them as JSON text.
_JSON_COLUMNS = {
    "runs": ["scenarios"],
    "tool_calls": ["inputs", "result"],
    "gateway": ["violations"],
}


def save_snapshot(data: RunData, snapshot_dir: Path) -> None:
    """Write every table of `data` to `<snapshot_dir>/<table>.parquet`."""
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in RunData._fields:
        df = getattr(data, name).copy()
        for col in _JSON_COLUMNS.get(name, []):
            df[col] = df[col].map(json.dumps)
        df.to_parquet(snapshot_dir / f"{name}.parquet", index=False)


def load_snapshot(snapshot_dir: Path) -> RunData:
    """Read the tables written by `save_snapshot`."""
    tables = {}
    for name in RunData._fields:
        df = pd.read_parquet(Path(snapshot_dir) / f"{name}.parquet")
        for col in _JSON_COLUMNS.get(name, []):
            df[col] = df[col].map(json.loads)
        tables[name] = df
    return RunData(**tables)
