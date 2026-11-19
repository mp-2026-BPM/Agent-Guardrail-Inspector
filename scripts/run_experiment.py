#!/usr/bin/env python3
"""Run a setup x scenario experiment grid as a sequence of isolated `simulate` cells.

One cell is one (setup, scenario) pair run as its own `simulate` process. Before
every cell the coffee-shop SQLite is deleted and the coffee-machine service is
restarted, so cells never share order history, stock drift, or brew-failure RNG.
Inventory is already reset before every single conversation by the simulator
itself, so no extra work is needed for that.

MLflow traces, `guardrail_log/events.jsonl` and `feedback_store.json` are the
experiment results and are deliberately never touched — they accumulate across
the whole run and stay separable via the `setup` / `scenario_index` trace tags.

Progress is journalled to `<run-dir>/ledger.json` after every cell, so an
interrupted sweep can be continued with `--resume <run-dir>`. Resume tops cells
up rather than restarting them: a cell that recorded 21 of 25 conversations runs
4 more and moves on. A cell counts as done only once it holds its full quota —
exiting 0 is not enough, since a cell whose conversations all failed exits 0 too.

    python scripts/run_experiment.py --dry-run
    python scripts/run_experiment.py --setups baseline --scenarios 1 --count 2
    python scripts/run_experiment.py
    python scripts/run_experiment.py --resume experiment_runs/20260731-224500
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import psutil
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.barista_agent import COFFEE_MACHINE_PORT, COFFEE_MACHINE_URL  # noqa: E402
from src.agents.customer_agent import CUSTOMER_SCENARIO_LABELS  # noqa: E402
from src.coffee_shop import LOG_DATE_FORMAT  # noqa: E402
from src.config import CoffeeShopConfig  # noqa: E402
from src.control_plane import AgentRepo, Catalog  # noqa: E402
from src.llm import create_chat_llm  # noqa: E402 — also loads .env
from src.setups import setup_dir  # noqa: E402
from src.trace_processing.mlflow_sqlite import (  # noqa: E402
    sqlite_path_from_uri,
    sqlite_trace_count,
)

DEFAULT_SETUPS = (
    "baseline",
    "baseline_flag",
    "baseline_soft",
    "baseline_hard",
    "strict_flow",
    "strict_flow_flag",
    "strict_flow_soft",
    "strict_flow_hard",
)
# Scenario 2 ("2 espressos (hurry)") is left out of the default grid; pass
# --scenarios to include it.
DEFAULT_SCENARIOS = (1, 3, 4)
DEFAULT_COUNT = 25
DEFAULT_MACHINE_SEED = 100

DASHBOARD_PORT = 5006
RUNS_ROOT = PROJECT_ROOT / "experiment_runs"
LEDGER_NAME = "ledger.json"

# `simulate` writes its banners at a custom STATUS level and errors at ERROR.
# Its lines are "[<timestamp>] [<LEVEL>] ..." (LOG_FORMAT in src/coffee_shop.py),
# so match the level bracket after the leading timestamp bracket to pick the
# interesting lines out of the child's stream for the console.
CONSOLE_LINE_RE = re.compile(r"^(?:\[[^\]]*\]\s*)?\[(?:STATUS|ERROR|CRITICAL)")


@dataclass(frozen=True)
class Cell:
    index: int
    setup: str
    scenario: int
    count: int

    @property
    def key(self) -> str:
        return f"{self.setup}:s{self.scenario}"

    @property
    def slug(self) -> str:
        return f"{self.index:02d}_{self.setup}_s{self.scenario}"


def build_cells(setups: list[str], scenarios: list[int], count: int) -> list[Cell]:
    """Build the grid setup-major, so the run walks the table row by row."""
    return [
        Cell(index=i, setup=setup, scenario=scenario, count=count)
        for i, (setup, scenario) in enumerate(
            ((s, sc) for s in setups for sc in scenarios), start=1
        )
    ]


def coffee_shop_db_paths() -> list[Path]:
    """The SQLite triplet the order store writes, honouring COFFEE_SHOP_DB."""
    db = Path(os.environ.get("COFFEE_SHOP_DB", PROJECT_ROOT / "coffee_shop.db"))
    return [db, Path(f"{db}-wal"), Path(f"{db}-shm")]


def wipe_coffee_shop_db() -> None:
    """Delete the coffee-shop DB. Safe here because no simulate process is
    alive — deleting an open SQLite file only unlinks the inode, and a live
    writer would silently recreate it at the next checkpoint."""
    for path in coffee_shop_db_paths():
        path.unlink(missing_ok=True)


def port_holders(port: int) -> list[psutil.Process]:
    """Return the processes listening on `port`.

    `lsof` first: `psutil.net_connections` needs root on macOS and otherwise
    reports nothing at all for another process's socket, which would make both
    the dashboard guard and the machine restart silently no-op.
    """
    pids: set[int] = set()
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        pids.update(int(token) for token in result.stdout.split() if token.isdigit())
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    if not pids:
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if (
                    conn.status == psutil.CONN_LISTEN
                    and conn.laddr
                    and conn.laddr.port == port
                    and conn.pid is not None
                ):
                    pids.add(conn.pid)
        except (psutil.AccessDenied, PermissionError):
            pass

    holders: list[psutil.Process] = []
    for pid in sorted(pids):
        try:
            holders.append(psutil.Process(pid))
        except psutil.NoSuchProcess:
            pass
    return holders


def stop_machine(timeout: float = 10.0) -> None:
    """Terminate whatever is listening on the coffee-machine port.

    Kills by port rather than by handle: a machine left over from an earlier
    run (the barista starts one lazily and nothing ever stops it) is exactly
    what we need to clear, and it is not our child.
    """
    holders = port_holders(COFFEE_MACHINE_PORT)
    if not holders:
        return
    for proc in holders:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(holders, timeout=timeout)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=timeout)

    deadline = time.monotonic() + timeout
    while port_holders(COFFEE_MACHINE_PORT) and time.monotonic() < deadline:
        time.sleep(0.2)


def start_machine(log_path: Path, seed: int, timeout: float = 30.0) -> subprocess.Popen:
    """Start a fresh coffee-machine service and wait for it to answer.

    Output goes to a file, never to a pipe: nothing drains the machine's
    stdout, and a full pipe buffer would wedge the service mid-cell.
    """
    env = dict(os.environ, COFFEE_MACHINE_SEED=str(seed))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as handle:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "services.coffee_machine.main:app",
                "--port",
                str(COFFEE_MACHINE_PORT),
                "--host",
                "127.0.0.1",
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"coffee machine exited immediately (code {proc.returncode}) — see {log_path}"
            )
        try:
            if requests.get(f"{COFFEE_MACHINE_URL}/healthz", timeout=2).ok:
                return proc
        except requests.RequestException:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError(
        f"coffee machine did not come up within {timeout:.0f}s — see {log_path}"
    )


def mlflow_db_path() -> Path | None:
    """Path behind the configured MLflow tracking URI, if it is sqlite-backed."""
    path = sqlite_path_from_uri(CoffeeShopConfig.mlflow_tracking_uri)
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def trace_stats() -> tuple[int, int]:
    """Return (trace count, conversation count) currently in the MLflow store.

    MLflow records one trace per conversation turn, so traces alone say little
    about progress. Conversations are counted as distinct LangGraph sessions
    among traces carrying a `setup` tag: only `ConversationEngine.send_message`
    tags traces, which excludes the untagged single-trace sessions that
    guardrail subgraphs and standalone judge calls leave behind — counting
    every session inflates the number several-fold.
    """
    path = mlflow_db_path()
    if path is None or not path.exists():
        return 0, 0
    try:
        traces = sqlite_trace_count(path)
        with closing(
            sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        ) as conn:
            (conversations,) = conn.execute(
                """
                SELECT COUNT(DISTINCT trm.value)
                  FROM trace_tags tt
                  JOIN trace_request_metadata trm
                    ON trm.request_id = tt.request_id
                   AND trm.key = 'mlflow.trace.session'
                 WHERE tt.key = 'setup'
                """
            ).fetchone()
    except (sqlite3.Error, OSError):
        return 0, 0
    return traces, int(conversations)


def git_provenance() -> dict:
    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def llm_identity() -> tuple[str, str]:
    """The provider/model `simulate` will actually use (src.llm loaded .env)."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
    if provider == "anthropic":
        return provider, os.getenv("ANTHROPIC_MODEL", "anthropic--claude-4.6-opus")
    return provider, os.getenv("OLLAMA_MODEL", "ministral-3:14b")


def validate_setup(setup: str) -> None:
    """Resolve every guardrail and guideline id each agent references.

    `setup_dir` only proves the directories exist. An agent YAML naming an id
    that its guardrails file does not define raises KeyError deep inside
    `build_coffee_shop_graph` — which, unchecked, burns one failed cell per
    scenario before anyone notices.
    """
    config_dir = setup_dir(setup)
    repo = AgentRepo(config_dir)
    catalog = Catalog(config_dir)
    for agent_id in repo.ids():
        definition = repo.get(agent_id)
        try:
            catalog.guardrails(list(definition.guardrail_ids))
            catalog.guidelines(list(definition.guideline_ids))
        except KeyError as exc:
            raise RuntimeError(f"setup {setup!r}, agent {agent_id!r}: {exc}") from exc


def preflight(cells: list[Cell], skip_llm_probe: bool) -> None:
    """Fail before any LLM work rather than hours into the sweep."""
    for setup in sorted({cell.setup for cell in cells}):
        validate_setup(setup)

    holders = port_holders(DASHBOARD_PORT)
    if holders:
        pids = ", ".join(str(p.pid) for p in holders)
        raise RuntimeError(
            f"the dashboard is listening on port {DASHBOARD_PORT} (pid {pids}). "
            "Stop it first — a live writer recreates the coffee-shop DB we delete "
            "between cells."
        )

    provider, model = llm_identity()
    print(f"LLM backend: provider={provider} model={model}")
    if skip_llm_probe:
        return
    try:
        create_chat_llm().invoke("ping")
    except Exception as exc:
        raise RuntimeError(
            f"LLM backend {provider}/{model} is not usable ({type(exc).__name__}: {exc}). "
            "Fix .env or start the backend, or pass --skip-llm-probe."
        ) from exc
    print("LLM backend responded.")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp() -> str:
    """Local timestamp for the driver's own progress lines, in the same shape
    the simulate cells log with."""
    return datetime.now().strftime(LOG_DATE_FORMAT)


def save_ledger(run_dir: Path, ledger: dict) -> None:
    tmp = run_dir / f"{LEDGER_NAME}.tmp"
    tmp.write_text(json.dumps(ledger, indent=2))
    tmp.replace(run_dir / LEDGER_NAME)


def load_ledger(run_dir: Path) -> dict:
    path = run_dir / LEDGER_NAME
    if not path.exists():
        raise RuntimeError(f"no {LEDGER_NAME} in {run_dir} — nothing to resume")
    return json.loads(path.read_text())


def run_cell(
    cell: Cell, log_path: Path, verbose: bool, log_level: str, count: int
) -> int:
    """Run `count` conversations of one cell as its own `simulate` process.

    `count` is the number still owed, which is below `cell.count` whenever a
    previous attempt got part of the way. The log is appended to, so a topped-up
    cell keeps the record of every attempt.
    """
    cmd = [
        sys.executable,
        "-m",
        "src.simulate",
        "--batches",
        f"{cell.setup}:{cell.scenario}:{count}",
        "--on-error",
        "skip",
        "--log-level",
        log_level,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        with open(log_path, "a", buffering=1) as handle:
            handle.write(
                f"\n===== {stamp()} | {cell.key} | {count} conversation(s) =====\n"
            )
            for line in proc.stdout:
                handle.write(line)
                if verbose or CONSOLE_LINE_RE.match(line):
                    print(line, end="")
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise


def export_event_log() -> None:
    """Build the event log once, at the end.

    Never per cell: the processor re-extracts every trace in the store on each
    call, so exporting 24 times over a growing store is quadratic work.
    """
    from src.trace_processing import TraceProcessor

    TraceProcessor().process_all_traces()


def print_plan(cells: list[Cell], log_level: str) -> None:
    total = sum(cell.count for cell in cells)
    print(f"{len(cells)} cell(s), {total} conversation(s):\n")
    for cell in cells:
        label = CUSTOMER_SCENARIO_LABELS[cell.scenario]
        print(
            f"  {cell.index:>2}. {cell.setup:<18} scenario {cell.scenario} "
            f"({label:<24}) x{cell.count}"
        )
    print(
        "\nper cell: wipe coffee_shop.db -> restart coffee machine -> "
        f"{sys.executable} -m src.simulate --batches <setup>:<scenario>:<count> "
        f"--on-error skip --log-level {log_level}"
    )


def outstanding(cell: Cell, entry: dict) -> int:
    """Conversations still owed for `cell`, given its ledger entry.

    `status: done` is authoritative and beats the arithmetic: it is how you
    tell a resume you are satisfied with a cell that came up short, whether the
    driver wrote it or you edited the ledger by hand.
    """
    if entry.get("status") == "done":
        return 0
    return max(0, cell.count - max(0, int(entry.get("conversations", 0))))


def print_summary(cells: list[Cell], ledger: dict) -> None:
    header = f"{'cell':<28}{'status':<12}{'conv':>9}{'traces':>8}{'tries':>7}{'duration':>12}"
    print(f"\n{header}")
    print("-" * len(header))
    owed = 0
    for cell in cells:
        entry = ledger["cells"].get(cell.key)
        if entry is None:
            print(f"{cell.key:<28}{'not run':<12}{'0/' + str(cell.count):>9}")
            owed += cell.count
            continue
        conversations = entry.get("conversations", 0)
        owed += outstanding(cell, entry)
        duration = entry.get("duration_s") or 0
        print(
            f"{cell.key:<28}{entry['status']:<12}"
            f"{f'{conversations}/{cell.count}':>9}{entry.get('traces', 0):>8}"
            f"{entry.get('attempts', 1):>7}{duration / 60:>10.1f}m"
        )
    if owed:
        print(f"\n{owed} conversation(s) still owed — resume to run only those.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--setups",
        default=",".join(DEFAULT_SETUPS),
        help=f"Comma-separated setups (default: {','.join(DEFAULT_SETUPS)})",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(str(s) for s in DEFAULT_SCENARIOS),
        help=f"Comma-separated scenario indices (default: {','.join(str(s) for s in DEFAULT_SCENARIOS)})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Conversations per cell (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--run-dir", help="Directory for logs and the ledger (default: timestamped)"
    )
    parser.add_argument(
        "--resume", help="Continue the run in this directory, skipping done cells"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the grid and exit"
    )
    parser.add_argument(
        "--stop-on-error", action="store_true", help="Abort the sweep when a cell fails"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Echo every simulate line to the terminal, not just [STATUS]/[ERROR]. "
            "Pair with --log-level debug to watch the run live."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level passed to each simulate cell (default: info)",
    )
    parser.add_argument(
        "--no-machine-restart",
        action="store_true",
        help="Leave the coffee-machine service alone between cells",
    )
    parser.add_argument(
        "--no-export", action="store_true", help="Skip the final event-log export"
    )
    parser.add_argument(
        "--skip-llm-probe", action="store_true", help="Skip the preflight LLM call"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_MACHINE_SEED,
        help=f"COFFEE_MACHINE_SEED for every cell (default: {DEFAULT_MACHINE_SEED})",
    )
    return parser.parse_args()


def resolve_grid(args: argparse.Namespace, resumed: dict | None) -> list[Cell]:
    """Grid from the CLI, falling back to the resumed run's own grid."""
    explicit = any(
        arg == f"--{name}" or arg.startswith(f"--{name}=")
        for name in ("setups", "scenarios", "count")
        for arg in sys.argv[1:]
    )
    if resumed and not explicit:
        grid = resumed["grid"]
        return build_cells(grid["setups"], grid["scenarios"], grid["count"])

    setups = [s.strip() for s in args.setups.split(",") if s.strip()]
    scenarios = []
    for token in args.scenarios.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if not 0 <= idx < len(CUSTOMER_SCENARIO_LABELS):
            raise SystemExit(
                f"scenario {idx} out of range 0-{len(CUSTOMER_SCENARIO_LABELS) - 1}"
            )
        scenarios.append(idx)
    if not setups or not scenarios:
        raise SystemExit("--setups and --scenarios must each name at least one value")
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    return build_cells(setups, scenarios, args.count)


def main() -> int:
    args = parse_args()
    if args.run_dir and args.resume:
        raise SystemExit("--run-dir and --resume are mutually exclusive")

    try:
        resumed = load_ledger(Path(args.resume)) if args.resume else None
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"Cannot resume: {exc}", file=sys.stderr)
        return 1
    cells = resolve_grid(args, resumed)

    if args.dry_run:
        print_plan(cells, args.log_level)
        return 0

    try:
        preflight(cells, args.skip_llm_probe)
    except RuntimeError as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 1

    if args.resume:
        run_dir = Path(args.resume)
        ledger = resumed
    else:
        run_dir = (
            Path(args.run_dir)
            if args.run_dir
            else (RUNS_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S"))
        )
        provider, model = llm_identity()
        ledger = {
            "created_at": now_iso(),
            "grid": {
                "setups": list(dict.fromkeys(c.setup for c in cells)),
                "scenarios": list(dict.fromkeys(c.scenario for c in cells)),
                "count": args.count,
            },
            "provenance": {
                "git": git_provenance(),
                "llm_provider": provider,
                "llm_model": model,
                "machine_seed": args.seed,
            },
            "cells": {},
        }
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    save_ledger(run_dir, ledger)

    todo = sum(outstanding(cell, ledger["cells"].get(cell.key) or {}) for cell in cells)
    print(
        f"Run directory: {run_dir}\n{len(cells)} cell(s), {todo} conversation(s) to run\n"
    )

    interrupted = False
    for cell in cells:
        if interrupted:
            break
        entry = ledger["cells"].get(cell.key) or {}
        done_before = max(0, int(entry.get("conversations", 0)))
        remaining = outstanding(cell, entry)

        if remaining == 0:
            # Nothing owed: either the cell is already flagged done, or it
            # reached its quota on an attempt that ended badly (interrupted, or
            # a non-zero exit on the tail conversation).
            if entry and entry.get("status") != "done":
                entry["status"] = "done"
                save_ledger(run_dir, ledger)
            print(
                f"[{stamp()}] [{cell.index}/{len(cells)}] {cell.key} — "
                f"already done ({done_before}/{cell.count}), skipping"
            )
            continue

        owed = (
            f"{remaining} conversation(s)"
            if not done_before
            else f"{remaining} conversation(s) still owed ({done_before}/{cell.count} recorded)"
        )
        print(f"\n[{stamp()}] [{cell.index}/{len(cells)}] {cell.key} — {owed}")
        try:
            wipe_coffee_shop_db()
            if not args.no_machine_restart:
                stop_machine()
                start_machine(run_dir / "coffee_machine.log", args.seed)
        except KeyboardInterrupt:
            interrupted = True
            break

        traces_before, conversations_before = trace_stats()
        log_path = run_dir / "logs" / f"{cell.slug}.log"
        started = time.monotonic()
        status, returncode = "done", 0
        try:
            returncode = run_cell(
                cell, log_path, args.verbose, args.log_level, remaining
            )
            if returncode != 0:
                status = "failed"
        except KeyboardInterrupt:
            status, interrupted = "interrupted", True

        traces_after, conversations_after = trace_stats()
        conversations = done_before + (conversations_after - conversations_before)
        # "done" means the cell holds its quota, not merely that the process
        # exited 0: a cell whose conversations all died still exits 0, and
        # calling that done would silently ship a short cell.
        if status == "done" and conversations < cell.count:
            status = "short"
        ledger["cells"][cell.key] = {
            "index": cell.index,
            "setup": cell.setup,
            "scenario": cell.scenario,
            "count": cell.count,
            "status": status,
            "returncode": returncode,
            "attempts": int(entry.get("attempts", 0)) + 1,
            "finished_at": now_iso(),
            "duration_s": round(
                float(entry.get("duration_s", 0.0)) + time.monotonic() - started, 1
            ),
            "traces": int(entry.get("traces", 0)) + (traces_after - traces_before),
            "conversations": conversations,
            "log": str(log_path.relative_to(run_dir)),
        }
        save_ledger(run_dir, ledger)
        print(
            f"[{stamp()}]     {status} — {conversations}/{cell.count} conversation(s), "
            f"{(time.monotonic() - started) / 60:.1f}m this attempt, log: {log_path}"
        )

        if interrupted:
            break
        if status == "failed" and args.stop_on_error:
            print("Stopping: cell failed and --stop-on-error is set.", file=sys.stderr)
            break

    if not args.no_machine_restart:
        stop_machine()

    print_summary(cells, ledger)
    print(f"\nLedger: {run_dir / LEDGER_NAME}")

    if interrupted:
        print(f"Interrupted. Continue with: --resume {run_dir}", file=sys.stderr)
        return 130

    if not args.no_export:
        print("\nExporting event log...")
        export_event_log()

    incomplete = [
        cell
        for cell in cells
        if ledger["cells"].get(cell.key, {}).get("status") != "done"
    ]
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
