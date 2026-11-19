import os
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TIMEOUT_SECONDS = 300


def _llm_unreachable_reason() -> str | None:
    """The simulation needs a live LLM. With the default Ollama provider, skip
    (instead of failing) when no Ollama server answers."""
    import src.llm  # noqa: F401 — loads .env, like the simulation does

    if os.getenv("LLM_PROVIDER", "ollama").lower().strip() != "ollama":
        return None
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith("http"):
        host = f"http://{host}"
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=2)
    except OSError as exc:
        return f"Ollama not reachable at {host}: {exc}"
    return None


_SKIP_REASON = _llm_unreachable_reason()


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class TestSimulationE2E(unittest.TestCase):
    """Integration test: run a full simulation and verify the order reaches COMPLETED."""

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._db_path = Path(self._tmp_dir) / "coffee_shop.db"

    def tearDown(self):
        if self._db_path.exists():
            try:
                self._db_path.unlink()
            except OSError:
                pass

    def test_scenario_0_completes_order(self):
        env = os.environ.copy()
        env["COFFEE_MACHINE_SEED"] = "100"
        env["COFFEE_SHOP_DB"] = str(self._db_path)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.simulate",
                    "--setup",
                    "baseline",
                    "--scenario",
                    "0",
                    "--log-level",
                    "debug",
                    "--traces",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode(errors="replace")
            stderr = (e.stderr or b"").decode(errors="replace")
            self.fail(
                f"Simulation timed out after {TIMEOUT_SECONDS}s\n"
                f"\n--- STDOUT (partial) ---\n{stdout}\n"
                f"\n--- STDERR (partial) ---\n{stderr}\n"
            )

        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode

        order = None
        order_repr = "Order not found in database"
        try:
            from sqlmodel import Session, select, create_engine as _ce
            from src.agents.shared_components import Order, OrderStatus

            eng = _ce(
                f"sqlite:///{self._db_path}",
                connect_args={"check_same_thread": False},
            )
            with Session(eng) as session:
                order = session.exec(select(Order)).first()
            if order:
                order_repr = (
                    f"Order(id={order.id}, customer={order.customer!r}, "
                    f"status={order.status.value!r}, total={order.total}, "
                    f"items={[f'{i.quantity}x {i.name}' for i in order.items]}, "
                    f"created_at={order.created_at}, last_modified={order.last_modified})"
                )
        except Exception as e:
            order_repr = f"Failed to query DB: {e}"

        diagnostics = (
            "\n"
            "=" * 80 + "\n"
            "SIMULATION DIAGNOSTIC DUMP\n"
            "=" * 80 + "\n"
            f"\n--- EXIT CODE: {exit_code} ---\n\n"
            f"--- STDOUT (full) ---\n{stdout}\n"
            f"--- STDERR (full) ---\n{stderr}\n"
            f"--- FINAL ORDER STATE ---\n{order_repr}\n"
            "=" * 80 + "\n"
        )

        if exit_code != 0:
            self.fail(f"Simulation exited with code {exit_code}{diagnostics}")

        self.assertIsNotNone(
            order,
            f"No order found in database after simulation{diagnostics}",
        )

        self.assertEqual(
            order.status,
            OrderStatus.COMPLETED,
            f"Order status is {order.status.value!r}, expected 'completed'{diagnostics}",
        )

        # Verify trace tagging: at least one LangGraph trace from this run
        # must carry setup=baseline + scenario_index=0. The subprocess writes
        # to the shared sqlite:///mlflow.db, so we filter to traces produced
        # since the test started to avoid accidentally matching traces from
        # earlier runs.
        try:
            import mlflow  # noqa: F401
            from mlflow import MlflowClient
        except ImportError:
            self.fail("mlflow not importable — cannot verify trace tags")
        client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
        experiments = client.search_experiments()
        found_tagged = False
        checked = 0
        for exp in experiments:
            traces = client.search_traces(
                experiment_ids=[exp.experiment_id], max_results=50
            )
            for t in traces:
                tags = dict(t.info.tags or {})
                if tags.get("mlflow.traceName") != "LangGraph":
                    continue
                checked += 1
                if (
                    tags.get("setup") == "baseline"
                    and tags.get("scenario_index") == "0"
                ):
                    found_tagged = True
                    break
            if found_tagged:
                break
        self.assertTrue(
            found_tagged,
            f"No LangGraph trace found with setup=baseline + scenario_index=0 "
            f"(checked {checked} LangGraph traces){diagnostics}",
        )


if __name__ == "__main__":
    unittest.main()
