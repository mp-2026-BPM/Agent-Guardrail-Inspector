"""Unit tests for the three new predicates added for the baseline "reasonable
defaults" refinement:

- ``process_order:items_on_menu`` (deterministic menu-adherence)
- ``transfer:context_summary_nonempty`` (silent-handover guard)
- ``clean_machine:only_after_error`` (no cleaning a healthy machine)

Test style mirrors ``tests/test_require_order_status.py``.
"""

import tempfile
import unittest
from pathlib import Path

from src.agents.order_store import (
    create_order_store_engine,
    get_engine,
    init_db,
    reset_inventory,
    save_order,
    set_engine,
)
from src.agents.shared_components import Order, OrderItem, OrderStatus
from src.control_plane.predicates import (
    PREDICATE_REGISTRY,
    clean_machine_only_after_error_predicate,
    process_order_items_on_menu_predicate,
    transfer_context_summary_nonempty_predicate,
)
from src.control_plane.types import Effect, GuardrailContext


def _ctx(tool_name: str, tool_args: dict) -> GuardrailContext:
    return GuardrailContext(
        agent_id="order_agent",
        tool_name=tool_name,
        tool_args=tool_args,
        state={},
        allowed_handovers=[],
    )


class TestProcessOrderItemsOnMenu(unittest.TestCase):
    def test_registered(self):
        self.assertIn("process_order_items_on_menu", PREDICATE_REGISTRY)

    def test_allows_on_menu_items(self):
        predicate = process_order_items_on_menu_predicate()
        ctx = _ctx("process_order", {"order": [{"name": "espresso", "quantity": 1}]})
        self.assertEqual(predicate(ctx).effect, Effect.ALLOW)

    def test_case_insensitive_match(self):
        predicate = process_order_items_on_menu_predicate()
        ctx = _ctx("process_order", {"order": [{"name": " ESPRESSO "}]})
        # tool normalizes with .lower().strip(); predicate must match this.
        self.assertEqual(predicate(ctx).effect, Effect.ALLOW)

    def test_denies_off_menu_item(self):
        predicate = process_order_items_on_menu_predicate()
        ctx = _ctx("process_order", {"order": [{"name": "mocha"}]})
        verdict = predicate(ctx)
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("mocha", verdict.reason_for_llm)

    def test_flag_effect_when_configured(self):
        predicate = process_order_items_on_menu_predicate(effect="flag")
        ctx = _ctx("process_order", {"order": [{"name": "mocha"}]})
        self.assertEqual(predicate(ctx).effect, Effect.FLAG)

    def test_missing_order_allows(self):
        predicate = process_order_items_on_menu_predicate()
        self.assertEqual(predicate(_ctx("process_order", {})).effect, Effect.ALLOW)

    def test_mixed_order_denies_on_first_offender(self):
        predicate = process_order_items_on_menu_predicate()
        ctx = _ctx(
            "process_order",
            {"order": [{"name": "latte"}, {"name": "unicorn_frappe"}]},
        )
        verdict = predicate(ctx)
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("unicorn_frappe", verdict.reason_for_llm)


class TestTransferContextSummaryNonempty(unittest.TestCase):
    def test_registered(self):
        self.assertIn("transfer_context_summary_nonempty", PREDICATE_REGISTRY)

    def test_denies_empty(self):
        predicate = transfer_context_summary_nonempty_predicate()
        ctx = _ctx(
            "transfer_to_agent",
            {"target_agent": "inventory_agent", "context_summary": ""},
        )
        self.assertEqual(predicate(ctx).effect, Effect.DENY)

    def test_denies_whitespace_only(self):
        predicate = transfer_context_summary_nonempty_predicate()
        ctx = _ctx(
            "transfer_to_agent",
            {"target_agent": "inventory_agent", "context_summary": "   \n  "},
        )
        self.assertEqual(predicate(ctx).effect, Effect.DENY)

    def test_denies_too_short(self):
        predicate = transfer_context_summary_nonempty_predicate(min_chars=20)
        ctx = _ctx(
            "transfer_to_agent",
            {"target_agent": "inventory_agent", "context_summary": "handoff"},
        )
        self.assertEqual(predicate(ctx).effect, Effect.DENY)

    def test_allows_real_summary(self):
        predicate = transfer_context_summary_nonempty_predicate(min_chars=20)
        ctx = _ctx(
            "transfer_to_agent",
            {
                "target_agent": "inventory_agent",
                "context_summary": (
                    "Order ORD0001 placed for 1 latte. Please confirm inventory."
                ),
            },
        )
        self.assertEqual(predicate(ctx).effect, Effect.ALLOW)

    def test_min_chars_honored(self):
        # min_chars=5 admits a very short summary that min_chars=20 rejects.
        short = "hello there"
        ctx = _ctx(
            "transfer_to_agent",
            {"target_agent": "x", "context_summary": short},
        )
        self.assertEqual(
            transfer_context_summary_nonempty_predicate(min_chars=5)(ctx).effect,
            Effect.ALLOW,
        )
        self.assertEqual(
            transfer_context_summary_nonempty_predicate(min_chars=20)(ctx).effect,
            Effect.DENY,
        )

    def test_flag_effect_when_configured(self):
        predicate = transfer_context_summary_nonempty_predicate(effect="flag")
        ctx = _ctx(
            "transfer_to_agent",
            {"target_agent": "x", "context_summary": ""},
        )
        self.assertEqual(predicate(ctx).effect, Effect.FLAG)


class TestCleanMachineOnlyAfterError(unittest.TestCase):
    def setUp(self):
        init_db()
        reset_inventory()

    def _seed_order(self, status: OrderStatus) -> None:
        order = Order(
            customer="CleanTest",
            status=status,
            total=4.0,
            items=[
                OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[])
            ],
        )
        save_order(order)

    def test_registered(self):
        self.assertIn("clean_machine_only_after_error", PREDICATE_REGISTRY)

    def test_allows_when_recent_order_in_error(self):
        self._seed_order(OrderStatus.PREPARATION_ERROR)
        predicate = clean_machine_only_after_error_predicate()
        self.assertEqual(predicate(_ctx("clean_machine", {})).effect, Effect.ALLOW)

    def test_denies_when_recent_order_healthy(self):
        self._seed_order(OrderStatus.IN_PREPARATION)
        predicate = clean_machine_only_after_error_predicate()
        verdict = predicate(_ctx("clean_machine", {}))
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("in_preparation", verdict.reason_internal)

    def test_allows_when_no_order(self):
        # Fresh DB, no orders — cannot resolve state; err toward ALLOW so
        # bootstrap / test-harness calls aren't spuriously blocked. Uses its own
        # empty DB: the shared one holds orders saved by earlier tests.
        previous = get_engine()
        with tempfile.TemporaryDirectory() as tmp:
            engine = create_order_store_engine(f"sqlite:///{Path(tmp) / 'orders.db'}")
            set_engine(engine)
            try:
                init_db()
                predicate = clean_machine_only_after_error_predicate()
                self.assertEqual(
                    predicate(_ctx("clean_machine", {})).effect, Effect.ALLOW
                )
            finally:
                set_engine(previous)
                engine.dispose()

    def test_flag_effect_when_configured(self):
        self._seed_order(OrderStatus.IN_PREPARATION)
        predicate = clean_machine_only_after_error_predicate(effect="flag")
        self.assertEqual(predicate(_ctx("clean_machine", {})).effect, Effect.FLAG)


if __name__ == "__main__":
    unittest.main()
