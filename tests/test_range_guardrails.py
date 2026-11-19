"""Range-constraint guardrail predicates: order size, refund %, order total, and the
discount `effect` param. These back the sensibility-spectrum setups (narrow_ranges,
sensible_ranges[_flag]). Shape follows tests/test_require_order_status.py.
"""

import tempfile
import unittest
from pathlib import Path

from src.agents.order_store import init_db, reset_inventory, save_order
from src.agents.shared_components import Order, OrderItem, OrderStatus
from src.control_plane.catalog import Catalog
from src.control_plane.predicates import (
    order_size_within_range_predicate,
    refund_within_limit_predicate,
    order_total_within_limit_predicate,
    discount_within_limit_predicate,
    PREDICATE_REGISTRY,
)
from src.control_plane.types import Effect, GuardrailContext


def _ctx(tool_name, **tool_args) -> GuardrailContext:
    return GuardrailContext(
        agent_id="order_agent",
        tool_name=tool_name,
        tool_args=dict(tool_args),
        state={},
        allowed_handovers=[],
    )


def _order_with_total(total: float) -> str:
    order = Order(
        customer="RangeTest",
        status=OrderStatus.PENDING,
        total=total,
        items=[OrderItem(name="latte", quantity=1, price=total, size=None, extras=[])],
    )
    save_order(order)
    return order.order_id_str


class TestRegistration(unittest.TestCase):
    def test_all_registered(self):
        for name in (
            "order_size_within_range",
            "refund_within_limit",
            "order_total_within_limit",
        ):
            self.assertIn(name, PREDICATE_REGISTRY)


class TestOrderSizeWithinRange(unittest.TestCase):
    def test_two_units_denied_at_max_one(self):
        pred = order_size_within_range_predicate(max_units=1, effect="deny")
        ctx = _ctx("process_order", order=[{"name": "espresso", "quantity": 2}])
        self.assertEqual(pred(ctx).effect, Effect.DENY)

    def test_two_units_allowed_at_max_six(self):
        pred = order_size_within_range_predicate(max_units=6, effect="deny")
        ctx = _ctx("process_order", order=[{"name": "espresso", "quantity": 2}])
        self.assertEqual(pred(ctx).effect, Effect.ALLOW)

    def test_units_summed_across_line_items(self):
        pred = order_size_within_range_predicate(max_units=6, effect="deny")
        order = [
            {"name": "latte", "quantity": 4},
            {"name": "muffin", "quantity": 3},
        ]  # 7 units
        self.assertEqual(pred(_ctx("process_order", order=order)).effect, Effect.DENY)

    def test_missing_order_allows(self):
        pred = order_size_within_range_predicate(max_units=1, effect="deny")
        self.assertEqual(pred(_ctx("process_order")).effect, Effect.ALLOW)

    def test_default_quantity_is_one(self):
        pred = order_size_within_range_predicate(max_units=1, effect="deny")
        # two line items without explicit quantity -> 2 units -> denied at max 1
        order = [{"name": "espresso"}, {"name": "latte"}]
        self.assertEqual(pred(_ctx("process_order", order=order)).effect, Effect.DENY)

    def test_flag_mode(self):
        pred = order_size_within_range_predicate(max_units=1, effect="flag")
        ctx = _ctx("process_order", order=[{"name": "espresso", "quantity": 2}])
        self.assertEqual(pred(ctx).effect, Effect.FLAG)


class TestRefundWithinLimit(unittest.TestCase):
    def test_over_limit_denied(self):
        pred = refund_within_limit_predicate(max_pct=50, effect="deny")
        self.assertEqual(
            pred(_ctx("offer_partial_refund", refund_percent=60)).effect, Effect.DENY
        )

    def test_at_limit_allowed(self):
        pred = refund_within_limit_predicate(max_pct=50, effect="deny")
        self.assertEqual(
            pred(_ctx("offer_partial_refund", refund_percent=50)).effect, Effect.ALLOW
        )

    def test_missing_uses_tool_default_50(self):
        pred = refund_within_limit_predicate(max_pct=0, effect="deny")
        # tool default is 50; at max 0 that is a violation
        self.assertEqual(pred(_ctx("offer_partial_refund")).effect, Effect.DENY)

    def test_flag_mode(self):
        pred = refund_within_limit_predicate(max_pct=50, effect="flag")
        self.assertEqual(
            pred(_ctx("offer_partial_refund", refund_percent=90)).effect, Effect.FLAG
        )

    def test_negative_allowed_without_min(self):
        # min_pct is opt-in; setups that omit it keep the historical upper-bound-only check.
        pred = refund_within_limit_predicate(max_pct=100, effect="deny")
        self.assertEqual(
            pred(_ctx("offer_partial_refund", refund_percent=-10)).effect, Effect.ALLOW
        )

    def test_below_min_denied(self):
        pred = refund_within_limit_predicate(max_pct=100, min_pct=0, effect="deny")
        verdict = pred(_ctx("offer_partial_refund", refund_percent=-10))
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("-10%", verdict.reason_for_llm)

    def test_at_min_allowed(self):
        pred = refund_within_limit_predicate(max_pct=100, min_pct=0, effect="deny")
        self.assertEqual(
            pred(_ctx("offer_partial_refund", refund_percent=0)).effect, Effect.ALLOW
        )


class TestOrderTotalWithinLimit(unittest.TestCase):
    def setUp(self):
        init_db()
        reset_inventory()

    def test_over_limit_denied(self):
        oid = _order_with_total(25.0)
        pred = order_total_within_limit_predicate(max_total=20, effect="deny")
        self.assertEqual(
            pred(_ctx("calculate_total", order_id=oid)).effect, Effect.DENY
        )

    def test_within_limit_allowed(self):
        oid = _order_with_total(10.0)
        pred = order_total_within_limit_predicate(max_total=20, effect="deny")
        self.assertEqual(
            pred(_ctx("calculate_total", order_id=oid)).effect, Effect.ALLOW
        )

    def test_unresolvable_allows(self):
        pred = order_total_within_limit_predicate(max_total=3.0, effect="deny")
        self.assertEqual(
            pred(_ctx("calculate_total", order_id="ORD999999")).effect, Effect.ALLOW
        )

    def test_flag_mode(self):
        oid = _order_with_total(25.0)
        pred = order_total_within_limit_predicate(max_total=20, effect="flag")
        self.assertEqual(
            pred(_ctx("calculate_total", order_id=oid)).effect, Effect.FLAG
        )


class TestDiscountEffectParam(unittest.TestCase):
    def test_default_is_flag_backward_compat(self):
        pred = discount_within_limit_predicate(max_pct=30)  # no effect arg
        self.assertEqual(
            pred(_ctx("calculate_total", discount_percent=40)).effect, Effect.FLAG
        )

    def test_deny_effect(self):
        pred = discount_within_limit_predicate(max_pct=0, effect="deny")
        self.assertEqual(
            pred(_ctx("calculate_total", discount_percent=10)).effect, Effect.DENY
        )

    def test_within_limit_allowed(self):
        pred = discount_within_limit_predicate(max_pct=30, effect="deny")
        self.assertEqual(
            pred(_ctx("calculate_total", discount_percent=30)).effect, Effect.ALLOW
        )


class TestRangeGuardrailsViaCatalog(unittest.TestCase):
    """Full YAML path: predicate_args carrying the range + effect reach the factory."""

    def _catalog(self, yaml_text: str) -> Catalog:
        d = tempfile.mkdtemp()
        setup = Path(d) / "s"
        (setup / "guardrails").mkdir(parents=True)
        (setup / "guidelines").mkdir(parents=True)
        (setup / "guardrails" / "coffee_shop.yaml").write_text(
            yaml_text, encoding="utf-8"
        )
        return Catalog(setup)

    def test_order_size_from_yaml(self):
        cat = self._catalog(
            "guardrails:\n"
            "  - id: create_order:size_within_range\n"
            "    type: hard\n"
            "    tools: [process_order]\n"
            "    effect: deny\n"
            "    predicate: order_size_within_range\n"
            "    predicate_args: {min_units: 1, max_units: 1, effect: deny}\n"
        )
        [gr] = cat.guardrails(["create_order:size_within_range"])
        ctx = _ctx("process_order", order=[{"name": "espresso", "quantity": 2}])
        self.assertEqual(gr.eval(ctx).effect, Effect.DENY)


if __name__ == "__main__":
    unittest.main()
