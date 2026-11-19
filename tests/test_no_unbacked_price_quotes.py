"""no_unbacked_price_quotes predicate: deterministic counterpart to the soft
`assistant_message:no_price_promises` judge, plus a wiring check that the
baseline_hard setup resolves every agent's guardrail ids.
Shape follows tests/test_max_tool_calls.py.
"""

import json
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.control_plane.agent_repo import AgentRepo
from src.control_plane.catalog import Catalog
from src.control_plane.predicates import (
    PREDICATE_REGISTRY,
    no_unbacked_price_quotes_predicate,
)
from src.control_plane.types import Effect, GuardrailContext

BASELINE_HARD_DIR = Path("config/setups/baseline_hard")
STRICT_FLOW_HARD_DIR = Path("config/setups/strict_flow_hard")

PROCESS_ORDER_RESULT = json.dumps(
    {
        "order_id": "ORD0001",
        "summary": "Order ORD0001\n  - 2x Latte (medium): $8.00\nTotal: $8.00\nStatus: pending",
    }
)
CALCULATE_TOTAL_RESULT = json.dumps(
    {
        "order_id": "ORD0001",
        "total": 7.2,
        "discount": 0.8,
        "summary": "Order ORD0001 total: $8.00, discount (10%): -$0.80, final: $7.20",
    }
)
# Shape of offer_partial_refund's result: original_total only exists as a JSON number.
PARTIAL_REFUND_RESULT = json.dumps(
    {
        "order_id": "ORD0001",
        "refund_amount": 2.0,
        "original_total": 4.0,
        "new_total": 2.0,
        "summary": "Partial refund (50%) of $2.00 for order ORD0001. New total: $2.00",
    }
)


def _ctx(content: str, messages=None) -> GuardrailContext:
    return GuardrailContext(
        agent_id="order_agent",
        tool_name="assistant_message",
        tool_args={"content": content, "agent_id": "order_agent"},
        state={"messages": messages or []},
        allowed_handovers=[],
    )


def _tool(name: str, content: str, status: str = "success") -> ToolMessage:
    return ToolMessage(
        content=content, name=name, tool_call_id=f"tc-{name}", status=status
    )


class TestNoUnbackedPriceQuotes(unittest.TestCase):
    def setUp(self):
        self.pred = no_unbacked_price_quotes_predicate()

    def test_registered(self):
        self.assertIn("no_unbacked_price_quotes", PREDICATE_REGISTRY)

    def test_no_amount_allows(self):
        self.assertEqual(
            self.pred(_ctx("Would you like a latte?")).effect, Effect.ALLOW
        )

    def test_unbacked_menu_price_denied(self):
        verdict = self.pred(_ctx("A latte is $4.00."))
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("$4.00", verdict.reason_for_llm)

    def test_amount_from_process_order_allows(self):
        msgs = [
            HumanMessage(content="2 lattes"),
            _tool("process_order", PROCESS_ORDER_RESULT),
        ]
        self.assertEqual(
            self.pred(_ctx("Your total is $8.00.", msgs)).effect, Effect.ALLOW
        )

    def test_amount_formats_compared_in_cents(self):
        msgs = [_tool("process_order", PROCESS_ORDER_RESULT)]
        self.assertEqual(
            self.pred(_ctx("That comes to $8.", msgs)).effect, Effect.ALLOW
        )
        self.assertEqual(
            self.pred(_ctx("That comes to 8 dollars.", msgs)).effect, Effect.ALLOW
        )

    def test_discount_figures_from_calculate_total_allow(self):
        msgs = [_tool("calculate_total", CALCULATE_TOTAL_RESULT)]
        ctx = _ctx("With 10% off you save $0.80, so you pay $7.20.", msgs)
        self.assertEqual(self.pred(ctx).effect, Effect.ALLOW)

    def test_one_unbacked_amount_denies(self):
        msgs = [_tool("process_order", PROCESS_ORDER_RESULT)]
        verdict = self.pred(_ctx("Total $8.00; add a muffin for $3.25?", msgs))
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("$3.25", verdict.reason_for_llm)
        self.assertNotIn("$8.00", verdict.reason_for_llm)

    def test_errored_tool_result_does_not_back(self):
        msgs = [_tool("process_order", PROCESS_ORDER_RESULT, status="error")]
        self.assertEqual(
            self.pred(_ctx("Your total is $8.00.", msgs)).effect, Effect.DENY
        )

    def test_other_tool_does_not_back(self):
        msgs = [_tool("check_inventory", "Latte costs $4.00")]
        self.assertEqual(self.pred(_ctx("A latte is $4.00.", msgs)).effect, Effect.DENY)

    def test_source_tools_configurable(self):
        pred = no_unbacked_price_quotes_predicate(source_tools=["offer_partial_refund"])
        msgs = [_tool("offer_partial_refund", '{"refund_amount": "$2.00"}')]
        self.assertEqual(pred(_ctx("We refunded $2.00.", msgs)).effect, Effect.ALLOW)

    def test_json_numbers_ignored_by_default(self):
        msgs = [_tool("offer_partial_refund", PARTIAL_REFUND_RESULT)]
        pred = no_unbacked_price_quotes_predicate(source_tools=["offer_partial_refund"])
        self.assertEqual(
            pred(_ctx("Your original total was $4.00.", msgs)).effect, Effect.DENY
        )

    def test_json_numbers_back_amounts_when_enabled(self):
        pred = no_unbacked_price_quotes_predicate(
            source_tools=["offer_partial_refund"], count_json_numbers=True
        )
        msgs = [_tool("offer_partial_refund", PARTIAL_REFUND_RESULT)]
        ctx = _ctx("We refunded $2.00 of your $4.00 order; $2.00 remains.", msgs)
        self.assertEqual(pred(ctx).effect, Effect.ALLOW)

    def test_json_numbers_skip_bools_and_non_json(self):
        pred = no_unbacked_price_quotes_predicate(
            source_tools=["offer_refund"], count_json_numbers=True
        )
        msgs = [
            _tool("offer_refund", '{"ok": true}'),  # True must not back $0.01
            _tool(
                "offer_refund", "Error-free plain text 3"
            ),  # not JSON, bare number ignored
        ]
        self.assertEqual(pred(_ctx("That's $0.01.", msgs)).effect, Effect.DENY)
        self.assertEqual(pred(_ctx("That's $3.", msgs)).effect, Effect.DENY)

    def test_vague_statements_allow(self):
        for text in (
            "A latte is around $4.",
            "That's roughly $8 in total.",
            "About 5 dollars each.",
            "Roughly ~$3 for a muffin.",
            "A couple of dollars each.",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.pred(_ctx(text)).effect, Effect.ALLOW)

    def test_flag_effect_when_configured(self):
        pred = no_unbacked_price_quotes_predicate(effect="flag")
        self.assertEqual(pred(_ctx("A latte is $4.00.")).effect, Effect.FLAG)

    def test_ignores_ai_message_amounts(self):
        # The agent's own earlier claims never back a price; only tool results do.
        msgs = [AIMessage(content="A latte is $4.00.")]
        self.assertEqual(self.pred(_ctx("So $4.00 it is.", msgs)).effect, Effect.DENY)


class TestBaselineHardWiring(unittest.TestCase):
    def test_every_agent_guardrail_resolves(self):
        catalog = Catalog(BASELINE_HARD_DIR)
        for agent in AgentRepo(BASELINE_HARD_DIR).all().values():
            with self.subTest(agent=agent.id):
                catalog.guardrails(list(agent.guardrail_ids))

    def test_no_soft_guardrails(self):
        catalog = Catalog(BASELINE_HARD_DIR)
        ids = {
            gid
            for a in AgentRepo(BASELINE_HARD_DIR).all().values()
            for gid in a.guardrail_ids
        }
        for gr in catalog.guardrails(sorted(ids)):
            with self.subTest(guardrail=gr.name):
                self.assertEqual(gr.type, "hard")

    def test_price_guardrail_applies_to_assistant_message(self):
        [gr] = Catalog(BASELINE_HARD_DIR).guardrails(
            ["assistant_message:no_price_promises"]
        )
        self.assertTrue(gr.applies_to("assistant_message"))
        self.assertEqual(gr.eval(_ctx("A latte is $4.00.")).effect, Effect.DENY)

    def test_price_guardrail_accepts_refund_amounts(self):
        # state_refund_details asks for refund, original total and remaining balance.
        # Both hard setups must agree, since they share the guideline.
        msgs = [_tool("offer_partial_refund", PARTIAL_REFUND_RESULT)]
        ctx = _ctx("Refunded $2.00 of your $4.00 order; you now owe $2.00.", msgs)
        for setup_dir in (BASELINE_HARD_DIR, STRICT_FLOW_HARD_DIR):
            with self.subTest(setup=setup_dir.name):
                [gr] = Catalog(setup_dir).guardrails(
                    ["assistant_message:no_price_promises"]
                )
                self.assertEqual(gr.eval(ctx).effect, Effect.ALLOW)


if __name__ == "__main__":
    unittest.main()
