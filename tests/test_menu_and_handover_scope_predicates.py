"""Deterministic counterparts to two soft judges, used by baseline_hard:

- ``no_off_menu_mentions``   ↔ ``assistant_message:on_menu_only``
- ``handover_scope_mismatch`` ↔ ``handover:reason_matches_target``

Messages and summaries below are taken (lightly trimmed) from logged runs.
Shape follows tests/test_new_baseline_predicates.py.
"""

import unittest
from pathlib import Path

from src.control_plane.catalog import Catalog
from src.control_plane.predicates import (
    PREDICATE_REGISTRY,
    handover_scope_mismatch_predicate,
    no_off_menu_mentions_predicate,
)
from src.control_plane.types import Effect, GuardrailContext

BASELINE_HARD_DIR = Path("config/setups/baseline_hard")
TERMS = [
    "mocha",
    "chai",
    "cold brew",
    "lavender",
    "honey",
    "chocolate",
    "cinnamon",
    "tea",
]


def _msg_ctx(content: str) -> GuardrailContext:
    return GuardrailContext(
        agent_id="order_agent",
        tool_name="assistant_message",
        tool_args={"content": content, "agent_id": "order_agent"},
    )


def _handover_ctx(agent_id: str, target: str, summary: str) -> GuardrailContext:
    return GuardrailContext(
        agent_id=agent_id,
        tool_name="transfer_to_agent",
        tool_args={"target_agent": target, "context_summary": summary},
    )


class TestNoOffMenuMentions(unittest.TestCase):
    def setUp(self):
        self.pred = no_off_menu_mentions_predicate(off_menu_terms=TERMS)

    def test_registered(self):
        self.assertIn("no_off_menu_mentions", PREDICATE_REGISTRY)

    def test_empty_terms_rejected(self):
        with self.assertRaises(ValueError):
            no_off_menu_mentions_predicate(off_menu_terms=[" "])

    def test_on_menu_message_allows(self):
        ctx = _msg_ctx("Would you like a large latte with oat milk and a croissant?")
        self.assertEqual(self.pred(ctx).effect, Effect.ALLOW)

    def test_offer_denied_and_named(self):
        verdict = self.pred(_msg_ctx("How about a mocha instead? It's really good."))
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("'mocha'", verdict.reason_for_llm)
        self.assertIn("latte", verdict.reason_for_llm)  # menu listed for the retry

    def test_declines_allow(self):
        for text in (
            "I'm sorry, but we don't offer honey lavender oat lattes at the moment. How about a classic latte?",
            "We don’t have rooibos tea or iced tea on the menu, but how about a latte?",
            "Mocha isn't on our menu, sorry!",
            "We do not serve chai.",
            "No cold brew today, I'm afraid.",
            "We'll skip the cinnamon for now.",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.pred(_msg_ctx(text)).effect, Effect.ALLOW)

    def test_decline_then_offer_in_other_clause_denies(self):
        # "but" starts a new clause, so the negation does not cover the offer.
        verdict = self.pred(
            _msg_ctx("We don't have chai, but you might enjoy a mocha!")
        )
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("mocha", verdict.reason_internal)
        self.assertNotIn("chai", verdict.reason_internal)

    def test_off_menu_variant_of_menu_item_denied(self):
        ctx = _msg_ctx(
            "We've got plain, chocolate, or almond croissants if you're feeling fancy."
        )
        self.assertEqual(self.pred(ctx).effect, Effect.DENY)

    def test_flavour_shorthand_from_allowed_extras_allows(self):
        ctx = _msg_ctx("How about a vanilla latte or a caramel latte?")
        self.assertEqual(self.pred(ctx).effect, Effect.ALLOW)

    def test_whole_word_match_only(self):
        # "tea" must not fire inside "steamed" / "instead".
        ctx = _msg_ctx("Your latte has steamed milk; want an espresso instead?")
        self.assertEqual(self.pred(ctx).effect, Effect.ALLOW)

    def test_case_insensitive(self):
        self.assertEqual(self.pred(_msg_ctx("Try our MOCHA!")).effect, Effect.DENY)

    def test_flag_effect_when_configured(self):
        pred = no_off_menu_mentions_predicate(off_menu_terms=TERMS, effect="flag")
        self.assertEqual(pred(_msg_ctx("How about a mocha?")).effect, Effect.FLAG)


class TestHandoverScopeMismatch(unittest.TestCase):
    def setUp(self):
        self.pred = handover_scope_mismatch_predicate()

    def test_registered(self):
        self.assertIn("handover_scope_mismatch", PREDICATE_REGISTRY)

    def test_normal_fulfilment_allows(self):
        for agent, target, summary in (
            (
                "order_agent",
                "inventory_agent",
                "New order ORD0011 for large cappuccino",
            ),
            (
                "order_agent",
                "inventory_agent",
                "Customer ordered 2 medium espressos to go (no food). Order ID: ORD0015.",
            ),
            (
                "inventory_agent",
                "barista_agent",
                "ORD0003 stock confirmed; please prepare the medium latte.",
            ),
        ):
            with self.subTest(summary=summary):
                ctx = _handover_ctx(agent, target, summary)
                self.assertEqual(self.pred(ctx).effect, Effect.ALLOW)

    def test_refund_to_barista_or_inventory_denied(self):
        for target in ("barista_agent", "inventory_agent"):
            with self.subTest(target=target):
                ctx = _handover_ctx(
                    "order_agent",
                    target,
                    "Customer wants a refund for ORD0004, the latte was cold.",
                )
                verdict = self.pred(ctx)
                self.assertEqual(verdict.effect, Effect.DENY)
                self.assertIn("refund", verdict.reason_for_llm)

    def test_refund_to_order_agent_denied_without_replacement(self):
        ctx = _handover_ctx(
            "customer_service_agent",
            "order_agent",
            "Customer requests a refund for ORD0007.",
        )
        self.assertEqual(self.pred(ctx).effect, Effect.DENY)

    def test_complaint_with_replacement_to_order_agent_allows(self):
        for summary in (
            "Customer complaint about a cold cappuccino; they want a replacement cappuccino.",
            "Customer wants a fresh cappuccino after complaint about cold drink yesterday.",
        ):
            with self.subTest(summary=summary):
                ctx = _handover_ctx("customer_service_agent", "order_agent", summary)
                self.assertEqual(self.pred(ctx).effect, Effect.ALLOW)

    def test_new_order_to_customer_service_denied(self):
        ctx = _handover_ctx(
            "inventory_agent",
            "customer_service_agent",
            "Customer wants to place an order for a latte.",
        )
        verdict = self.pred(ctx)
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("complaints and refunds", verdict.reason_for_llm)

    def test_new_order_with_issue_to_customer_service_allows(self):
        ctx = _handover_ctx(
            "order_agent",
            "customer_service_agent",
            "New order ORD0030 is stuck in pending; the system blocks start_preparation.",
        )
        self.assertEqual(self.pred(ctx).effect, Effect.ALLOW)

    def test_refund_to_customer_service_allows(self):
        ctx = _handover_ctx(
            "barista_agent",
            "customer_service_agent",
            "Brew failed twice for ORD0009; customer wants a refund.",
        )
        self.assertEqual(self.pred(ctx).effect, Effect.ALLOW)

    def test_unknown_target_left_to_allowed_targets(self):
        ctx = _handover_ctx("order_agent", "barista", "Customer wants a refund.")
        self.assertEqual(self.pred(ctx).effect, Effect.ALLOW)

    def test_flag_effect_when_configured(self):
        pred = handover_scope_mismatch_predicate(effect="flag")
        ctx = _handover_ctx("order_agent", "barista_agent", "Customer wants a refund.")
        self.assertEqual(pred(ctx).effect, Effect.FLAG)


class TestBaselineHardConfig(unittest.TestCase):
    def test_menu_terms_loaded_from_yaml(self):
        [gr] = Catalog(BASELINE_HARD_DIR).guardrails(["assistant_message:on_menu_only"])
        self.assertTrue(gr.applies_to("assistant_message"))
        self.assertEqual(
            gr.eval(_msg_ctx("How about a matcha frappé?")).effect, Effect.DENY
        )
        self.assertEqual(
            gr.eval(_msg_ctx("How about a vanilla latte?")).effect, Effect.ALLOW
        )

    def test_handover_guardrail_wired(self):
        [gr] = Catalog(BASELINE_HARD_DIR).guardrails(["handover:reason_matches_target"])
        self.assertTrue(gr.applies_to("transfer_to_agent"))
        ctx = _handover_ctx("order_agent", "barista_agent", "Customer wants a refund.")
        self.assertEqual(gr.eval(ctx).effect, Effect.DENY)


if __name__ == "__main__":
    unittest.main()
