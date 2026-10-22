"""strict_flow_hard: strict_flow with every rule enforced by a hard predicate,
equalized to strict_flow_soft. Covers the configurable transfer_includes_order_id
factory and the setup's wiring. Shape follows tests/test_max_tool_calls.py.
"""

import json
import unittest
from pathlib import Path

from langchain_core.messages import ToolMessage

from src.control_plane.agent_repo import AgentRepo
from src.control_plane.catalog import Catalog
from src.control_plane.predicates import transfer_includes_order_id_predicate
from src.control_plane.types import Effect, GuardrailContext

STRICT_FLOW_HARD_DIR = Path("config/setups/strict_flow_hard")

PLACED = ToolMessage(
    content=json.dumps({"order_id": "ORD0007", "summary": "Total: $4.00"}),
    name="process_order",
    tool_call_id="tc1",
)
FAILED = ToolMessage(
    content="Error processing order.\nItems not on menu: mocha",
    name="process_order",
    tool_call_id="tc1",
)


def _ctx(summary: str, messages=None) -> GuardrailContext:
    return GuardrailContext(
        agent_id="order_agent",
        tool_name="transfer_to_agent",
        tool_args={"target_agent": "inventory_agent", "context_summary": summary},
        state={"messages": messages or []},
        allowed_handovers=["inventory_agent"],
    )


class TestTransferIncludesOrderId(unittest.TestCase):
    def test_direct_call_keeps_flag_behavior(self):
        self.assertEqual(
            transfer_includes_order_id_predicate(
                _ctx("Customer wants a latte.")
            ).effect,
            Effect.FLAG,
        )

    def test_factory_deny(self):
        pred = transfer_includes_order_id_predicate(effect="deny")
        self.assertEqual(pred(_ctx("Customer wants a latte.")).effect, Effect.DENY)
        self.assertEqual(pred(_ctx("Check stock for ORD0007.")).effect, Effect.ALLOW)

    def test_only_after_order_placed_allows_before_any_order(self):
        pred = transfer_includes_order_id_predicate(
            effect="deny", only_after_order_placed=True
        )
        self.assertEqual(
            pred(_ctx("Customer complains about yesterday's coffee.")).effect,
            Effect.ALLOW,
        )

    def test_only_after_order_placed_denies_once_order_exists(self):
        pred = transfer_includes_order_id_predicate(
            effect="deny", only_after_order_placed=True
        )
        verdict = pred(_ctx("Customer wants a latte.", [PLACED]))
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("ORDXXXX", verdict.reason_for_llm)

    def test_failed_process_order_is_not_a_placed_order(self):
        pred = transfer_includes_order_id_predicate(
            effect="deny", only_after_order_placed=True
        )
        self.assertEqual(
            pred(_ctx("Customer wants a latte.", [FAILED])).effect, Effect.ALLOW
        )


class TestStrictFlowHardSetup(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog(STRICT_FLOW_HARD_DIR)
        self.agents = AgentRepo(STRICT_FLOW_HARD_DIR).all()

    def test_every_agent_guardrail_resolves_and_is_hard(self):
        for agent in self.agents.values():
            for gr in self.catalog.guardrails(list(agent.guardrail_ids)):
                with self.subTest(agent=agent.id, guardrail=gr.name):
                    self.assertEqual(gr.type, "hard")

    def test_agents_wire_same_ids_as_strict_flow(self):
        strict = AgentRepo(Path("config/setups/strict_flow")).all()
        for agent_id, agent in self.agents.items():
            with self.subTest(agent=agent_id):
                self.assertEqual(agent.guardrail_ids, strict[agent_id].guardrail_ids)

    # Enforcement strength against strict_flow / strict_flow_soft is checked for
    # every setup class in tests/test_setup_coherence.py.

    def test_partial_refund_rejects_negative(self):
        [gr] = self.catalog.guardrails(["offer_partial_refund:within_policy_cap"])
        ctx = GuardrailContext(
            agent_id="customer_service_agent",
            tool_name="offer_partial_refund",
            tool_args={"order_id": "ORD0007", "refund_percent": -10},
        )
        self.assertEqual(gr.eval(ctx).effect, Effect.DENY)

    def test_includes_order_id_wired_with_exemption(self):
        # Flag only, as in strict_flow; no flag before an order exists.
        [gr] = self.catalog.guardrails(["transfer:includes_order_id"])
        self.assertEqual(gr.eval(_ctx("Customer wants a latte.")).effect, Effect.ALLOW)
        self.assertEqual(
            gr.eval(_ctx("Customer wants a latte.", [PLACED])).effect, Effect.FLAG
        )


if __name__ == "__main__":
    unittest.main()
