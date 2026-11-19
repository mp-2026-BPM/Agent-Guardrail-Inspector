"""max_tool_calls predicate: caps per-conversation invocations of a named tool.
Used by the strict_flow setup to enforce that order_agent may call process_order
at most once per conversation.
"""

import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.control_plane.catalog import Catalog
from src.control_plane.predicates import (
    PREDICATE_REGISTRY,
    max_tool_calls_predicate,
)
from src.control_plane.types import Effect, GuardrailContext


def _ctx(messages=None, tool_name="process_order", **tool_args) -> GuardrailContext:
    return GuardrailContext(
        agent_id="order_agent",
        tool_name=tool_name,
        tool_args=dict(tool_args),
        state={"messages": messages or []},
        allowed_handovers=[],
    )


class TestRegistration(unittest.TestCase):
    def test_registered(self):
        self.assertIn("max_tool_calls", PREDICATE_REGISTRY)


class TestMaxToolCalls(unittest.TestCase):
    def test_first_call_allowed(self):
        pred = max_tool_calls_predicate("process_order", max_calls=1)
        self.assertEqual(pred(_ctx()).effect, Effect.ALLOW)

    def test_first_call_allowed_when_history_has_other_tools(self):
        pred = max_tool_calls_predicate("process_order", max_calls=1)
        history = [
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[]),
            ToolMessage(content="{}", name="calculate_total", tool_call_id="tc1"),
        ]
        self.assertEqual(pred(_ctx(messages=history)).effect, Effect.ALLOW)

    def test_second_call_denied_after_prior_success(self):
        pred = max_tool_calls_predicate("process_order", max_calls=1)
        history = [
            HumanMessage(content="hi"),
            ToolMessage(
                content='{"order_id": "ORD001"}',
                name="process_order",
                tool_call_id="tc1",
            ),
        ]
        verdict = pred(_ctx(messages=history))
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("process_order", verdict.reason_for_llm)

    def test_prior_denied_call_does_not_count(self):
        pred = max_tool_calls_predicate("process_order", max_calls=1)
        history = [
            ToolMessage(
                content="Denied by guardrail.",
                name="process_order",
                tool_call_id="tc1",
                status="error",
            ),
        ]
        self.assertEqual(pred(_ctx(messages=history)).effect, Effect.ALLOW)

    def test_prior_returned_failure_does_not_count(self):
        # Tools report bad input in their reply while the ToolMessage status stays
        # success; such a rejected attempt must not use up the only allowed call.
        pred = max_tool_calls_predicate("process_order", max_calls=1)
        for content in (
            "Error: Unknown extras: vanilla. Allowed: caramel syrup, ...",
            "Error processing order: bad payload. Please use specified format.",
            '{"status": "error", "message": "Coffee machine is not available."}',
        ):
            with self.subTest(content=content):
                history = [
                    ToolMessage(
                        content=content, name="process_order", tool_call_id="tc1"
                    )
                ]
                self.assertEqual(pred(_ctx(messages=history)).effect, Effect.ALLOW)

    def test_success_after_failure_still_counts(self):
        pred = max_tool_calls_predicate("process_order", max_calls=1)
        history = [
            ToolMessage(
                content="Error: Unknown extras: vanilla.",
                name="process_order",
                tool_call_id="tc1",
            ),
            ToolMessage(
                content='{"order_id": "ORD001"}',
                name="process_order",
                tool_call_id="tc2",
            ),
        ]
        self.assertEqual(pred(_ctx(messages=history)).effect, Effect.DENY)

    def test_max_calls_two_permits_second_call(self):
        pred = max_tool_calls_predicate("process_order", max_calls=2)
        history = [
            ToolMessage(content="{}", name="process_order", tool_call_id="tc1"),
        ]
        self.assertEqual(pred(_ctx(messages=history)).effect, Effect.ALLOW)

    def test_max_calls_two_denies_third_call(self):
        pred = max_tool_calls_predicate("process_order", max_calls=2)
        history = [
            ToolMessage(content="{}", name="process_order", tool_call_id="tc1"),
            ToolMessage(content="{}", name="process_order", tool_call_id="tc2"),
        ]
        self.assertEqual(pred(_ctx(messages=history)).effect, Effect.DENY)

    def test_flag_effect(self):
        pred = max_tool_calls_predicate("process_order", max_calls=1, effect="flag")
        history = [
            ToolMessage(content="{}", name="process_order", tool_call_id="tc1"),
        ]
        self.assertEqual(pred(_ctx(messages=history)).effect, Effect.FLAG)

    def test_only_counts_matching_tool_name(self):
        pred = max_tool_calls_predicate("process_order", max_calls=1)
        history = [
            ToolMessage(content="{}", name="calculate_total", tool_call_id="tc1"),
            ToolMessage(content="{}", name="check_inventory", tool_call_id="tc2"),
        ]
        self.assertEqual(pred(_ctx(messages=history)).effect, Effect.ALLOW)


class TestStrictFlowSetupWiring(unittest.TestCase):
    """The strict_flow setup wires this predicate onto order_agent's process_order."""

    def test_catalog_builds_guardrail(self):
        catalog = Catalog(Path("config/setups/strict_flow"))
        [guardrail] = catalog.guardrails(["process_order:max_calls"])
        self.assertEqual(list(guardrail.tools), ["process_order"])
        history = [
            ToolMessage(content="{}", name="process_order", tool_call_id="tc1"),
        ]
        verdict = guardrail.eval(_ctx(messages=history))
        self.assertEqual(verdict.effect, Effect.DENY)


if __name__ == "__main__":
    unittest.main()
