"""Soft-judge plumbing: the conversation / order context judges receive, judging
tool calls without args, retrying and labelling unreadable judge output, and the
judge LLM settings. Judges load from the real setup YAML with a stub invoker,
as in tests/test_baseline_soft_guardrails.py.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.order_store import (
    create_order_store_engine,
    get_engine,
    init_db,
    save_order,
    set_engine,
)
from src.agents.shared_components import Order, OrderItem, OrderStatus
from src.control_plane import Catalog, SoftGuardrail
from src.control_plane.guardrails import (
    JUDGE_ERROR,
    JUDGE_UNPARSEABLE,
    recent_dialogue,
    tool_result_log,
)
from src.control_plane.types import Effect, GuardrailContext

STRICT_FLOW_SOFT_DIR = Path("config/setups/strict_flow_soft")
ALLOW = '{"reason": "fine", "decision": "allow"}'
DENY = '{"reason": "not fine", "decision": "deny"}'


def _call(name: str, args: dict, call_id: str, agent: str = "order_agent") -> AIMessage:
    return AIMessage(
        content="", name=agent, tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


def _result(
    call_id: str, name: str, content: str, status: str = "success"
) -> ToolMessage:
    return ToolMessage(content=content, name=name, tool_call_id=call_id, status=status)


PLACED = [
    _call("process_order", {"order": [{"name": "latte"}]}, "c1"),
    _result(
        "c1",
        "process_order",
        '{"order_id": "ORD0007", "summary": "Order ORD0007 Total: $4.00 Status: pending"}',
    ),
]


class _Recorder:
    """Stub judge: records prompts, replies with the queued responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        response = (
            self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        )
        if isinstance(response, Exception):
            raise response
        return response


def _judge(guardrail_id: str, invoker) -> SoftGuardrail:
    [gr] = Catalog(STRICT_FLOW_SOFT_DIR).guardrails([guardrail_id])
    gr.judge_invoker = invoker
    return gr


def _ctx(tool_name: str, tool_args: dict, messages=None) -> GuardrailContext:
    return GuardrailContext(
        agent_id="order_agent",
        tool_name=tool_name,
        tool_args=tool_args,
        state={"messages": messages or []},
        allowed_handovers=["inventory_agent"],
    )


class TestToolResultLog(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(tool_result_log([]), "(no tool calls yet)")

    def test_outcomes_and_summary(self):
        log = tool_result_log(
            PLACED
            + [
                _call("calculate_total", {"order_id": "ORD0007"}, "c2"),
                _result("c2", "calculate_total", "Error: Order 'ORD0007' not found."),
                _call(
                    "offer_refund",
                    {"order_id": "ORD0007"},
                    "c3",
                    agent="customer_service_agent",
                ),
                _result(
                    "c3",
                    "offer_refund",
                    "The offer_refund:order_status guardrail rejected the tool call 'offer_refund'.",
                    status="error",
                ),
                _call("clean_machine", {}, "c4", agent="barista_agent"),
                _result("c4", "clean_machine", "Error: boom", status="error"),
            ]
        )
        lines = log.splitlines()
        self.assertTrue(lines[0].startswith("1. order_agent → process_order("))
        # JSON results show their summary, not the raw JSON.
        self.assertEqual(
            lines[1].strip(), 'ok: "Order ORD0007 Total: $4.00 Status: pending"'
        )
        self.assertTrue(
            lines[3].strip().startswith("failed:")
        )  # returned an error reply
        self.assertTrue(lines[5].strip().startswith("blocked:"))  # gateway denial
        self.assertTrue(lines[7].strip().startswith("failed:"))  # tool raised
        self.assertIn("customer_service_agent → offer_refund", lines[4])

    def test_pending_call_left_out(self):
        log = tool_result_log(PLACED + [_call("process_order", {}, "c9")])
        self.assertNotIn("2.", log)

    def test_keeps_last_entries(self):
        messages = []
        for i in range(25):
            messages += [
                _call("get_order", {"i": i}, f"g{i}"),
                _result(f"g{i}", "get_order", "ok"),
            ]
        log = tool_result_log(messages)
        self.assertTrue(log.startswith("(5 earlier calls omitted)"))
        self.assertIn('25. order_agent → get_order({"i": 24})', log)
        self.assertNotIn('{"i": 4}', log)


class TestRecentDialogue(unittest.TestCase):
    def test_skips_tool_traffic_and_corrections(self):
        messages = [
            HumanMessage(content="Hi, a latte please"),
            AIMessage(content="Large or medium?", name="order_agent"),
            HumanMessage(
                content="rewrite that",
                additional_kwargs={"response_guardrail_correction": True},
            ),
            *PLACED,
            HumanMessage(content="Medium"),
        ]
        self.assertEqual(
            recent_dialogue(messages),
            "customer: Hi, a latte please\norder_agent: Large or medium?\ncustomer: Medium",
        )

    def test_keeps_last_turns(self):
        messages = [HumanMessage(content=f"m{i}") for i in range(6)]
        self.assertEqual(recent_dialogue(messages).splitlines()[0], "customer: m2")


class TestJudgeContext(unittest.TestCase):
    def test_max_calls_judge_sees_tool_results(self):
        judge = _Recorder(DENY)
        gr = _judge("soft_process_order:max_calls", judge)
        verdict = gr.eval(_ctx("process_order", {"order": []}, PLACED))
        self.assertEqual(verdict.effect, Effect.DENY)
        system_prompt = judge.calls[0][0]
        self.assertIn("1. order_agent → process_order(", system_prompt)
        self.assertNotIn("{tool_results}", system_prompt)

    def test_price_judge_sees_tool_results(self):
        judge = _Recorder(ALLOW)
        [gr] = Catalog(STRICT_FLOW_SOFT_DIR).guardrails(
            ["assistant_message:no_price_promises"]
        )
        gr.judge_invoker = judge
        gr.eval(_ctx("assistant_message", {"content": "That's $4.00."}, PLACED))
        self.assertIn("Total: $4.00", judge.calls[0][0])

    def test_handover_judge_sees_dialogue_and_tools(self):
        judge = _Recorder(ALLOW)
        [gr] = Catalog(STRICT_FLOW_SOFT_DIR).guardrails(
            ["handover:reason_matches_target"]
        )
        gr.judge_invoker = judge
        messages = [HumanMessage(content="A latte please"), *PLACED]
        gr.eval(
            _ctx(
                "transfer_to_agent",
                {
                    "target_agent": "inventory_agent",
                    "context_summary": "Check stock for ORD0007.",
                },
                messages,
            )
        )
        self.assertIn("customer: A latte please", judge.calls[0][0])
        self.assertIn("process_order(", judge.calls[0][0])


class TestOrderContext(unittest.TestCase):
    def setUp(self):
        self._previous = get_engine()
        self._tmp = tempfile.TemporaryDirectory()
        self._engine = create_order_store_engine(
            f"sqlite:///{Path(self._tmp.name) / 'orders.db'}"
        )
        set_engine(self._engine)
        init_db()

    def tearDown(self):
        set_engine(self._previous)
        self._engine.dispose()
        self._tmp.cleanup()

    def _order(self, status: OrderStatus, total: float = 4.0) -> str:
        order = Order(
            customer="JudgeCtx",
            status=status,
            total=total,
            items=[
                OrderItem(name="latte", quantity=1, price=total, size=None, extras=[])
            ],
        )
        save_order(order)
        return order.order_id_str

    def test_clean_machine_judged_without_args(self):
        # clean_machine has no args; it must still reach the judge.
        judge = _Recorder(DENY)
        gr = _judge("soft_clean_machine:only_after_error", judge)
        verdict = gr.eval(_ctx("clean_machine", {}))
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertEqual(len(judge.calls), 1)
        self.assertIn(
            "Most recent order: no order has been placed yet", judge.calls[0][0]
        )

    def test_clean_machine_sees_recent_order(self):
        order_id = self._order(OrderStatus.PREPARATION_ERROR)
        judge = _Recorder(ALLOW)
        gr = _judge("soft_clean_machine:only_after_error", judge)
        gr.eval(_ctx("clean_machine", {}))
        self.assertIn(f'{order_id}, status "preparation_error"', judge.calls[0][0])

    def test_total_judge_sees_stored_total(self):
        order_id = self._order(OrderStatus.PENDING, total=123.5)
        judge = _Recorder(DENY)
        gr = _judge("soft_calculate_total:total_within_limit", judge)
        gr.eval(_ctx("calculate_total", {"order_id": order_id}))
        self.assertIn(f"Order {order_id}, stored total: $123.50", judge.calls[0][0])

    def test_total_unknown_for_missing_order(self):
        judge = _Recorder(ALLOW)
        gr = _judge("soft_calculate_total:total_within_limit", judge)
        gr.eval(_ctx("calculate_total", {"order_id": "ORD9999"}))
        self.assertIn("stored total: unknown", judge.calls[0][0])


class TestJudgeRetryAndLabels(unittest.TestCase):
    def _gr(self, invoker):
        return _judge("soft_calculate_total:discount_within_limit_20pct", invoker)

    def test_unreadable_then_valid_uses_retry(self):
        judge = _Recorder("no json here", DENY)
        verdict = self._gr(judge).eval(
            _ctx("calculate_total", {"discount_percent": 50})
        )
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertEqual(len(judge.calls), 2)

    def test_unreadable_twice_allows_with_label(self):
        judge = _Recorder('{"reason": "x", "decision": "denay"}')
        verdict = self._gr(judge).eval(
            _ctx("calculate_total", {"discount_percent": 50})
        )
        self.assertEqual(verdict.effect, Effect.ALLOW)
        self.assertTrue(verdict.reason_internal.startswith(JUDGE_UNPARSEABLE))
        self.assertEqual(len(judge.calls), 2)

    def test_errors_twice_allow_with_label(self):
        judge = _Recorder(RuntimeError("ollama down"))
        verdict = self._gr(judge).eval(
            _ctx("calculate_total", {"discount_percent": 50})
        )
        self.assertEqual(verdict.effect, Effect.ALLOW)
        self.assertTrue(verdict.reason_internal.startswith(JUDGE_ERROR))
        self.assertIn("ollama down", verdict.reason_internal)

    def test_valid_first_answer_is_not_retried(self):
        judge = _Recorder(ALLOW)
        self._gr(judge).eval(_ctx("calculate_total", {}))
        self.assertEqual(len(judge.calls), 1)

    def test_deny_without_reason_names_guardrail(self):
        judge = _Recorder('{"decision": "deny"}')
        verdict = self._gr(judge).eval(
            _ctx("calculate_total", {"discount_percent": 50})
        )
        self.assertIn("discount_within_limit_20pct", verdict.reason_for_llm)


class TestJudgeLLM(unittest.TestCase):
    def test_ollama_judge_is_deterministic_json(self):
        from src.llm import create_judge_llm

        with mock.patch.dict(
            os.environ, {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "ministral-3:14b"}
        ):
            llm = create_judge_llm()
        self.assertEqual(llm.model, "ministral-3:14b")
        self.assertEqual(llm.temperature, 0)
        self.assertEqual(llm.format, "json")
        # No load-time option that would make Ollama reload the agents' model.
        self.assertIsNone(llm.num_ctx)


if __name__ == "__main__":
    unittest.main()
