import logging
from dataclasses import dataclass
from typing import Any

from .guardrails import Guardrail
from .log_sink import JsonlLogSink, NullLogSink
from .types import Effect, GuardrailContext, Verdict

logger = logging.getLogger("coffee_shop.control_plane.gateway")


@dataclass
class CallDecision:
    """Per-tool-call evaluation result."""

    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any]
    verdicts: list[Verdict]
    final_decision: Effect  # ALLOW or DENY (FLAG collapses to ALLOW for routing)
    deny_reason_for_llm: str = ""


class Gateway:
    """Per-agent gateway. Evaluates proposed tool calls against guardrails,
    logs every decision, and returns CallDecision objects so the subgraph
    can route allowed/denied calls.
    """

    def __init__(
        self,
        agent_id: str,
        guardrails: list[Guardrail],
        allowed_handovers: list[str],
        snapshot_id: str,
        log_sink: JsonlLogSink | NullLogSink,
    ):
        self.agent_id = agent_id
        self.guardrails = guardrails
        self.allowed_handovers = list(allowed_handovers)
        self.snapshot_id = snapshot_id
        self.log_sink = log_sink

    def evaluate_call(
        self,
        tool_call: dict[str, Any],
        state: dict[str, Any],
        thread_id: str | None = None,
    ) -> CallDecision:
        tool_name = tool_call.get("name", "")
        tool_args = dict(tool_call.get("args", {}))
        tool_call_id = tool_call.get("id", "")

        logger.debug(
            f"Evaluating tool call: {tool_name} with args {tool_args} and id {tool_call_id}"
        )

        context = GuardrailContext(
            agent_id=self.agent_id,
            tool_name=tool_name,
            tool_args=tool_args,
            state=state,
            allowed_handovers=list(self.allowed_handovers),
        )

        applicable = [
            guardrail
            for guardrail in self.guardrails
            if guardrail.applies_to(tool_name)
        ]
        applicable.sort(key=lambda g: 0 if g.type == "hard" else 1)

        logger.debug(
            f"Applicable guardrails for {tool_name}: {[g.name for g in applicable]}"
        )

        verdicts: list[Verdict] = []
        verdict_versions: list[str] = []
        final = Effect.ALLOW
        deny_reason = ""
        for guardrail in applicable:
            verdict = guardrail.eval(context)
            verdicts.append(verdict)
            verdict_versions.append(guardrail.version)
            if verdict.effect == Effect.DENY and final != Effect.DENY:
                final = Effect.DENY
                deny_reason = verdict.reason_for_llm or verdict.reason_internal
                logger.debug(f"Tool call denied by guardrail: {guardrail.name}")
                break
            if verdict.effect == Effect.FLAG and final == Effect.ALLOW:
                final = Effect.FLAG  # observability only; doesn't block
                logger.debug(f"Tool call flagged by guardrail: {guardrail.name}")

        logger.debug(
            f"Final decision for tool call {tool_name}: {final.value}, reason: {deny_reason}"
        )
        decision = CallDecision(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args=tool_args,
            verdicts=verdicts,
            final_decision=final,
            deny_reason_for_llm=deny_reason,
        )
        self._log_decision(decision, thread_id, verdict_versions)
        return decision

    def log_response_fallback(
        self,
        thread_id: str | None,
        rejecting_guardrail: str,
        rejected_content: str,
        reason: str,
        corrections: int,
        fallback_content: str,
    ) -> None:
        """Record that the response guardrail retry cap was hit and the customer got
        the canned fallback instead of the agent's (rejected) message. Readers that
        only want verdicts filter on `event_type == "gateway_decision"`."""
        self.log_sink.append(
            {
                "event_type": "response_guardrail_fallback",
                "snapshot_id": self.snapshot_id,
                "agent_id": self.agent_id,
                "thread_id": thread_id,
                "rejecting_guardrail": rejecting_guardrail,
                "rejected_content": rejected_content,
                "reason": reason,
                "corrections_before_fallback": corrections,
                "fallback_content": fallback_content,
            }
        )

    def _log_decision(
        self,
        decision: CallDecision,
        thread_id: str | None,
        verdict_versions: list[str],
    ) -> None:
        self.log_sink.append(
            {
                "event_type": "gateway_decision",
                "snapshot_id": self.snapshot_id,
                "agent_id": self.agent_id,
                "thread_id": thread_id,
                "tool_name": decision.tool_name,
                "tool_call_id": decision.tool_call_id,
                "tool_args": decision.tool_args,
                "final_decision": decision.final_decision.value,
                "verdicts": [
                    {
                        "guardrail_name": v.guardrail_name,
                        "guardrail_version": ver,
                        "guardrail_type": v.guardrail_type,
                        "effect": v.effect.value,
                        "reason_internal": v.reason_internal,
                        "reason_for_llm": v.reason_for_llm,
                    }
                    for v, ver in zip(decision.verdicts, verdict_versions)
                ],
            }
        )
