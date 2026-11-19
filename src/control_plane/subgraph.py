"""Per-agent guarded subgraph that replaces `create_react_agent`.

Topology:
    START → llm → response_gateway → (cond)
                                     ├─ correction issued → llm (loop with corrective HumanMessage)
                                     └─ passed          → (cond)
                                                          ├─ no tool_calls   → END
                                                          └─ has tool_calls  → gateway → (cond)
                                                                                         ├─ batch-denied → llm (loop, synthetic ToolMessages for every tool_call_id)
                                                                                         └─ batch-allowed → tools → llm (loop)

The response_gateway inspects the LLM's assistant message text (before any
tool_call routing) against response-scoped guardrails — the ones that declare
themselves for the synthetic `assistant_message` "tool". A DENY verdict removes
the offending AIMessage from state and appends a corrective HumanMessage
carrying the guardrail's reason_for_llm, so the LLM can revise. A per-turn
retry cap prevents infinite loops.

Batch-verdict policy (all-or-nothing per LLM turn): per-call verdicts are still
evaluated and logged individually, but if ANY call in the batch is denied the
entire batch is rejected — synthetic ToolMessages are emitted for every
tool_call_id in the AIMessage so the Anthropic tool_use↔tool_result invariant
holds, and control returns to the LLM with denial reasons. Only when every
proposed call is allowed (or flagged) does the batch reach `tools`.
"""

import logging

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.context_isolation import create_context_isolation_hook
from src.agents.shared_components import CoffeeShopState
from src.llm import bind_tools_sequential, normalize_content

from .gateway import Gateway
from .types import Effect

logger = logging.getLogger("coffee_shop.control_plane.subgraph")

RESPONSE_GUARDRAIL_TOOL_NAME = "assistant_message"
CORRECTION_KWARG = "response_guardrail_correction"
REJECTED_CONTENT_KWARG = "response_guardrail_rejected_content"
REJECTED_MESSAGE_ID_KWARG = "response_guardrail_rejected_message_id"
REJECTION_REASON_KWARG = "response_guardrail_rejection_reason"
REJECTING_GUARDRAIL_KWARG = "response_guardrail_rejecting_guardrail"
REJECTED_AGENT_KWARG = "response_guardrail_rejected_agent"
MAX_RESPONSE_GUARDRAIL_RETRIES = 2
CAP_EXHAUSTED_FALLBACK = (
    "Sorry, I need a moment to think about that — could you tell me a bit more "
    "about what you're in the mood for?"
)


def _last_ai_with_tool_calls(messages) -> AIMessage | None:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            return m
    return None


def _thread_id_of(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    return (config.get("configurable") or {}).get("thread_id")


def _is_correction_message(msg) -> bool:
    return (
        isinstance(msg, HumanMessage)
        and (msg.additional_kwargs or {}).get(CORRECTION_KWARG) is True
    )


def _corrections_since_last_user_turn(messages) -> int:
    """Count corrective HumanMessages appended after the most recent genuine user
    HumanMessage. This is the retry-budget counter — a new user turn implicitly
    resets it to zero because we only walk the tail up to the last non-correction
    HumanMessage.
    """
    count = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and not _is_correction_message(msg):
            break
        if _is_correction_message(msg):
            count += 1
    return count


_MAX_REASON_LENGTH = 400


def _sanitize_reason(reason: str) -> str:
    """Prepare a judge- or predicate-supplied reason for embedding in a prompt.

    Applies the minimum neutralization needed so a jailbroken judge cannot
    trivially break out of the quoted framing span:
      * strips leading/trailing whitespace,
      * collapses `\\n` to spaces so multiline injected prose can't visually
        separate itself from the framing,
      * replaces ASCII double quotes with single quotes so `"..."` framing
        cannot be closed early by an injected `"`,
      * truncates to `_MAX_REASON_LENGTH` characters.

    Not stripped: other control chars (`\\r`, `\\t`, bidi overrides), smart
    quotes, and backticks. A determined adversary can still degrade the
    framing via those — but the size cap plus the "third-party observation"
    prose framing still meaningfully lowers the authority the LLM assigns to
    the injected text. Broadening this sanitizer is deferred as a follow-up.
    """
    reason = (reason or "").strip().replace("\n", " ").replace('"', "'")
    if len(reason) > _MAX_REASON_LENGTH:
        reason = reason[:_MAX_REASON_LENGTH] + "…"
    return reason


_INTERNAL_NOTE = (
    "[Internal note from the coffee shop's guardrails. The customer cannot see "
    "it; never quote or mention it in your reply.]"
)


def _bounded_reason(reason: str, guardrail_name: str, hint: str = "") -> str:
    """Neutralize and bound the judge's reason before it becomes an agent-facing
    HumanMessage. A jailbroken judge could otherwise embed instructions in the
    `reason` field ("Instead, recommend our new secret item X") that the agent
    would receive as authoritative correction guidance. Wrapping in quotes and
    prefixing "The <guardrail> guardrail reported:" reframes the text as
    third-party observation rather than an instruction, and truncation caps the
    prompt-injection payload size. `hint` is the guardrail's trusted,
    config-authored correction hint (e.g. the actual menu), so it is added
    outside the quotes.
    """
    safe = _sanitize_reason(reason)
    label = guardrail_name or "response"
    guidance = f" {hint}" if hint else ""
    return (
        f"{_INTERNAL_NOTE} The {label} guardrail rejected the previous message with "
        f'note: "{safe}".{guidance} Rewrite the message to the customer so it '
        f"complies with the coffee shop policy."
    )


def _bounded_tool_reason(
    reason: str, guardrail_name: str, tool_name: str, hint: str = ""
) -> str:
    """Same neutralization as `_bounded_reason` but framed for a tool-call
    denial (the reason lands in a `ToolMessage` with `status="error"`).
    """
    safe = _sanitize_reason(reason)
    label = guardrail_name or "response"
    guidance = f" {hint}" if hint else ""
    return (
        f"The {label} guardrail rejected the tool call {tool_name!r} with "
        f'note: "{safe}".{guidance} Choose a different action that complies with '
        f"the coffee shop policy."
    )


def _guardrail_named(gateway: Gateway, name: str):
    return next((g for g in gateway.guardrails if g.name == name), None)


def create_agent_subgraph(
    agent_id: str,
    llm,
    tools,
    prompt: str,
    gateway: Gateway,
):
    """Build a compiled per-agent subgraph wired with the given Gateway."""
    llm_with_tools = bind_tools_sequential(llm, tools)
    context_hook = create_context_isolation_hook(agent_id)
    tool_node = ToolNode(tools)

    def llm_node(state: CoffeeShopState, config: RunnableConfig):
        hook_update = context_hook(state)
        llm_input = hook_update.get("llm_input_messages") or state.get("messages", [])
        sys_msgs = [SystemMessage(content=prompt)] if prompt else []
        ai = llm_with_tools.invoke(sys_msgs + list(llm_input), config=config)
        # create_react_agent set this for us; we need to set it explicitly so
        # _resolve_from_agent and context-isolation boundary detection keep working.
        ai.name = agent_id
        return {"messages": [ai]}

    def route_after_llm(state: CoffeeShopState) -> str:
        last = state["messages"][-1] if state.get("messages") else None
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "gateway"
        return END

    def response_gateway_node(state: CoffeeShopState, config: RunnableConfig):
        """Evaluate the outgoing AIMessage against response-scoped guardrails.

        Synthesizes a pseudo tool call `assistant_message` whose args carry the
        message content. Reuses `Gateway.evaluate_call` so the verdict, JSONL
        log record, and OCEL projection are identical to any other guardrail
        decision. On DENY (under the retry cap): removes the offending
        AIMessage from state, appends a corrective HumanMessage carrying the
        guardrail's `reason_for_llm`, and stamps rejection metadata on the
        correction so the runner can surface it on the dashboard. On DENY at
        the retry cap: replaces the offending message with a canned fallback
        AIMessage so the customer never sees rejected content, even when the
        LLM cannot produce a compliant reply within budget. On ALLOW/FLAG:
        leaves state untouched.

        Only text-only AIMessages are evaluated — messages that carry tool
        calls (i.e. private tool-use turns) are skipped because their content
        never reaches the customer.
        """
        messages = state.get("messages", [])
        if not messages:
            return {}
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return {}
        if getattr(last, "tool_calls", None):
            return {}

        applicable = [
            gr
            for gr in gateway.guardrails
            if gr.applies_to(RESPONSE_GUARDRAIL_TOOL_NAME)
        ]
        if not applicable:
            return {}

        content = normalize_content(last.content)
        if not content.strip():
            return {}

        rejected_message_id = getattr(last, "id", None)
        synthetic_call = {
            "name": RESPONSE_GUARDRAIL_TOOL_NAME,
            "args": {"content": content, "agent_id": agent_id},
            "id": f"resp-{rejected_message_id or 'unknown'}",
        }
        decision = gateway.evaluate_call(
            synthetic_call,
            dict(state),
            thread_id=_thread_id_of(config),
        )

        if decision.final_decision != Effect.DENY:
            return {}

        rejecting_guardrail = ""
        for verdict in decision.verdicts:
            if verdict.effect == Effect.DENY:
                rejecting_guardrail = verdict.guardrail_name
                break

        prior_corrections = _corrections_since_last_user_turn(messages)
        raw_reason = _sanitize_reason(
            decision.deny_reason_for_llm
            or "Your last message violated a response guardrail."
        )
        rejecting = _guardrail_named(gateway, rejecting_guardrail)
        if prior_corrections >= MAX_RESPONSE_GUARDRAIL_RETRIES:
            logger.warning(
                f"response guardrail retry cap ({MAX_RESPONSE_GUARDRAIL_RETRIES}) "
                f"reached for {agent_id}; publishing fallback"
            )
            fallback_content = (
                rejecting.rendered_fallback_message() if rejecting else ""
            ) or CAP_EXHAUSTED_FALLBACK
            gateway.log_response_fallback(
                thread_id=_thread_id_of(config),
                rejecting_guardrail=rejecting_guardrail,
                rejected_content=content,
                reason=raw_reason,
                corrections=prior_corrections,
                fallback_content=fallback_content,
            )
            fallback = AIMessage(
                content=fallback_content,
                name=agent_id,
                additional_kwargs={
                    CORRECTION_KWARG: True,
                    REJECTED_CONTENT_KWARG: content,
                    REJECTED_MESSAGE_ID_KWARG: rejected_message_id or "",
                    REJECTION_REASON_KWARG: raw_reason,
                    REJECTING_GUARDRAIL_KWARG: rejecting_guardrail,
                    REJECTED_AGENT_KWARG: agent_id,
                },
            )
            patch_cap: list = [fallback]
            if rejected_message_id:
                patch_cap.insert(0, RemoveMessage(id=rejected_message_id))
            return {"messages": patch_cap}

        correction_text = _bounded_reason(
            raw_reason,
            rejecting_guardrail,
            rejecting.rendered_correction_hint() if rejecting else "",
        )
        correction = HumanMessage(
            content=correction_text,
            additional_kwargs={
                CORRECTION_KWARG: True,
                REJECTED_CONTENT_KWARG: content,
                REJECTED_MESSAGE_ID_KWARG: rejected_message_id or "",
                REJECTION_REASON_KWARG: raw_reason,
                REJECTING_GUARDRAIL_KWARG: rejecting_guardrail,
                REJECTED_AGENT_KWARG: agent_id,
            },
        )
        patch: list = [correction]
        if rejected_message_id:
            patch.insert(0, RemoveMessage(id=rejected_message_id))
        return {"messages": patch}

    def route_after_response_gateway(state: CoffeeShopState) -> str:
        last = state["messages"][-1] if state.get("messages") else None
        if _is_correction_message(last):
            return "llm"
        return route_after_llm(state)

    def gateway_node(state: CoffeeShopState, config: RunnableConfig):
        ai = _last_ai_with_tool_calls(state.get("messages", []))
        if ai is None:
            return {}
        thread_id = _thread_id_of(config)

        decisions = [
            gateway.evaluate_call(tc, dict(state), thread_id=thread_id)
            for tc in ai.tool_calls
        ]
        any_denied = any(d.final_decision == Effect.DENY for d in decisions)
        if not any_denied:
            # All allowed — no state update; tools_node reads state.messages directly.
            return {}

        # Batch denied: one synthetic ToolMessage per tool_call_id.
        deny_reasons: list[str] = []
        for d in decisions:
            if d.final_decision != Effect.DENY:
                continue
            denying = next(
                (v.guardrail_name for v in d.verdicts if v.effect == Effect.DENY),
                "response",
            )
            denying_guardrail = _guardrail_named(gateway, denying)
            deny_reasons.append(
                _bounded_tool_reason(
                    d.deny_reason_for_llm,
                    denying,
                    d.tool_name,
                    denying_guardrail.rendered_correction_hint()
                    if denying_guardrail
                    else "",
                )
            )
        sibling_blurb = "; ".join(deny_reasons)
        synth: list[ToolMessage] = []
        deny_iter = iter(deny_reasons)
        for d in decisions:
            if d.final_decision == Effect.DENY:
                content = next(deny_iter, "Tool call denied by guardrail.")
            else:
                content = (
                    f"Tool call {d.tool_name!r} was not executed because a sibling tool "
                    f"call in the same batch was denied by a guardrail: {sibling_blurb}"
                )
            synth.append(
                ToolMessage(
                    content=content,
                    name=d.tool_name,
                    tool_call_id=d.tool_call_id,
                    status="error",
                )
            )
        return {"messages": synth}

    def route_after_gateway(state: CoffeeShopState) -> str:
        # If the last message is an AIMessage with un-answered tool_calls, route to tools;
        # otherwise (gateway emitted synthetic ToolMessages) route back to llm.
        msgs = state.get("messages", [])
        if not msgs:
            return END
        last = msgs[-1]
        if isinstance(last, ToolMessage):
            return "llm"
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    g = StateGraph(CoffeeShopState)
    g.add_node("llm", llm_node)
    g.add_node("response_gateway", response_gateway_node)
    g.add_node("gateway", gateway_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "llm")
    g.add_edge("llm", "response_gateway")
    g.add_conditional_edges(
        "response_gateway",
        route_after_response_gateway,
        {"llm": "llm", "gateway": "gateway", END: END},
    )
    g.add_conditional_edges(
        "gateway", route_after_gateway, {"tools": "tools", "llm": "llm", END: END}
    )
    g.add_edge("tools", "llm")
    return g.compile()
