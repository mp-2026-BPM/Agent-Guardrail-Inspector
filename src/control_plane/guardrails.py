import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from .types import Effect, GuardrailContext, Verdict

logger = logging.getLogger("coffee_shop.control_plane.guardrails")

MAX_JUDGE_MESSAGE_LENGTH = 8000
MAX_JUDGE_REASON_LENGTH = 400
JUDGE_ATTEMPTS = 2
# Prefixes of `reason_internal` on fail-open allows, so they can be filtered out
# of the real allow decisions in the guardrail log.
JUDGE_UNPARSEABLE = "[judge_unparseable]"
JUDGE_ERROR = "[judge_error]"

# Kept in sync with subgraph.RESPONSE_GUARDRAIL_TOOL_NAME. Duplicated here (not
# imported) to avoid a circular dependency: subgraph.py already imports from
# this module. The identity check in `_extract_message` picks a payload shape
# per tool kind — outgoing free text for response guardrails, JSON-serialized
# args for every real tool call.
RESPONSE_GUARDRAIL_TOOL_NAME = "assistant_message"


@dataclass
class Guardrail(ABC):
    """Base guardrail. Subclassed by evaluation mechanism."""

    name: str
    version: str = "unversioned"
    tools: list[str] = field(default_factory=list)
    effect: Effect = Effect.DENY
    description: str = ""

    @abstractmethod
    def eval(self, context: GuardrailContext) -> Verdict: ...

    @property
    def type(self) -> str:
        return "guardrail"

    def applies_to(self, tool_name: str) -> bool:
        return not self.tools or tool_name in self.tools


@dataclass
class HardGuardrail(Guardrail):
    """Deterministic rule, deterministic evaluation."""

    predicate: Callable[[GuardrailContext], Verdict] | None = None
    predicate_args: dict | None = None

    def eval(self, context: GuardrailContext) -> Verdict:
        if self.predicate is None:
            raise ValueError(f"HardGuardrail {self.name!r} has no predicate")
        verdict = self.predicate(context)
        if not verdict.guardrail_name:
            verdict.guardrail_name = self.name
        if not verdict.guardrail_type:
            verdict.guardrail_type = self.type
        return verdict

    @property
    def type(self) -> str:
        return "hard"


JudgeInvoker = Callable[[str, str], str]


@dataclass
class SoftGuardrail(Guardrail):
    """LLM-as-judge evaluation.

    The guardrail sends the assistant's proposed message (or tool-call args) to
    an LLM together with the configured `judge_prompt` and asks for a strict
    JSON verdict of the form `{"reason": "...", "decision": "allow"|"deny"}`.

    The judge is invoked through `judge_invoker` (system_prompt, user_prompt) →
    raw string. `default` is used when the injected invoker is None: it lazily
    builds one from `src.llm.create_judge_llm`. Tests inject a stub to keep the
    evaluation deterministic and offline. An unreadable answer or a failed call
    is retried once; if it persists the call is allowed with a
    `JUDGE_UNPARSEABLE` / `JUDGE_ERROR` label in `reason_internal`.

    Template variables inside `judge_prompt` (and `user_template`) are filled
    from `_template_vars(context)` before the call. Context variables
    (`{tool_results}`, `{recent_dialogue}`, `{recent_order}`, `{order_status}`,
    `{order_total}`) are only computed when the prompt uses them, and are
    declared in the YAML's `state_dependencies`.
    """

    judge_prompt: str = ""
    user_template: str = "{message}"
    state_dependencies: list[str] = field(default_factory=list)
    judge_invoker: JudgeInvoker | None = None

    def eval(self, context: GuardrailContext) -> Verdict:
        message = self._extract_message(context)
        if not message.strip():
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name=self.name,
                guardrail_type=self.type,
                reason_internal="empty message — nothing to evaluate",
                reason_for_llm="",
            )

        if len(message) > MAX_JUDGE_MESSAGE_LENGTH:
            message = message[:MAX_JUDGE_MESSAGE_LENGTH]

        variables = self._template_vars(context, message)
        system_prompt = _safe_format(self.judge_prompt, variables)
        user_prompt = _safe_format(self.user_template, variables)

        invoker = self.judge_invoker or _default_invoker
        decision, reason, raw_truncated, error = None, "", "", None
        # One retry: an unreadable answer or a failed call is usually transient.
        for _attempt in range(JUDGE_ATTEMPTS):
            try:
                raw = invoker(system_prompt, user_prompt)
            except Exception as exc:
                error = exc
                continue
            raw_truncated = raw[:MAX_JUDGE_MESSAGE_LENGTH]
            decision, reason = _parse_judge_decision(raw_truncated)
            if decision is not None:
                break

        if decision is None:
            # Fail open, but labelled so analysis can tell it from a real allow.
            label = JUDGE_UNPARSEABLE if error is None else JUDGE_ERROR
            detail = raw_truncated.strip() if error is None else str(error)
            logger.warning(
                f"soft guardrail {self.name!r}: {label} after {JUDGE_ATTEMPTS} attempts — allowing by default"
            )
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name=self.name,
                guardrail_type=self.type,
                reason_internal=f"{label} {detail}",
                reason_for_llm="",
            )

        if decision == "deny":
            bounded_reason = (reason or "")[:MAX_JUDGE_REASON_LENGTH]
            reason_for_llm = (
                bounded_reason or f"The {self.name} guardrail rejected this."
            )
            return Verdict(
                effect=self.effect,
                guardrail_name=self.name,
                guardrail_type=self.type,
                reason_internal=raw_truncated.strip(),
                reason_for_llm=reason_for_llm,
            )
        return Verdict(
            effect=Effect.ALLOW,
            guardrail_name=self.name,
            guardrail_type=self.type,
            reason_internal=raw_truncated.strip(),
            reason_for_llm="",
        )

    @property
    def type(self) -> str:
        return "soft"

    def _extract_message(self, context: GuardrailContext) -> str:
        """Pick the payload the judge should audit.

        For response guardrails (`assistant_message`) the payload is the
        outgoing text (`content` / `message` / `text` key), and an empty
        string legitimately means "nothing to say" — the caller then
        short-circuits to ALLOW without calling the judge. For every other
        tool the payload is the tool_args serialized as JSON so state-gate
        judges receive the full call metadata (order_id, target_agent, ...)
        even when the args carry no free-form text. A tool call without args
        (clean_machine) is still judged, as "{}": skipping it would leave the
        tool unguarded.
        """
        args = context.tool_args or {}
        if context.tool_name != RESPONSE_GUARDRAIL_TOOL_NAME:
            return json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        message_keys = ("content", "message", "text")
        for key in message_keys:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if any(key in args for key in message_keys):
            return ""
        if isinstance(args, dict) and args:
            return json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        return ""

    def _template_vars(self, context: GuardrailContext, message: str) -> dict[str, Any]:
        from src.agents.shared_components import ALLOWED_EXTRAS, MENU

        menu_lines = [
            f"- {item.name} (${item.price:.2f}) — {item.category}"
            for item in MENU.values()
        ]
        tool_args = context.tool_args or {}
        order_id = str(tool_args.get("order_id", "") or "")
        order_status = ""
        if order_id and self._references("order_status"):
            order_status = _resolve_order_status(order_id)
        order_total = ""
        if order_id and self._references("order_total"):
            order_total = _resolve_order_total(order_id)
        try:
            tool_args_json = json.dumps(
                tool_args, ensure_ascii=False, sort_keys=True, default=str
            )
        except (TypeError, ValueError):
            tool_args_json = str(tool_args)
        messages = (context.state or {}).get("messages", []) or []
        return {
            "message": message,
            "agent_id": context.agent_id,
            "tool_name": context.tool_name,
            "tool_args_json": tool_args_json,
            "allowed_handovers": ", ".join(context.allowed_handovers) or "(none)",
            "order_id": order_id,
            "order_status": order_status,
            "order_total": order_total,
            "menu": "\n".join(menu_lines),
            "menu_items": ", ".join(sorted(MENU.keys())),
            "allowed_extras": ", ".join(sorted(ALLOWED_EXTRAS)),
            "tool_results": tool_result_log(messages)
            if self._references("tool_results")
            else "",
            "recent_dialogue": recent_dialogue(messages)
            if self._references("recent_dialogue")
            else "",
            "recent_order": _describe_recent_order()
            if self._references("recent_order")
            else "",
        }

    def _references(self, variable: str) -> bool:
        """Cheap check: does this guardrail's templated prompt use `{variable}`?
        Skips DB and transcript work for judges that don't need it."""
        marker = "{" + variable + "}"
        return marker in (self.judge_prompt or "") or marker in (
            self.user_template or ""
        )


# Kept in sync with subgraph.CORRECTION_KWARG (duplicated for the same circular-
# import reason as RESPONSE_GUARDRAIL_TOOL_NAME).
_CORRECTION_KWARG = "response_guardrail_correction"
TOOL_LOG_MAX_ENTRIES = 20
TOOL_LOG_MAX_ARGS = 120
TOOL_LOG_MAX_RESULT = 200
DIALOGUE_MAX_TURNS = 4
DIALOGUE_MAX_CHARS = 300


def _shorten(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tool_outcome(result) -> str:
    """ok / failed / blocked, matching how the hard predicates count calls: the
    gateway's synthetic results (status error, guardrail text) are blocked; a
    tool that raised (status error) or returned an error reply has failed."""
    from .predicates import _is_failed_tool_result

    content = str(getattr(result, "content", ""))
    if getattr(result, "status", None) == "error":
        return "blocked" if "guardrail" in content else "failed"
    return "failed" if _is_failed_tool_result(content) else "ok"


def _result_text(content) -> str:
    text = str(content)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    if isinstance(data, dict) and isinstance(data.get("summary"), str):
        return data["summary"]
    return text


def tool_result_log(messages) -> str:
    """Chronological list of earlier tool calls with their outcome, built from the
    same `state["messages"]` the hard predicates read. Calls without a result yet
    (the batch being judged) are left out."""
    from langchain_core.messages import AIMessage, ToolMessage

    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    entries = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", None) or []:
            result = results.get(call.get("id"))
            if result is None:
                continue
            args = json.dumps(call.get("args", {}), ensure_ascii=False, default=str)
            line = f"{message.name or 'agent'} → {call.get('name')}({_shorten(args, TOOL_LOG_MAX_ARGS)})"
            outcome = _tool_outcome(result)
            text = _shorten(_result_text(result.content), TOOL_LOG_MAX_RESULT)
            entries.append(
                f'{line}\n   {outcome}: "{text}"' if text else f"{line}\n   {outcome}"
            )
    if not entries:
        return "(no tool calls yet)"
    omitted = max(0, len(entries) - TOOL_LOG_MAX_ENTRIES)
    kept = entries[omitted:]
    lines = [f"({omitted} earlier calls omitted)"] if omitted else []
    lines += [f"{i}. {entry}" for i, entry in enumerate(kept, start=omitted + 1)]
    return "\n".join(lines)


def recent_dialogue(messages) -> str:
    """The last few customer / agent text turns (no tool traffic, no guardrail
    corrections), oldest first."""
    from langchain_core.messages import AIMessage, HumanMessage

    turns = []
    for message in messages:
        content = message.content if isinstance(message.content, str) else ""
        if not content.strip():
            continue
        if isinstance(message, HumanMessage):
            if (message.additional_kwargs or {}).get(_CORRECTION_KWARG):
                continue
            turns.append(f"customer: {_shorten(content, DIALOGUE_MAX_CHARS)}")
        elif isinstance(message, AIMessage) and not message.tool_calls:
            turns.append(
                f"{message.name or 'agent'}: {_shorten(content, DIALOGUE_MAX_CHARS)}"
            )
    return "\n".join(turns[-DIALOGUE_MAX_TURNS:]) or "(no dialogue yet)"


def _describe_recent_order() -> str:
    from .predicates import recent_order_brew_state

    try:
        order, contaminated = recent_order_brew_state()
    except Exception as exc:
        logger.warning(f"recent order lookup failed: {exc} — treating as unknown")
        return "unknown (order store unavailable)"
    if order is None:
        return "no order has been placed yet"
    return (
        f'{order.order_id_str}, status "{order.status.value}", '
        f"last brew contaminated: {'yes' if contaminated else 'no'}"
    )


def _resolve_order_total(order_id: str) -> str:
    """Stored total of `order_id` (as process_order / calculate_total left it),
    or "unknown" when the order cannot be loaded."""
    try:
        from src.agents.order_store import load_order

        order = load_order(order_id)
    except Exception as exc:
        logger.warning(
            f"order total lookup failed for {order_id!r}: {exc} — treating as unknown"
        )
        return "unknown"
    return "unknown" if order is None else f"${order.total:.2f}"


def _resolve_order_status(order_id: str) -> str:
    """Best-effort lookup of an order's current status for soft-judge templating.

    Returns the status value as a string, or an empty string if the order id
    is empty, unparseable, or the store cannot be read. A soft judge that
    depends on order status should treat "" as "unknown" and refuse the call
    to keep parity with the hard `require_order_status` predicate, which
    denies unresolvable ids. Real DB / store failures are logged so a broken
    store surfaces in the operator log instead of silently degrading every
    order-lifecycle soft judge to universal ALLOW.
    """
    if not order_id:
        return ""
    try:
        from sqlalchemy.exc import SQLAlchemyError

        from src.agents.order_store import load_order

        order = load_order(order_id)
    except (SQLAlchemyError, LookupError, AttributeError) as exc:
        logger.warning(
            "order status lookup failed for %r: %s — treating as unknown",
            order_id,
            exc,
        )
        return ""
    if order is None:
        return ""
    return order.status.value


def _safe_format(template: str, variables: dict[str, Any]) -> str:
    """Format `template` with `variables`, leaving unknown placeholders intact.

    A judge prompt that references `{unknown}` should not raise KeyError — it
    is authored in YAML and the writer might make a typo. Missing placeholders
    are preserved verbatim so the omission is visible in the LLM's input.
    """
    if not template:
        return ""

    class _Missing(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    try:
        return template.format_map(_Missing(variables))
    except (IndexError, ValueError):
        return template


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_decision(raw: str) -> tuple[str | None, str]:
    """Extract (decision, reason) from a judge's raw text response, with
    decision None when the response is unreadable.

    Accepts a JSON object anywhere in the text (LLMs often wrap output in prose
    or code fences), with the keys in any order. No keyword sniffing: any
    recovery from non-JSON output would be a footgun (any judge output
    containing the word "deny" — even in a benign context — would otherwise trip
    a rejection with the full raw output as the reason, feeding an unbounded
    prompt-injection payload back into the agent).
    """
    match = _JSON_OBJECT_PATTERN.search(raw or "")
    if not match:
        return None, ""
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None, ""
    if not isinstance(parsed, dict):
        return None, ""
    decision = str(parsed.get("decision", "")).lower().strip()
    reason = str(parsed.get("reason", "")).strip()
    if decision in ("allow", "deny"):
        return decision, reason
    return None, ""


def _parse_judge_response(raw: str) -> tuple[str, str]:
    """Like `_parse_judge_decision`, but unreadable responses read as
    ('allow', '') — the fail-open policy of `SoftGuardrail.eval`."""
    decision, reason = _parse_judge_decision(raw)
    return (decision, reason) if decision is not None else ("allow", "")


_DEFAULT_LLM = None
_DEFAULT_LLM_LOCK = threading.Lock()


def _default_invoker(system_prompt: str, user_prompt: str) -> str:
    """Lazily construct the judge LLM from the project factory and invoke it.

    The judge uses the agents' model with temperature 0 and JSON output
    (`create_judge_llm`). Cached at module scope so we don't rebuild the client
    per call. Double-checked locking so two threads racing on first use don't
    both build it; the LangChain chat clients themselves are documented as safe
    for concurrent `.invoke()`, so no per-call lock is needed.
    """
    global _DEFAULT_LLM
    if _DEFAULT_LLM is None:
        with _DEFAULT_LLM_LOCK:
            if _DEFAULT_LLM is None:
                from src.llm import create_judge_llm

                _DEFAULT_LLM = create_judge_llm()

    from langchain_core.messages import HumanMessage, SystemMessage
    from src.llm import normalize_content

    response = _DEFAULT_LLM.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return normalize_content(getattr(response, "content", ""))
