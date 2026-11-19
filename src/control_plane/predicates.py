import json
import re

from src.agents.order_store import load_order, load_recent_order

from .types import Effect, GuardrailContext, Verdict

_ORDER_ID_RE = re.compile(r"\bORD\d{3,}\b")


def _allowed_handover_targets_eval(
    context: GuardrailContext, effect: str = "deny"
) -> Verdict:
    target = context.tool_args.get("target_agent", "")
    allowed = context.allowed_handovers
    if target in allowed:
        return Verdict(
            effect=Effect.ALLOW,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"{target!r} is in allowed_handovers={allowed}",
        )
    return Verdict(
        effect=Effect(effect),
        guardrail_name="",
        guardrail_type="",
        reason_internal=f"{target!r} not in allowed_handovers={allowed} for agent {context.agent_id!r}",
        reason_for_llm=(
            f"Handover to {target!r} is not allowed from {context.agent_id!r}. "
            f"You may only transfer to: {', '.join(allowed) or '(none)'}."
        ),
    )


def allowed_handover_targets_predicate(*args, **kwargs):
    """Dual-use: called directly by the gateway with a GuardrailContext, OR called
    as a factory with keyword `effect` to bind a non-default violation effect.

    Setups without `predicate_args` (baseline, most others) get direct-call
    semantics equivalent to the historical hard-DENY behavior. Setups that supply
    `predicate_args: {effect: flag}` (e.g. baseline_flag) get a configured
    evaluator that flags instead of denies.
    """
    if args and isinstance(args[0], GuardrailContext):
        return _allowed_handover_targets_eval(args[0])
    effect = kwargs.get("effect", "deny")

    def _bound(context: GuardrailContext) -> Verdict:
        return _allowed_handover_targets_eval(context, effect=effect)

    return _bound


def discount_within_limit_predicate(max_pct: int, effect: str = "flag"):
    """Factory: verdict when calculate_total is called with discount_percent above max_pct.

    `effect` selects deny vs flag on violation; defaults to flag so existing setups
    (which pass only `max_pct`) keep their observe-only behavior.
    """
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        pct = int(context.tool_args.get("discount_percent", 0) or 0)
        if pct <= max_pct:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"discount_percent={pct} within limit {max_pct}",
            )
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"discount_percent={pct} exceeds limit {max_pct}",
            reason_for_llm=(
                f"A discount of {pct}% is not allowed; the maximum permitted discount is {max_pct}%."
            ),
        )

    return _eval


def _order_placed_in_conversation(context: GuardrailContext) -> bool:
    """True once a process_order result carrying an ORD id is in the conversation.
    process_order reports failures as plain "Error ..." strings without an id, so the
    id is the success signal."""
    return any(
        getattr(m, "name", None) == "process_order"
        and getattr(m, "status", None) != "error"
        and _ORDER_ID_RE.search(str(getattr(m, "content", "")))
        for m in context.state.get("messages", []) or []
    )


def _transfer_includes_order_id_eval(
    context: GuardrailContext,
    effect: str = "flag",
    only_after_order_placed: bool = False,
) -> Verdict:
    summary = str(context.tool_args.get("context_summary", ""))
    if _ORDER_ID_RE.search(summary):
        return Verdict(
            effect=Effect.ALLOW,
            guardrail_name="",
            guardrail_type="",
            reason_internal="transfer includes order_id",
        )
    if only_after_order_placed and not _order_placed_in_conversation(context):
        return Verdict(
            effect=Effect.ALLOW,
            guardrail_name="",
            guardrail_type="",
            reason_internal="transfer missing order_id, but no order placed yet in this conversation",
        )
    return Verdict(
        effect=Effect(effect),
        guardrail_name="",
        guardrail_type="",
        reason_internal=f"transfer missing order_id ({effect})",
        reason_for_llm=(
            "When transferring to another agent, always include the order id so the next agent can look up order details."
            " Include the order id when handing over in this pattern: ORDXXXX"
        ),
    )


def transfer_includes_order_id_predicate(*args, **kwargs):
    """Dual-use, like `allowed_handover_targets_predicate`: called directly with a
    GuardrailContext it FLAGs a transfer_to_agent without an ORD#### order id (the
    historical behavior). Called as a factory with `effect` and/or
    `only_after_order_placed` it binds those; the latter exempts handovers that happen
    before any process_order succeeded, when there is no id to reference yet.
    """
    if args and isinstance(args[0], GuardrailContext):
        return _transfer_includes_order_id_eval(args[0])
    effect = kwargs.get("effect", "flag")
    only_after_order_placed = bool(kwargs.get("only_after_order_placed", False))

    def _bound(context: GuardrailContext) -> Verdict:
        return _transfer_includes_order_id_eval(
            context, effect=effect, only_after_order_placed=only_after_order_placed
        )

    return _bound


def require_order_status_predicate(allowed: list[str], effect: str = "deny"):
    """Factory: block (or flag) a tool call unless the order's *current* status is in
    `allowed`. This is how the order lifecycle is enforced now that there is no state
    machine — each state-gated tool declares the source statuses it may be called from.

    The order is resolved from the tool's `order_id` argument. The transition target the
    tool computes at runtime is always legal from an allowed source, so a source-status
    precondition is equivalent to validating the transition. A missing or unresolvable
    `order_id` yields the same violation effect as an out-of-set status: hallucinated
    IDs must not silently bypass the gate. `effect` selects deny vs flag.
    """
    allowed_set = {str(s) for s in allowed}
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        order_id = str(context.tool_args.get("order_id", ""))
        order = load_order(order_id) if order_id else None
        if order is None:
            return Verdict(
                effect=violation_effect,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"order {order_id!r} not resolvable; treated as violation",
                reason_for_llm=(
                    f"Cannot call {context.tool_name!r} for order {order_id!r}: "
                    f"the order does not exist. Verify the order id before retrying."
                ),
            )
        current = order.status.value
        if current in allowed_set:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"order {order_id} status={current} in allowed={sorted(allowed_set)}",
            )
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"order {order_id} status={current} not in allowed={sorted(allowed_set)}",
            reason_for_llm=(
                f"Cannot call {context.tool_name!r} while order {order_id} is '{current}'. "
                f"This tool is only valid when the order status is one of: "
                f"{', '.join(sorted(allowed_set))}."
            ),
        )

    return _eval


def order_size_within_range_predicate(
    max_units: int, min_units: int = 1, effect: str = "deny"
):
    """Factory: constrain order size at process_order time. Size is counted in *units*
    (summed quantities) read straight from the proposed `tool_args["order"]` — the order
    row does not exist yet, so there is no order_id to load. Violation when total units
    fall outside [min_units, max_units]. Missing/malformed `order` → ALLOW (nothing to
    evaluate; the tool itself validates the payload).
    """
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        order = context.tool_args.get("order")
        if not isinstance(order, list):
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal="no order list in tool_args; size not evaluated",
            )
        units = sum(
            int(item.get("quantity", 1) or 1)
            for item in order
            if isinstance(item, dict)
        )
        if min_units <= units <= max_units:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"order size {units} units within [{min_units}, {max_units}]",
            )
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"order size {units} units outside [{min_units}, {max_units}]",
            reason_for_llm=(
                f"This order has {units} item(s), which is outside the permitted range of "
                f"{min_units}–{max_units}. Adjust the order to fit within the allowed size."
            ),
        )

    return _eval


def refund_within_limit_predicate(
    max_pct: int, min_pct: int | None = None, effect: str = "deny"
):
    """Factory: constrain offer_partial_refund's `refund_percent`. Violation when the
    requested percentage exceeds `max_pct`, or falls below `min_pct` when one is set
    (e.g. `min_pct: 0` rejects negative refunds). Mirrors the tool default of 50 when
    absent.
    """
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        pct = int(context.tool_args.get("refund_percent", 50) or 0)
        if min_pct is not None and pct < min_pct:
            return Verdict(
                effect=violation_effect,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"refund_percent={pct} below minimum {min_pct}",
                reason_for_llm=(
                    f"A partial refund of {pct}% is not allowed; the refund must be at least {min_pct}%."
                ),
            )
        if pct <= max_pct:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"refund_percent={pct} within limit {max_pct}",
            )
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"refund_percent={pct} exceeds limit {max_pct}",
            reason_for_llm=(
                f"A partial refund of {pct}% is not allowed; the maximum permitted refund is {max_pct}%."
            ),
        )

    return _eval


def _is_failed_tool_result(content) -> bool:
    """True when a tool *returned* a failure instead of raising: the tools report
    bad input as `"Error..."` text (process_order, refunds) or as JSON with
    `"status": "error"` (barista, tray), while the ToolMessage status stays success."""
    text = str(content).lstrip()
    if text.startswith("Error"):
        return True
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(data, dict) and data.get("status") == "error"


def max_tool_calls_predicate(tool_name: str, max_calls: int, effect: str = "deny"):
    """Factory: cap the number of successful invocations of `tool_name` in one
    conversation. Counts prior ToolMessages in `state["messages"]` whose `name`
    matches, whose `status` is not "error" (i.e. previously allowed by the
    gateway and executed by the tool) and whose content is not a failure the tool
    returned (see `_is_failed_tool_result`), so a rejected process_order does not
    use up the only allowed call. Violation when the count is already at or
    above `max_calls` at the moment of evaluation — remember the gateway runs
    *before* the current call, so `>= max_calls` blocks the (max+1)-th attempt.
    """
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        messages = context.state.get("messages", []) or []
        prior = sum(
            1
            for m in messages
            if getattr(m, "name", None) == tool_name
            and getattr(m, "status", None) != "error"
            and not _is_failed_tool_result(getattr(m, "content", ""))
        )
        if prior < max_calls:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"prior {tool_name!r} calls={prior} < max={max_calls}",
            )
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"prior {tool_name!r} calls={prior} >= max={max_calls}",
            reason_for_llm=(
                f"You have already called {tool_name!r} {prior} time(s) in this "
                f"conversation; the limit is {max_calls}. Continue with this order "
                f", you cannot use this tool again."
            ),
        )

    return _eval


def order_total_within_limit_predicate(max_total: float, effect: str = "deny"):
    """Factory: constrain the order's total (in dollars). Resolves the order from
    `tool_args["order_id"]` and reads `order.total`; violation when it exceeds `max_total`.
    Unresolvable orders → ALLOW. Enforced at calculate_total, so it reads the pre-discount
    total set by process_order.
    """
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        order_id = str(context.tool_args.get("order_id", ""))
        order = load_order(order_id) if order_id else None
        if order is None:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"order {order_id!r} not resolvable; total not evaluated",
            )
        if order.total <= max_total:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"order {order_id} total ${order.total:.2f} within limit ${max_total:.2f}",
            )
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"order {order_id} total ${order.total:.2f} exceeds limit ${max_total:.2f}",
            reason_for_llm=(
                f"Order {order_id} totals ${order.total:.2f}, which exceeds the maximum permitted "
                f"order total of ${max_total:.2f}."
            ),
        )

    return _eval


def process_order_items_on_menu_predicate(effect: str = "deny"):
    """Factory: deterministic counterpart to the soft `assistant_message:on_menu_only`
    judge. Blocks process_order when any item name (after lower/strip) is not in MENU.
    Missing/malformed `order` → ALLOW; the tool itself validates payload shape.
    """
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        # Lazy import: mirrors _template_vars in guardrails.py; avoids agent-package
        # cold-import cost on control-plane load.
        from src.agents.shared_components import ALLOWED_EXTRAS, MENU

        order = context.tool_args.get("order")
        if not isinstance(order, list):
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal="no order list in tool_args; menu adherence not evaluated",
            )
        bad = []
        for item in order:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).lower().strip()
            if name and name not in MENU:
                bad.append(name)
        if not bad:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal="all order items on MENU",
            )
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"off-menu items in order: {bad}",
            reason_for_llm=(
                f"Cannot place this order: item(s) {bad} are not on the menu. "
                f"The menu is: {sorted(MENU.keys())}. "
                f"Allowed extras/modifiers (which are NOT standalone items): {sorted(ALLOWED_EXTRAS)}. "
                f"Adjust the order to only include on-menu items."
            ),
        )

    return _eval


def transfer_context_summary_nonempty_predicate(
    min_chars: int = 20, effect: str = "deny"
):
    """Factory: deny transfer_to_agent when `context_summary` (after strip) is
    shorter than `min_chars`. Catches empty strings and bare filler ("handoff").
    """
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        summary = str(context.tool_args.get("context_summary", "")).strip()
        if len(summary) >= min_chars:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"context_summary length {len(summary)} >= min {min_chars}",
            )
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"context_summary length {len(summary)} < min {min_chars}",
            reason_for_llm=(
                f"Cannot hand off: the context_summary is too short "
                f"({len(summary)} chars; need at least {min_chars}). Provide a "
                f"short sentence describing what the next agent should do, including "
                f"the order id if one exists."
            ),
        )

    return _eval


def recent_order_brew_state():
    """(most recent order or None, whether the barista cache flags its last brew
    as contaminated). Shared by the hard clean_machine predicate and the soft
    judges' `{recent_order}` context, so both read the same facts."""
    order = load_recent_order()
    if order is None:
        return None, False
    try:
        from src.agents.barista_agent import ORDER_STATUS_CACHE

        cache_entry = ORDER_STATUS_CACHE.get(
            str(order.id), {}
        ) or ORDER_STATUS_CACHE.get(f"ORD{int(order.id):04d}", {})
        return order, bool(cache_entry.get("last_brew_contaminated"))
    except Exception:
        return order, False


def clean_machine_only_after_error_predicate(effect: str = "deny"):
    """Factory: allow clean_machine only when the most-recent order is in
    `preparation_error` or when the barista cache flags the last brew as
    contaminated. `clean_machine` has no `order_id` arg, so we fall back to
    `load_recent_order()` — correct for the single-track swarm.
    """
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        order, contaminated = recent_order_brew_state()
        if order is None:
            # No recent order (fresh session / test harness) — nothing to gate.
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal="no recent order; clean_machine not gated",
            )
        status = order.status.value
        if status == "preparation_error" or contaminated:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=(
                    f"order {order.id} status={status} contaminated={contaminated}; clean allowed"
                ),
            )
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=(
                f"order {order.id} status={status} contaminated={contaminated}; clean_machine not needed"
            ),
            reason_for_llm=(
                f"clean_machine is only needed after a brew failure or contamination. "
                f"The current order is '{status}' and the last brew was clean — cleaning "
                f"now will not help. Continue with the normal preparation flow."
            ),
        )

    return _eval


_DOLLAR_AMOUNT_RE = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)")
# A message amount is either "$4.50" or "4.50 dollars"/"4 USD". The optional leading
# hedge word marks vague statements ("around $5"), which the rule lets through.
_QUOTED_PRICE_RE = re.compile(
    r"(?:\b(around|about|roughly|approximately|approx\.?|under|over|less than|more than)\s+|(~)\s*)?"
    r"(?:\$\s?(\d+(?:\.\d{1,2})?)|\b(\d+(?:\.\d{1,2})?)\s*(?:dollars?|usd|bucks)\b)",
    re.IGNORECASE,
)


def _to_cents(amount: str | float) -> int:
    return round(float(amount) * 100)


def _json_number_cents(content: str) -> set[int]:
    """Every numeric value in a JSON tool result, in cents (bools excluded). Tool results
    like offer_partial_refund carry amounts such as `original_total` only as JSON numbers,
    never as `$` text."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return set()
    cents: set[int] = set()
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            cents.add(_to_cents(node))
    return cents


def no_unbacked_price_quotes_predicate(
    source_tools: list[str] | None = None,
    count_json_numbers: bool = False,
    effect: str = "deny",
):
    """Factory: deterministic counterpart to the soft `assistant_message:no_price_promises`
    judge. Applied to the synthetic `assistant_message` tool: every specific dollar amount
    in the outgoing message must already appear as a `$` amount in an earlier successful
    result of one of `source_tools` (default: calculate_total, process_order) in this
    conversation. With `count_json_numbers`, numeric fields of a JSON result (e.g.
    `original_total`) also back an amount. Hedged amounts ("around $5") and spelled-out
    figures ("five dollars") are not specific quotes and pass.
    """
    sources = set(source_tools or ["calculate_total", "process_order"])
    violation_effect = Effect(effect)

    def _backed_cents(content: str) -> set[int]:
        cents = {_to_cents(a) for a in _DOLLAR_AMOUNT_RE.findall(content)}
        if count_json_numbers:
            cents |= _json_number_cents(content)
        return cents

    def _eval(context: GuardrailContext) -> Verdict:
        content = str(context.tool_args.get("content", ""))
        quoted = {
            dollars or plain
            for hedge, tilde, dollars, plain in _QUOTED_PRICE_RE.findall(content)
            if not (hedge or tilde)
        }
        if not quoted:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal="no specific dollar amount in message",
            )
        backed = {
            cents
            for m in context.state.get("messages", []) or []
            if getattr(m, "name", None) in sources
            and getattr(m, "status", None) != "error"
            for cents in _backed_cents(str(getattr(m, "content", "")))
        }
        unbacked = sorted(
            (a for a in quoted if _to_cents(a) not in backed), key=_to_cents
        )
        if not unbacked:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"all quoted amounts {sorted(quoted, key=_to_cents)} backed by {sorted(sources)}",
            )
        amounts = ", ".join(f"${a}" for a in unbacked)
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"unbacked price quote(s) {unbacked}; backed cents={sorted(backed)}",
            reason_for_llm=(
                f"Your message quotes {amounts}, but no {' or '.join(sorted(sources))} result "
                f"in this conversation produced that amount. Run the tool first, or do not "
                f"quote a specific price."
            ),
        )

    return _eval


# Clause boundaries for the off-menu check: punctuation, line breaks, and "but"
# ("we don't have chai, but how about a mocha?" splits into two clauses).
_CLAUSE_SPLIT_RE = re.compile(r"[.!?\n;,:—–()]+|\bbut\b", re.IGNORECASE)
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|cannot|without|unavailable|skip|skipping|instead of)\b|n['’]t\b",
    re.IGNORECASE,
)


def no_off_menu_mentions_predicate(off_menu_terms: list[str], effect: str = "deny"):
    """Factory: deterministic counterpart to the soft `assistant_message:on_menu_only`
    judge. Splits the outgoing message into clauses and violates when a clause names
    one of `off_menu_terms` (whole-word, case-insensitive) without a negation cue, so
    "we don't serve mocha" passes while "how about a mocha?" does not. Deliberately
    a blocklist: terms not listed pass, trading recall for few false refusals.
    Flavour shorthand built from allowed extras ("vanilla latte") is not listed.
    """
    terms = sorted(
        {t.lower().strip() for t in off_menu_terms if t.strip()}, key=len, reverse=True
    )
    if not terms:
        raise ValueError("off_menu_terms must not be empty")
    term_re = re.compile(
        r"\b(" + "|".join(map(re.escape, terms)) + r")\b", re.IGNORECASE
    )
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        content = str(context.tool_args.get("content", ""))
        hits: list[str] = []
        for clause in _CLAUSE_SPLIT_RE.split(content):
            found = [t.lower() for t in term_re.findall(clause)]
            if found and not _NEGATION_RE.search(clause):
                hits.extend(t for t in found if t not in hits)
        if not hits:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal="no un-negated off-menu term in message",
            )
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"un-negated off-menu term(s): {hits}",
            # The menu itself reaches the agent through the guardrail's correction
            # hint, which the soft judge gets too.
            reason_for_llm=(
                f"Your message offers {', '.join(repr(h) for h in hits)}, which is not on the menu."
            ),
        )

    return _eval


_REFUND_CUE_RE = re.compile(
    r"\b(?:refund\w*|money back|reimburs\w*|compensat\w*|complain\w*|dissatisf\w*|unhappy|store credit)\b",
    re.IGNORECASE,
)
_NEW_ORDER_CUE_RE = re.compile(
    r"\b(?:new order|place (?:an?|their|the) order|wants? to order|take (?:an?|their|the) order|pick ?up|ready for pickup)\b",
    re.IGNORECASE,
)
_REPLACEMENT_CUE_RE = re.compile(
    r"\b(?:new order|replace\w*|re-?order\w*|remake|fresh|another)\b", re.IGNORECASE
)
_ISSUE_CUE_RE = re.compile(
    r"\b(?:refund\w*|complain\w*|issue|problem|error|fail\w*|stuck|wrong|cancel\w*|missing|cold|lukewarm|blocked?)\b",
    re.IGNORECASE,
)
# (cue in context_summary, target it must not go to, cue that makes the summary
# ambiguous and therefore allowed). Only clear negative evidence denies.
# A replacement order after a complaint runs the normal order -> inventory ->
# barista flow, and its summaries keep mentioning the complaint, so a replacement
# cue exempts every one of those targets.
_HANDOVER_MISMATCHES = (
    (_REFUND_CUE_RE, "barista_agent", _REPLACEMENT_CUE_RE),
    (_REFUND_CUE_RE, "inventory_agent", _REPLACEMENT_CUE_RE),
    (_REFUND_CUE_RE, "order_agent", _REPLACEMENT_CUE_RE),
    (_NEW_ORDER_CUE_RE, "customer_service_agent", _ISSUE_CUE_RE),
)
_AGENT_SCOPES = {
    "order_agent": "new orders and pricing",
    "inventory_agent": "stock checks and food on the tray",
    "barista_agent": "brewing coffee",
    "customer_service_agent": "complaints and refunds",
}


def handover_scope_mismatch_predicate(effect: str = "deny"):
    """Factory: deterministic counterpart to the soft `handover:reason_matches_target`
    judge. Violates only on clear negative evidence from `_HANDOVER_MISMATCHES`:
    refund/complaint work sent to barista, inventory or order_agent without a
    replacement cue, or new-order/pickup work sent to customer_service without an
    issue cue. Anything ambiguous or procedural is allowed; unknown targets are left
    to `allowed_handover_targets`. Brewing/preparation words are deliberately not
    used — normal order → inventory summaries mention them all the time.
    """
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        target = str(context.tool_args.get("target_agent", ""))
        summary = str(context.tool_args.get("context_summary", ""))
        for cue_re, forbidden, exception_re in _HANDOVER_MISMATCHES:
            if target != forbidden:
                continue
            match = cue_re.search(summary)
            if match and not (exception_re and exception_re.search(summary)):
                cue = match.group(0)
                return Verdict(
                    effect=violation_effect,
                    guardrail_name="",
                    guardrail_type="",
                    reason_internal=f"context_summary cue {cue!r} does not fit target {target!r}",
                    reason_for_llm=(
                        f"Handover to {target!r} does not fit the summary (mentions {cue!r}). "
                        f"{target} handles {_AGENT_SCOPES[target]}. Hand off to the agent "
                        f"responsible for this work instead."
                    ),
                )
        return Verdict(
            effect=Effect.ALLOW,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"no scope mismatch for target {target!r}",
        )

    return _eval


PREDICATE_REGISTRY = {
    "allowed_handover_targets": allowed_handover_targets_predicate,
    "discount_within_limit": discount_within_limit_predicate,
    "transfer_includes_order_id": transfer_includes_order_id_predicate,
    "require_order_status": require_order_status_predicate,
    "order_size_within_range": order_size_within_range_predicate,
    "refund_within_limit": refund_within_limit_predicate,
    "order_total_within_limit": order_total_within_limit_predicate,
    "max_tool_calls": max_tool_calls_predicate,
    "process_order_items_on_menu": process_order_items_on_menu_predicate,
    "transfer_context_summary_nonempty": transfer_context_summary_nonempty_predicate,
    "clean_machine_only_after_error": clean_machine_only_after_error_predicate,
    "no_unbacked_price_quotes": no_unbacked_price_quotes_predicate,
    "no_off_menu_mentions": no_off_menu_mentions_predicate,
    "handover_scope_mismatch": handover_scope_mismatch_predicate,
}
