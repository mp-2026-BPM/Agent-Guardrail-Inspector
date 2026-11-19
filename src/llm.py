import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger("coffee_shop.llm")

load_dotenv()


def create_chat_llm():
    """Create a chat LLM instance based on environment configuration."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER is set to 'anthropic' but ANTHROPIC_API_KEY is not set. "
                "Please set it in your .env file."
            )
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "anthropic--claude-4.6-opus"),
            base_url=os.getenv(
                "ANTHROPIC_BASE_URL", "http://localhost:6655/anthropic/"
            ),
            api_key=api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "ministral-3:14b"))

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{provider}'. Supported values: 'ollama', 'anthropic'."
    )


def create_judge_llm():
    """Chat LLM for soft-guardrail judges: the agents' model, called with
    temperature 0 and (for Ollama) JSON output mode.

    Both are per-request options, so Ollama reuses the model it already has
    loaded for the agents; nothing that forces a reload (such as `num_ctx`) is
    changed here.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()

    if provider == "anthropic":
        llm = create_chat_llm()
        return llm.model_copy(update={"temperature": 0})

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "ministral-3:14b"),
            temperature=0,
            format="json",
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{provider}'. Supported values: 'ollama', 'anthropic'."
    )


class _HandoffDeferrer:
    """Callable that defers handoff tool calls when they co-occur with
    other tool calls, then re-injects them once the model next responds
    with no tool calls (i.e. after it finishes its non-handoff work).

    At most one handoff is stored; if the model emits multiple handoffs
    in a single response only the first is kept.

    Each agent gets its own instance (created in ``bind_tools_sequential``),
    so state never leaks between agents.  Within a single conversation the
    react-agent loop calls the model sequentially, which keeps the
    ``_pending`` bookkeeping safe without locks.
    """

    def __init__(self):
        self._pending = None
        self._defer_count = 0

    def __call__(self, message):
        tool_calls = getattr(message, "tool_calls", None) or []
        names = [tc["name"] for tc in tool_calls]
        logger.debug(
            "model returned tool_calls=%s | pending=%s",
            names,
            self._pending["name"] if self._pending else None,
        )

        if self._pending is not None:
            has_handoff_now = any(
                tc["name"].startswith("transfer_to_") for tc in tool_calls
            )
            if has_handoff_now:
                logger.info(
                    "LLM re-issued handoff — discarding old pending "
                    "(new message will be re-evaluated)"
                )
                self._pending = None
                self._defer_count = 0
            elif not tool_calls:
                logger.info(
                    "injecting deferred handoff %s into text-only response",
                    self._pending["name"],
                )
                message.tool_calls = [self._pending]
                self._pending = None
                self._defer_count = 0
                return message
            else:
                self._defer_count += 1
                logger.debug(
                    "LLM made more non-handoff calls %s — keeping pending "
                    "(defer count: %d)",
                    names,
                    self._defer_count,
                )

        if len(tool_calls) <= 1:
            return message

        handoff = [tc for tc in tool_calls if tc["name"].startswith("transfer_to_")]
        non_handoff = [
            tc for tc in tool_calls if not tc["name"].startswith("transfer_to_")
        ]

        if handoff and non_handoff:
            if len(handoff) > 1:
                logger.warning(
                    "multiple handoffs in one response — keeping only first: %s",
                    [tc["name"] for tc in handoff],
                )
            logger.info(
                "stripping %s (will defer); keeping %s",
                handoff[0]["name"],
                [tc["name"] for tc in non_handoff],
            )
            self._pending = handoff[0]
            self._defer_count = 0
            message.tool_calls = non_handoff

        return message


def bind_tools_sequential(llm, tools):
    """bind_tools with parallel-call mitigation.

    ChatAnthropic accepts ``parallel_tool_calls=False``; ChatOllama forwards
    the kwarg to its underlying client, which raises TypeError at invoke time.
    For Ollama we instead post-process: if a handoff tool call co-occurs with
    other tool calls, the handoff is deferred and automatically re-injected
    once the LLM finishes its non-handoff work.
    """
    if type(llm).__name__ == "ChatOllama":
        return llm.bind_tools(tools) | _HandoffDeferrer()
    return llm.bind_tools(tools, parallel_tool_calls=False)


def normalize_content(content):
    """Normalize LLM message content to a plain string.

    Anthropic returns content as a list of blocks when tool calls are present;
    Ollama always returns a string. This ensures a consistent string everywhere.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content)
