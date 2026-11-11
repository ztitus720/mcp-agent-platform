"""Retry, timeout and fallback middleware: a failure degrades the answer, never crashes the agent.

Layers (outermost first):
  model:  last-resort catcher -> fallback model (optional)      [transient HTTP retries live in app/llm.py]
  tools:  last-resort catcher -> retry transport failures -> per-call timeout
          [business errors arrive as ToolErrors (error ToolMessages), never retried]
"""

import asyncio

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from mcp import MCPError

from app.policy import AgentPolicy

# Failures where trying the same call again can succeed: the server process died or hung.
# Wrong arguments (ValueError, pydantic ValidationError) are deliberately NOT here.
TRANSIENT_TOOL_ERRORS: tuple[type[Exception], ...] = (MCPError, TimeoutError, ConnectionError, OSError, RuntimeError)


class ToolErrorGuardMiddleware(AgentMiddleware):
    """Last-resort catcher: ANY tool exception (e.g. bad arguments) becomes an error result
    the model can read and react to, instead of crashing the whole turn."""

    @staticmethod
    def _as_message(request, exc: Exception) -> ToolMessage:
        call = request.tool_call
        return ToolMessage(content=tool_error_message(exc), tool_call_id=call["id"],
                           name=call["name"], status="error")

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as exc:
            return self._as_message(request, exc)

    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except Exception as exc:
            return self._as_message(request, exc)


class ToolTimeoutMiddleware(AgentMiddleware):
    """Abort a tool call that does not answer in time, so a hung server cannot freeze the turn."""

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds

    async def awrap_tool_call(self, request, handler):
        return await asyncio.wait_for(handler(request), self.seconds)

    def wrap_tool_call(self, request, handler):
        return handler(request)  # sync path (unit tests only); MCP tools always run async


def model_failure_message(exc: Exception) -> str:
    return (
        f"Sorry, the language model is unavailable right now ({type(exc).__name__}). "
        "Please try again in a moment."
    )


def tool_error_message(exc: Exception) -> str:
    return (
        f"Tool failed ({type(exc).__name__}: {exc}). Check the arguments, or answer without "
        "this tool and tell the user which information could not be retrieved."
    )


def tool_failure_message(exc: Exception) -> str:
    return (
        f"Tool unavailable after retries ({type(exc).__name__}: {exc}). "
        "Answer without this tool and tell the user which information could not be retrieved."
    )


def build_resilience_middleware(
    policy: AgentPolicy, fallback_model: BaseChatModel | None = None
) -> list[AgentMiddleware]:
    middleware: list[AgentMiddleware] = [
        # Outermost model layer: no extra retries (the HTTP client already retries), it only
        # turns a final failure into a polite answer instead of an exception.
        # retry_on=(Exception,) is required: by default "non-retryable" errors such as a 404
        # unknown model are re-raised untouched and would crash the turn.
        ModelRetryMiddleware(max_retries=0, retry_on=(Exception,), on_failure=model_failure_message),
    ]
    if fallback_model is not None:
        middleware.append(ModelFallbackMiddleware(fallback_model))
    middleware += [
        ToolErrorGuardMiddleware(),
        ToolRetryMiddleware(
            max_retries=policy.tool_max_retries,
            retry_on=TRANSIENT_TOOL_ERRORS,
            on_failure=tool_failure_message,
            initial_delay=policy.retry_initial_delay_seconds,
        ),
        ToolTimeoutMiddleware(policy.tool_timeout_seconds),
    ]
    return middleware
