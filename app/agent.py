"""Agent layer: assemble model + MCP tools + policy into a LangGraph ReAct agent.

This file must never name a specific tool. Tools come from app.mcp_client (servers.yaml),
limits and the system prompt from config/policy.yaml, the model from app.llm (.env).
"""

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from app.llm import get_chat_model, get_fallback_model
from app.mcp_client import LoadReport, load_mcp_tools
from app.policy import AgentPolicy, load_policy
from app.resilience import build_resilience_middleware


def build_agent(
    tools: list[BaseTool],
    policy: AgentPolicy,
    model: BaseChatModel | None = None,
    checkpointer: Any = None,
    fallback_model: BaseChatModel | None = None,
):
    """Create the ReAct agent graph. Models are injectable so tests can use scripted ones."""
    return create_agent(
        model=model or get_chat_model(),
        tools=tools,
        system_prompt=policy.system_prompt,
        middleware=[
            # Too many LLM calls in one turn: stop and return what we have.
            ModelCallLimitMiddleware(run_limit=policy.max_model_calls, exit_behavior="end"),
            # Too many tool calls in one turn: refuse further calls, let the model answer.
            ToolCallLimitMiddleware(run_limit=policy.max_tool_calls, exit_behavior="continue"),
            # Retries, timeouts and model fallback: failures degrade the answer, never crash.
            *build_resilience_middleware(policy, fallback_model),
        ],
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )


@dataclass
class AgentRuntime:
    """Everything a front end (CLI, FastAPI) needs: the graph, its policy and the tool report."""

    graph: Any
    policy: AgentPolicy
    tool_report: LoadReport

    def run_config(self, thread_id: str) -> dict:
        """Per-request config: thread_id selects the conversation memory."""
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.policy.recursion_limit,
        }


async def create_runtime() -> AgentRuntime:
    """Load policy and MCP tools, then build the agent. Call once per process / event loop."""
    policy = load_policy()
    report = await load_mcp_tools()
    if not report.tools:
        raise RuntimeError(f"No MCP tools could be loaded: {report.failed}")
    graph = build_agent(report.tools, policy, fallback_model=get_fallback_model())
    return AgentRuntime(graph=graph, policy=policy, tool_report=report)
