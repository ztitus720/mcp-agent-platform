"""Interactive command-line chat with the agent (stage-5 debugging front end).

Run from the project root:
    python -m scripts.cli_chat
Commands:  /new  start a new conversation   /tools  list tools   /exit  quit
Every tool call and tool result is printed, so you can see the ReAct loop happen.
"""

import asyncio
import sys
import uuid

from langchain_core.messages import AIMessage, ToolMessage

from app.agent import AgentRuntime, create_runtime

PREVIEW_CHARS = 200


def text_of(content) -> str:
    """Message content can be a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def show(message) -> None:
    if isinstance(message, AIMessage):
        for call in message.tool_calls:
            print(f"  -> {call['name']}({call['args']})")
        text = text_of(message.content).strip()
        if text and not message.tool_calls:
            print(f"\nassistant> {text}\n")
        elif text:
            print(f"  (model) {text[:PREVIEW_CHARS]}")
    elif isinstance(message, ToolMessage):
        status = "ERROR" if message.status == "error" else "ok"
        preview = text_of(message.content).replace("\n", " ")[:PREVIEW_CHARS]
        print(f"  <- {message.name} [{status}] {preview}")


async def chat_turn(runtime: AgentRuntime, thread_id: str, user_text: str) -> None:
    """Run one user turn and print each step as the graph produces it."""
    async for update in runtime.graph.astream(
        {"messages": [("user", user_text)]},
        runtime.run_config(thread_id),
        stream_mode="updates",
    ):
        for node_output in update.values():
            if isinstance(node_output, dict):
                for message in node_output.get("messages", []):
                    show(message)


async def main() -> int:
    print("Loading MCP tools ...")
    runtime = await create_runtime()
    report = runtime.tool_report
    print(f"{len(report.tools)} tools from {list(report.loaded)}")
    if report.failed:
        print(f"FAILED servers: {report.failed}")

    thread_id = uuid.uuid4().hex[:8]
    print(f"Conversation {thread_id}. Commands: /new /tools /exit\n")
    while True:
        try:
            user_text = (await asyncio.to_thread(input, "you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/new":
            thread_id = uuid.uuid4().hex[:8]
            print(f"New conversation {thread_id}\n")
            continue
        if user_text == "/tools":
            print("\n".join(f"  {tool.name}" for tool in report.tools) + "\n")
            continue
        try:
            await chat_turn(runtime, thread_id, user_text)
        except Exception as exc:  # keep the session alive; stage 6 adds proper retry/fallback
            print(f"  [error] {type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
