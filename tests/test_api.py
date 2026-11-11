"""Stage-7 tests for the FastAPI layer, with a scripted runtime (no network).

    python -m pytest tests/test_api.py -v
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.agent import AgentRuntime, build_agent
from app.api import create_app
from app.mcp_client import LoadReport
from test_agent import ScriptedModel
from test_approval import POLICY, calls, executed, risky, safe


def make_client(*script: AIMessage) -> TestClient:
    async def factory() -> AgentRuntime:
        graph = build_agent([risky, safe], POLICY, model=ScriptedModel(script=list(script)))
        report = LoadReport(tools=[risky, safe], loaded={"demo": ["risky", "safe"]}, failed={"broken": "boom"})
        return AgentRuntime(graph=graph, policy=POLICY, tool_report=report)
    return TestClient(create_app(factory))


def sse(response) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs."""
    events = []
    for block in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


@pytest.fixture(autouse=True)
def reset():
    executed.clear()


def test_health_and_tools():
    with make_client(AIMessage(content="hi")) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded" and health["servers_failed"] == {"broken": "boom"}
        assert [t["name"] for t in client.get("/tools").json()] == ["risky", "safe"]


def test_chat_streams_tool_steps_and_answer():
    with make_client(calls("safe"), AIMessage(content="all good")) as client:
        response = client.post("/chat", json={"message": "go", "thread_id": "t1"})
        assert response.headers["content-type"].startswith("text/event-stream")
        names = [e for e, _ in sse(response)]
        assert names[0] == "thread" and names[-1] == "done"
        assert "tool_call" in names and "tool_result" in names
        assert sse(response)[-1][1] == {"thread_id": "t1", "answer": "all good"}


def test_approval_round_trip_over_http():
    with make_client(calls("risky"), AIMessage(content="done after approval")) as client:
        first = sse(client.post("/chat", json={"message": "go", "thread_id": "t2"}))
        assert first[-1][0] == "approval_required"
        assert first[-1][1]["requests"][0]["name"] == "risky"
        assert executed == []

        # a new message while paused is refused; wrong number of decisions is a 400
        assert client.post("/chat", json={"message": "again", "thread_id": "t2"}).status_code == 409
        assert client.post("/chat/approve", json={"thread_id": "t2", "decisions": [
            {"type": "approve"}, {"type": "approve"}]}).status_code == 400

        second = sse(client.post("/chat/approve", json={"thread_id": "t2", "decisions": [{"type": "approve"}]}))
        assert executed == ["risky"] and second[-1] == ("done", {"thread_id": "t2", "answer": "done after approval"})


def test_bad_requests_are_rejected():
    with make_client(AIMessage(content="hi")) as client:
        assert client.post("/chat", json={"message": ""}).status_code == 422
        assert client.post("/chat", json={"message": "x", "thread_id": "../etc"}).status_code == 422
        assert client.post("/chat/approve", json={"thread_id": "nope", "decisions": [
            {"type": "approve"}]}).status_code == 409
        assert client.post("/chat/approve", json={"thread_id": "t", "decisions": [
            {"type": "maybe"}]}).status_code == 422
