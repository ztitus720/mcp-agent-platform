"""Stage-8 bonus tests: strict approval-list check and POST /admin/reload (no network).

    python -m pytest tests/test_hot_reload.py -v
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import AgentRuntime, build_agent, check_high_risk_tools
from app.api import create_app
from app.mcp_client import LoadReport
from app.policy import AgentPolicy
from core.settings import get_settings
from test_agent import ScriptedModel
from test_approval import POLICY, calls, executed, risky, safe

TOKEN = "test-admin-token"

# ---------------------------------------------------------------------------
# (a) strict approval list
# ---------------------------------------------------------------------------


def report(loaded: dict, failed: dict | None = None, tools=(), disabled=()) -> LoadReport:
    return LoadReport(tools=list(tools), loaded=loaded, failed=failed or {}, disabled=list(disabled))


class Named:
    def __init__(self, name):
        self.name = name


def policy_with(*names):
    return AgentPolicy(**{**POLICY.__dict__, "high_risk_tools": names})


def test_typo_in_a_loaded_server_refuses_to_start():
    r = report({"files": ["files_write_file"]}, tools=[Named("files_write_file")])
    with pytest.raises(ValueError, match="files_writefile"):
        check_high_risk_tools(policy_with("files_writefile"), r)


def test_unknown_server_prefix_refuses_to_start():
    r = report({"files": ["files_write_file"]}, tools=[Named("files_write_file")])
    with pytest.raises(ValueError, match="web_search"):
        check_high_risk_tools(policy_with("web_search"), r)        # missing "search_" prefix


def test_tool_of_a_failed_server_is_only_a_warning():
    r = report({"files": ["files_write_file"]}, failed={"search": "boom"}, tools=[Named("files_write_file")])
    assert check_high_risk_tools(policy_with("search_web_search", "files_write_file"), r) == ["search_web_search"]


def test_tool_of_a_disabled_server_is_only_a_warning():
    r = report({"search": ["search_web_search"]}, tools=[Named("search_web_search")], disabled=["files"])
    assert check_high_risk_tools(policy_with("search_web_search", "files_write_file"), r) == ["files_write_file"]


# ---------------------------------------------------------------------------
# (b) POST /admin/reload
# ---------------------------------------------------------------------------


@pytest.fixture
def admin_token(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", TOKEN)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def no_admin_token(monkeypatch):
    """Simulate a deployment WITHOUT ADMIN_TOKEN, whatever the developer's own .env contains."""
    monkeypatch.setattr("app.api.get_settings",
                        lambda: get_settings().model_copy(update={"admin_token": None}))


@pytest.fixture(autouse=True)
def reset():
    executed.clear()


def make_client(tool_sets: list, script: list, fail_on_call: int | None = None) -> TestClient:
    """Each factory call (startup, then every reload) gets the next tool set."""
    state = {"calls": 0}

    async def factory(checkpointer=None) -> AgentRuntime:
        state["calls"] += 1
        if fail_on_call == state["calls"]:
            raise ValueError("servers.yaml is broken")
        tools = tool_sets[min(state["calls"], len(tool_sets)) - 1]
        saver = checkpointer if checkpointer is not None else InMemorySaver()
        graph = build_agent(tools, POLICY, model=ScriptedModel(script=list(script)), checkpointer=saver)
        return AgentRuntime(graph=graph, policy=POLICY, checkpointer=saver,
                            tool_report=report({"demo": [t.name for t in tools]}, tools=tools))

    return TestClient(create_app(factory))


def reload(client, token=TOKEN):
    return client.post("/admin/reload", headers={"X-Admin-Token": token} if token else {})


def test_reload_is_off_without_admin_token(no_admin_token):
    with make_client([[safe]], [AIMessage(content="hi")]) as client:
        assert reload(client).status_code == 403


def test_reload_rejects_wrong_token(admin_token):
    with make_client([[safe]], [AIMessage(content="hi")]) as client:
        assert reload(client, token="nope").status_code == 401
        assert reload(client, token=None).status_code == 401


def test_reload_adds_tools_without_restart(admin_token):
    with make_client([[safe], [safe, risky]], [AIMessage(content="hi")]) as client:
        assert [t["name"] for t in client.get("/tools").json()] == ["safe"]
        body = reload(client).json()
        assert body["added_tools"] == ["risky"] and body["removed_tools"] == []
        assert [t["name"] for t in client.get("/tools").json()] == ["safe", "risky"]


def test_paused_approval_survives_reload(admin_token):
    with make_client([[risky, safe], [risky, safe]], [calls("risky"), AIMessage(content="done")]) as client:
        client.post("/chat", json={"message": "go", "thread_id": "p1"})
        assert reload(client).status_code == 200                      # same checkpointer kept
        resumed = client.post("/chat/approve", json={"thread_id": "p1", "decisions": [{"type": "approve"}]})
        assert resumed.status_code == 200 and executed == ["risky"]


def test_broken_reload_keeps_the_old_runtime(admin_token):
    with make_client([[safe], [safe]], [AIMessage(content="hi")], fail_on_call=2) as client:
        response = reload(client)
        assert response.status_code == 422 and "nothing changed" in response.json()["detail"]
        assert [t["name"] for t in client.get("/tools").json()] == ["safe"]
