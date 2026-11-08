"""End-to-end SSE streaming through the real HTTP surface."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.llm import LLMResult, ToolCall


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture
def auth(client):
    import uuid

    email = f"sse{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/auth/signup", json={"email": email, "password": "password123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def _events(resp) -> list[dict]:
    out = []
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            try:
                out.append(json.loads(line[5:].strip()))
            except json.JSONDecodeError:
                pass
    return out


def test_chat_streams_full_event_sequence(client, auth, fake_llm, tool_call):
    fake_llm(script=[
        tool_call("calculator", {"expression": "17*23"}),
        LLMResult(content="The answer is 391.", provider="fake", model="fake-1"),
    ])
    r = client.post("/api/chat", headers=auth, json={"message": "What is 17*23?"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    events = _events(r)
    kinds = [e["type"] for e in events]
    for expected in ("conversation", "meta", "run_start", "tool_call", "guard", "tool_result", "answer", "end"):
        assert expected in kinds, f"missing {expected} in {kinds}"
    answer = next(e for e in events if e["type"] == "answer")
    assert "391" in answer["text"]


def test_chat_persists_conversation_and_messages(client, auth, fake_llm):
    fake_llm(default="A persisted answer.")
    r = client.post("/api/chat", headers=auth, json={"message": "Remember this exchange."})
    events = _events(r)
    conv_id = next(e for e in events if e["type"] == "conversation")["conversation_id"]

    detail = client.get(f"/api/conversations/{conv_id}", headers=auth).json()
    roles = [m["role"] for m in detail["messages"]]
    assert "user" in roles and "assistant" in roles
    assert any("persisted answer" in m["content"] for m in detail["messages"])


def test_chat_continues_existing_conversation(client, auth, fake_llm):
    fake_llm(default="First answer.")
    r1 = client.post("/api/chat", headers=auth, json={"message": "First question."})
    conv_id = next(e for e in _events(r1) if e["type"] == "conversation")["conversation_id"]

    fake_llm(default="Second answer.")
    r2 = client.post("/api/chat", headers=auth, json={"message": "Second question.", "conversation_id": conv_id})
    assert r2.status_code == 200
    detail = client.get(f"/api/conversations/{conv_id}", headers=auth).json()
    assert len([m for m in detail["messages"] if m["role"] == "user"]) == 2


def test_chat_rejects_empty_message(client, auth):
    assert client.post("/api/chat", headers=auth, json={"message": "   "}).status_code == 400


def test_chat_rejects_foreign_conversation(client, auth, fake_llm):
    import uuid

    other = client.post("/api/auth/signup", json={"email": f"o{uuid.uuid4().hex[:8]}@example.com",
                                                  "password": "password123"}).json()["token"]
    conv = client.post("/api/conversations", headers={"Authorization": f"Bearer {other}"}).json()
    r = client.post("/api/chat", headers=auth, json={"message": "hi", "conversation_id": conv["id"]})
    assert r.status_code == 404


def test_council_endpoint(client, auth, fake_llm):
    fake_llm(script=[
        LLMResult(content='{"lanes":["sage"],"why":"analysis"}', provider="fake", model="fake-1"),
        LLMResult(content="Sage finding.", provider="fake", model="fake-1"),
        LLMResult(content="Council synthesis that is comfortably long enough to be critiqued "
                          "properly by Verity on every rubric dimension.",
                  provider="fake", model="fake-1"),
        LLMResult(content='{"verdict":"pass","score":0.9,"issues":[]}', provider="fake", model="fake-1"),
    ])
    r = client.post("/api/council", headers=auth, json={"message": "Analyse this deeply."})
    assert r.status_code == 200
    events = _events(r)
    assert any(e["type"] == "council" for e in events)
    assert any(e["type"] == "answer" for e in events)


def test_run_audit_trail(client, auth, fake_llm, tool_call):
    fake_llm(script=[
        tool_call("calculator", {"expression": "2+2"}),
        LLMResult(content="Four.", provider="fake", model="fake-1"),
    ])
    r = client.post("/api/chat", headers=auth, json={"message": "What is 2+2?"})
    run_id = next(e for e in _events(r) if e["type"] == "meta")["run_id"]
    detail = client.get(f"/api/runs/{run_id}", headers=auth).json()
    assert detail["run"]["status"] == "done"
    assert any(s["kind"] == "tool" for s in detail["steps"])


def test_reflection_event_emitted(client, auth, fake_llm):
    fake_llm(default="An answer long enough to be scored by the reflection loop.")
    r = client.post("/api/chat", headers=auth, json={"message": "Explain the architecture in detail."})
    events = _events(r)
    assert any(e["type"] == "reflection" for e in events)


def test_goals_api(client, auth, fake_llm):
    created = client.post("/api/goals", headers=auth,
                          json={"title": "Ship v12", "success_criterion": "pushed"}).json()
    assert created["status"] == "active"
    assert client.get("/api/goals", headers=auth).json()["goals"]

    fake_llm(script=[
        LLMResult(content='{"thought":"t","action":"run tests","status":"continue","note":"green"}',
                  provider="fake", model="fake-1"),
    ])
    r = client.post(f"/api/goals/{created['id']}/advance", headers=auth)
    assert r.status_code == 200
    assert any(e["type"] == "goal_checkpoint" for e in _events(r))

    assert client.patch(f"/api/goals/{created['id']}", headers=auth, json={"status": "paused"}).status_code == 200
    assert client.delete(f"/api/goals/{created['id']}", headers=auth).status_code == 200


def test_memory_api(client, auth):
    r = client.post("/api/memories", headers=auth, json={"content": "The user ships weekly.", "layer": "semantic"})
    assert r.status_code == 200
    mem_id = r.json()["id"]
    assert client.get("/api/memories", headers=auth).json()["memories"]
    assert client.post("/api/memories/consolidate", headers=auth).status_code == 200
    assert client.delete(f"/api/memories/{mem_id}", headers=auth).status_code == 200


def test_forge_api(client, auth):
    assert client.get("/api/forge/stats", headers=auth).status_code == 200
    assert client.get("/api/forge/lessons", headers=auth).status_code == 200
    variants = client.get("/api/forge/variants", headers=auth).json()["variants"]
    assert variants
    assert client.post("/api/forge/variants", headers=auth,
                       json={"task_class": "general", "name": "custom", "body": "Be terse."}).status_code == 200
    assert client.get("/api/forge/dataset", headers=auth).status_code == 200
    assert client.get("/api/forge/preferences", headers=auth).status_code == 200


def test_evals_api(client, auth, fake_llm):
    suites = client.get("/api/evals", headers=auth).json()["evals"]
    assert suites
    suite_id = suites[0]["id"]
    fake_llm(default="391 red blue")
    r = client.post(f"/api/evals/{suite_id}/run", headers=auth)
    assert r.status_code == 200 and "score" in r.json()
    assert client.get(f"/api/evals/{suite_id}/results", headers=auth).status_code == 200


def test_tools_api(client, auth):
    assert client.get("/api/tools", headers=auth).json()["tools"]
    r = client.post("/api/tools/run", headers=auth,
                    json={"name": "calculator", "arguments": {"expression": "10*10"}})
    assert r.status_code == 200 and "100" in r.json()["output"]


def test_tools_api_guard_blocks(client, auth):
    r = client.post("/api/tools/run", headers=auth,
                    json={"name": "fetch_url", "arguments": {"url": "https://x.test/api_key"}})
    body = r.json()
    assert body["ok"] is False
    assert body["guard"]["verdict"] == "block"


def test_approvals_api(client, auth, fake_llm, tool_call):
    fake_llm(script=[
        tool_call("fetch_url", {"url": "https://x.test/send-email"}),
        LLMResult(content="Held.", provider="fake", model="fake-1"),
    ])
    client.post("/api/chat", headers=auth, json={"message": "Send an email please."})
    pending = client.get("/api/approvals", headers=auth).json()["approvals"]
    assert pending
    r = client.post(f"/api/approvals/{pending[0]['id']}", headers=auth, json={"status": "approved"})
    assert r.status_code == 200
