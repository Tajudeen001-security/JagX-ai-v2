"""API tests — no external providers required."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ["JAGX_ADMIN_SECRET"] = "test-admin-secret"
os.environ["JAGX_KEYS_FILE"] = "test_keys.json"

from jagx_api.app import create_app
from jagx_api import knowledge


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAGX_ADMIN_SECRET", "test-admin-secret")
    monkeypatch.setenv("JAGX_KEYS_FILE", str(tmp_path / "keys.json"))
    monkeypatch.setenv("JAGX_MEMORY_FILE", str(tmp_path / "mem.json"))
    knowledge.load_brain(str(tmp_path))
    app = create_app()
    with TestClient(app) as c:
        yield c


def _make_key(client) -> str:
    r = client.post(
        "/create-key",
        json={"owner_label": "tester", "admin_secret": "test-admin-secret", "tier": "free"},
    )
    assert r.status_code == 200
    return r.json()["api_key"]


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "8.0" in r.json()["version"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_key_and_chat_fallback(client):
    key = _make_key(client)
    r = client.post(
        "/chat",
        headers={"x-api-key": key},
        json={"message": "hello jagx"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "response" in body
    assert len(body["response"]) > 0


def test_invalid_key(client):
    r = client.post("/chat", headers={"x-api-key": "bad"}, json={"message": "hi"})
    assert r.status_code == 401


def test_calc(client):
    key = _make_key(client)
    r = client.post(
        "/calc",
        headers={"x-api-key": key},
        json={"expression": "3*7+1"},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["value"] == 22


def test_memory_roundtrip(client):
    key = _make_key(client)
    r = client.post(
        "/memory",
        headers={"x-api-key": key},
        json={"action": "remember", "user_id": "u1", "text": "deadline is Friday"},
    )
    assert r.json()["ok"] is True
    r2 = client.post(
        "/memory",
        headers={"x-api-key": key},
        json={"action": "recall", "user_id": "u1", "query": "deadline"},
    )
    assert r2.json()["ok"] is True
    assert len(r2.json()["hits"]) >= 1


def test_keys_me(client):
    key = _make_key(client)
    r = client.get("/keys/me", headers={"x-api-key": key})
    assert r.status_code == 200
    assert r.json()["tier"] == "free"
