from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from jagx_api import conversation_store, knowledge_pipeline
from jagx_api.app import create_app
from jagx_api.config import settings
from jagx_api import providers


def test_provider_falls_through_to_nvidia(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "groq-test")
    monkeypatch.setattr(settings, "nvidia_api_key", "nvidia-test")
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    monkeypatch.setattr(settings, "hf_token", "")
    calls = []
    def fake_post(url, **kwargs):
        calls.append(url)
        if "groq.com" in url:
            return SimpleNamespace(status_code=429, text="rate limited")
        return SimpleNamespace(status_code=200, text="", json=lambda: {"choices":[{"message":{"content":"NVIDIA fallback answer"}}]})
    monkeypatch.setattr(providers.HTTP, "post", fake_post)
    text, provider = providers.chat_completion([{"role":"user","content":"hello"}])
    assert text == "NVIDIA fallback answer"
    assert provider == "nvidia"
    assert len(calls) == 2


def test_provider_malformed_success_response_falls_through(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "groq-test")
    monkeypatch.setattr(settings, "nvidia_api_key", "")
    monkeypatch.setattr(settings, "openrouter_api_key", "router-test")
    monkeypatch.setattr(settings, "hf_token", "")
    def fake_post(url, **kwargs):
        if "groq.com" in url:
            return SimpleNamespace(status_code=200, text="", json=lambda: {"unexpected": True})
        return SimpleNamespace(status_code=200, text="", json=lambda: {"choices":[{"message":{"content":"Router answer"}}]})
    monkeypatch.setattr(providers.HTTP, "post", fake_post)
    text, provider = providers.chat_completion([{"role":"user","content":"hello"}])
    assert text == "Router answer"
    assert provider == "openrouter"


def test_conversation_owner_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(conversation_store, "DB_PATH", str(tmp_path / "conversations.sqlite3"))
    created = conversation_store.create("key-A", "private")
    cid = created["id"]
    assert conversation_store.append("key-A", cid, "user", "hello")
    assert conversation_store.get("key-A", cid)["messages"][0]["content"] == "hello"
    assert conversation_store.get("key-B", cid) is None
    assert conversation_store.delete("key-B", cid) is False
    assert conversation_store.get("key-A", cid) is not None


def test_knowledge_validation_rejects_bad_rows():
    with pytest.raises(ValueError):
        knowledge_pipeline.validate_rows([{"question":"only question"}])
    rows = knowledge_pipeline.validate_rows([{"q":"What is JagX?","a":"JagX AI"}])
    assert rows == [{"question":"What is JagX?","answer":"JagX AI"}]


def test_conversation_api_auth_and_isolation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAGX_ADMIN_SECRET", "test-admin-secret")
    monkeypatch.setenv("JAGX_KEYS_FILE", str(tmp_path / "keys.json"))
    monkeypatch.setattr(settings, "admin_secret", "test-admin-secret")
    monkeypatch.setattr(settings, "keys_file", str(tmp_path / "keys.json"))
    monkeypatch.setenv("JAGX_MEMORY_FILE", str(tmp_path / "memory.json"))
    monkeypatch.setattr(conversation_store, "DB_PATH", str(tmp_path / "conversations.sqlite3"))
    with TestClient(create_app()) as client:
        key_response = client.post("/create-key", json={"owner_label":"a","admin_secret":"test-admin-secret"})
        key = key_response.json()["api_key"]
        other_key = client.post("/create-key", json={"owner_label":"b","admin_secret":"test-admin-secret"}).json()["api_key"]
        created = client.post("/conversations", headers={"x-api-key":key}, json={"title":"History"})
        assert created.status_code == 200
        cid = created.json()["conversation"]["id"]
        assert client.get(f"/conversations/{cid}", headers={"x-api-key":other_key}).status_code == 404
        assert client.delete(f"/conversations/{cid}", headers={"x-api-key":key}).json()["deleted"] is True


def test_engineering_estimate_has_explicit_warning():
    from jagx_api.engineering import endurance_estimate
    result = endurance_estimate(100, 50, 0.8)
    assert result["estimated_minutes"] == 96
    assert "Idealized estimate" in result["warning"]
