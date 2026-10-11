from __future__ import annotations
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from jagx_api import conversation_store, providers
from jagx_api.app import create_app
from jagx_api.config import settings


def make_key(client, owner="tester"):
    response=client.post("/create-key",json={"owner_label":owner,"admin_secret":"test-admin-secret","tier":"free"})
    assert response.status_code==200
    return response.json()["api_key"]


def test_conversation_history_isolated_and_persistent(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings,"admin_secret","test-admin-secret")
    monkeypatch.setattr(settings,"keys_file",str(tmp_path/"keys.json"))
    monkeypatch.setattr(settings,"conversation_db",str(tmp_path/"conversations.sqlite3"))
    import jagx_api.app as app_module
    monkeypatch.setattr(app_module,"generate_response",lambda message,history,memory_context=None,training_consent=False:{"response":"Hello from test","provider":"test","local_hit":False})
    with TestClient(create_app()) as client:
        key_a=make_key(client,"A")
        key_b=make_key(client,"B")
        conv=client.post("/conversations",headers={"x-api-key":key_a},json={"title":"Test"})
        assert conv.status_code==200
        cid=conv.json()["conversation"]["id"]
        reply=client.post("/chat",headers={"x-api-key":key_a},json={"message":"hello","conversation_id":cid})
        assert reply.status_code==200
        history=client.get(f"/conversations/{cid}",headers={"x-api-key":key_a})
        assert [m["role"] for m in history.json()["conversation"]["messages"]]==["user","assistant"]
        assert client.get(f"/conversations/{cid}",headers={"x-api-key":key_b}).status_code==404
        assert client.delete(f"/conversations/{cid}",headers={"x-api-key":key_b}).status_code==404


def test_personal_memory_requires_consent_and_isolated(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings,"admin_secret","test-admin-secret")
    monkeypatch.setattr(settings,"keys_file",str(tmp_path/"keys.json"))
    monkeypatch.setattr(settings,"memory_file",str(tmp_path/"memory.json"))
    with TestClient(create_app()) as client:
        key_a=make_key(client,"A")
        key_b=make_key(client,"B")
        denied=client.post("/memory",headers={"x-api-key":key_a},json={"action":"remember","text":"My secret project name"})
        assert denied.status_code==400
        saved=client.post("/memory",headers={"x-api-key":key_a},json={"action":"remember","text":"JagX project deadline Friday","consent":True,"user_id":"spoofed-user"})
        assert saved.status_code==200
        own=client.post("/memory",headers={"x-api-key":key_a},json={"action":"recall","query":"deadline"})
        other=client.post("/memory",headers={"x-api-key":key_b},json={"action":"recall","query":"deadline"})
        assert len(own.json()["hits"])==1
        assert other.json()["hits"]==[]


def test_provider_fallbacks_through_nvidia(monkeypatch):
    monkeypatch.setattr(settings,"groq_api_key","test-groq")
    monkeypatch.setattr(settings,"nvidia_api_key","test-nvidia")
    monkeypatch.setattr(settings,"openrouter_api_key","")
    monkeypatch.setattr(settings,"hf_token","")
    calls=[]
    def fake_post(url,**kwargs):
        calls.append(url)
        if "groq.com" in url:
            return SimpleNamespace(status_code=429,text="rate limited",json=lambda:{})
        return SimpleNamespace(status_code=200,text="",json=lambda:{"choices":[{"message":{"content":"NVIDIA response"}}]})
    monkeypatch.setattr(providers.HTTP,"post",fake_post)
    result,name=providers.chat_completion([{"role":"user","content":"hi"}])
    assert (result,name)==("NVIDIA response","nvidia")
    assert len(calls)==2


def test_provider_fallbacks_reject_malformed_success(monkeypatch):
    monkeypatch.setattr(settings,"groq_api_key","test-groq")
    monkeypatch.setattr(settings,"nvidia_api_key","")
    monkeypatch.setattr(settings,"openrouter_api_key","test-router")
    monkeypatch.setattr(settings,"hf_token","")
    def fake_post(url,**kwargs):
        if "groq.com" in url:
            return SimpleNamespace(status_code=200,text="",json=lambda:{"unexpected":True})
        return SimpleNamespace(status_code=200,text="",json=lambda:{"choices":[{"message":{"content":"router response"}}]})
    monkeypatch.setattr(providers.HTTP,"post",fake_post)
    result,name=providers.chat_completion([{"role":"user","content":"hi"}])
    assert (result,name)==("router response","openrouter")


def test_engineering_endurance_is_estimate_only():
    from jagx_api.engineering import endurance_estimate
    result=endurance_estimate(100,50,.8)
    assert result["estimated_minutes"]==96
    assert "Idealized estimate" in result["warning"]


def test_offline_mode_never_calls_cloud_provider(monkeypatch):
    monkeypatch.setattr(settings,"offline_mode",True)
    def should_not_call(*args,**kwargs):
        raise AssertionError("cloud provider called while offline mode is enabled")
    monkeypatch.setattr(providers.HTTP,"post",should_not_call)
    result,name=providers.chat_completion([{"role":"user","content":"hello"}])
    assert result is None and name is None


def test_local_model_status_is_explicit_when_no_model_configured(monkeypatch):
    from jagx_api import local_model
    monkeypatch.delenv("JAGX_GGUF_MODEL",raising=False)
    result=local_model.status()
    assert result["model_configured"] is False
    assert result["enabled"] is False
