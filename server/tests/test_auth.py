"""Auth, conversation isolation and settings."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app import security


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _signup(client, email="a@example.com", password="password123"):
    r = client.post("/api/auth/signup", json={"email": email, "password": password})
    return r


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "llm" in body and "sandbox" in body


def test_config_exposes_architecture(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["local_first"] is True
    assert set(body["modes"]) == {"loop", "council"}
    assert body["council"]["coordinator"]["name"] == "Atlas"
    assert body["council"]["critic"]["name"] == "Verity"
    assert "send" in body["guard"]["hard_boundary"]
    assert len(body["tools"]) >= 7


def test_signup_login_me(client):
    r = _signup(client, "auth1@example.com")
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    assert token

    r = client.post("/api/auth/login", json={"email": "auth1@example.com", "password": "password123"})
    assert r.status_code == 200

    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["email"] == "auth1@example.com"


def test_duplicate_signup_rejected(client):
    _signup(client, "dup@example.com")
    r = _signup(client, "dup@example.com")
    assert r.status_code == 409


def test_short_password_rejected(client):
    r = _signup(client, "short@example.com", "123")
    assert r.status_code == 400


def test_bad_login_rejected(client):
    _signup(client, "bad@example.com")
    r = client.post("/api/auth/login", json={"email": "bad@example.com", "password": "wrong"})
    assert r.status_code == 401


def test_missing_token_rejected(client):
    assert client.get("/api/conversations").status_code == 401
    assert client.get("/api/conversations", headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_password_hash_roundtrip():
    h = security.hash_password("password123")
    assert security.verify_password("password123", h)
    assert not security.verify_password("password124", h)
    assert h.startswith("pbkdf2$")


def test_token_roundtrip():
    t = security.create_token("usr_x", "x@example.com")
    payload = security.decode_token(t)
    assert payload and payload["sub"] == "usr_x"
    assert security.decode_token("garbage") is None


def test_conversation_isolation(client):
    a = _signup(client, "iso_a@example.com").json()["token"]
    b = _signup(client, "iso_b@example.com").json()["token"]
    ha = {"Authorization": f"Bearer {a}"}
    hb = {"Authorization": f"Bearer {b}"}

    conv = client.post("/api/conversations", headers=ha).json()
    assert client.get(f"/api/conversations/{conv['id']}", headers=ha).status_code == 200
    # B must not see A's conversation.
    assert client.get(f"/api/conversations/{conv['id']}", headers=hb).status_code == 404
    assert client.delete(f"/api/conversations/{conv['id']}", headers=hb).status_code == 404
    assert len(client.get("/api/conversations", headers=hb).json()["conversations"]) == 0


def test_settings_roundtrip(client):
    token = _signup(client, "set@example.com").json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    r = client.put("/api/settings", headers=h, json={"temperature": 0.9, "max_steps": 5, "bogus": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["temperature"] == 0.9
    assert body["max_steps"] == 5
    assert "bogus" not in body
