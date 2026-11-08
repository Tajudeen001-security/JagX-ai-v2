"""Shared pytest fixtures. Everything runs offline with a fake LLM."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Isolate the data dir before importing the app.
_TMP = tempfile.mkdtemp(prefix="jagx_test_")
os.environ["JAGX_DATA_DIR"] = _TMP
os.environ["JAGX_JWT_SECRET"] = "test-secret-not-for-production-use-only-in-tests-0123456789"
os.environ["JAGX_ENABLE_WEB_TOOLS"] = "false"
os.environ["JAGX_ENABLE_CODE_EXECUTION"] = "true"
os.environ["JAGX_SANDBOX_TIMEOUT"] = "5"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db, llm  # noqa: E402
from app.llm import LLMResult, ToolCall  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _init():
    db.init_db()
    yield


@pytest.fixture
def user():
    import uuid

    email = f"u{uuid.uuid4().hex[:8]}@example.com"
    from app import security

    return db.create_user(email, security.hash_password("password123"))


class FakeLLM:
    """Scripted LLM: returns queued results in order, then a default answer."""

    def __init__(self, script=None, default: str = "Final answer from the fake model."):
        self.script = list(script or [])
        self.default = default
        self.calls: list[list[dict]] = []

    def _next(self, messages):
        self.calls.append(messages)
        if self.script:
            item = self.script.pop(0)
            return item(messages) if callable(item) else item
        return LLMResult(content=self.default, provider="fake", model="fake-1")

    def complete(self, messages, **kwargs):
        return self._next(messages)

    def stream_turn(self, messages, **kwargs):
        result = self._next(messages)
        if result.content:
            for word in result.content.split(" "):
                yield {"type": "token", "text": word + " "}
        yield {
            "type": "end",
            "content": result.content,
            "tool_calls": result.tool_calls,
            "provider": result.provider or "fake",
            "model": result.model or "fake-1",
        }


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a FakeLLM and return a setter for its script."""

    def _install(script=None, default="Final answer from the fake model."):
        fake = FakeLLM(script, default)
        monkeypatch.setattr(llm, "complete", fake.complete)
        monkeypatch.setattr(llm, "stream_turn", fake.stream_turn)
        return fake

    return _install


@pytest.fixture
def tool_call():
    def _make(name: str, arguments: dict, call_id: str = "call_1") -> LLMResult:
        return LLMResult(
            content="",
            tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
            provider="fake",
            model="fake-1",
        )

    return _make
