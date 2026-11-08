"""Tool layer, sandbox and the Guard policy engine."""

from __future__ import annotations

import pytest

from app import guard, sandbox, tools


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
def test_tool_schemas_are_openai_shaped():
    schemas = tools.schemas()
    assert len(schemas) >= 7
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"


def test_calculator():
    out, ok = tools.execute("calculator", {"expression": "(1250*0.07)+89"}, user_id="u1")
    assert ok and "176.5" in out


def test_calculator_rejects_code():
    out, ok = tools.execute("calculator", {"expression": "__import__('os').system('ls')"}, user_id="u1")
    assert not ok or "Error" in out


def test_unknown_tool():
    out, ok = tools.execute("nope", {}, user_id="u1")
    assert not ok and "unknown tool" in out


def test_bad_arguments_do_not_crash():
    out, ok = tools.execute("calculator", {"wrong": 1}, user_id="u1")
    assert not ok and "Error" in out


def test_remember_and_recall(user):
    out, ok = tools.execute("remember", {"content": "The user prefers dark mode interfaces."}, user_id=user["id"])
    assert ok and "Saved" in out
    out, ok = tools.execute("recall", {"query": "dark mode preference"}, user_id=user["id"])
    assert ok and "dark mode" in out.lower()


def test_list_memories(user):
    tools.execute("remember", {"content": "Project codename is JagX."}, user_id=user["id"])
    out, ok = tools.execute("list_memories", {}, user_id=user["id"])
    assert ok and "JagX" in out


def test_web_tools_disabled_returns_message():
    out, ok = tools.execute("web_search", {"query": "anything"}, user_id="u1")
    assert ok and "disabled" in out.lower()


# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------
def test_sandbox_runs_python():
    out = sandbox.run_code("python", "print(6*7)")
    assert "42" in out


def test_sandbox_captures_errors():
    out = sandbox.run_code("python", "raise ValueError('boom')")
    assert "boom" in out


def test_sandbox_timeout():
    out = sandbox.run_code("python", "while True: pass", timeout=2)
    assert "timed out" in out.lower()


def test_sandbox_unsupported_language():
    out = sandbox.run_code("cobol", "DISPLAY 'HI'")
    assert "unsupported language" in out.lower()


def test_sandbox_empty_code():
    assert "empty" in sandbox.run_code("python", "   ").lower()


def test_sandbox_status():
    st = sandbox.status()
    assert st["backend"] in ("local-subprocess", "piston")
    assert "python" in st["languages"]


# ---------------------------------------------------------------------------
# guard
# ---------------------------------------------------------------------------
def test_safe_tool_allowed():
    v = guard.check("web_search", '{"query":"news"}', tool="web_search")
    assert v.verdict == guard.ALLOW


def test_send_requires_review():
    v = guard.check("send email to the client", "draft body", tool="send_email")
    assert v.verdict == guard.REVIEW
    assert v.matched == "send"


def test_pay_requires_review():
    v = guard.check("pay the invoice", "amount 500", tool="payments")
    assert v.verdict == guard.REVIEW and v.matched == "pay"


def test_delete_requires_review():
    v = guard.check("delete the old records", "", tool="db_admin")
    assert v.verdict == guard.REVIEW and v.matched == "delete"


def test_credential_handling_blocked():
    v = guard.check("read the api key", "token abc", tool="files")
    assert v.verdict == guard.BLOCK


def test_always_blocked_verbs():
    assert guard.check("change password", "", tool="x").verdict == guard.BLOCK
    assert guard.check("rm -rf /", "", tool="x").verdict == guard.BLOCK


def test_user_rule_block_wins():
    v = guard.check("send the report", "", tool="email", user_rules="block send")
    assert v.verdict == guard.BLOCK


def test_user_rule_allow_wins():
    v = guard.check("send the report", "", tool="email", user_rules="allow send")
    assert v.verdict == guard.ALLOW


def test_unknown_action_defaults_to_review():
    v = guard.check("frobnicate the widget", "", tool="mystery")
    assert v.verdict == guard.REVIEW


def test_guard_describe():
    d = guard.describe()
    assert set(d["verdicts"]) == {"allow", "review", "block"}
    assert "send" in d["hard_boundary"]
