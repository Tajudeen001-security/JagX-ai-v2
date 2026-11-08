"""The JagX Loop: planning, tool calls, guard gating, budget, critique."""

from __future__ import annotations

from app import agent, db, guard
from app.llm import LLMResult, ToolCall


def _settings(**over):
    base = {
        "model": "",
        "temperature": 0.2,
        "max_steps": 4,
        "system_prompt": "",
        "auto_memory": False,
        "critique": False,
        "guard_rules": "",
    }
    base.update(over)
    return base


def _run(user, message, settings=None, history=None):
    conv = db.create_conversation(user["id"], "t")
    run_id = db.create_run(conv["id"], user["id"])
    return list(agent.run(
        user_id=user["id"],
        conversation_id=conv["id"],
        user_message=message,
        history=history or [],
        user_settings=settings or _settings(),
        run_id=run_id,
    ))


def test_simple_answer_no_tools(user, fake_llm):
    fake_llm(default="Hello there, this is a direct answer.")
    events = _run(user, "Say hello.")
    kinds = [e["type"] for e in events]
    assert "run_start" in kinds and "answer" in kinds and "done" in kinds
    answer = next(e for e in events if e["type"] == "answer")
    assert "direct answer" in answer["text"]


def test_tool_call_then_answer(user, fake_llm, tool_call):
    fake_llm(script=[
        tool_call("calculator", {"expression": "17*23"}),
        LLMResult(content="The answer is 391.", provider="fake", model="fake-1"),
    ])
    events = _run(user, "What is 17*23?")
    kinds = [e["type"] for e in events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["ok"] is True
    assert "391" in result["output"]
    answer = next(e for e in events if e["type"] == "answer")
    assert "391" in answer["text"]


def test_guard_blocks_credential_action(user, fake_llm, tool_call):
    fake_llm(script=[
        tool_call("fetch_url", {"url": "https://x.test/api_key"}),
        LLMResult(content="I could not do that.", provider="fake", model="fake-1"),
    ])
    events = _run(user, "Fetch the api key from that page.")
    verdicts = [e for e in events if e["type"] == "guard"]
    assert verdicts and verdicts[0]["verdict"] == guard.BLOCK
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["ok"] is False
    assert "REFUSED" in result["output"]


def test_guard_holds_send_for_review(user, fake_llm, tool_call):
    fake_llm(script=[
        tool_call("fetch_url", {"url": "https://x.test/send-email-now"}),
        LLMResult(content="Held for review.", provider="fake", model="fake-1"),
    ])
    events = _run(user, "Send an email to the client.")
    verdicts = [e for e in events if e["type"] == "guard"]
    assert verdicts and verdicts[0]["verdict"] == guard.REVIEW
    approvals = db.list_approvals(user["id"], status="pending")
    assert approvals, "a review verdict must queue an approval"
    answer = next(e for e in events if e["type"] == "answer")
    assert "approval" in answer["text"].lower()


def test_step_budget_forces_synthesis(user, fake_llm, tool_call):
    # Always asks for a tool -> must hit the budget and then synthesise.
    fake_llm(script=[tool_call("calculator", {"expression": "1+1"})] * 10,
             default="Synthesised answer after the step limit.")
    events = _run(user, "Keep going forever.", settings=_settings(max_steps=2))
    assert any(e["type"] == "answer" for e in events)
    done = next(e for e in events if e["type"] == "done")
    assert done["steps_used"] >= 2


def test_planning_emitted_for_multistep(user, fake_llm):
    fake_llm(script=[
        LLMResult(content='["Search for pricing", "Compare options"]', provider="fake", model="fake-1"),
        LLMResult(content="Here is the comparison.", provider="fake", model="fake-1"),
    ])
    events = _run(user, "Research and compare the latest pricing options for me.")
    plans = [e for e in events if e["type"] == "plan"]
    assert plans and len(plans[0]["steps"]) == 2


def test_no_plan_for_simple_question(user, fake_llm):
    fake_llm(default="Short answer.")
    events = _run(user, "Hi")
    assert not any(e["type"] == "plan" for e in events)


def test_critique_triggers_revision(user, fake_llm):
    fake_llm(script=[
        LLMResult(content="A first draft that is comfortably long enough to be critiqued "
                          "properly by Verity, the critic inside JagX, on every rubric dimension.",
                   provider="fake", model="fake-1"),
        LLMResult(content='{"verdict":"revise","score":0.4,"issues":["claim without source"]}',
                   provider="fake", model="fake-1"),
        LLMResult(content="The improved answer with the source added.", provider="fake", model="fake-1"),
    ])
    events = _run(user, "Explain something at length please.", settings=_settings(critique=True))
    crits = [e for e in events if e["type"] == "critique"]
    assert crits and crits[0]["verdict"] == "revise"
    answer = next(e for e in events if e["type"] == "answer")
    assert "improved answer" in answer["text"]


def test_critique_pass_keeps_answer(user, fake_llm):
    fake_llm(script=[
        LLMResult(content="A solid draft answer that is comfortably long enough to be critiqued "
                          "properly by Verity on every rubric dimension without any issue.",
                   provider="fake", model="fake-1"),
        LLMResult(content='{"verdict":"pass","score":0.9,"issues":[]}', provider="fake", model="fake-1"),
    ])
    events = _run(user, "Explain something at length please.", settings=_settings(critique=True))
    crits = [e for e in events if e["type"] == "critique"]
    assert crits and crits[0]["verdict"] == "pass"
    answer = next(e for e in events if e["type"] == "answer")
    assert "solid draft" in answer["text"]


def test_llm_error_is_surfaced_not_crashed(user, monkeypatch):
    from app import llm

    def boom(*a, **k):
        raise llm.LLMError("no provider reachable")

    monkeypatch.setattr(llm, "stream_turn", boom)
    events = _run(user, "Hello?")
    assert any(e["type"] == "error" for e in events)
    assert any(e["type"] == "done" for e in events)


def test_steps_are_persisted(user, fake_llm, tool_call):
    fake_llm(script=[
        tool_call("calculator", {"expression": "2+2"}),
        LLMResult(content="Four.", provider="fake", model="fake-1"),
    ])
    conv = db.create_conversation(user["id"], "persist")
    run_id = db.create_run(conv["id"], user["id"])
    list(agent.run(
        user_id=user["id"], conversation_id=conv["id"], user_message="2+2?",
        history=[], user_settings=_settings(), run_id=run_id,
    ))
    steps = db.list_steps(run_id)
    assert any(s["kind"] == "tool" and s["name"] == "calculator" for s in steps)


def test_parse_plan_handles_fences():
    assert agent._parse_plan('```json\n["a","b"]\n```') == ["a", "b"]
    assert agent._parse_plan("not json") == []
    assert agent._parse_plan('{"step":"x"}') == []


def test_parse_critique_defaults():
    assert agent._parse_critique("garbage")["verdict"] == "pass"
    assert agent._parse_critique('{"verdict":"revise","score":2,"issues":"x"}')["score"] == 1.0


def test_looks_multistep():
    assert agent._looks_multistep("Research the latest AI agent frameworks and compare them")
    assert not agent._looks_multistep("Hi")
