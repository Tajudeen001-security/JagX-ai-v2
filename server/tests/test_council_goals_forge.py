"""Council fan-out, goal runner, Forge self-improvement and the eval harness."""

from __future__ import annotations

from app import council, db, evals, goals, memory, selfimprove
from app.llm import LLMResult


def _settings(**over):
    base = {"model": "", "temperature": 0.2, "max_steps": 3, "auto_memory": False, "guard_rules": ""}
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# council
# ---------------------------------------------------------------------------
def test_council_describe():
    d = council.describe()
    assert d["coordinator"]["name"] == "Atlas"
    assert d["critic"]["name"] == "Verity"
    assert {l["key"] for l in d["lanes"]} == {"scout", "forge", "sage"}


def test_council_decompose_falls_back(monkeypatch):
    from app import llm

    def boom(*a, **k):
        raise llm.LLMError("down")

    monkeypatch.setattr(llm, "complete", boom)
    lanes = council.decompose("anything")
    assert lanes and all(l in council.LANES for l in lanes)


def test_council_decompose_parses(monkeypatch):
    from app import llm

    monkeypatch.setattr(
        llm, "complete",
        lambda *a, **k: LLMResult(content='{"lanes":["scout","forge"],"why":"needs both"}',
                                  provider="fake", model="fake-1"),
    )
    assert council.decompose("build and research") == ["scout", "forge"]


def test_council_run_end_to_end(user, fake_llm):
    fake_llm(script=[
        LLMResult(content='{"lanes":["scout","sage"],"why":"research and analysis"}',
                  provider="fake", model="fake-1"),
        LLMResult(content="Scout finding: the market grew 20%.", provider="fake", model="fake-1"),
        LLMResult(content="Sage finding: growth is driven by adoption.", provider="fake", model="fake-1"),
        LLMResult(content="Synthesised council answer that is comfortably long enough to be "
                          "critiqued properly by Verity on every rubric dimension.",
                  provider="fake", model="fake-1"),
        LLMResult(content='{"verdict":"pass","score":0.9,"issues":[]}', provider="fake", model="fake-1"),
    ])
    conv = db.create_conversation(user["id"], "c")
    run_id = db.create_run(conv["id"], user["id"], mode="council")
    events = list(council.run(
        user_id=user["id"], conversation_id=conv["id"], question="Research and analyse the market.",
        history=[], user_settings=_settings(), run_id=run_id,
    ))
    kinds = [e["type"] for e in events]
    assert "council" in kinds
    phases = {e.get("phase") for e in events if e["type"] == "council"}
    assert {"decompose", "fanout", "lane", "synthesis"} <= phases
    assert any(e["type"] == "answer" for e in events)
    assert any(e["type"] == "critique" for e in events)


def test_council_lane_tools_are_restricted():
    scout = {s["function"]["name"] for s in council._lane_tools("scout")}
    assert "web_search" in scout
    assert "run_code" not in scout


# ---------------------------------------------------------------------------
# goals
# ---------------------------------------------------------------------------
def test_goal_lifecycle(user, fake_llm):
    goal = goals.create(user["id"], "Ship the JagX v12 release", success_criterion="Repo pushed and tests green")
    assert goal["status"] == "active"

    fake_llm(script=[
        LLMResult(content='{"thought":"start with tests","action":"run the test suite",'
                          '"status":"continue","note":"tests pass"}', provider="fake", model="fake-1"),
    ])
    events = list(goals.advance(user["id"], goal["id"]))
    assert any(e["type"] == "goal_checkpoint" for e in events)
    updated = db.get_goal(user["id"], goal["id"])
    assert updated["steps_used"] == 1
    assert updated["status"] == "active"
    assert db.list_checkpoints(goal["id"])


def test_goal_completes(user, fake_llm):
    goal = goals.create(user["id"], "Write the design doc")
    fake_llm(script=[
        LLMResult(content='{"thought":"done","action":"finalise","status":"done","note":"doc written"}',
                  provider="fake", model="fake-1"),
    ])
    list(goals.advance(user["id"], goal["id"]))
    assert db.get_goal(user["id"], goal["id"])["status"] == "done"


def test_goal_blocked_on_guard(user, fake_llm):
    goal = goals.create(user["id"], "Handle credentials")
    fake_llm(script=[
        LLMResult(content='{"thought":"need it","action":"change password for the account",'
                          '"status":"continue","note":"x"}', provider="fake", model="fake-1"),
    ])
    events = list(goals.advance(user["id"], goal["id"]))
    assert any(e["type"] == "guard" and e["verdict"] == "block" for e in events)
    assert db.get_goal(user["id"], goal["id"])["status"] == "blocked"


def test_goal_budget_exhausted(user, fake_llm):
    goal = goals.create(user["id"], "Tiny budget", budget=1)
    fake_llm(script=[
        LLMResult(content='{"thought":"go","action":"step one","status":"continue","note":"n"}',
                  provider="fake", model="fake-1"),
    ])
    list(goals.advance(user["id"], goal["id"]))
    events = list(goals.advance(user["id"], goal["id"]))
    assert any(e["type"] == "error" for e in events)


def test_goal_tick(user, fake_llm):
    goals.create(user["id"], "Tick goal one")
    fake_llm(script=[
        LLMResult(content='{"thought":"t","action":"a","status":"continue","note":"n"}',
                  provider="fake", model="fake-1"),
    ])
    result = goals.tick(user["id"])
    assert isinstance(result, list)


def test_goal_describe():
    assert goals.describe()["unit"] == "goal"


# ---------------------------------------------------------------------------
# forge
# ---------------------------------------------------------------------------
def test_score_run_bounds():
    good = selfimprove.score_run(steps_used=2, tool_calls=2, tool_failures=0,
                                 critique={"verdict": "pass", "score": 0.9}, answer="x" * 200)
    bad = selfimprove.score_run(steps_used=12, tool_calls=0, tool_failures=0,
                                critique={"verdict": "revise", "score": 0.2}, answer="")
    assert 0.0 <= bad["score"] < good["score"] <= 1.0


def test_reflect_writes_lessons(user):
    out = selfimprove.reflect(
        user["id"], run_id="run_x", question="A long multi-part question " * 3,
        answer="A" * 200, steps_used=2, tool_calls=0, tool_failures=0, critique=None,
    )
    assert out["score"] > 0
    assert out["lessons"]
    assert db.list_lessons(user["id"])


def test_reflect_stores_exemplar(user):
    selfimprove.reflect(
        user["id"], run_id="run_y", question="How do I deploy the agent?",
        answer="B" * 300, steps_used=2, tool_calls=2, tool_failures=0,
        critique={"verdict": "pass", "score": 0.95},
    )
    procs = db.list_memories(user["id"], layer="procedural", limit=50)
    assert any("exemplar" in (p["tags"] or "") for p in procs)


def test_replay_examples(user):
    selfimprove.reflect(
        user["id"], run_id="run_z", question="Deploy the agent to production",
        answer="C" * 300, steps_used=2, tool_calls=2, tool_failures=0,
        critique={"verdict": "pass", "score": 0.95},
    )
    block = selfimprove.replay_examples(user["id"], "deploy the agent", limit=2)
    assert isinstance(block, str)


def test_classify_task():
    assert selfimprove.classify_task("fix this python bug") == "code"
    assert selfimprove.classify_task("research the latest market news") == "research"
    assert selfimprove.classify_task("hello") == "general"


def test_seed_and_pick_variant(user):
    created = selfimprove.seed_variants(user["id"])
    assert created >= 3
    assert selfimprove.seed_variants(user["id"]) == 0  # idempotent
    v = selfimprove.pick_variant(user["id"], "general")
    assert v and v["task_class"] == "general"


def test_bandit_prefers_winner(user):
    selfimprove.seed_variants(user["id"])
    variants = db.list_variants(user["id"], task_class="general", active_only=True)
    winner = variants[0]
    for _ in range(12):
        db.record_variant_outcome(winner["id"], won=True, score=0.95)
    for v in variants[1:]:
        for _ in range(12):
            db.record_variant_outcome(v["id"], won=False, score=0.2)
    picked = selfimprove.pick_variant(user["id"], "general")
    assert picked["id"] == winner["id"]


def test_variant_prompt(user):
    selfimprove.seed_variants(user["id"])
    task_class, fragment = selfimprove.variant_prompt(user["id"], "write some python code")
    assert task_class == "code"
    assert fragment


def test_optimize_retires_losers(user):
    selfimprove.seed_variants(user["id"])
    variants = db.list_variants(user["id"], task_class="general", active_only=True)
    for _ in range(8):
        db.record_variant_outcome(variants[0]["id"], won=True, score=0.95)
    for v in variants[1:]:
        for _ in range(8):
            db.record_variant_outcome(v["id"], won=False, score=0.1)
    report = selfimprove.optimize(user["id"])
    assert report["promoted"]
    assert report["retired"]


def test_export_dataset(user, fake_llm):
    from app import agent

    conv = db.create_conversation(user["id"], "ds")
    run_id = db.create_run(conv["id"], user["id"])
    list(agent.run(
        user_id=user["id"], conversation_id=conv["id"], user_message="Explain X",
        history=[], user_settings=_settings(), run_id=run_id,
    ))
    db.add_message(conv["id"], "user", "Explain X")
    db.add_message(conv["id"], "assistant", "X is explained.")
    db.add_step(run_id, 1, "answer", name="answer", output="X is explained.")
    db.finish_run(run_id, status="done", steps_used=1, provider="fake", model="fake-1")

    rows = selfimprove.export_dataset(user["id"], fmt="messages")
    assert rows and "messages" in rows[0]
    rows = selfimprove.export_dataset(user["id"], fmt="sharegpt")
    assert rows and "conversations" in rows[0]


def test_export_preferences(user):
    db.add_lesson(user["id"], "Prefer evidence-first answers.", kind="reflection", score=0.9)
    rows = selfimprove.export_preferences(user["id"])
    assert rows and "chosen_score" in rows[0]


def test_forge_stats_and_describe(user):
    st = selfimprove.stats(user["id"])
    assert "loops" in st and st["loops"]["reflection"] == "on"
    d = selfimprove.describe()
    assert len(d["loops"]) == 4
    assert "honest_limits" in d


# ---------------------------------------------------------------------------
# evals
# ---------------------------------------------------------------------------
def test_seed_suites(user):
    assert evals.seed_suites(user["id"]) >= 1
    assert evals.seed_suites(user["id"]) == 0
    assert db.list_evals(user["id"])


def test_checker_logic():
    assert evals._check({"min_len": 2}, "ok")[0]
    assert not evals._check({"min_len": 50}, "short")[0]
    assert evals._check({"expect_any": ["391"]}, "the answer is 391")[0]
    assert not evals._check({"expect_any": ["391"]}, "the answer is 392")[0]
    assert evals._check({"expect_all": ["a", "b"]}, "a and b")[0]
    assert not evals._check({"expect_all": ["a", "z"]}, "a and b")[0]
    assert evals._check({"regex": r"\d+"}, "42")[0]


def test_run_eval(user, fake_llm):
    evals.seed_suites(user["id"])
    suite = next(e for e in db.list_evals(user["id"]) if e["name"] == "core")
    fake_llm(default="The answer is 391 and red is a primary colour.")
    result = evals.run_eval(user["id"], suite["id"])
    assert 0.0 <= result["score"] <= 1.0
    assert result["total"] == len(suite["cases"])
    assert db.list_eval_results(suite["id"])


def test_compare_variants(user, fake_llm):
    evals.seed_suites(user["id"])
    selfimprove.seed_variants(user["id"])
    suite = next(e for e in db.list_evals(user["id"]) if e["name"] == "core")
    fake_llm(default="391 red blue green")
    result = evals.compare_variants(user["id"], suite["id"])
    assert result["ranking"]
    assert result["winner"]


def test_eval_describe():
    assert "min_len" in evals.describe()["checkers"]
