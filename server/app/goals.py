"""JagX v13 — JagX Goals: the always-on goal runner.

The unit of work is a **goal**, not a conversation. A goal has an identity, a
title, a success criterion, a status, a step budget and a checkpoint log. The
runner advances every active goal on a tick: it plans the next step, executes it
through the same Loop and Guard, records the checkpoint, and either completes,
pauses for approval, or reschedules.

This is what makes JagX always-on: the goal survives the session, and the work
continues when nobody is watching.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Iterator, Optional

from . import agent, db, guard, llm, memory
from .config import settings

logger = logging.getLogger("jagx.goals")

NEXT_STEP_PROMPT = """You are the goal runner inside JagX. You advance one long-running goal by exactly one step.

You are given the goal, its success criterion, and the checkpoints so far.

Decide the single most useful next action. Return ONLY JSON:
{
  "thought": "one line on why this is the right next step",
  "action": "a short imperative description of the step",
  "status": "continue" | "done" | "blocked",
  "note": "what you accomplished or learned in this step (1-3 sentences)"
}

Use "done" only when the success criterion is genuinely met. Use "blocked" when you need
something from the user that you cannot obtain yourself.
"""


def _parse(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1] if "```" in raw[3:] else raw[3:]
        raw = raw.lstrip("json").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        data = json.loads(raw[start : end + 1])
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def create(
    user_id: str,
    title: str,
    *,
    success_criterion: str = "",
    priority: int = 3,
    budget: int = 20,
) -> dict:
    return db.create_goal(
        user_id, title, success_criterion=success_criterion, priority=priority, budget=budget
    )


def advance(user_id: str, goal_id: str, *, user_settings: Optional[dict] = None) -> Iterator[dict]:
    """Advance one goal by one step, yielding progress events."""
    goal = db.get_goal(user_id, goal_id)
    if not goal:
        yield {"type": "error", "message": "Goal not found."}
        return

    if goal["status"] not in ("active", "paused"):
        yield {"type": "error", "message": f"Goal is {goal['status']} and cannot be advanced."}
        return

    if goal["steps_used"] >= goal["budget"]:
        db.update_goal(user_id, goal_id, status="exhausted")
        yield {"type": "error", "message": "Goal step budget exhausted."}
        return

    settings_map = user_settings or db.get_settings(user_id)
    model = (settings_map.get("model") or "").strip()
    user_rules = settings_map.get("guard_rules") or ""

    checkpoints = db.list_checkpoints(goal_id)
    history_block = "\n".join(
        f"{c['idx']}. [{c['status']}] {c['note']}" for c in checkpoints[-12:]
    ) or "(no checkpoints yet)"

    yield {"type": "goal_start", "goal_id": goal_id, "title": goal["title"], "step": goal["steps_used"] + 1}

    try:
        result = llm.complete(
            [
                {"role": "system", "content": NEXT_STEP_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"GOAL: {goal['title']}\n"
                        f"SUCCESS CRITERION: {goal['success_criterion'] or '(not stated)'}\n"
                        f"STEPS USED: {goal['steps_used']} / {goal['budget']}\n\n"
                        f"CHECKPOINTS SO FAR:\n{history_block}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=500,
            model=model,
        )
        decision = _parse(result.content)
    except llm.LLMError as exc:
        yield {"type": "error", "message": str(exc)}
        return

    if not decision:
        decision = {"thought": "Model returned no usable decision.", "action": "re-plan",
                    "status": "blocked", "note": "Could not decide a next step."}

    thought = str(decision.get("thought", ""))[:400]
    action = str(decision.get("action", ""))[:400]
    status = str(decision.get("status", "continue")).lower()
    note = str(decision.get("note", ""))[:1500]

    yield {"type": "goal_thought", "thought": thought, "action": action}

    # Guard the proposed action before it is treated as done. A goal step is a
    # description rather than a tool call, so it is allowed unless it matches a
    # boundary pattern (send / pay / sign / delete / credential).
    verdict = guard.check(action, note, tool="goal_step", user_rules=user_rules)
    yield {"type": "guard", "verdict": verdict.verdict, "reason": verdict.reason}

    idx = goal["steps_used"] + 1
    if verdict.verdict == guard.BLOCK:
        db.add_checkpoint(goal_id, idx, f"Blocked: {verdict.reason}", "blocked")
        db.update_goal(user_id, goal_id, steps_used=idx, status="blocked")
        yield {"type": "goal_checkpoint", "index": idx, "status": "blocked", "note": verdict.reason}
        yield {"type": "done", "goal_id": goal_id, "status": "blocked"}
        return

    if verdict.verdict == guard.REVIEW:
        db.create_approval(user_id, f"goal:{goal['title']}", action, run_id="")
        db.add_checkpoint(goal_id, idx, f"Awaiting approval: {action}", "review")
        db.update_goal(user_id, goal_id, steps_used=idx, status="paused", next_run_at=time.time() + 3600)
        yield {"type": "goal_checkpoint", "index": idx, "status": "review", "note": action}
        yield {"type": "done", "goal_id": goal_id, "status": "paused"}
        return

    # Record the step and decide the goal's fate.
    db.add_checkpoint(goal_id, idx, note or action, "ok")
    memory.remember(
        user_id,
        f"Goal '{goal['title']}' step {idx}: {note or action}",
        layer="episodic",
        tags="goal",
        strength=0.8,
    )

    if status == "done":
        db.update_goal(user_id, goal_id, steps_used=idx, status="done")
        memory.remember(
            user_id,
            f"Completed goal: {goal['title']}. Outcome: {note or action}",
            layer="semantic",
            tags="goal,completed",
        )
        yield {"type": "goal_checkpoint", "index": idx, "status": "done", "note": note}
        yield {"type": "done", "goal_id": goal_id, "status": "done"}
        return

    if status == "blocked":
        db.update_goal(user_id, goal_id, steps_used=idx, status="blocked")
        yield {"type": "goal_checkpoint", "index": idx, "status": "blocked", "note": note}
        yield {"type": "done", "goal_id": goal_id, "status": "blocked"}
        return

    # continue: reschedule with a gentle backoff so a tick loop stays cheap.
    backoff = min(3600, 60 * (2 ** min(idx, 6)))
    db.update_goal(
        user_id, goal_id, steps_used=idx, status="active", next_run_at=time.time() + backoff
    )
    yield {"type": "goal_checkpoint", "index": idx, "status": "ok", "note": note}
    yield {"type": "done", "goal_id": goal_id, "status": "active", "next_in_s": backoff}


def tick(user_id: str, *, max_goals: int = 3) -> list[dict]:
    """Advance every due goal once. Returns a summary per goal."""
    out = []
    for goal in db.due_goals(user_id)[:max_goals]:
        events = list(advance(user_id, goal["id"]))
        final = next((e for e in reversed(events) if e.get("type") == "done"), {})
        out.append({
            "goal_id": goal["id"],
            "title": goal["title"],
            "status": final.get("status", "unknown"),
            "events": len(events),
        })
    return out


def describe() -> dict:
    return {
        "unit": "goal",
        "statuses": ["active", "paused", "blocked", "done", "exhausted"],
        "note": "Goals advance server-side on a tick, independent of any client session.",
    }
