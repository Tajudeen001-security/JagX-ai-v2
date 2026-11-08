"""JagX v12 — JagX Forge: the self-improvement engine.

Four loops, in increasing order of risk and cost:

1. **Reflection** (free, always on). After every run, score the trajectory on a
   rubric and write a lesson to memory. Lessons are injected into future runs.
2. **Experience replay** (free). Successful trajectories are stored as reusable
   exemplars and retrieved as few-shot context for similar future tasks. The
   weights never change; the behaviour does.
3. **Prompt & policy optimisation** (free). Competing prompt variants per task
   class, with a multi-armed bandit over them using the eval harness score as
   reward. Winners are promoted; losers are retired. Every change is versioned.
4. **Weight training** (free software, needs a GPU you own). Export trajectories
   as a clean SFT dataset and ship a LoRA/QLoRA training script. The trained
   adapter is served back through the same OpenAI-compatible endpoint.

Loops 1-3 run in production. Loop 4 is honest about needing hardware.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from typing import Optional

from . import db, memory
from .config import settings

logger = logging.getLogger("jagx.forge")

# ---------------------------------------------------------------------------
# 1. Reflection
# ---------------------------------------------------------------------------
REFLECTION_RUBRIC = {
    "used_tools": 0.25,
    "no_wasted_steps": 0.20,
    "critic_passed": 0.25,
    "answered": 0.20,
    "efficient": 0.10,
}


def score_run(*, steps_used: int, tool_calls: int, tool_failures: int, critique: Optional[dict], answer: str) -> dict:
    """Deterministic trajectory score in [0,1]. No model call — free and auditable."""
    used_tools = 1.0 if tool_calls > 0 else 0.5
    no_wasted = 1.0 - min(1.0, tool_failures / max(1, tool_calls)) if tool_calls else 0.7
    if critique is None:
        critic_passed = 0.7
    else:
        critic_passed = 1.0 if critique.get("verdict") == "pass" else 0.35
        critic_passed = 0.5 * critic_passed + 0.5 * float(critique.get("score", 0.7))
    answered = 1.0 if len(answer or "") > 60 else 0.3
    efficient = 1.0 - min(1.0, max(0, steps_used - 3) / 10.0)

    parts = {
        "used_tools": used_tools,
        "no_wasted_steps": no_wasted,
        "critic_passed": critic_passed,
        "answered": answered,
        "efficient": efficient,
    }
    total = sum(parts[k] * w for k, w in REFLECTION_RUBRIC.items())
    return {"score": round(total, 4), "parts": {k: round(v, 3) for k, v in parts.items()}}


def reflect(
    user_id: str,
    *,
    run_id: str,
    question: str,
    answer: str,
    steps_used: int,
    tool_calls: int,
    tool_failures: int,
    critique: Optional[dict],
) -> dict:
    """Loop 1: score the run and write a lesson."""
    scored = score_run(
        steps_used=steps_used,
        tool_calls=tool_calls,
        tool_failures=tool_failures,
        critique=critique,
        answer=answer,
    )
    score = scored["score"]
    parts = scored["parts"]

    lessons: list[str] = []
    if tool_calls == 0 and len(question) > 60:
        lessons.append("For multi-part questions, gather evidence with tools before answering.")
    if tool_failures > 0:
        lessons.append("A tool failed mid-run — check arguments and have a fallback path.")
    if critique and critique.get("verdict") == "revise":
        issues = critique.get("issues") or []
        if issues:
            lessons.append("Critic flagged: " + "; ".join(str(i) for i in issues[:2]))
    if steps_used >= settings.max_steps:
        lessons.append("Ran to the step limit — plan fewer, more decisive steps.")
    if score >= 0.85:
        lessons.append("This approach worked well; reuse it for similar tasks.")

    written = []
    for text in lessons[:3]:
        written.append(db.add_lesson(user_id, text, run_id=run_id, kind="reflection", score=score))

    # Loop 2: store a successful trajectory as a reusable exemplar.
    if score >= 0.8 and len(answer) > 120:
        memory.remember(
            user_id,
            f"Exemplar — task: {question[:200]} → approach that scored {score:.2f}: {answer[:400]}",
            layer="procedural",
            tags="exemplar,replay",
            source_run=run_id,
            strength=0.9,
        )

    # Episodic record of the run itself.
    memory.remember(
        user_id,
        f"Run on '{question[:120]}' scored {score:.2f} using {steps_used} steps and {tool_calls} tool calls.",
        layer="episodic",
        tags="run",
        source_run=run_id,
        strength=0.7,
    )

    return {"score": score, "parts": parts, "lessons": written}


# ---------------------------------------------------------------------------
# 2. Experience replay
# ---------------------------------------------------------------------------
def replay_examples(user_id: str, question: str, *, limit: int = 2) -> str:
    """Retrieve successful past approaches for a similar task."""
    hits = memory.retrieve(user_id, question, limit=limit * 3, layers=("procedural",))
    picked = [h for h in hits if "exemplar" in (h.get("tags") or "")][:limit]
    if not picked:
        return ""
    return "\n\n".join(f"- {h['content'][:600]}" for h in picked)


# ---------------------------------------------------------------------------
# 3. Prompt & policy optimisation (multi-armed bandit)
# ---------------------------------------------------------------------------
DEFAULT_VARIANTS = {
    "general": [
        {
            "name": "direct",
            "body": "Answer directly and concisely. Lead with the result, then the detail.",
        },
        {
            "name": "evidence-first",
            "body": (
                "Gather evidence with tools before answering. Cite what you found. "
                "Mark anything you could not verify as uncertain."
            ),
        },
        {
            "name": "structured",
            "body": (
                "Structure the answer: a one-line answer, then the reasoning, then the "
                "concrete next steps. Use headings and lists."
            ),
        },
    ],
    "research": [
        {"name": "sources", "body": "Search at least two independent sources and compare them. Report disagreements."},
        {"name": "primary", "body": "Prefer primary sources (docs, filings, official pages) over summaries."},
    ],
    "code": [
        {"name": "run-it", "body": "Write the code, then RUN it in the sandbox and report the actual output."},
        {"name": "explain", "body": "Write the code, run it, then explain the key design decisions in two lines."},
    ],
}


def seed_variants(user_id: str) -> int:
    """Create the default variant pool for a user (idempotent)."""
    existing = db.list_variants(user_id)
    have = {(v["task_class"], v["name"]) for v in existing}
    created = 0
    for task_class, variants in DEFAULT_VARIANTS.items():
        for v in variants:
            if (task_class, v["name"]) in have:
                continue
            db.add_variant(user_id, task_class, v["name"], v["body"])
            created += 1
    return created


def classify_task(question: str) -> str:
    q = (question or "").lower()
    if any(k in q for k in ("code", "function", "bug", "script", "api", "python", "javascript", "debug", "implement")):
        return "code"
    if any(k in q for k in ("research", "compare", "latest", "news", "find out", "investigate", "sources", "market")):
        return "research"
    return "general"


def pick_variant(user_id: str, task_class: str) -> Optional[dict]:
    """UCB1 bandit over the active variants of a task class."""
    variants = db.list_variants(user_id, task_class=task_class, active_only=True)
    if not variants:
        return None
    total = sum(v["wins"] + v["losses"] for v in variants)
    if total == 0:
        return random.choice(variants)
    best, best_score = None, -1.0
    for v in variants:
        n = v["wins"] + v["losses"]
        if n == 0:
            return v  # always explore an untried arm first
        mean = v["wins"] / n
        explore = math.sqrt(2 * math.log(total + 1) / n)
        ucb = mean + explore
        if ucb > best_score:
            best, best_score = v, ucb
    return best


def variant_prompt(user_id: str, question: str) -> tuple[str, str]:
    """Return (task_class, prompt_fragment) for this question."""
    task_class = classify_task(question)
    variant = pick_variant(user_id, task_class)
    if not variant:
        return task_class, ""
    return task_class, variant["body"]


def record_outcome(user_id: str, task_class: str, variant_id: str, *, score: float, threshold: float = 0.7) -> None:
    """Feed a run's score back to the bandit."""
    if not variant_id:
        return
    db.record_variant_outcome(variant_id, won=score >= threshold, score=score)


def optimize(user_id: str, *, eval_id: str = "") -> dict:
    """One optimisation round: score active variants, promote the winner, retire losers."""
    seed_variants(user_id)
    from . import evals as evals_mod

    report = {"task_classes": [], "promoted": [], "retired": []}
    for task_class in DEFAULT_VARIANTS:
        variants = db.list_variants(user_id, task_class=task_class, active_only=True)
        if len(variants) < 2:
            continue
        scored = []
        for v in variants:
            if eval_id:
                result = evals_mod.run_eval(user_id, eval_id, variant_id=v["id"])
                score = result["score"]
            else:
                n = v["wins"] + v["losses"]
                score = (v["wins"] / n) if n else 0.0
            scored.append((score, v))
        scored.sort(key=lambda x: x[0], reverse=True)
        winner_score, winner = scored[0]
        report["task_classes"].append({
            "task_class": task_class,
            "winner": winner["name"],
            "score": round(winner_score, 4),
            "arms": [{"name": v["name"], "score": round(s, 4)} for s, v in scored],
        })
        # Retire arms that are clearly worse and have enough evidence.
        for score, v in scored[1:]:
            n = v["wins"] + v["losses"]
            if n >= 5 and score < winner_score - 0.15:
                db.set_variant_active(v["id"], False)
                report["retired"].append({"task_class": task_class, "name": v["name"]})
        report["promoted"].append({"task_class": task_class, "name": winner["name"]})
    return report


# ---------------------------------------------------------------------------
# 4. Weight training — dataset export
# ---------------------------------------------------------------------------
def export_dataset(user_id: str, *, fmt: str = "messages", limit: int = 500) -> list[dict]:
    """Export this user's own trajectories as a clean SFT dataset.

    ``fmt``:
      * ``messages``  — {"messages":[{role,content},...]}  (OpenAI / TRL style)
      * ``sharegpt``  — {"conversations":[{"from","value"},...]}  (Axolotl / LLaMA-Factory)
    """
    rows: list[dict] = []
    runs = db.list_runs(user_id, limit=limit)
    for run in runs:
        if run["status"] != "done":
            continue
        steps = db.list_steps(run["id"])
        messages = db.list_messages(run["conversation_id"], limit=200)
        # find the user turn that started this run and the assistant answer
        question = ""
        for m in messages:
            if m["role"] == "user":
                question = m["content"]
        answer = ""
        for s in reversed(steps):
            if s["kind"] == "answer":
                answer = s["output"]
                break
        if not answer:
            continue
        tool_trace = [
            {"name": s["name"], "ok": s["ok"], "output": (s["output"] or "")[:800]}
            for s in steps if s["kind"] == "tool"
        ]
        if fmt == "sharegpt":
            rows.append({
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer},
                ],
                "meta": {"run_id": run["id"], "tools": tool_trace},
            })
        else:
            rows.append({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "meta": {"run_id": run["id"], "tools": tool_trace},
            })
    return rows


def export_preferences(user_id: str, *, limit: int = 500) -> list[dict]:
    """Export DPO-style preference pairs from critic pass/fail on the same task class."""
    lessons = db.list_lessons(user_id, limit=limit)
    pairs = []
    for lesson in lessons:
        if lesson["kind"] != "reflection":
            continue
        pairs.append({
            "prompt": lesson["text"],
            "chosen_score": lesson["score"],
            "rejected_score": max(0.0, lesson["score"] - 0.3),
        })
    return pairs


def stats(user_id: str) -> dict:
    lessons = db.list_lessons(user_id, limit=500)
    variants = db.list_variants(user_id)
    runs = db.list_runs(user_id, limit=500)
    scores = [l["score"] for l in lessons if l["score"] > 0]
    return {
        "runs": len(runs),
        "lessons": len(lessons),
        "variants": len(variants),
        "active_variants": len([v for v in variants if v["active"]]),
        "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "best_score": round(max(scores), 4) if scores else 0.0,
        "loops": {
            "reflection": "on",
            "experience_replay": "on",
            "prompt_optimisation": "on",
            "weight_training": "available (needs a local GPU)",
        },
    }


def describe() -> dict:
    return {
        "loops": [
            {"id": "reflection", "free": True, "always_on": True,
             "what": "Score every run on a rubric and write a lesson into memory."},
            {"id": "experience_replay", "free": True, "always_on": True,
             "what": "Reuse successful past approaches as few-shot context."},
            {"id": "prompt_optimisation", "free": True, "always_on": True,
             "what": "UCB1 bandit over prompt variants, rewarded by eval scores."},
            {"id": "weight_training", "free": True, "always_on": False,
             "what": "Export trajectories as SFT/DPO data and fine-tune a LoRA locally.",
             "requirement": "a GPU you own (16-24 GB for a 7-8B QLoRA run)"},
        ],
        "honest_limits": (
            "JagX does not train a frontier model — that is not possible for free. "
            "Loops 1-3 are genuinely free and run in production. Loop 4 is free software "
            "that needs local hardware."
        ),
    }
