"""JagX v13 — the eval harness.

The measurable half of self-improvement: named suites of cases with a checker,
scored per prompt variant, so a prompt change is only promoted when it actually
scores better.

A case is:
    {"input": "...", "expect_any": ["..."], "expect_all": ["..."], "min_len": 40}

Checkers are deterministic (substring / length / regex) so scoring is free,
instant and reproducible. A model-graded checker can be layered on later.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from . import db, llm
from .config import settings

logger = logging.getLogger("jagx.evals")

DEFAULT_SUITES = {
    "core": [
        {"input": "What is 17 * 23?", "expect_any": ["391"], "min_len": 2},
        {"input": "Say hello in one short sentence.", "min_len": 5},
        {"input": "Name three primary colours.", "expect_any": ["red"], "min_len": 10},
    ],
    "tools": [
        {"input": "Use the calculator to compute 1234 * 5678 and report the exact number.",
         "expect_any": ["7006652"], "min_len": 5},
        {"input": "Search the web for the current year and tell me what you find.",
         "min_len": 20},
    ],
    "reasoning": [
        {"input": "If a train travels 60 km in 45 minutes, what is its average speed in km/h?",
         "expect_any": ["80"], "min_len": 5},
        {"input": "Which is larger: 2^10 or 10^3? Answer with the number.",
         "expect_any": ["1024"], "min_len": 3},
    ],
}


def seed_suites(user_id: str) -> int:
    existing = {e["name"] for e in db.list_evals(user_id)}
    created = 0
    for name, cases in DEFAULT_SUITES.items():
        if name in existing:
            continue
        db.create_eval(user_id, name, cases)
        created += 1
    return created


def _check(case: dict, answer: str) -> tuple[bool, str]:
    text = (answer or "").lower()
    if len(answer or "") < int(case.get("min_len", 1)):
        return False, f"too short ({len(answer or '')} chars)"
    for needle in case.get("expect_all") or []:
        if str(needle).lower() not in text:
            return False, f"missing required: {needle}"
    any_list = case.get("expect_any") or []
    if any_list and not any(str(n).lower() in text for n in any_list):
        return False, f"none of: {any_list}"
    pattern = case.get("regex")
    if pattern and not re.search(pattern, answer or "", re.IGNORECASE):
        return False, f"regex did not match: {pattern}"
    return True, "ok"


def run_eval(user_id: str, eval_id: str, *, variant_id: str = "", model: str = "") -> dict:
    """Run one suite. If a variant is given, its prompt fragment is prepended."""
    suite = db.get_eval(user_id, eval_id)
    if not suite:
        return {"score": 0.0, "error": "eval not found", "cases": []}

    fragment = ""
    if variant_id:
        for v in db.list_variants(user_id):
            if v["id"] == variant_id:
                fragment = v["body"]
                break

    results = []
    passed = 0
    for case in suite["cases"]:
        system = "You are JagX, a capable autonomous agent. Answer accurately and concisely."
        if fragment:
            system += "\n\nStyle guidance: " + fragment
        try:
            out = llm.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": str(case.get("input", ""))[:2000]},
                ],
                temperature=0.2,
                max_tokens=600,
                model=model,
            )
            answer = out.content
        except llm.LLMError as exc:
            results.append({"input": case.get("input"), "ok": False, "reason": f"llm error: {exc}"})
            continue
        ok, reason = _check(case, answer)
        passed += 1 if ok else 0
        results.append({
            "input": case.get("input"),
            "ok": ok,
            "reason": reason,
            "answer": answer[:400],
        })

    total = max(1, len(suite["cases"]))
    score = round(passed / total, 4)
    db.add_eval_result(eval_id, variant_id, score, {"results": results})
    return {"eval_id": eval_id, "name": suite["name"], "score": score, "passed": passed, "total": total, "cases": results}


def compare_variants(user_id: str, eval_id: str, *, model: str = "") -> dict:
    """Score every active variant on a suite and rank them."""
    variants = db.list_variants(user_id, active_only=True)
    if not variants:
        return {"error": "no active variants", "ranking": []}
    ranking = []
    for v in variants:
        res = run_eval(user_id, eval_id, variant_id=v["id"], model=model)
        ranking.append({
            "variant_id": v["id"],
            "task_class": v["task_class"],
            "name": v["name"],
            "score": res.get("score", 0.0),
        })
    ranking.sort(key=lambda r: r["score"], reverse=True)
    return {"eval_id": eval_id, "ranking": ranking, "winner": ranking[0] if ranking else None}


def describe() -> dict:
    return {
        "checkers": ["min_len", "expect_any", "expect_all", "regex"],
        "default_suites": list(DEFAULT_SUITES),
        "note": "Deterministic checkers keep scoring free, instant and reproducible.",
    }
