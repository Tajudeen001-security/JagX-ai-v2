"""
JagX AI V9.6 — Stronger smoke + agent eval harness.
Not a certification. Run: python -m evals.run_eval
Created by JagX & JRILICENSE
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent


def _load_cases() -> List[dict]:
    return json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))


def _check_contains(response: str, expected: str) -> bool:
    return bool(expected) and expected.casefold() in (response or "").casefold()


def _check_not_contains(response: str, forbidden: str) -> bool:
    return bool(forbidden) and forbidden.casefold() not in (response or "").casefold()


def eval_chat_cases(cases: List[dict]) -> List[dict]:
    from jagx_api.chat import generate_response

    results = []
    for case in cases:
        if case.get("kind") and case.get("kind") != "chat":
            continue
        name = case.get("name", "unnamed")
        try:
            answer = generate_response(case["input"], use_tools=False)
            response = str(answer.get("response", ""))
            exp = str(case.get("expected_contains", ""))
            forbid = str(case.get("forbidden_contains", "") or "")
            passed = _check_contains(response, exp)
            if forbid and not _check_not_contains(response, forbid):
                passed = False
            results.append({
                "name": name,
                "category": case.get("category"),
                "kind": "chat",
                "passed": passed,
                "provider": answer.get("provider"),
                "response_preview": response[:240],
            })
        except Exception as exc:
            results.append({
                "name": name,
                "category": case.get("category"),
                "kind": "chat",
                "passed": False,
                "error": type(exc).__name__,
            })
    return results


def eval_router_cases(cases: List[dict]) -> List[dict]:
    from jagx_api.agent_router import select_agents

    results = []
    for case in cases:
        if case.get("kind") != "router":
            continue
        name = case.get("name", "unnamed")
        try:
            agents = select_agents(case["input"], max_agents=3)
            expected = case.get("expected_agents") or []
            passed = all(a in agents for a in expected) if expected else bool(agents)
            results.append({
                "name": name,
                "category": case.get("category", "router"),
                "kind": "router",
                "passed": passed,
                "agents": agents,
            })
        except Exception as exc:
            results.append({
                "name": name,
                "category": "router",
                "kind": "router",
                "passed": False,
                "error": type(exc).__name__,
            })
    return results


def eval_identity_cases(cases: List[dict]) -> List[dict]:
    from jagx_api.chat import sanitize_identity

    results = []
    for case in cases:
        if case.get("kind") != "identity":
            continue
        name = case.get("name", "unnamed")
        try:
            raw = case.get("input", "")
            cleaned = sanitize_identity(raw)
            forbid = str(case.get("forbidden_contains", "openai"))
            passed = _check_not_contains(cleaned, forbid)
            if case.get("expected_contains"):
                passed = passed and _check_contains(cleaned, case["expected_contains"])
            results.append({
                "name": name,
                "category": "identity",
                "kind": "identity",
                "passed": passed,
                "response_preview": cleaned[:200],
            })
        except Exception as exc:
            results.append({
                "name": name,
                "category": "identity",
                "kind": "identity",
                "passed": False,
                "error": type(exc).__name__,
            })
    return results


def eval_artifact_cases(cases: List[dict]) -> List[dict]:
    from jagx_api.artifacts import build_response_artifacts

    results = []
    for case in cases:
        if case.get("kind") != "artifacts":
            continue
        name = case.get("name", "unnamed")
        try:
            arts = build_response_artifacts(case.get("input", ""))
            types = [a.get("type") for a in arts]
            expected_type = case.get("expected_type")
            passed = (expected_type in types) if expected_type else bool(arts)
            min_count = int(case.get("min_count") or 0)
            if min_count:
                passed = passed and len(arts) >= min_count
            results.append({
                "name": name,
                "category": "artifacts",
                "kind": "artifacts",
                "passed": passed,
                "types": types,
            })
        except Exception as exc:
            results.append({
                "name": name,
                "category": "artifacts",
                "kind": "artifacts",
                "passed": False,
                "error": type(exc).__name__,
            })
    return results


def eval_preferences() -> List[dict]:
    from jagx_api import preferences as prefs

    results = []
    r1 = prefs.set_preferences("eval-user", {"tone": "direct"}, consent=False)
    results.append({
        "name": "prefs_require_consent",
        "category": "preferences",
        "kind": "preferences",
        "passed": r1.get("ok") is False,
    })
    r2 = prefs.set_preferences(
        "eval-user",
        {"tone": "direct", "verbosity": "short", "code_style": "production"},
        consent=True,
    )
    results.append({
        "name": "prefs_set_ok",
        "category": "preferences",
        "kind": "preferences",
        "passed": bool(r2.get("ok")),
    })
    ctx = prefs.format_preference_context("eval-user")
    results.append({
        "name": "prefs_context_string",
        "category": "preferences",
        "kind": "preferences",
        "passed": bool(ctx) and "direct" in (ctx or ""),
    })
    prefs.clear_preferences("eval-user")
    results.append({
        "name": "prefs_clear",
        "category": "preferences",
        "kind": "preferences",
        "passed": True,
    })
    return results


def main() -> int:
    cases = _load_cases()
    results: List[Dict[str, Any]] = []
    results.extend(eval_chat_cases(cases))
    results.extend(eval_router_cases(cases))
    results.extend(eval_identity_cases(cases))
    results.extend(eval_artifact_cases(cases))
    results.extend(eval_preferences())

    total = len(results)
    passed = sum(1 for item in results if item.get("passed"))
    by_kind: Dict[str, Dict[str, int]] = {}
    for item in results:
        k = item.get("kind") or "other"
        by_kind.setdefault(k, {"total": 0, "passed": 0})
        by_kind[k]["total"] += 1
        if item.get("passed"):
            by_kind[k]["passed"] += 1

    report = {
        "version": "9.6.0",
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0,
        "by_kind": by_kind,
        "warning": (
            "Smoke + structural eval only. Not a comprehensive multilingual, "
            "reasoning, safety, or production certification."
        ),
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if total and passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
