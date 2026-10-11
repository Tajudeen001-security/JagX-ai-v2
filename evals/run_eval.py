"""Run the small smoke evaluation set against the configured provider chain."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jagx_api.chat import generate_response

def main() -> int:
    cases = json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))
    results = []
    for case in cases:
        try:
            answer = generate_response(case["input"], use_tools=False)
            text = str(answer.get("response", "")).casefold()
            expected = str(case.get("expected_contains", "")).casefold()
            passed = bool(expected) and expected in text
            results.append({"name":case.get("name","unnamed"),"passed":passed,"provider":answer.get("provider")})
        except Exception as exc:
            results.append({"name":case.get("name","unnamed"),"passed":False,"error":type(exc).__name__})
    total=len(results); passed=sum(bool(r["passed"]) for r in results)
    report={"total":total,"passed":passed,"pass_rate":passed/total if total else 0.0,"results":results}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if passed==total else 1

if __name__ == "__main__":
    raise SystemExit(main())
