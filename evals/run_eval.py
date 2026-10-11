"""Small smoke-evaluation runner. Results are not a certification of model quality."""
from __future__ import annotations
import json
from pathlib import Path
from jagx_api.chat import generate_response

def main() -> int:
    root=Path(__file__).resolve().parent
    cases=json.loads((root/"cases.json").read_text(encoding="utf-8"))
    results=[]
    for case in cases:
        try:
            answer=generate_response(case["input"],use_tools=False)
            response=str(answer.get("response","")).casefold()
            expected=str(case.get("expected_contains","")).casefold()
            results.append({"name":case.get("name","unnamed"),"category":case.get("category"),
                            "passed":bool(expected and expected in response),
                            "provider":answer.get("provider"),"response_preview":response[:240]})
        except Exception as exc:
            results.append({"name":case.get("name","unnamed"),"category":case.get("category"),
                            "passed":False,"error":type(exc).__name__})
    total=len(results)
    passed=sum(1 for item in results if item["passed"])
    report={"total":total,"passed":passed,"pass_rate":round(passed/total,3) if total else 0,
            "warning":"Small smoke set only; not a comprehensive multilingual, reasoning, or safety evaluation.",
            "results":results}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if total and passed==total else 1

if __name__=="__main__":
    raise SystemExit(main())
