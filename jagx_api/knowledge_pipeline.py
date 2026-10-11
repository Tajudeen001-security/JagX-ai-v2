"""Local knowledge ingestion helpers with strict size and schema limits."""
import json
from pathlib import Path
MAX_BYTES=2_000_000; MAX_ROWS=10_000; MAX_TEXT=20_000
def validate_rows(payload)->list[dict]:
    if not isinstance(payload,list): raise ValueError("knowledge input must be a JSON list")
    if len(payload)>MAX_ROWS: raise ValueError(f"maximum {MAX_ROWS} knowledge rows")
    clean=[]
    for i,row in enumerate(payload):
        if not isinstance(row,dict): raise ValueError(f"row {i} must be an object")
        q=str(row.get("question") or row.get("q") or "").strip(); a=str(row.get("answer") or row.get("a") or "").strip()
        if not q or not a: raise ValueError(f"row {i} needs question/q and answer/a")
        if len(q)>MAX_TEXT or len(a)>MAX_TEXT: raise ValueError(f"row {i} exceeds text limit")
        clean.append({"question":q,"answer":a})
    return clean
def ingest_json_file(source:str,destination:str)->dict:
    src=Path(source)
    if not src.is_file(): raise ValueError("source file does not exist")
    if src.stat().st_size>MAX_BYTES: raise ValueError("source exceeds 2 MB limit")
    rows=validate_rows(json.loads(src.read_text(encoding="utf-8")))
    dest=Path(destination); dest.parent.mkdir(parents=True,exist_ok=True); tmp=dest.with_suffix(dest.suffix+".tmp")
    tmp.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8"); tmp.replace(dest)
    return {"ok":True,"rows":len(rows),"destination":str(dest)}
def evaluate_exact_match(cases:list[dict],responder)->dict:
    results=[]
    for case in cases:
        expected=str(case.get("expected_contains","")).casefold(); actual=str(responder(str(case.get("input",""))) or "")
        results.append({"name":str(case.get("name","unnamed")),"passed":bool(expected) and expected in actual.casefold()})
    total=len(results); passed=sum(x["passed"] for x in results)
    return {"total":total,"passed":passed,"pass_rate":passed/total if total else 0.0,"results":results}
