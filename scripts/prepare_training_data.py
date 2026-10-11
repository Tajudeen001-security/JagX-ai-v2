"""Filter a JSONL training candidate to explicitly consented, bounded examples."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

SECRET_PATTERNS = [
    re.compile(r"(?i)\b(api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
]

def clean_text(value: object, limit: int) -> str:
    text=str(value or "").strip()[:limit]
    for pattern in SECRET_PATTERNS:
        text=pattern.sub("[REDACTED]",text)
    return text

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("input",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    accepted=0
    rejected=0
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w",encoding="utf-8") as dest:
        for line in source:
            try:
                row=json.loads(line)
                if row.get("consent") is not True:
                    rejected+=1
                    continue
                prompt=clean_text(row.get("prompt") or row.get("input"),4000)
                response=clean_text(row.get("response") or row.get("output"),8000)
                if len(prompt)<2 or len(response)<2:
                    rejected+=1
                    continue
                dest.write(json.dumps({"prompt":prompt,"response":response},ensure_ascii=False)+"\n")
                accepted+=1
            except (json.JSONDecodeError,AttributeError,TypeError):
                rejected+=1
    print(json.dumps({"accepted":accepted,"rejected":rejected,"output":str(args.output)},indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
