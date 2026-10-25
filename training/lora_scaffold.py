#!/usr/bin/env python3
"""
JagX AI — QLoRA fine-tune scaffold (Phase 1 of ROADMAP_100B.md)

Run this ONLY when you have a GPU (local 24GB+ or cloud A10/A100/H100).
Does not train by default — it validates data and documents the run.

Usage:
  python training/lora_scaffold.py --data data/sft_jagx.jsonl --dry-run
  python training/lora_scaffold.py --data data/sft_jagx.jsonl --base Qwen/Qwen2.5-7B-Instruct --out models/jagx-7b-lora

Dataset format (JSONL):
  {"messages": [{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

Identity rule: every assistant turn should be pure JagX AI (JagX & JRILICENSE).
Never train on secrets, API keys, or non-consented private memory.

Created by JagX & JRILICENSE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON on line {i}: {e}") from e
            if "messages" not in obj or not isinstance(obj["messages"], list):
                raise SystemExit(f"Line {i}: expected {{'messages': [...]}}")
            rows.append(obj)
    return rows


def validate_identity(rows: list) -> int:
    bad = 0
    forbidden = ("openai", "chatgpt", "claude", "anthropic", "gemini", "google ai", "built by meta")
    for i, row in enumerate(rows):
        for m in row["messages"]:
            if m.get("role") != "assistant":
                continue
            text = (m.get("content") or "").lower()
            if any(f in text for f in forbidden):
                bad += 1
                print(f"  WARN row {i}: possible non-JagX identity leak")
                break
    return bad


def main() -> int:
    p = argparse.ArgumentParser(description="JagX QLoRA scaffold")
    p.add_argument("--data", type=Path, required=True, help="JSONL SFT dataset")
    p.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct", help="HF base model id")
    p.add_argument("--out", type=Path, default=Path("models/jagx-lora"), help="Output adapter dir")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="Validate only; do not train")
    args = p.parse_args()

    if not args.data.exists():
        print(f"Dataset not found: {args.data}")
        print("Create JSONL rows with messages[] (system/user/assistant).")
        return 1

    rows = load_jsonl(args.data)
    print(f"Loaded {len(rows)} examples from {args.data}")
    if len(rows) < 50:
        print("WARN: <50 examples — fine-tune will overfit. Aim for 500+ quality pairs.")
    leaks = validate_identity(rows)
    if leaks:
        print(f"Identity warnings: {leaks} (clean before production train)")

    if args.dry_run:
        print("Dry-run OK. No training started.")
        print("Next: provision GPU, then re-run without --dry-run.")
        return 0

    try:
        import torch
    except ImportError as e:
        print("Missing train deps. On GPU machine install:")
        print("  pip install torch transformers peft bitsandbytes datasets accelerate trl")
        print(f"Import error: {e}")
        return 2

    if not torch.cuda.is_available():
        print("No CUDA GPU detected. Aborting (use cloud GPU or --dry-run).")
        return 3

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Base: {args.base}")
    print(f"Out:  {args.out}")
    print(
        "\nScaffold ready. Wire TRL SFTTrainer here with:\n"
        "  - 4-bit load (BitsAndBytesConfig)\n"
        "  - LoraConfig(r=16, lora_alpha=32, target_modules=q/k/v/o_proj)\n"
        "  - max_seq_length=2048\n"
        "  - eval on held-out coding + Pidgin + identity tests\n"
        "See docs/ROADMAP_100B.md Phase 1.\n"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "base": args.base,
        "examples": len(rows),
        "created_by": "JagX & JRILICENSE",
        "note": "Scaffold only — full trainer to be completed on GPU host",
    }
    (args.out / "scaffold_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out / 'scaffold_meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
