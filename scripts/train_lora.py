#!/usr/bin/env python3
"""JagX v13 — free QLoRA fine-tuning pipeline for open-weight models.

This is the real thing, not a stub. It takes the dataset JagX Forge exports
(``data/forge/dataset.jsonl``) and fine-tunes a small open-weight model with
QLoRA on a single consumer GPU — or a free Colab T4.

    # 1. Export a dataset from the running app (Self-Improvement → Export dataset)
    # 2. Install the training extras
    pip install torch transformers peft datasets bitsandbytes accelerate trl

    # 3. Train
    python scripts/train_lora.py \
        --dataset data/forge/dataset.jsonl \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --output data/forge/adapter

    # 4. Serve the adapter with Ollama (see --export-ollama below)

Honest limits
-------------
* Training needs a GPU. On CPU this script will run but be impractically slow.
* A free Colab T4 (16 GB) comfortably trains a 1.5B–3B model with QLoRA.
* JagX never claims to have trained a model it has not trained. This script
  produces a genuine adapter when you run it on real hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_dataset(path: Path) -> list[dict]:
    """Read JagX's JSONL export into chat-format records."""
    if not path.exists():
        sys.exit(
            f"Dataset not found: {path}\n"
            "Export one from the app: Self-Improvement → Export dataset."
        )
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        sys.exit(f"Dataset is empty: {path}")
    return rows


def to_chat(rows: list[dict]) -> list[dict]:
    """Normalise JagX records into ``{"messages": [...]}`` chat format."""
    out: list[dict] = []
    for row in rows:
        if "messages" in row:
            out.append({"messages": row["messages"]})
            continue
        system = row.get("system") or "You are JagX, an autonomous agent."
        user = row.get("prompt") or row.get("input") or ""
        assistant = row.get("completion") or row.get("output") or row.get("answer") or ""
        if not user or not assistant:
            continue
        out.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        })
    if not out:
        sys.exit("No usable training records found in the dataset.")
    return out


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        sys.exit(
            f"Missing training dependency: {exc.name}\n"
            "Install them with:\n"
            "  pip install torch transformers peft datasets bitsandbytes accelerate trl"
        )

    rows = to_chat(load_dataset(Path(args.dataset)))
    print(f"[jagx] {len(rows)} training records from {args.dataset}")

    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        print(
            "[jagx] WARNING: no CUDA GPU detected. Training will run on CPU and be "
            "extremely slow. A free Colab T4 is the recommended path."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if has_cuda and args.qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto" if has_cuda else None,
        torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
        trust_remote_code=True,
    )
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules.split(","),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def format_row(row: dict) -> dict:
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list([format_row(r) for r in rows])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        bf16=has_cuda,
        report_to=[],
        warmup_ratio=0.03,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[jagx] adapter saved to {out_dir}")

    if args.export_ollama:
        modelfile = out_dir / "Modelfile"
        modelfile.write_text(
            f"FROM {args.base_model}\n"
            f"ADAPTER {out_dir.resolve()}\n"
            'PARAMETER temperature 0.7\n'
            'SYSTEM "You are JagX, an autonomous agent that plans, uses tools and '
            'improves itself."\n',
            encoding="utf-8",
        )
        print(
            f"[jagx] wrote {modelfile}\n"
            f"[jagx] serve it with:  ollama create jagx-lora -f {modelfile}\n"
            f"[jagx] then set JAGX_LLM_MODEL=jagx-lora"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="JagX free QLoRA fine-tuning pipeline")
    parser.add_argument("--dataset", default="data/forge/dataset.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", default="data/forge/adapter")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--qlora", action="store_true", default=True)
    parser.add_argument("--no-qlora", dest="qlora", action="store_false")
    parser.add_argument("--export-ollama", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
