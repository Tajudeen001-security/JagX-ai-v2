# JagX Training

Phase 1 of `docs/ROADMAP_100B.md`.

## Quick start (when you have a GPU)

1. Export consented SFT data to `data/sft_jagx.jsonl`.
2. Validate:
   ```bash
   python training/lora_scaffold.py --data data/sft_jagx.jsonl --dry-run
   ```
3. On a GPU host:
   ```bash
   pip install torch transformers peft bitsandbytes datasets accelerate trl
   python training/lora_scaffold.py --data data/sft_jagx.jsonl --base Qwen/Qwen2.5-7B-Instruct --out models/jagx-7b-lora
   ```
4. Quantize to GGUF and point `jagx_api/local_model.py` at the file.

## Rules

- Only consented production data
- Strip secrets and private memory
- Enforce JagX identity in assistant turns
- Eval: coding, Pidgin/Africa, identity refusal

Built by JagX & JRILICENSE.
