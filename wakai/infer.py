#!/usr/bin/env python3
"""Load a Wakai checkpoint and generate text (local / GPU)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import Wakai

try:
    import sentencepiece as spm
except ImportError:
    spm = None


def load_checkpoint(path: Path, device: torch.device):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"], state
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"], state
    return state, {}


def build_model(state_dict, device: torch.device) -> Wakai:
    # Infer vocab size from embedding weight if present
    vocab_size = None
    for k, v in state_dict.items():
        if k.endswith("tok_emb.weight") or k == "tok_emb.weight":
            vocab_size = v.shape[0]
            break
    if vocab_size is None:
        vocab_size = 32000
    model = Wakai(vocab_size=vocab_size)
    # strip possible module. prefix
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print("Missing keys (sample):", list(missing)[:5])
    if unexpected:
        print("Unexpected keys (sample):", list(unexpected)[:5])
    model.to(device)
    model.eval()
    print(f"Params: {model.get_num_params():,}")
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, help="Path to wakai.model (SentencePiece)")
    p.add_argument("--prompt", type=str, default="Hello, I am JagX AI")
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    state_dict, meta = load_checkpoint(args.ckpt, device)
    print("Meta keys:", list(meta.keys())[:10] if isinstance(meta, dict) else type(meta))
    model = build_model(state_dict, device)

    if args.tokenizer and spm is not None:
        sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
        ids = sp.encode(args.prompt, out_type=int)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=args.max_new, temperature=args.temperature)
        text = sp.decode(out[0].tolist())
        print("\n=== GENERATION ===\n")
        print(text)
    else:
        print("No tokenizer — loaded weights only. Install sentencepiece and pass --tokenizer.")
        print("Model ready on", device)


if __name__ == "__main__":
    main()
