#!/usr/bin/env python3
"""Download Wakai checkpoint + tokenizer from the public Google Drive folder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Install gdown: pip install gdown")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "checkpoints_manifest.json").read_text())


def download(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading -> {dest}")
    gdown.download(url, str(dest), quiet=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Download Wakai weights from Drive")
    p.add_argument(
        "--which",
        choices=["clean", "final", "longest"],
        default="clean",
        help="Which recommended checkpoint (default: clean = step 10000 cleaned)",
    )
    p.add_argument("--out", type=Path, default=ROOT / "weights", help="Output directory")
    p.add_argument("--tokenizer-only", action="store_true")
    args = p.parse_args()

    out: Path = args.out
    tok = MANIFEST["tokenizer"]
    download(tok["model"]["file_id"], out / "wakai.model")
    download(tok["vocab"]["file_id"], out / "wakai.vocab")

    if not args.tokenizer_only:
        ckpt = MANIFEST["recommended"][args.which]
        download(ckpt["file_id"], out / ckpt["name"])
        print(f"Done. Checkpoint: {out / ckpt['name']}")
    else:
        print("Tokenizer only downloaded.")


if __name__ == "__main__":
    main()
