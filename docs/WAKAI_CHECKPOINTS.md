# Wakai trained checkpoints (JagX)

**Created by:** JagX & JRILICENSE  
**Drive folder:** https://drive.google.com/drive/folders/1-cPZzUpmDRWox3Yl_HruNnbNgGVSmiUi

## What you trained

| Asset | Location | Notes |
|-------|----------|-------|
| Model code | `code/wakai_model.py` | Transformer: dim=576, 14 layers, 8 heads, ctx 512 |
| Checkpoints | `checkpoints/*.pt` | ~315 MB each |
| Tokenizer | `tokenizer/wakai.model` + `.vocab` | SentencePiece |
| Data | `data/train.txt` (~98MB), `train_clean.txt` (~10MB) | Training corpora |

### Recommended weights
1. **`wakai_clean_step_10000.pt`** — cleaned data run (preferred for text quality)
2. **`wakai_final.pt`** — marked final from first run
3. **`wakai_step_30000.pt`** — longest step count on the first run

## Why not on GitHub?

Each `.pt` is **~315 MB**. GitHub file limit is 100 MB (Git LFS is possible but heavy). Weights stay on Drive; **code + download script** are on GitHub.

## Download locally

```bash
cd JagX-ai-v2
pip install gdown torch sentencepiece
python wakai/download_weights.py --which clean
# files land in wakai/weights/
python wakai/infer.py --ckpt wakai/weights/wakai_clean_step_10000.pt --tokenizer wakai/weights/wakai.model --prompt "Who are you?"
```

## Render / production note

**Render free tier cannot load a 315MB GPU-style model reliably.**  
Production API stays:
- local **knowledge packs** (fast)
- **Groq / cloud LLM** (quality)

Use Wakai on:
- your laptop / Colab / Kaggle GPU
- a paid VM with enough RAM if you want self-hosted generation

## Resume more training

```bash
# On Kaggle/Colab with GPU — after downloading a checkpoint
# Load state and continue training with your train_clean.txt corpus
```

Always copy new `step-*.pt` files back into the Drive `checkpoints/` folder before the session ends.

## Manifest

See `wakai/checkpoints_manifest.json` for all Drive file IDs used by the downloader.
