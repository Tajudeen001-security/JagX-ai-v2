# Resume Wakai training on Kaggle

1. New Kaggle notebook → GPU T4
2. Enable Internet
3. Run:

```python
!pip -q install gdown sentencepiece
!mkdir -p /kaggle/working/wakai
# clean checkpoint + tokenizer
!gdown --id 1EBinyXVylBgbLmQhWKiToOWUgGBHn3it -O /kaggle/working/wakai/wakai_clean_step_10000.pt
!gdown --id 1w4C67pqqSwTtZHtfgC1Yzib40407JR7L -O /kaggle/working/wakai/wakai.model
!gdown --id 1JdNf9za4A0-ibSq69WGKl0KDZLG1qOqA -O /kaggle/working/wakai/wakai.vocab
!gdown --id 1gQAna_HmVC127bJRQYpIwvNuPczylV1L -O /kaggle/working/wakai/train_clean.txt
print('ready')
```

4. Clone JagX-ai-v2 for model code, or copy `wakai/model.py`.
5. Load checkpoint, continue training, save `wakai_clean_step_12000.pt` etc.
6. **Before the session dies:** download the new `.pt` and upload to Drive `checkpoints/`.

Do not rely on Kaggle output surviving overnight.
