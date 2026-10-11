# JagX AI model improvement plan (free-first)
## Measure before fine-tuning
Fine-tuning can improve a narrow behavior; it does not automatically add reliable facts, language coverage, reasoning, or safe tool use. First build a versioned evaluation set and establish a baseline for each configured provider and local fallback.
## Stages
1. Curate opt-in, consented examples; remove secrets, personal data, credentials, and material without permission.
2. Split train/validation/test by source and time to prevent near-duplicate leakage.
3. Score language matching, code-switching, factuality, instruction following, tool-call validity, safety, latency, errors, and cost. Preserve the no-paid-key constraint.
4. Improve prompts, retrieval, deterministic tools, and provider routing before training.
5. Evaluate open-weight models locally only if hardware can run them; account for RAM, disk, speed, and license terms.
6. Consider LoRA/QLoRA only on free compute actually available or owned hardware; free GPU quota is not guaranteed.
7. Promote a model only if held-out tests improve without safety or language regressions.
## Data governance
Training capture must be opt-in. Never silently train on private conversations. Support deletion, provenance, redaction, and restricted dataset/checkpoint access.
## Current status
This repository includes the improvement plan and evaluation helper. It does not claim to have trained or deployed a fine-tuned model.
