# JagX AI model improvement and fine-tuning plan

## 1. Baseline before training
Run `python -m pytest -q` and `python -m evals.run_eval`. Keep evaluation prompts and expected checks fixed so a new model can be compared with the current provider chain. Record the model name, quantization, prompt version, provider, pass rate, latency, and failures.

## 2. Small local inference model
A compact GGUF model can provide basic local responses without a per-request API key. A 0.5B parameter model in 4-bit quantization is a reasonable low-memory experiment, but it will not match larger models in reasoning or language breadth. Download the GGUF separately, verify its licence, configure `JAGX_GGUF_MODEL`, and check `GET /offline/status`. The repository does not redistribute model weights.

## 3. Training-data requirements
Only use examples with explicit, recorded permission for model training. Do not silently train on ordinary chats. Remove credentials, private identifiers, and unnecessary personal details. Maintain a deletion process and retention period. The normal API logging path is disabled unless both the operator enables `JAGX_TRAINING_DATA_ENABLED=true` and the user sets `training_consent=true` for that turn.

The dataset preparation helper accepts only JSONL examples with `consent: true`; it is a validation/filtering step, not a trainer.

## 4. Fine-tuning
For a small open-weight model, evaluate LoRA/QLoRA with a compatible training library on a GPU-equipped machine or genuinely available free notebook quota. Do not assume free GPU time exists. Start with a small domain-specific dataset and keep an untouched test set. Check model licence, redistribution terms, privacy obligations, and hardware requirements first.

## 5. Release gates
A candidate must not be released unless it improves the target metrics without unacceptable regressions in multilingual instructions, factuality, refusal/safety boundaries, tool execution, latency, or memory use. Keep the previous model available for rollback. Fine-tuning changes model weights; it does not automatically create reliable reasoning or verified engineering knowledge.
