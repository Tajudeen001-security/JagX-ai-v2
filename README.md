# JagX v13 — an autonomous agent that owns its own brain

**JagX** is a full-stack autonomous AI agent in the class of Grok bot, Dots, Muse and
Claude Duet: it **plans**, **calls tools**, **browses**, **writes and runs code**,
**sees and makes images**, **reads documents**, **remembers**, **critiques itself**,
**runs long-lived goals unattended**, and **improves itself** — with **no third-party
company API key** required to run.

It is local-first: point it at Ollama, vLLM or llama.cpp and it works entirely on your
own machine, offline, for free. A built-in mock provider lets the whole product run
end to end with no model at all. The multimodal engine follows the same rule — image
generation, image editing, vision and document parsing all run locally with no key.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  JagX Web (React + TypeScript + Vite)                                    │
│  Chat · timeline · Council · Media Studio · Goals · Memory · Forge       │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │ REST + SSE
┌───────────────────────────────▼──────────────────────────────────────────┐
│  JagX API (FastAPI)                                                      │
│  auth · conversations · chat SSE · goals · memory · council · forge      │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  JagX Loop    │     │  JagX Council   │     │  JagX Forge      │
│  plan→act→    │     │  coordinator +  │     │  reflection ·    │
│  observe→     │     │  specialists +  │     │  replay · prompt │
│  critique     │     │  critic         │     │  optimisation ·  │
└───────┬───────┘     └─────────────────┘     │  LoRA pipeline   │
        │                                     └──────────────────┘
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Tools · Sandbox · Guard · Memory (4 layers) · Multimodal · Embeddings   │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  LLM client — OpenAI-compatible, local-first                             │
│  Ollama · vLLM · llama.cpp · LM Studio · mock (no model needed)          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Why it is different

| Requirement | How JagX does it |
|---|---|
| **No third-party API keys** | The provider chain is local-first: Ollama → vLLM → llama.cpp → LM Studio → any OpenAI-compatible endpoint. A built-in mock provider runs the whole product with no model. Optional hosted providers exist but are never required. |
| **Self-training, kept free** | JagX Forge runs four loops: reflection, experience replay, prompt/policy optimisation, and a **LoRA/QLoRA fine-tuning pipeline** that exports a real dataset and a runnable `train_lora.py` for open-weight models. |
| **Autonomous work** | The JagX Loop plans, acts, observes, critiques and re-plans under a step budget. JagX Goals runs long-lived objectives unattended, step by step. |
| **Multi-agent orchestration** | The JagX Council fans out to parallel specialists, then a critic reviews, then a synthesiser merges — streamed live. |
| **Sandboxed execution** | `run_code` executes Python in a subprocess with a timeout, a memory cap, a scratch directory and no network. |
| **Safety** | JagX Guard classifies every proposed action as allow / review / block, with a human approval queue. |
| **Multimodal, local & free** | Generate images, edit them (inpaint, outpaint, background removal, upscale, style, transform), understand images (vision + OCR) and read documents (PDF, DOCX, XLSX, CSV, TXT, images) — all on your own machine. A real SD/SDXL/FLUX pipeline is used when installed; otherwise a built-in procedural renderer keeps every feature working. **No image API, no key.** |
| **Polished design** | A hand-built design system: dark/light themes, responsive layout, streaming UI, live activity timeline, a full Media Studio, accessible components. |
| **Reverse-engineered, our own style** | Behaviour and architecture of leading agents were studied and re-implemented independently as original JagX code — our own naming, our own design language. No proprietary code, weights or secrets. |

---

## Multimodal — images, vision and documents (all local, all free)

JagX v13 adds a complete multimodal surface. **Nothing in it calls a company's image
or vision API, and no API key is read anywhere.** Every capability has two tiers:

| Capability | Tier 1 — the real model (when installed) | Tier 2 — the always-available fallback |
|---|---|---|
| **Generate** | local `diffusers` SD / SDXL / FLUX checkpoint, or a local ComfyUI server | built-in procedural renderer: real PNG art seeded deterministically from the prompt |
| **Edit** | diffusion inpainting pipeline | content-aware patch fill, edge-extend outpaint, flood-fill matting, Lanczos+sharpen upscale, deterministic style filters |
| **Vision** | a local vision-language model (e.g. `llava`) on an OpenAI-compatible endpoint | OCR (tesseract) + structural analysis: dimensions, palette, brightness, contrast, edge density |
| **Documents** | — | `pypdf` / `pdftotext` (with OCR fallback for scanned PDFs), `python-docx`, `openpyxl`, CSV/TSV, plain text and code |

The fallback is not a stub. It produces real, valid artifacts so the agent loop, the
tool registry, the API, the UI and the gallery all work end to end on a laptop with no
GPU and no model. When a real model is present, the same code path silently upgrades.
`GET /api/media/capabilities` reports which tier is live, and the UI shows it honestly.

### The five agent tools

The agent can call these autonomously mid-run, exactly like `web_search` or `run_code`:

| Tool | What it does |
|---|---|
| `generate_image` | text → image; returns an `asset_id` |
| `edit_image` | inpaint · outpaint · remove-background · upscale · style · transform |
| `understand_image` | describe an image, answer a question about it, OCR any text in it |
| `read_document` | parse PDF / DOCX / XLSX / CSV / TSV / TXT / MD / JSON / code / images |
| `list_media` | browse the gallery to find an `asset_id` |

### Upgrading to a real diffusion model (still free, still no key)

```bash
# Option A — diffusers (SD / SDXL / FLUX)
pip install torch diffusers accelerate
export JAGX_IMAGE_MODEL=stabilityai/sd-turbo     # or any local checkpoint

# Option B — ComfyUI (already running locally)
export JAGX_COMFYUI_URL=http://localhost:8188
export JAGX_COMFYUI_CHECKPOINT=sd_xl_base_1.0.safetensors

# Optional extras
pip install rembg          # better background removal
ollama pull llava          # local vision model
export JAGX_VISION_MODEL=llava
apt-get install tesseract-ocr   # OCR
```

### Media API

```
GET    /api/media/capabilities      which tier is live, and the honest limits
GET    /api/media                   gallery + storage stats
POST   /api/media/generate          { prompt, width, height, style, seed }
POST   /api/media/upload            multipart file upload (image or document)
GET    /api/media/{id}              asset metadata
GET    /api/media/{id}/raw          the bytes
GET    /api/media/{id}/thumb        a thumbnail
POST   /api/media/{id}/edit         { operation, ... }
POST   /api/media/{id}/understand   { question }
POST   /api/media/{id}/read         parse as a document
DELETE /api/media/{id}              remove an asset
```

### In the UI

The **Media Studio** (sidebar) has four tabs — Generate, Edit, Read & Inspect, Gallery —
plus a live preview pane with download, edit and inspect shortcuts. The chat composer
also takes attachments: drop in an image or a document and the agent receives its
`asset_id` so it can read it with its own tools.

---

## Quick start

### 1. Backend

```bash
cd server
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env          # then edit if you like
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd web
npm install
npm run dev                          # http://localhost:5173
```

Open http://localhost:5173, create an account, and start a task.

### 3. Give it a real brain (free, local)

Install [Ollama](https://ollama.com) and pull an open-weight model with tool calling:

```bash
ollama pull qwen2.5:7b
ollama serve                         # http://localhost:11434
```

JagX auto-detects it. Nothing else to configure — no key, no account, no cost.

**Prefer vLLM or llama.cpp?** Both expose an OpenAI-compatible server; set
`VLLM_BASE_URL` or `LLAMACPP_BASE_URL` to that endpoint and JagX uses it.

### 4. Run with no model at all

```bash
python3 scripts/mock_llm.py --port 11434     # terminal 1
cd server && OLLAMA_BASE_URL=http://127.0.0.1:11434/v1 \
  OLLAMA_MODEL=mock uvicorn app.main:app --port 8000
```

The mock is a real OpenAI-compatible server: it emits tool calls and streams tokens,
so the entire product — loop, guard, council, goals, forge — is exercisable offline.

---

## Verify it works

```bash
# 156 unit + integration tests
cd server && . .venv/bin/activate && python -m pytest -q

# 52 end-to-end checks against a running server
bash scripts/e2e_check.sh
```

Both suites are green. `scripts/e2e_check.sh` exercises auth, conversations, SSE
streaming, the agent loop, the guard review path, the council, goals, memory,
self-improvement, evals, tools, the sandbox and settings over real HTTP.

---

## The nine dimensions

### 1. Agent loop
`server/app/agent.py` — plan → act → observe → critique → re-plan. Native function
calling, a step budget, forced synthesis when the budget runs out, and a self-critique
pass that can trigger one revision. Every step is emitted as an SSE event, so the UI
shows the trajectory live.

### 2. Planning
The loop opens with an explicit plan step; the plan is visible in the timeline and
revisited after each observation. Goals add a second, coarser layer of planning.

### 3. Tool / function calling
`server/app/tools.py` — native OpenAI-style tool schemas, executed through the Guard.
Tools: `web_search` (keyless, with a local fallback), `fetch_url`, `run_code`
(sandboxed), `calculator`, `remember`, `recall`, `list_memories`, `write_file`,
`read_file`, `list_files`, `goal_step`, plus the five multimodal tools
(`generate_image`, `edit_image`, `understand_image`, `read_document`, `list_media`).

### 4. Memory
`server/app/memory.py` — four layers: **working** (the live context window),
**episodic** (what happened), **semantic** (durable facts), **procedural** (lessons
learned). Hybrid retrieval blends local embeddings with lexical overlap. A
consolidation pass promotes repeated episodes into semantic facts.

### 5. Retrieval
`server/app/embeddings.py` — local, offline embeddings (a deterministic hashing
embedder by default, with an optional sentence-transformers backend). No embedding API.

### 6. Streaming
SSE end to end: token deltas, plan steps, tool calls, tool results, guard verdicts,
critique notes and the final answer — all as typed events the UI renders live.

### 7. Sandboxed execution
`server/app/sandbox.py` — subprocess isolation, wall-clock timeout, memory cap,
scratch directory, no network, output truncation.

### 8. Multi-agent orchestration
`server/app/council.py` — a coordinator decomposes the question, specialists run in
parallel, a critic reviews their work, and a synthesiser merges everything into one
answer. Streamed as a council view.

### 9. Self-critique
Two levels: the in-loop critique pass (rubric-scored, can force a revision) and the
Council's dedicated critic agent.

---

## Self-improvement (JagX Forge)

`server/app/selfimprove.py` — four loops, in increasing order of ambition:

1. **Reflection** — after each run, JagX writes a lesson to procedural memory and
   injects relevant lessons into future runs.
2. **Experience replay** — successful trajectories are stored and replayed as
   few-shot exemplars.
3. **Prompt / policy optimisation** — candidate system prompts are scored against the
   eval harness and the best variant is promoted.
4. **LoRA / QLoRA fine-tuning** — JagX exports a real training dataset (JSONL) and a
   runnable `scripts/train_lora.py` for open-weight models.

### Honest limits

- Loops 1–3 are **real and free** — they run today, offline, with no GPU.
- Loop 4 **generates a genuine, runnable pipeline**, but actually training a model
  needs a GPU and time. On a free Colab T4 or a local GPU, `train_lora.py` fine-tunes
  a small open-weight model with QLoRA. JagX does not pretend to have trained a model
  it has not trained.
- JagX does **not** modify its own weights at runtime. It improves its prompts,
  memory and exemplars — which is what is genuinely achievable for free.

---

## Repository layout

```
jagx-v13/
├── server/                     FastAPI backend
│   ├── app/
│   │   ├── agent.py            the JagX Loop
│   │   ├── council.py          multi-agent orchestration
│   │   ├── goals.py            always-on goal runner
│   │   ├── selfimprove.py      JagX Forge
│   │   ├── evals.py            eval harness
│   │   ├── memory.py           4-layer memory
│   │   ├── embeddings.py       local embeddings
│   │   ├── tools.py            tool layer
│   │   ├── sandbox.py          code sandbox
│   │   ├── guard.py            control plane
│   │   ├── media.py            multimodal engine (generate/edit/vision/docs)
│   │   ├── media_tools.py      the five multimodal agent tools
│   │   ├── media_api.py        /api/media/* router
│   │   ├── llm.py              local-first LLM client
│   │   ├── db.py               SQLite persistence
│   │   ├── security.py         PBKDF2 + JWT
│   │   ├── config.py           configuration
│   │   ├── api.py              REST + SSE
│   │   └── main.py             entrypoint
│   └── tests/                  156 tests (105 core + 51 multimodal)
├── web/                        React + TypeScript frontend
│   └── src/
│       ├── views/              Chat · Media Studio · Goals · Memory · Forge · Settings
│       ├── components/         Sidebar · Timeline · AuthView
│       ├── api.ts              typed client + SSE
│       ├── markdown.tsx        dependency-free renderer
│       ├── icons.tsx           inline SVG icon set
│       └── styles.css          the design system
├── scripts/
│   ├── mock_llm.py             OpenAI-compatible mock server
│   ├── train_lora.py           free QLoRA fine-tuning pipeline
│   └── e2e_check.sh            52 end-to-end checks
├── docs/design.md              the design document
├── .github/workflows/ci.yml    CI
├── Dockerfile · docker-compose.yml
└── Makefile
```

---

## Configuration

Everything is environment-driven; see `.env.example`. The important ones:

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model name |
| `JAGX_JWT_SECRET` | dev value | **Set this in production** |
| `JAGX_DATA_DIR` | `./data` | SQLite + workspace location |
| `JAGX_MAX_STEPS` | `12` | Agent loop step budget |
| `JAGX_SANDBOX_TIMEOUT` | `10` | Sandbox seconds |
| `JAGX_CRITIQUE_PASSES` | `2` | Self-critique passes per run |

---

## Docker

```bash
docker compose up --build          # API on :8000, web on :5173
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## A note on reverse engineering

JagX studies how leading agents behave — their interaction patterns, tool surfaces,
memory models and UX — and re-implements those capabilities as original code. It does
not copy proprietary code, weights or secrets, and it respects licenses and terms.
The result is recognisably JagX: our own naming, our own architecture, our own design
language.
