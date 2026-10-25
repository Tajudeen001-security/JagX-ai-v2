# JagX AI Backend v10.2

**Multi-agent production API — built by JagX & JRILICENSE**

FastAPI backend with parallel specialists, tools, durable jobs, priority queue, manual OPay billing, streaming, memory, and Africa-first offline packs.

**Repo:** https://github.com/Tajudeen001-security/JagX-ai-v2

---

## Features (v10.2)

| Area | Capability |
|------|------------|
| **Chat** | Multi-agent default (`mode=agents\|reason\|fast`), SSE streaming |
| **Agents** | Engineer, Researcher, Architect, Creator, Africa, Media |
| **Tools** | web_search, fetch_url, browse_research, run_code, calc, PDF/CV, image |
| **Jobs** | Durable SQLite queue, tier priority, reason/reflect, worker status |
| **Billing** | Manual OPay payment → admin confirm → tier upgrade |
| **Memory** | Consented recall, style preferences |
| **Training** | LoRA flywheel export (`/training/stats`, `/training/export`) |
| **Evals** | Router, identity, artifacts, tools, priority in CI |

---

## Quick start

```bash
pip install -r requirements.txt
export JAGX_ADMIN_SECRET="your-strong-secret"
export GROQ_API_KEY="..."   # recommended
python main.py              # :10000
```

### Create API key

```bash
curl -X POST http://localhost:10000/create-key \
  -H "Content-Type: application/json" \
  -d '{"owner_label":"me","admin_secret":"your-strong-secret","tier":"free"}'
```

### Chat (multi-agent)

```bash
curl -X POST http://localhost:10000/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: jagx-xxxxxxxx" \
  -d '{"message":"Design a Flutter auth flow","mode":"agents"}'
```

### Stream

```bash
curl -N -X POST http://localhost:10000/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: jagx-xxxxxxxx" \
  -d '{"message":"Explain recursion","stream":true,"mode":"agents"}'
```

---

## Core endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | no | Service info + features |
| GET | `/health` | no | Health + providers |
| GET | `/docs` | no | OpenAPI UI |
| POST | `/create-key` | admin_secret | Issue permanent key |
| GET | `/keys/me` | x-api-key | Key + tier |
| POST | `/chat` | x-api-key | Chat (`mode`, `stream`, `reflect`) |
| POST | `/jobs` | x-api-key | Durable multi-agent job |
| GET | `/jobs/worker/status` | x-api-key | Queue depth / priority backlog |
| POST | `/agents/run` | x-api-key | Explicit multi-agent run |
| POST | `/tool` | x-api-key | Direct tool call |
| GET | `/billing/catalog` | no | Prices + OPay account |
| GET | `/billing/payment-note` | no | Copy-paste transfer note |
| POST | `/billing/upgrade-request` | x-api-key | Submit payment ref |
| GET | `/billing/admin/panel` | browser | Admin confirm UI |

**Provider order:** Groq → NVIDIA → OpenRouter → Hugging Face → local GGUF → local knowledge.

---

## Billing (manual OPay)

**Settlement account**

| Field | Value |
|-------|--------|
| Bank | OPay |
| Account | `9160654415` |
| Name | Gbadamosi Tajudeen Olajide |

| Tier | Price (NGN) | Hourly limit | Job priority |
|------|-------------|--------------|--------------|
| free | 0 | 100 | 10 |
| premium | 15,000 | 400 | 50 |
| premium_plus | 35,000 | 1,000 | 80 |

1. Transfer the amount to the OPay account  
2. `GET /billing/payment-note?tier=premium` for narration text  
3. `POST /billing/upgrade-request` with `payment_ref`  
4. Admin opens `/billing/admin/panel` and confirms  

**Authorized admins:** `gbadamositajudeenwan@gmail.com`, `gbadamositajudeeneh@gmail.com`  
(Confirm requires `JAGX_ADMIN_SECRET` + one of those emails.)

---

## Environment

| Variable | Description |
|----------|-------------|
| `JAGX_ADMIN_SECRET` | Required for create-key + billing confirm |
| `GROQ_API_KEY` | Primary LLM |
| `NVIDIA_API_KEY` / `OPENROUTER_API_KEY` / `HF_TOKEN` | Fallbacks |
| `JAGX_PERMANENT_KEYS` | Comma-separated always-valid keys |
| `JAGX_OFFLINE_MODE` | `true` = local only |
| `JAGX_TRAINING_DATA_ENABLED` | Consented SFT logging |
| `JAGX_JOBS_DB` | Persistent path for durable jobs |
| `JAGX_DATA_ENCRYPTION_KEY` | Optional Fernet key for memory/conversations |
| `PORT` | Default 10000 |

---

## Tests & evals

```bash
python -m pytest -q
python -m evals.run_eval   # structural + optional live chat
```

CI runs pytest + offline structural evals (router, identity, artifacts, preferences, tools, priority).

---

## Deploy notes

- Prefer a host with **persistent disk** for jobs DB, keys, and billing file.
- Free ephemeral hosts lose queue/billing state on redeploy.
- Never commit API keys or `JAGX_ADMIN_SECRET`.

---

## Safety

JagX AI assists with software and planning. It is **not** a certification authority and must not directly control safety-critical vehicles or actuators.

---

Built by **JagX & JRILICENSE**
