# JagX AI Backend v8.0

**Text + coding focused production API**  
Created by **JagX & JRILICENSE**

Clean modular FastAPI service — no remote `exec()`, permanent API keys, multi-provider LLM fallback, local knowledge brain, memory, calc, streaming.

**Repo:** https://github.com/Tajudeen001-security/JagX-ai-v2

---

## What changed in v8.0

| Before (v7) | After (v8) |
|-------------|------------|
| `exec()` of remote source at boot | Self-contained `jagx_api/` package |
| Single monolith | Modules: auth, providers, chat, knowledge, tools, memory |
| Boot fails if GitHub unreachable | Always boots; cloud optional |
| Limited structure | `/docs` OpenAPI, streaming, calc, memory |

---

## Quick start

```bash
pip install -r requirements.txt
export JAGX_ADMIN_SECRET="your-strong-secret"
export GROQ_API_KEY="..."   # recommended
python main.py              # :10000
```

### Create key

```bash
curl -X POST http://localhost:10000/create-key \
  -H "Content-Type: application/json" \
  -d '{"owner_label":"me","admin_secret":"your-strong-secret","tier":"free"}'
```

### Chat

```bash
curl -X POST http://localhost:10000/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: jagx-xxxxxxxx" \
  -d '{"message":"Write a Python function that reverses a string"}'
```

### Stream

```bash
curl -N -X POST http://localhost:10000/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: jagx-xxxxxxxx" \
  -d '{"message":"Explain recursion","stream":true}'
```

---

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | no | Service info |
| GET | `/health` | no | Health + providers |
| GET | `/docs` | no | OpenAPI UI |
| POST | `/create-key` | admin_secret | Issue permanent key |
| GET | `/keys/me` | x-api-key | Key metadata |
| POST | `/chat` | x-api-key | Chat (`stream` optional) |
| POST | `/memory` | x-api-key | remember / recall / forget |
| POST | `/calc` | x-api-key | Safe arithmetic |

**Provider order:** Groq → OpenRouter → Hugging Face → local knowledge brain.

## Env

| Variable | Description |
|----------|-------------|
| `JAGX_ADMIN_SECRET` | Required for `/create-key` |
| `GROQ_API_KEY` | Fast primary LLM |
| `OPENROUTER_API_KEY` | Fallback |
| `HF_TOKEN` | Fallback |
| `JAGX_PERMANENT_KEYS` | Comma-separated always-valid keys |
| `PORT` | Default 10000 |

## Deploy (Render)

Docker web service → set env → health check `/health` → create key.

Built by **JagX & JRILICENSE**
