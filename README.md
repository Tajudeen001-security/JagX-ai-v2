# JagX AI Backend v8.0

**Text + coding focused production API**  
Created by **JagX & JRILICENSE**

Clean modular FastAPI service — provider fallback, local knowledge ingestion, API-key-scoped persistent history and memory, optional speech, and bounded engineering decision-support utilities.

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
| POST | `/memory` | x-api-key | remember / recall / forget (API-key isolated; file storage is not encrypted) |\n| POST/GET | `/conversations` | x-api-key | Create/list persistent conversations |\n| GET/DELETE | `/conversations/{id}` | x-api-key | Read/delete only the caller’s conversations |\n| GET | `/connected-accounts/providers` | x-api-key | Show OAuth readiness; providers are not configured yet |\n| GET | `/connected-accounts` | x-api-key | List connected accounts (empty until OAuth is configured) |\n| POST | `/knowledge/ingest` | x-api-key + x-admin-secret | Validate and replace local knowledge pack |\n| GET | `/engineering/capabilities` | x-api-key | Engineering support areas and safety boundary |\n| POST | `/engineering/endurance` | x-api-key | Idealized battery/endurance estimate |\n| POST | `/engineering/fmea-template` | x-api-key | Create an FMEA worksheet template |
| POST | `/calc` | x-api-key | Safe arithmetic |

**Provider order:** Groq → NVIDIA NIM → OpenRouter → Hugging Face → local knowledge brain.

## Multilingual and voice support

JagX AI is instructed to reply in the user's language, handle code-switching where the underlying model supports it, and preserve Unicode letters in local knowledge matching. Language coverage depends on the selected cloud model; a local keyword knowledge base alone is not a general multilingual reasoning model.

Speech is optional so the base API remains lightweight and can run without paid speech API keys:

- Install speech extras: `pip install -r requirements-speech.txt`.
- `POST /speech/transcribe`: upload an audio file as multipart field `audio`; local faster-whisper detects many languages. First use downloads the selected model, so the host needs internet access, disk space, and enough RAM/CPU.
- `POST /speech/speak`: JSON body `{\"text\": \"Hello from JagX AI\", \"voice\": \"en-NG-AbeoNeural\"}`; returns MP3 audio. Edge TTS requires internet and uses an external service; it is keyless, but not offline or guaranteed.
- Set `JAGX_WHISPER_MODEL=tiny` for lower resource use, or `small` for a stronger local recognition model. `JAGX_TTS_VOICE` selects the default TTS voice.

These endpoints provide speech building blocks; a voice conversation experience chains transcription → chat → synthesis. The frontend integration is separate.

## Safety-critical engineering

JagX AI can assist with design reviews, simulation plans, maintenance, hazard analysis, and operator decision support for vehicles, drones, aircraft, and industrial systems. Do not use generated answers as direct control signals. Real systems require qualified engineering review, independent safety interlocks, fail-safe behavior, and testing before operation.

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


## Backend milestones and honest limitations

- Conversation history is stored in SQLite and scoped to a hash of the API key. This is not yet a real user-account identity system; shared API keys share history. Connect a verified JWT/session subject before exposing this as multi-user consumer account history.
- Long-term memory is persistent and API-key scoped, with restrictive file permissions where supported. The JSON file is **not encrypted at rest**; use encrypted managed storage before storing sensitive information.
- Knowledge ingestion requires both a valid API key and the configured admin secret. The local knowledge pack may be ephemeral on some hosting providers; mount persistent storage or migrate to a database for production.
- Connected-account endpoints intentionally report OAuth as not configured. No account is claimed connected until provider client registration, verified identity, callback URLs, consent scopes, and secure token storage exist.
- Engineering endpoints are decision-support only. Endurance calculations are simplified estimates; they are not flight-time guarantees or a safety certification.
- A CI workflow now runs pytest on branch pushes and pull requests. A workflow file is not proof that the suite passed; inspect the actual run result before merging.
- Fine-tuning remains an evaluated future option. No training has occurred, and no free GPU quota or provider availability is guaranteed.
