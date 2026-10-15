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

**Provider order:** Groq → NVIDIA NIM → OpenRouter → Hugging Face → local GGUF model → local knowledge brain.

## Env

| Variable | Description |
|----------|-------------|
| `JAGX_ADMIN_SECRET` | Required for `/create-key` |
| `GROQ_API_KEY` | Fast primary LLM |
| `NVIDIA_API_KEY` | Optional NVIDIA NIM fallback |
| `OPENROUTER_API_KEY` | Fallback |
| `HF_TOKEN` | Fallback |
| `JAGX_PERMANENT_KEYS` | Comma-separated always-valid keys |
| `JAGX_OFFLINE_MODE` | Set `true` to skip cloud providers and use only local inference/knowledge |\n| `JAGX_DATA_ENCRYPTION_KEY` | Optional Fernet key to encrypt new conversation titles/messages and the memory JSON file at rest; generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` and store only in host secrets |
| `JAGX_GGUF_MODEL` | Optional path to local GGUF model weights |
| `PORT` | Default 10000 |

## Deploy (Render)

Docker web service → set env → health check `/health` → create key.

Built by **JagX & JRILICENSE**


## JagX AI Autonomous Reasoning and Interaction Engine

The backend now has a staged foundation for multilingual conversation, bounded tool use, persistent history, consent-based memory, voice, offline local inference, evaluations, and engineering decision support.

### Provider fallback

Configured providers are tried in order: **Groq → NVIDIA NIM → OpenRouter → Hugging Face**. Streaming uses the same configured provider order and falls back to one-shot generation if no stream yields text. The local GGUF model and local knowledge base are used when cloud responses are unavailable. Provider status only reports whether a key is configured; it does not prove the key works or that a provider is available.

All providers can be configured with free-tier credentials where offered. There is no guarantee of unlimited free usage or uptime. Never commit API keys.

### Persistent history and personal memory

- `POST /conversations`, `GET /conversations`, `GET /conversations/{id}`, and `DELETE /conversations/{id}` manage history.
- Send `conversation_id` to `POST /chat` to load and persist a conversation.
- History and memory are currently scoped to a server-derived hash of the API key. This prevents a caller from spoofing the `user_id` field, but it is not a replacement for verified user-account/JWT authentication. Do not share API keys between users.\n- Set `JAGX_DATA_ENCRYPTION_KEY` to a generated Fernet key to encrypt new conversation titles and message contents at rest. Keep the key in the hosting provider secret manager and back it up securely: losing it makes encrypted records unreadable. Existing plaintext records remain readable and are not automatically migrated. Without this setting, conversation text remains plaintext in SQLite; file permissions alone are not encryption.
- `POST /memory` with `action=remember` requires `consent: true`. Recall and forget operate only on the current API-key namespace. When `JAGX_DATA_ENCRYPTION_KEY` is configured, the memory JSON file is also Fernet-encrypted at rest; legacy plaintext memory remains readable and is encrypted on its next successful write. If the key is missing or incorrect for an encrypted memory file, memory operations fail rather than silently erasing or replacing the data. Without the key, memory remains plaintext JSON.
- Training-data logging is disabled by default (`JAGX_TRAINING_DATA_ENABLED=false`) and requires both operator opt-in and explicit per-turn `training_consent: true` (or the voice form field). Enable it only after suitable user notice/consent, data minimisation, retention, and deletion controls are in place.
- Configure `JAGX_CONVERSATION_DB` and `JAGX_MEMORY_FILE` on persistent storage for a hosted service. Ephemeral hosting disks can lose data on restart or redeploy.

### Complete voice pipeline (optional)

Install optional dependencies with `pip install -r requirements-speech.txt`.

- `POST /speech/transcribe`: multipart audio upload, local Whisper transcription, language detection.
- `POST /speech/speak`: text-to-speech; returns MPEG audio.
- `POST /voice/turn`: audio → transcription → model reasoning → spoken response. It returns the response audio as base64.

Voice endpoints are optional because Whisper dependencies and model files can be large. Whisper downloads a local model on first use; TTS uses an external internet service and is not guaranteed offline. Default Whisper model is `tiny`; use `JAGX_WHISPER_MODEL` to choose a larger model. A language detection label is not proof of accurate transcription.

### Small local model / offline mode

The local inference runtime is optional so the normal Render deployment stays small. Install it on a machine with compatible build support:

```bash
pip install -r requirements-local.txt
```

Then download a compatible, licensed GGUF model file separately and set:

```bash
export JAGX_GGUF_MODEL="/absolute/path/to/model.gguf"
export JAGX_LOCAL_N_CTX=2048
export JAGX_LOCAL_THREADS=4
export JAGX_OFFLINE_MODE=true
```

A small candidate to evaluate on limited hardware is **Qwen2.5-0.5B-Instruct in a Q4 GGUF quantization**. Small models can run with less memory and no inference API key, but their reasoning, factuality, and language quality are significantly weaker than larger models. Model binaries are not committed to this repository. Verify the model licence and use a reputable model repository. `GET /offline/status` reports whether the runtime and configured file are present.

Set `JAGX_OFFLINE_MODE=true` to skip all cloud provider calls and use only the local GGUF model/local knowledge. If no local model is configured, JagX AI returns a clear offline-unavailable response instead of contacting cloud APIs. Set it back to `false` to restore cloud fallback.\n\nLocal inference is most useful on a device or server with persistent storage and enough RAM. The free Render web service is not a dependable offline-model host; it may not retain a model file and may lack enough memory/CPU.

### Knowledge ingestion and evaluations

- `POST /knowledge/ingest` requires a valid API key and the configured `x-admin-secret`. Imported knowledge is local-file based; configure persistent storage.
- `GET /engineering/capabilities`
- `POST /engineering/endurance`
- `POST /engineering/fmea-template`

Run regression tests with `python -m pytest -q`. A small smoke-evaluation suite can be run with `python -m evals.run_eval` after adding `evals/cases.json` and configuring a model. Evaluation outputs are measurements, not proof of general intelligence or safety certification.

### Engineering safety boundary

JagX AI can assist with diagnostics, maintenance, planning, hazard analysis, simulation, and documentation for vehicles, drones, aircraft, robotics, and industrial systems. Its estimates and generated checklists are not certifications and must not directly control safety-critical actuators. Qualified engineers, independent safety controls, hardware interlocks, validated simulations, and fail-safe designs remain mandatory.

Connected-account routes currently report OAuth integrations as not configured. Real Google/Microsoft/GitHub account linking still requires verified user sessions, registered OAuth clients, callbacks, consent scopes, and secure token storage.


### Image generation API

- `POST /images/generate` accepts `prompt`, optional `width` and `height` (256–1024), and requires `x-api-key`.
- The backend tries Hugging Face hosted image inference when `HF_TOKEN` is configured, then Pollinations when `POLLINATIONS_API_KEY` is configured. Both are remote hosted services; Pollinations documents API-key authentication. The response contains base64 image data and MIME type for the client to render.
- This uses hosted inference; it does not train a model or run image generation on the Render instance. Free access/credits, model availability, quotas, latency, and service terms can change. Configure `POLLINATIONS_API_KEY` in Render if you want that fallback; don't put it in source code. A configured token is not a guarantee that inference is enabled for the selected model.
- Images are size-limited and validated before returning. Do not treat base64 responses as permanent storage; configure object storage if generated images must persist.


### Real-time bidirectional chat

The backend supports both HTTP streaming and WebSocket streaming:

- `POST /chat` with `{"message":"Hello","stream":true,"stream_format":"sse"}` streams Server-Sent Events.
- `/ws/chat` is a WebSocket endpoint. Authenticate with the `x-api-key` handshake header. After connecting, the server sends a `ready` event. Send JSON frames such as `{"message":"Hello"}` or `{"message":"Continue","conversation_id":"<id>"}`. The server streams `start`, `token`, and `done` JSON events, or a generic `error` event.
- WebSocket chat supports a persistent `conversation_id` and consented memory context. Keep API keys out of browser URLs and logs.
- This is real-time streamed text chat, not continuous live microphone audio. Continuous voice requires a separate streaming speech-recognition/speech-synthesis integration. Actual response speed and availability still depend on configured model providers and host limits.


### Durable background jobs

Authenticated job endpoints let clients submit a request and disconnect while the backend processes it:

- `POST /jobs` JSON: `{"message":"Prepare a detailed implementation plan","conversation_id":null}` returns HTTP 202 with a job ID.
- `GET /jobs/{job_id}` returns `queued`, `running`, `succeeded`, `failed`, or `cancelled` and the result when complete.
- `GET /jobs` lists the authenticated API key owner's recent jobs.
- `POST /jobs/{job_id}/cancel` requests cancellation. A running model call may continue until it returns, but its result will not overwrite a cancelled status.

Jobs are stored in SQLite and running jobs are re-queued on process startup. Set `JAGX_JOBS_DB` to a path on a persistent disk/volume to preserve jobs across Render restarts and deploys. The Render free filesystem may be ephemeral, and a free web service may sleep; therefore this is restart-recoverable queue logic, not a guarantee of 24/7 execution. For dependable continuous coding while the user is away, deploy a continuously running worker plus persistent storage and explicitly configured repository access. This queue currently performs queued AI chat work; it does not autonomously edit or commit arbitrary GitHub repositories.

### Operations and reliability

- `/health` is a liveness check. Provider configuration status only means a credential is present; it does not prove that a provider is reachable, accepts the key, has quota, or exposes the selected model.
- Provider calls use bounded timeouts and fallback ordering. Retries and fallback can increase total latency; no free provider guarantees uninterrupted service.
- Run the GitHub Actions test suite before treating new endpoints as production-ready.


### Resumable coding jobs, provider resilience, voice streaming and video

- `POST /coding/jobs` accepts a software task and returns a durable job ID. The worker checkpoints planning progress; use `GET /jobs/{job_id}` after disconnecting to retrieve status and output.
- Coding work is deliberately in planning/safe-test mode by default. To enable test execution, configure `JAGX_CODING_WORKSPACE` to a trusted Git checkout containing the project. The only test command currently allowed is `python -m pytest -q`; task text is never executed as a shell command. Repository edits, commits, pushes, and pull requests are not enabled by this worker yet.
- Job retries use exponential delay and are capped at three attempts. Job status/checkpoints live in SQLite; configure `JAGX_JOBS_DB` to a path on persistent storage to survive redeploys. Running jobs are recovered from their last saved checkpoint where the task supports it.
- `GET /providers/health` reports network reachability, breaker state and recent failure details. Reachability does not validate credentials, model access, or quota. Provider breakers open after repeated failures and cool down before retries.
- `POST /speech/speak/stream` streams MP3 chunks as the speech service produces them. It reduces buffering, but speech generation still relies on an external service and does not guarantee low latency or uninterrupted availability.
- `POST /video/generate` accepts `prompt`, optional `duration` (2–8 seconds), `model`, and `aspect_ratio`. The hosted Pollinations video API returns MP4 data, bounded to 20 MB. Current default model is `google/veo-3.1-fast`; alternatives in this backend include `bytedance/seedance-1-pro-fast`, `alibaba/wan-2.2-fast`, and `prunaai/p-video`. Access, credits and pricing are controlled by the provider, not JagX. Set `POLLINATIONS_API_KEY` in Render; never put it in source code or frontend code.
- Video generation can take several minutes and currently uses a synchronous provider call. For production, submit video as a background job and store output in object storage instead of placing large base64 video data in SQLite.
- Free Render web services have ephemeral filesystems, can sleep, and do not support persistent disks. A continuous worker with persistent local storage requires a paid Render service with a disk, or a separate worker plus a shared persistent database/queue. Do not promise unattended coding on the free web plan.


### Autonomous systems and vehicle safety (advisory/simulation only)

- `POST /autonomy/assess` evaluates whether a simulated/advisory scenario has required safety signals. It fails closed when sensor health, communications, operator authorization or independent emergency stop are missing.
- `POST /autonomy/driving/stopping-distance` calculates an idealized educational stopping-distance estimate for self-driving-car research. It is not a brake command or a validated vehicle-dynamics model.
- `POST /autonomy/simulation/risk-check` screens simulated scenarios for low sensor confidence, high latency, missing human override, and missing/near obstacle distance. Its thresholds are illustrative defaults, not certified safety limits.
- `GET /autonomy/checklists/self_driving_car`, `/autonomy/checklists/drone`, `/autonomy/checklists/aviation`, and `/autonomy/checklists/industrial_robot` return safety-oriented review checklists.
- These routes do not connect to real vehicles or actuators. Real-world autonomy requires simulation, controlled test environments, certified control systems, redundant sensing, geofencing, independent emergency stops, regulatory compliance and qualified human oversight. Do not use language-model output as a real-time flight/driving controller or weapon-targeting system.
