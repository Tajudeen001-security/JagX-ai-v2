# JagX AI Backend v10.2

**Multi-agent production API — built by JagX & JRILICENSE**

FastAPI backend with parallel specialists, tools, durable jobs, priority queue, manual OPay billing, streaming, memory, and Africa-first offline packs.

**Repo:** https://github.com/Tajudeen001-security/JagX-ai-v2

---

## Features (v10.2)

| Area | Capability |
|------|------------|
| **Chat** | Multi-agent default (`mode=agents|reason|fast`), SSE streaming |
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

## Frontend

A modern React chat UI lives in [`frontend/`](frontend/README.md) — streaming chat, agent modes (fast / agents / reason), Reflect toggle, durable jobs panel, billing panel, API-key sign-in.

```bash
# terminal 1 — backend
python main.py                # :10000

# terminal 2 — frontend
cd frontend
npm install
npm run dev                   # :5173 (proxies API calls to :10000)
```

Open http://localhost:5173, paste your `jagx-…` API key and chat. For production, `npm run build` and deploy `frontend/dist` to any static host (Vercel/Netlify). Backend CORS already allows `localhost:5173` and Vercel preview URLs; for a custom domain set `JAGX_CORS_ORIGINS=https://yourdomain.com`.

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
| `JAGX_CORS_ORIGINS` | Comma-separated extra CORS origins |
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

## Secure GitHub execution

JagX AI now includes a backend-owned GitHub App OAuth flow and user-scoped repository API. Browser clients do **not** receive or store GitHub access tokens. Tokens are encrypted at rest using Fernet and associated with the authenticated Supabase user. Repository browsing and writes are checked against repositories where the JagX AI GitHub App is installed. File updates use an expected SHA to reject stale writes. OAuth uses short-lived, single-use state plus PKCE; disconnect attempts server-side authorization revocation before deleting the local token.

### Required setup before enabling it

1. Create a **GitHub App** in GitHub Developer Settings. Set the user authorization callback URL to `https://jagx-ai-v2.onrender.com/github/callback` (or the actual deployed backend URL). Because JagX starts the OAuth flow itself and validates its own state/PKCE, leave **Request user authorization (OAuth) during installation** off.
2. Grant only **Repository permissions → Contents: Read and write** and **Metadata: Read**. Do not grant Actions, administration, secrets, or organization permissions unless a separate reviewed feature requires them.
3. Enable user authorization for the app and set its homepage to `https://jagxai.name.ng`. Copy the App slug, client ID, and client secret into the backend hosting environment.
4. Set the variables listed in `.env.example` in the backend host. Generate a unique Fernet key for `GITHUB_TOKEN_ENCRYPTION_KEY`; never commit it. Use a persistent disk for `GITHUB_CONNECTION_DB` or move token/state storage to a managed encrypted database before relying on it across restarts.
5. Install the GitHub App only on the repositories the user wants JagX AI to access. Users must sign in to JagX AI before connecting GitHub.
6. Verify OAuth callback, repository allowlisting, read access, stale-file conflict rejection, disconnect, and a test commit on a non-production branch before enabling production use.

### API

- `GET /github/connect` — create short-lived state and return the GitHub authorization URL (requires Supabase bearer token).
- `GET /github/callback` — validate one-use state, exchange OAuth code, encrypt and store the user token server-side.
- `GET /github/status`, `GET /github/repos` — check the user's connection and list installed repositories.
- `GET /github/repo/{owner}/{repo}/contents` — browse/read installed repository files.
- `PUT /github/repo/{owner}/{repo}/contents/{path}` — write a file only after validating repository installation and expected SHA.
- `DELETE /github/disconnect` — delete the current user's stored connection.

The GitHub integration is **not operational until the GitHub App and backend environment variables are configured and the deployed end-to-end checks pass**. The SQLite token/state database must use persistent storage; otherwise a host restart can lose connections. Production hardening should add token revocation on disconnect, rate limits, audit records, and a background agent job that plans/tests changes before any confirmed commit.
