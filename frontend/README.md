# JagX AI Frontend

Modern React chat UI for the JagX AI multi-agent backend — Grok-style dark interface with streaming, agent modes, durable jobs, and billing panel.

## Features

- 💬 **Streaming chat** — real-time SSE token streaming from `POST /chat`
- 🤖 **Agent modes** — `fast` / `agents` (multi-agent team) / `reason` (deep reasoning) + Reflect toggle
- 🗂️ **Conversations** — persistent in localStorage
- ⚙️ **Durable jobs** — send any request as a background job, track status
- 💳 **Billing panel** — tier catalog + OPay settlement details
- 🔑 **API key sign-in** — stored locally, sent as `x-api-key`
- 📊 **Health indicator** — backend version + online status

## Quick start (local)

```bash
# 1. Start the backend (from repo root)
pip install -r requirements.txt
export JAGX_ADMIN_SECRET="your-strong-secret"
export GROQ_API_KEY="..."
python main.py              # runs on :10000

# 2. Start the frontend
cd frontend
npm install
npm run dev                # runs on :5173, proxies API to :10000
```

Open http://localhost:5173, paste your `jagx-…` API key (create one with `POST /create-key`), and chat.

## Build for production

```bash
npm run build   # outputs static site to frontend/dist
```

Deploy `dist/` to any static host (Vercel, Netlify, Render static site). The backend already allows CORS for Vercel previews and `localhost:5173`. For a custom domain, set `JAGX_CORS_ORIGINS=https://yourdomain.com` on the backend, or set `JAGX_CORS_ALLOW_ALL=true` during testing.

In the app's settings (⚙) you can point the frontend at any backend URL (e.g. `https://jagx-ai-v2.onrender.com`).

## Stack

- React 18 + Vite
- No other dependencies — plain fetch + SSE parsing, custom CSS


> CI: a build check runs on every change under `frontend/` (`.github/workflows/frontend-build.yml`).