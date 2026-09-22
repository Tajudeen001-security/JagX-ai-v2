# JagX Backend v1.1

**Text + Coding focused AI API**  
Created by **JagX & JRILICENSE** (single developer)

Permanent API keys that **never expire**.  
Fast + accurate answers. Ready for Render free tier.

---

## What changed in v1.1

- Complete rewrite (old truncated `app.py` replaced by `main.py`)
- Faster default model: `llama-3.1-8b-instant`
- Lower temperature (0.35) for more accurate answers
- Stronger system prompt for precise coding + factual replies
- All old API keys cleared (`keys.json` is empty)
- Permanent keys that never die
- Complete endpoints: `/`, `/health`, `/create-key`, `/chat`, `/keys/me`
- Cleaner dependencies

---

## Create a permanent API key

```bash
curl -X POST https://YOUR-SERVICE.onrender.com/create-key \
  -H "Content-Type: application/json" \
  -d '{
    "owner_label": "Friend Name",
    "admin_secret": "YOUR_JAGX_ADMIN_SECRET",
    "tier": "free"
  }'
```

Response includes `never_expires: true`.

**Tiers**

| Tier          | Requests / hour |
|---------------|-----------------|
| free          | 80              |
| premium       | 350             |
| premium_plus  | 900             |
| master / admin| unlimited       |

---

## Chat

```bash
curl -X POST https://YOUR-SERVICE.onrender.com/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: jagx-xxxxxxxx" \
  -d '{"message": "Write a Python function that reverses a string"}'
```

Optional direct code run:

```json
{
  "message": "run this",
  "run_code": {"language": "python", "code": "print(2+2)"}
}
```

Optional search:

```json
{
  "message": "search",
  "search": "latest AI news"
}
```

---

## Render env vars (required)

| Key | Required |
|-----|----------|
| `JAGX_ADMIN_SECRET` | Yes (strong secret) |
| `GROQ_API_KEY` | Yes (recommended) |
| `OPENROUTER_API_KEY` | Optional fallback |
| `HF_TOKEN` | Optional fallback |
| `JAGX_PERMANENT_KEYS` | Optional comma-separated permanent keys |

---

## Keep Render alive

Free Render sleeps after ~15 min.  
Ping `/health` every 5 minutes with **UptimeRobot** (free):

1. https://uptimerobot.com
2. Add monitor → HTTP(s)
3. URL: `https://YOUR-SERVICE.onrender.com/health`
4. Interval: 5 minutes

---

## Deploy

1. Render → Web Service → connect this repo
2. Runtime: **Docker**
3. Set the env vars above
4. Deploy
5. Set UptimeRobot on `/health`
6. Create your first key with `/create-key`

---

Built by **JagX & JRILICENSE**
