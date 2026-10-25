# JagX AI V9 — Integration Notes

Already integrated on `main`:

1. `agents_router` imported and included in `create_app()`
2. Version bumped to `9.0.0`
3. Features list includes `multi_agent_v9` and `agents_api`

## Test after deploy

```bash
curl https://YOUR_HOST/agents/capabilities

curl -X POST https://YOUR_HOST/agents/run \
  -H "Content-Type: application/json" \
  -H "x-api-key: jagx-xxxxxxxx" \
  -d '{"message":"Build a FastAPI health endpoint with logging"}'
```

Existing `/chat`, jobs, memory, tools remain unchanged.
