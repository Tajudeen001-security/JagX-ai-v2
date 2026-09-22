# Fix models on Render (required)

Your Groq key is valid length, but model IDs were wrong/blocked.

Set these **exactly** in Render Environment, then Save:

```
GROQ_MODEL=openai/gpt-oss-20b
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
JAGX_MAX_OUTPUT_TOKENS=800
JAGX_LLM_TIMEOUT=12
```

If `openai/gpt-oss-20b` fails, try in order:
1. `llama-3.3-70b-versatile`
2. `openai/gpt-oss-120b`

Check live: GET https://jagx-ai-v2.onrender.com/debug/llm
