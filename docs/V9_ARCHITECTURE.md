# JagX AI V9 — Multi-Agent Core Architecture

**Created by JagX & JRILICENSE**  
**Date:** 2026-10-25  
**Status:** Production — integrated into main

## Goal

Turn JagX AI into a multi-agent intelligence layer that can power the JagX & JRILICENSE family.

## Design

```
User Request → Agent Router → Specialists (Engineer / Researcher / Architect / Creator / Africa) → Synthesizer → Final answer
```

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/agents/capabilities` | no | List agents |
| POST | `/agents/run` | x-api-key | Full multi-agent run |
| POST | `/agents/coding` | x-api-key | Engineer + Architect |
| POST | `/agents/research` | x-api-key | Researcher + Africa |

## Modules

- `jagx_api/agent_router.py` — intent classification
- `jagx_api/agents.py` — specialists + synthesizer
- `jagx_api/agents_api.py` — FastAPI routes

## Identity

Every response is pure JagX AI (JagX & JRILICENSE). Never claim other labs.

**Built by JagX & JRILICENSE**
