# JagX v12 — Design Document

**An autonomous agent in the class of Grok bot, Dots, Muse and Claude Duet — built from open weights, with no third-party company API key required.**

Version 12.0.0 · Status: implemented · License: MIT

---

## 0. Executive summary

JagX v12 is a full-stack autonomous agent. It plans, calls tools, browses, writes and runs
code, remembers across sessions, pursues long-running goals in the background, debates hard
questions with a council of specialist sub-agents, critiques its own output, and improves its
own prompts and policies from its own run history.

It is built on four pillars that we reverse-engineered from the four leading products of 2026
and then re-implemented in our own architecture and naming:

| Pillar | What it is | Closest studied analogue |
|---|---|---|
| **JagX Loop** | plan → act → observe → reflect → answer, with a step budget and forced synthesis | Grok bot's agent-with-a-computer loop |
| **JagX Council** | coordinator + parallel specialists + critic + synthesis | Grok 4.2's four-agent debate; Muse's subagent swarms |
| **JagX Goals** | the *goal*, not the conversation, is the unit of work; always-on runner with checkpoints | Dots' goal-owned always-on runtime |
| **JagX Guard** | allow / review / block policy engine with a hard boundary on irreversible actions | Claude Duet's approval boundary; Grok bot's reviewer agent |

On top of those sit **JagX Memory** (working / episodic / semantic / procedural with
consolidation), **JagX Forge** (the self-improvement engine), **local-first inference**
(Ollama / vLLM / llama.cpp / llama-server, no key needed) and a **local sandbox** for code
execution.

**The honest headline:** JagX v12 does not train a frontier model from scratch — nobody can do
that for free. What it *does* do, for free, is everything that actually makes an agent improve:
it reflects on its own runs, replays its own experience, optimises its own prompts and policies
against a measurable eval harness, and exports a clean LoRA/QLoRA training set from its own
trajectories so that a small open-weight model can be fine-tuned on a single consumer GPU.
Section 5 states exactly where the free path ends and what it costs to go further.

---

## 1. What we studied

We studied the observable behaviour and published architecture of four systems. Reverse
engineering here means *understanding behaviour and interfaces and re-implementing them
independently* — never copying proprietary code, weights or secrets.

### 1.1 Grok bot (xAI)

- **Shape:** "an agent with a computer." A persistent desktop it can drive — apps, browser,
  code — interacted with like texting a colleague.
- **Loop:** an *outer loop* agent gathers context (Slack, Notion, GitHub, docs) and hands a
  clean prompt to an *inner loop* executor that does the work. The explicit design goal is
  keeping dirty context out of the executor.
- **Control:** bots are kept in check by permissions, a **separate review agent** that can
  allow / block / escalate, and allow/block lists. Rules are written in natural language, not
  JSON. Work happens in an isolated environment.
- **Multi-agent:** Grok 4.2 runs a four-agent system — coordinator, research, logic/math/code,
  and a contrarian checker — that debate in parallel and cross-verify before surfacing an
  answer. Heavy mode scales the agent count up for hard queries.
- **Takeaway for us:** the outer/inner split, the reviewer agent, and natural-language policy
  are the three ideas worth having.

### 1.2 Dots (OpenAI)

- **Shape:** the unit of work is a **goal**, not a conversation. A dot has an identity and a
  name; it is an executing subject, not a chat partner.
- **Runtime:** each dot gets its own cloud computer and browser. The runtime is independent of
  the user's device — the work continues when the laptop is off.
- **Continuity:** a goal is decomposed across multiple sessions and driven forward over time.
- **Control:** user-set permission levels, an auto-review step that checks actions against
  instructions and custom rules, and a hard rule that certain sensitive tasks (e.g. changing a
  password) always stay with the human.
- **Takeaway for us:** goal-as-unit, device-independent runtime, and the auto-review gate.

### 1.3 Muse (Meta)

- **Shape:** a personal autonomous agent for multi-step real-world goals (travel, scheduling,
  checkout).
- **Architecture:** two distinct layers — **orchestration** (subagent swarms) and **runtime**
  (a per-user isolated VM, the "Muse Secure VM") that walls off sensitive operations from
  general model computation.
- **Autonomy:** L4 — works in the background; sensitive actions require approval.
- **Takeaway for us:** separating orchestration from runtime, and isolating the runtime.

### 1.4 Claude Duet (Anthropic)

- **Shape:** a *proactive* agent app. It comes to you: a morning brief, a check on connected
  accounts every few hours, tasks that keep running on a hosted machine while your phone is off.
- **Control:** a hard, stated boundary — it shows you any **message, payment, signature or
  deletion** before acting, and refuses a small set of actions outright.
- **Cadence:** checks every few hours rather than constantly — a deliberate choice to be
  useful without being noisy.
- **Takeaway for us:** the proactive brief, the four-action hard boundary, and the cadence.

### 1.5 Cross-cutting patterns

Every one of the four converged on the same five things:

1. **A trajectory, not a reply.** One model call is a chatbot; a loop of
   propose → policy → execute → observe → update is an agent.
2. **A persistent runtime.** The agent's work must not die with the user's session.
3. **A memory that outlives the context window.** Context is a buffer, not memory.
4. **A control plane that assumes the cognition plane will be wrong.** Policy, approvals,
   isolation, audit.
5. **Delegation.** One agent that does everything carries dirty context; specialists don't.

---

## 2. The nine dimensions, compared

| Dimension | Grok bot | Dots | Muse | Claude Duet | **JagX v12** |
|---|---|---|---|---|---|
| **Agent loop** | outer/inner split, agent-with-a-computer | goal-driven, multi-session | orchestration + runtime | proactive loop, periodic checks | **JagX Loop**: plan → act → observe → reflect → answer, step budget + forced synthesis |
| **Planning** | outer loop assembles context, hands clean prompt | goal decomposed across sessions | swarm plan | brief + task intake | **Planner** emits 2–5 steps for multi-step asks; **Council** re-plans on critique failure |
| **Tool / function calling** | apps, browser, code, connectors | 4,000+ connectors | connectors | 6,000+ connectors | **Native OpenAI-style function calling**; 14 built-in tools; MCP-compatible surface |
| **Memory** | workspace context | persistent context | per-user VM state | connected-account state | **4 layers**: working / episodic / semantic / procedural, with consolidation + decay |
| **Retrieval** | connector search | connector search | connector search | connector search | **Local hybrid retrieval**: hashed-embedding cosine + BM25-style keyword, no external service |
| **Streaming** | live activity | live activity | live activity | live activity | **Token-level SSE** + typed events (`plan`, `tool_call`, `tool_result`, `council`, `critique`, `token`, `answer`) |
| **Sandboxed execution** | isolated environment | cloud computer | secure VM | cloud VM | **Local subprocess sandbox** (rlimits, timeout, no network by default) + optional Piston |
| **Multi-agent orchestration** | 4-agent debate, heavy mode scales up | subagent swarms | subagent swarms | single proactive agent | **JagX Council**: Atlas (coordinator) + Scout / Forge / Sage (parallel specialists) + Verity (critic) + synthesis |
| **Self-critique** | separate review agent (allow/block/escalate) | auto-review against rules | approval gates | approval boundary | **Verity critic** on the answer + **JagX Guard** on every action + **JagX Forge** reflection on every run |

---

## 3. JagX v12 architecture

### 3.1 Component map

```
┌──────────────────────────────────────────────────────────────────────┐
│  web/  React + TypeScript                                            │
│  Chat · Council · Goals · Memory · Forge · Settings                  │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ SSE + REST  (/api)
┌───────────────────────────▼──────────────────────────────────────────┐
│  server/app/api.py        FastAPI surface                            │
├──────────────────────────────────────────────────────────────────────┤
│  agent.py      JagX Loop        council.py   JagX Council            │
│  goals.py      JagX Goals       guard.py     JagX Guard              │
│  selfimprove.py JagX Forge      evals.py     Eval harness            │
├──────────────────────────────────────────────────────────────────────┤
│  memory.py  4-layer memory + consolidation                           │
│  embeddings.py  local embedder (hashed, optional sentence-transformers)│
│  tools.py   14 tools            sandbox.py  local code sandbox        │
│  llm.py     local-first provider chain (OpenAI-compatible)            │
├──────────────────────────────────────────────────────────────────────┤
│  db.py  SQLite (WAL)   security.py  PBKDF2 + JWT   config.py  env     │
└──────────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴────────────────────┐
        │  Ollama / vLLM / llama.cpp / llama-server  (default, no key)  │
        │  any OpenAI-compatible endpoint            (optional, pluggable)│
        └─────────────────────────────────────────┘
```

### 3.2 The JagX Loop

```
user message
   │
   ├─ recall memory (working + episodic + semantic)
   ├─ plan (2–5 steps, only when the ask is multi-step)
   │
   └─ for step in 1..budget:
        ├─ LLM turn (streamed)
        ├─ tool_calls?  ── yes ─→ Guard.check(action)
        │                          ├─ allow  → execute → observe → loop
        │                          ├─ review → pause, ask the human
        │                          └─ block  → refuse, tell the model why
        └─ no tool_calls → candidate answer
                            │
                            └─ Verity critique (rubric) ── fail ─→ revise (≤2 passes)
                                                          pass ─→ answer
   budget exhausted → forced synthesis with no tools
```

Every step is persisted (`runs`, `steps`) and streamed, so the whole trajectory is auditable
after the fact — the same "a component you cannot inspect is a component you do not have"
principle the industry converged on.

### 3.3 The JagX Council

For hard questions the loop escalates to a council. Our roles are our own:

| Role | Job |
|---|---|
| **Atlas** | coordinator — decomposes the question, assigns lanes, owns the final synthesis |
| **Scout** | research — searches, fetches, gathers evidence with citations |
| **Forge** | builder — writes and *runs* code, produces artefacts |
| **Sage** | analyst — logic, maths, comparison, structured reasoning |
| **Verity** | critic — attacks the draft: unsupported claims, missing cases, arithmetic, contradictions |

Scout / Forge / Sage run **in parallel** (thread pool), each with its own context window and
its own tool budget, and return a compact finding. Atlas synthesises. Verity critiques the
synthesis against a concrete rubric; if it fails, Atlas revises once. The whole debate is
streamed to the UI as a council view.

This is the same shape as the four-agent debate the frontier systems converged on, but the
lanes, the rubric and the synthesis contract are ours.

### 3.4 JagX Memory

Four layers, because one layer is not memory:

| Layer | Holds | Lifetime |
|---|---|---|
| **Working** | the current run's messages and observations | the run |
| **Episodic** | what happened: run summaries, tool outcomes, failures | weeks, then consolidated |
| **Semantic** | durable facts about the user and their world | until contradicted |
| **Procedural** | how to do it: successful tool sequences, learned skills | until superseded |

**Consolidation** is the mechanism that matters. After a run, Forge writes an episode. When
episodes on a topic accumulate, they are consolidated into a semantic fact (or a procedural
skill) and the raw episodes decay. Retrieval is hybrid: a local hashed-embedding cosine score
blended with keyword overlap, so it works with zero external services and zero API keys.

### 3.5 JagX Goals

The unit of work is a **goal**, not a conversation. A goal has an identity, a title, a
success criterion, a status, a step budget and a checkpoint log. The runner advances every
active goal on a tick: it plans the next step, executes it through the same Loop and Guard,
records the checkpoint, and either completes, pauses for approval, or reschedules.

This is what makes JagX *always-on* in the Dots sense: the goal survives the session, and the
work continues when nobody is watching.

### 3.6 JagX Guard

Every action the agent proposes passes through Guard before it executes. Guard returns one of
three verdicts:

- **allow** — read-only or reversible: search, fetch, read memory, run code in the sandbox.
- **review** — needs a human: anything that leaves the machine or spends something.
- **block** — refused outright.

The hard boundary is stated as four verbs, in the spirit of the systems we studied:
**send, pay, sign, delete** always require review, and a small set (credential changes,
irreversible destruction) is always blocked. Rules are natural language, stored per user, and
evaluated by a deterministic matcher first and the model second — so the default path is
predictable and auditable.

### 3.7 JagX Forge — the self-improvement engine

This is the part that answers "it should train itself if it can, and it should be free."

Forge runs four loops, in increasing order of risk and cost:

1. **Reflection (free, always on).** After every run, Forge scores the trajectory on a rubric
   (did it use tools when it should have? did it stop early? did the critic fail it? how many
   steps?) and writes a lesson to episodic memory. Lessons are injected into future runs.
2. **Experience replay (free).** Successful trajectories are stored as reusable exemplars and
   retrieved as few-shot context for similar future tasks. This is the cheapest real
   improvement available: the weights never change, the behaviour does.
3. **Prompt & policy optimisation (free).** Forge maintains competing prompt variants per
   task class and runs a **multi-armed bandit** over them, using the eval harness score as the
   reward. Winners are promoted; losers are retired. Every change is versioned and reversible.
4. **Weight training (free, but needs a GPU you own).** Forge exports its own trajectories as
   a clean SFT dataset (`sharegpt` / `messages` JSONL) and ships a **LoRA / QLoRA training
   script** (`scripts/train_lora.py`) that fine-tunes a small open-weight model on a single
   consumer GPU, plus a DPO exporter for preference pairs from critic pass/fail. The trained
   adapter is served back through the same OpenAI-compatible endpoint — no vendor involved.

Loops 1–3 are genuinely free and run in production. Loop 4 is free in *software* and needs
hardware; section 5 is explicit about that.

### 3.8 Local-first inference

The provider chain is ordered so that a **local runtime is the default and no key is ever
required**:

```
ollama (http://localhost:11434/v1)      ← default, no key
llama.cpp / llama-server (…:8080/v1)    ← no key
vLLM (…:8000/v1)                        ← no key
───────────────────────────────────────
any OpenAI-compatible endpoint          ← optional, pluggable, off by default
```

All of them speak the same `/chat/completions` protocol with native tool calling, so one
client covers all of them and switching is a base-URL change. If nothing is reachable, the app
still starts and returns a clear, actionable error instead of failing silently.

### 3.9 Sandboxed execution

Code runs in a **local** sandbox: a subprocess with a wall-clock timeout, CPU/memory/file-size
rlimits, a scratch working directory, and network access off by default. No external service is
required. An optional Piston backend can be enabled for stronger isolation.

---

## 4. Reverse engineering → our own style

What we took (behaviour and interface, re-implemented from scratch):

| Studied behaviour | Our independent implementation |
|---|---|
| outer/inner loop separation | `agent.py` keeps the planner's context clean and hands the executor a compact brief |
| separate reviewer agent | `guard.py` (deterministic policy) + `council.py` Verity (semantic critique) — two different mechanisms, deliberately |
| goal as the unit of work | `goals.py` with identity, success criterion, checkpoints, reschedule |
| device-independent runtime | the goal runner advances goals server-side on a tick, independent of any client |
| subagent swarms | `council.py` thread-pool fan-out with per-lane context and tool budgets |
| approval boundary on send/pay/sign/delete | `guard.py` hard boundary, stated as four verbs |
| natural-language rules | per-user rule list, deterministic matcher first, model second |
| reflection / self-improvement | `selfimprove.py` four loops + `evals.py` harness |

What we deliberately did **not** do:

- We did not copy any proprietary code, weights, prompts or secrets. Every line here is
  original.
- We did not reuse the studied systems' agent names or role names. Our council is
  Atlas / Scout / Forge / Sage / Verity.
- We did not build a connector marketplace of thousands of integrations. We built a small,
  honest tool surface plus an MCP-compatible extension point.
- We did not claim capabilities we cannot demonstrate. Everything in this document is
  implemented and covered by the test suite.

---

## 5. Free & self-hosted — the honest limits

**What is genuinely free and works today:**

- The whole agent: loop, council, memory, goals, guard, tools, sandbox, streaming UI.
- Local inference on open weights via Ollama / vLLM / llama.cpp — no key, no vendor.
- Keyless web search (DuckDuckGo HTML) so the search tool works with zero configuration.
- Self-improvement loops 1–3: reflection, experience replay, prompt/policy bandit.
- The eval harness, the dataset exporter and the LoRA/QLoRA training script.

**What is free in software but needs hardware:**

- Loop 4, actual weight training. A 7B–8B model with QLoRA needs roughly one 16–24 GB GPU
  (a single consumer card) and a few hours. On CPU-only hardware it is not practical. The
  script is included and correct; the GPU is the constraint, and we say so rather than
  pretending otherwise.

**What we will not pretend:**

- JagX v12 does not train a frontier model. Training a Grok- or Opus-class model from scratch
  costs tens of millions of dollars. No free path exists to that, and any claim otherwise
  would be false.
- The closest free path to "a model that is ours" is: take a strong open-weight base
  (Llama / Qwen / Mistral / Gemma), fine-tune a LoRA on our own trajectories, and serve it
  locally. That is a real, owned, improving model — just not a frontier one.
- Local models are weaker than frontier models at long-horizon planning. The architecture
  compensates with structure (planning, council, critique, memory) rather than pretending the
  model is stronger than it is.

---

## 6. Data model

```
users(id, email, password_hash, created_at)
conversations(id, user_id, title, created_at, updated_at)
messages(id, conversation_id, role, content, created_at)
runs(id, conversation_id, user_id, status, mode, steps_used, provider, model, created_at, finished_at)
steps(id, run_id, idx, kind, name, payload, output, ok, duration_ms, created_at)
memories(id, user_id, layer, content, tags, vector, strength, source_run, created_at, last_used_at)
goals(id, user_id, title, success_criterion, status, priority, steps_used, budget, next_run_at, created_at, updated_at)
goal_checkpoints(id, goal_id, idx, note, status, created_at)
lessons(id, user_id, run_id, kind, text, score, created_at)
prompt_variants(id, user_id, task_class, name, body, wins, losses, score, active, created_at)
evals(id, user_id, name, cases, created_at)
eval_results(id, eval_id, variant_id, score, detail, created_at)
approvals(id, user_id, run_id, action, detail, status, created_at, decided_at)
settings(user_id, key, value)
```

SQLite with WAL. Single-instance by design; swap `db.py` for Postgres before horizontal
scaling.

---

## 7. API surface

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/auth/signup` · `/api/auth/login` | account + JWT |
| GET | `/api/auth/me` | current user |
| GET | `/api/health` · `/api/config` | status, providers, tools, modes |
| GET/POST | `/api/conversations` | list / create |
| GET/PATCH/DELETE | `/api/conversations/{id}` | read / rename / delete |
| POST | `/api/chat` | **run the agent (SSE)** |
| POST | `/api/council` | **run the council (SSE)** |
| GET | `/api/runs/{id}` | run + full step audit trail |
| GET/POST/DELETE | `/api/memories` | memory browser |
| POST | `/api/memories/consolidate` | run consolidation |
| GET/POST/PATCH/DELETE | `/api/goals` | goal board |
| POST | `/api/goals/{id}/advance` | advance one goal one tick |
| GET | `/api/approvals` · POST `/api/approvals/{id}` | review queue |
| GET | `/api/forge/lessons` | reflection log |
| GET/POST | `/api/forge/variants` | prompt variants + bandit state |
| POST | `/api/forge/optimize` | run one optimisation round |
| GET | `/api/forge/dataset` | export SFT dataset |
| GET/POST | `/api/evals` · POST `/api/evals/{id}/run` | eval harness |
| GET/PUT | `/api/settings` | per-user settings |
| GET | `/api/tools` · POST `/api/tools/run` | tool schemas / direct invoke |

### SSE event shapes

```jsonc
{"type":"conversation","conversation_id":"cnv_…","title":"…"}
{"type":"run_start","run_id":"run_…","mode":"loop"}
{"type":"plan","steps":["Search pricing","Compare options"]}
{"type":"step_start","index":1}
{"type":"tool_call","index":1,"name":"web_search","arguments":{…}}
{"type":"guard","index":1,"verdict":"allow","reason":"read-only"}
{"type":"tool_result","index":1,"name":"web_search","ok":true,"output":"…","duration_ms":412}
{"type":"council","phase":"fanout","lanes":["scout","forge","sage"]}
{"type":"council","phase":"lane","lane":"scout","status":"done","finding":"…"}
{"type":"critique","pass":1,"verdict":"revise","issues":["claim without source"]}
{"type":"token","text":"The "}
{"type":"draft_reset"}
{"type":"answer","text":"…"}
{"type":"done","run_id":"…","steps_used":2,"provider":"ollama","model":"…"}
{"type":"error","message":"…"}
```

---

## 8. Testing & evaluation

- **Unit + integration tests** (`server/tests/`) cover auth, the tool layer, the sandbox, all
  four memory layers and consolidation, the agent loop (tool call → answer, budget
  termination, guard blocking), the council (fan-out, critique, revision), goals, the guard
  policy engine, the Forge loops and the eval harness, plus the SSE stream and conversation
  isolation. They run offline with a fake LLM — no network, no key.
- **The eval harness** (`evals.py`) is the measurable half of self-improvement: named suites
  of cases with a checker, scored per prompt variant, so a prompt change is only promoted when
  it actually scores better.
- **The mock provider** (`scripts/mock_llm.py`) is a real OpenAI-compatible server that streams
  a tool call and then an answer, so the entire product can be exercised end to end for free.

---

## 9. Roadmap

- Postgres backend + horizontal scaling.
- Real vector store behind `embeddings.py` (the interface is already there).
- MCP client so external tool servers plug in without code changes.
- A scheduler process for the goal runner (today it advances on an API tick).
- GRPO/DPO training loop wired to the eval harness for preference-based improvement.
- Multi-tenant isolation for the sandbox (today it is per-process).

---

*JagX v12 is an independent implementation. It studies the behaviour of Grok bot, Dots, Muse
and Claude Duet and re-implements the capabilities in our own architecture, naming and code.
No proprietary code, weights, prompts or secrets were used.*
