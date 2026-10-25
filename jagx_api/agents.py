"""
JagX AI V9 — Multi-Agent Engine

Core orchestration: run specialists, synthesize final answer,
preserve JagX identity, and stay fully compatible with existing
providers, tools, memory, and knowledge modules.

Created by JagX & JRILICENSE
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from jagx_api.agent_router import select_agents
from jagx_api.providers import chat_completion
from jagx_api import knowledge
from jagx_api.chat import sanitize_identity, add_watermark

logger = logging.getLogger("jagx-ai")

_BASE_IDENTITY = """You are JagX AI — built by JagX & JRILICENSE (an independent African tech team).
You are only JagX AI. Never claim you were built by OpenAI, Google, Anthropic, Meta, xAI, or any other lab.
Be calm, direct, and useful. No corporate fluff.
Current UTC date: {date}.
"""

SPECIALIST_PROMPTS: Dict[str, str] = {
    "engineer": _BASE_IDENTITY + """
You are the Engineer specialist of JagX AI.
Focus exclusively on software engineering: clean, complete, production-minded code.
Strong at Flutter/Dart, React Native, Python/FastAPI, Node, mobile backends, auth, APIs.
- Deliver complete files in fenced code blocks with path comments when useful.
- Prefer maintainable, testable solutions with error handling.
- Never invent package versions.
- For "build X": short plan → full files → how to run.
- Treat engineering advice as decision support, not certification.
""",
    "researcher": _BASE_IDENTITY + """
You are the Researcher specialist of JagX AI.
Provide clear, evidence-led answers. Prefer truth over sounding smart.
- Lead with the useful result, then supporting detail.
- When you use external knowledge, be honest about uncertainty.
- Distinguish measured facts from estimates.
- Keep answers structured and easy to scan.
""",
    "architect": _BASE_IDENTITY + """
You are the Architect specialist of JagX AI.
Think about systems, scalability, reliability, and long-term maintainability.
- Prefer practical designs that can be implemented by a small team.
- Discuss trade-offs explicitly (cost, complexity, latency, operational burden).
- Give concrete recommendations, not vague principles.
""",
    "creator": _BASE_IDENTITY + """
You are the Creator specialist of JagX AI.
Help with writing, naming, branding, product language, and storytelling.
- Be original, clear, and culturally aware.
- Prefer short, high-signal copy over long marketing speak.
- When asked for names or slogans, give several strong options with brief rationale.
""",
    "africa": _BASE_IDENTITY + """
You are the Africa Context specialist of JagX AI.
You understand Nigerian and broader African realities: languages (including Pidgin),
mobile-first usage, payments, infrastructure constraints, local culture, and markets.
- Reply in the language the user is using (including natural code-switching).
- Prefer practical advice that works in Lagos, Nairobi, Accra, Johannesburg, etc.
- Be respectful and precise; never stereotype.
""",
}

SYNTHESIZER_PROMPT = _BASE_IDENTITY + """
You are the final synthesizer of JagX AI.
You receive outputs from specialist agents. Your job is to produce one clean,
coherent final answer for the user.

Rules:
1. Merge the best parts of each specialist response.
2. Remove duplication and internal agent chatter.
3. Keep the voice of JagX AI (calm, direct, useful).
4. Lead with the useful result.
5. Never mention that you are "synthesizing" or "combining agents" unless the user asks how the system works.
6. Preserve any complete code blocks that the Engineer produced.
"""


def _date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _run_specialist(
    agent_name: str,
    user_message: str,
    history: Optional[List[dict]] = None,
    memory_context: Optional[str] = None,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    """Run one specialist and return structured result."""
    prompt = SPECIALIST_PROMPTS.get(agent_name, SPECIALIST_PROMPTS["engineer"])
    system = prompt.format(date=_date())

    messages: List[dict] = [{"role": "system", "content": system}]
    if memory_context:
        messages.append({
            "role": "system",
            "content": "Consented user memory (use only when relevant): " + memory_context[:4000],
        })
    if history:
        for m in history[-8:]:
            role = m.get("role")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": str(m.get("content") or "")[:3000]})
    messages.append({"role": "user", "content": user_message})

    text, provider = chat_completion(messages, max_tokens=max_tokens)
    if not text:
        local = knowledge.match(user_message)
        if local:
            text, provider = local, "local_brain"
        else:
            text, provider = f"(No response from {agent_name} specialist)", "fallback"

    return {
        "agent": agent_name,
        "response": sanitize_identity(text.strip()),
        "provider": provider,
    }


def _synthesize(
    user_message: str,
    specialist_results: List[Dict[str, Any]],
    max_tokens: int = 1400,
) -> str:
    """Merge specialist outputs into one final answer."""
    if len(specialist_results) == 1:
        return specialist_results[0]["response"]

    parts = []
    for r in specialist_results:
        parts.append(f"### {r['agent'].title()} specialist output:\n{r['response']}")

    synthesis_input = (
        f"User request:\n{user_message}\n\n"
        f"Specialist outputs:\n\n" + "\n\n".join(parts)
    )

    messages = [
        {"role": "system", "content": SYNTHESIZER_PROMPT.format(date=_date())},
        {"role": "user", "content": synthesis_input},
    ]

    text, _ = chat_completion(messages, max_tokens=max_tokens)
    if text:
        return sanitize_identity(text.strip())

    return "\n\n".join(r["response"] for r in specialist_results)


def run_multi_agent(
    user_message: str,
    history: Optional[List[dict]] = None,
    *,
    memory_context: Optional[str] = None,
    force_agents: Optional[List[str]] = None,
    max_agents: int = 3,
) -> Dict[str, Any]:
    """
    Main entry point for JagX AI V9 multi-agent runs.

    Returns:
        {
            "response": str,
            "agents_used": list,
            "specialist_results": list,
            "provider": str,
        }
    """
    if force_agents:
        agents = force_agents[:max_agents]
    else:
        agents = select_agents(user_message, max_agents=max_agents)

    if not agents:
        agents = ["engineer"]

    logger.info("V9 multi-agent run → agents=%s", agents)

    specialist_results: List[Dict[str, Any]] = []
    for name in agents:
        try:
            result = _run_specialist(
                name,
                user_message,
                history=history,
                memory_context=memory_context,
            )
            specialist_results.append(result)
        except Exception as exc:
            logger.warning("Specialist %s failed: %s", name, exc)
            specialist_results.append({
                "agent": name,
                "response": f"(Specialist {name} encountered an error)",
                "provider": "error",
            })

    final_text = _synthesize(user_message, specialist_results)
    final_text = add_watermark(final_text)

    return {
        "response": final_text,
        "agents_used": agents,
        "specialist_results": [
            {"agent": r["agent"], "provider": r["provider"]} for r in specialist_results
        ],
        "provider": "multi-agent",
    }


def run_coding_agent(
    user_message: str,
    history: Optional[List[dict]] = None,
    memory_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Fast path that forces the Engineer (+ optional Architect)."""
    return run_multi_agent(
        user_message,
        history=history,
        memory_context=memory_context,
        force_agents=["engineer", "architect"],
        max_agents=2,
    )


def run_research_agent(
    user_message: str,
    history: Optional[List[dict]] = None,
    memory_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Fast path that forces Researcher (+ optional Africa Context)."""
    return run_multi_agent(
        user_message,
        history=history,
        memory_context=memory_context,
        force_agents=["researcher", "africa"],
        max_agents=2,
    )


def list_capabilities() -> Dict[str, Any]:
    """Public capability discovery."""
    return {
        "version": "9.0.0",
        "created_by": "JagX & JRILICENSE",
        "agents": [
            {"id": "engineer", "label": "Engineer", "description": "Coding, debugging, architecture, Flutter/Python/FastAPI"},
            {"id": "researcher", "label": "Researcher", "description": "Evidence-led answers and synthesis"},
            {"id": "architect", "label": "Architect", "description": "Systems design, scale, reliability"},
            {"id": "creator", "label": "Creator", "description": "Writing, naming, branding, product language"},
            {"id": "africa", "label": "Africa Context", "description": "Nigeria/Africa realities, languages, local knowledge"},
        ],
        "endpoints": [
            "POST /agents/run",
            "POST /agents/coding",
            "POST /agents/research",
            "GET /agents/capabilities",
        ],
    }
