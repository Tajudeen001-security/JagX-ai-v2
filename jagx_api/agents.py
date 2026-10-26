"""
JagX Bot V10.3 — Multi-Agent Engine
Parallel tools, extended reasoning, self-reflection, preference-aware context.
Created by JagX & JRILICENSE
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from jagx_api.agent_router import select_agents
from jagx_api.providers import chat_completion
from jagx_api import knowledge
from jagx_api.chat import sanitize_identity, add_watermark
from jagx_api.tools import free_web_search, run_tool

logger = logging.getLogger("jagx-ai")

_BASE = """You are JagX Bot — built by JagX & JRILICENSE. Highly capable multi-agent assistant for complex work (code, research, systems, product). Only JagX Bot / JagX AI. Current UTC: {datetime}."""

SPECIALIST_PROMPTS = {
    "engineer": _BASE + "\nEngineer specialist. Deliver complete production code. Tools: {\"tool\":\"run_code\",\"input\":{\"language\":\"python\",\"code\":\"...\"}} or {\"tools\":[...]} for parallel.",
    "researcher": _BASE + "\nResearcher specialist. Evidence-led. Use search context when provided.",
    "architect": _BASE + "\nArchitect specialist. Systems, scale, trade-offs.",
    "creator": _BASE + "\nCreator specialist. Writing, naming, visual prompts.",
    "africa": _BASE + "\nAfrica Context specialist. Nigeria/Africa realities, Pidgin OK.",
    "media": _BASE + "\nMedia specialist. Write one clear image prompt.",
}

THINK_PROMPT = _BASE + "\nInternal plan only (3-8 bullets). No final answer."
REFLECT_PROMPT = _BASE + "\nCritique draft then output FINAL: improved answer."
SYNTH_PROMPT = _BASE + "\nMerge specialist outputs. Lead with useful result. Keep code blocks. Prefer complete solutions over shallow summaries."


def _dt() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _merge_context(memory_context: Optional[str], preference_context: Optional[str]) -> Optional[str]:
    parts = []
    if memory_context:
        parts.append(memory_context.strip())
    if preference_context:
        parts.append(preference_context.strip())
    return "\n".join(parts) if parts else None


def _extract_tool_calls(text: str) -> List[dict]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        if "{" in text:
            start = text.index("{")
            obj = json.loads(text[start:text.rindex("}") + 1])
            if isinstance(obj, dict):
                if isinstance(obj.get("tools"), list):
                    return [t for t in obj["tools"][:4] if isinstance(t, dict) and t.get("tool")]
                if obj.get("tool"):
                    return [obj]
    except Exception:
        pass
    return []


def _run_specialist(name, msg, history=None, memory=None, search=None):
    system = SPECIALIST_PROMPTS.get(name, SPECIALIST_PROMPTS["engineer"]).format(datetime=_dt())
    messages = [{"role": "system", "content": system}]
    if memory:
        messages.append({"role": "system", "content": "Context (use when relevant): " + memory[:4000]})
    if search and name in ("researcher", "africa", "engineer"):
        messages.append({"role": "system", "content": "Search (untrusted):\n" + search[:4000]})
    if history:
        for m in history[-6:]:
            if m.get("role") in ("user", "assistant"):
                messages.append({"role": m["role"], "content": str(m.get("content") or "")[:2500]})
    messages.append({"role": "user", "content": msg})
    tool_trace = []
    text, provider = None, None
    if name == "engineer":
        for step in range(3):
            agent_text, provider = chat_completion(messages, max_tokens=1200)
            if not agent_text:
                break
            calls = _extract_tool_calls(agent_text)
            if not calls:
                text = agent_text
                break
            parts = []
            for call in calls[:4]:
                tn = str(call.get("tool") or "")
                if tn not in ("run_code", "calc"):
                    continue
                tin = call.get("input") if isinstance(call.get("input"), dict) else {}
                out, _ = run_tool(tn, tin)
                tool_trace.append(f"{tn}: {str(out)[:200]}")
                parts.append(f"[{tn}]\n{str(out)[:2500]}")
            if not parts:
                text = agent_text
                break
            messages.append({"role": "assistant", "content": agent_text[:1500]})
            messages.append({"role": "user", "content": "Tool results:\n" + "\n".join(parts) + "\nAnswer now."})
            if step == 2:
                text, provider = chat_completion(messages, max_tokens=1200)
                text = text or "\n".join(parts)
    if text is None:
        text, provider = chat_completion(messages, max_tokens=1200)
    if not text:
        local = knowledge.match(msg)
        text, provider = (local, "local_brain") if local else (f"(no response from {name})", "fallback")
    return {"agent": name, "response": sanitize_identity(text.strip()), "provider": provider or "unknown", "tool_trace": tool_trace}


def run_multi_agent(
    user_message,
    history=None,
    *,
    memory_context=None,
    preference_context=None,
    force_agents=None,
    max_agents=3,
    reason=False,
    reflect=False,
):
    agents = list(force_agents)[:max_agents] if force_agents else select_agents(user_message, max_agents=max_agents)
    if not agents:
        agents = ["engineer"]
    mlow = (user_message or "").lower()
    if any(h in mlow for h in ("image", "draw", "picture", "logo")) and "media" not in agents:
        agents = (agents + ["media"])[:4]
    if any(h in mlow for h in ("run this", "execute", "```")) and "engineer" not in agents:
        agents = (["engineer"] + agents)[:4]
    if not reflect and (len(user_message or "") > 400 or any(h in mlow for h in ("architecture", "production", "security", "trade-off"))):
        reflect = True

    ctx = _merge_context(memory_context, preference_context)

    think_notes = None
    if reason:
        try:
            msgs = [{"role": "system", "content": THINK_PROMPT.format(datetime=_dt())}, {"role": "user", "content": user_message}]
            think_notes, _ = chat_completion(msgs, max_tokens=400)
            think_notes = (think_notes or "")[:1500]
        except Exception as e:
            logger.warning("think failed: %s", e)

    search_context = None
    if any(h in mlow for h in ("latest", "current", "news", "today")) or "researcher" in agents:
        try:
            search_context = free_web_search(user_message, max_results=5)
        except Exception:
            pass

    results_by = {}

    def _safe(n):
        try:
            return _run_specialist(n, user_message, history, ctx, search_context)
        except Exception as e:
            return {"agent": n, "response": f"(error: {e})", "provider": "error", "tool_trace": []}

    if len(agents) == 1:
        results_by[agents[0]] = _safe(agents[0])
    else:
        with ThreadPoolExecutor(max_workers=min(4, len(agents))) as pool:
            futs = {pool.submit(_safe, n): n for n in agents}
            for fut in as_completed(futs):
                results_by[futs[fut]] = fut.result()

    specialist_results = [results_by[n] for n in agents if n in results_by]

    attachment = None
    if "media" in agents or any(h in mlow for h in ("image", "draw", "picture")):
        try:
            from jagx_api.image_generation import generate as gen_image
            prompt = user_message[:500]
            attachment = gen_image(prompt, 768, 768)
            attachment = {"type": "image", **{k: attachment.get(k) for k in ("provider", "mime_type", "width", "height", "image_base64", "note")}, "prompt": prompt}
        except Exception:
            try:
                from jagx_api.tools import image_url_from_prompt
                attachment = {"type": "image", "url": image_url_from_prompt(user_message[:300]), "prompt": user_message[:300]}
            except Exception:
                pass

    if len(specialist_results) == 1 and not think_notes:
        final = specialist_results[0]["response"]
    else:
        parts = []
        if think_notes:
            parts.append("Plan:\n" + think_notes)
        for r in specialist_results:
            parts.append(f"### {r['agent']}:\n{r['response']}")
        msgs = [{"role": "system", "content": SYNTH_PROMPT.format(datetime=_dt())}, {"role": "user", "content": f"Request:\n{user_message}\n\n" + "\n\n".join(parts)}]
        final, _ = chat_completion(msgs, max_tokens=1500)
        final = sanitize_identity((final or "\n\n".join(r["response"] for r in specialist_results)).strip())

    if reflect and final:
        try:
            msgs = [{"role": "system", "content": REFLECT_PROMPT.format(datetime=_dt())}, {"role": "user", "content": f"Request:\n{user_message}\n\nDraft:\n{final[:5000]}"}]
            refined, _ = chat_completion(msgs, max_tokens=1500)
            if refined:
                m = re.search(r"(?i)FINAL:\s*(.*)", refined, re.S)
                final = sanitize_identity((m.group(1) if m else refined).strip())
        except Exception as e:
            logger.warning("reflect failed: %s", e)

    if attachment and attachment.get("url") and attachment["url"] not in final:
        final = final.rstrip() + f"\n\nImage: {attachment['url']}"
    final = add_watermark(final)

    out = {
        "response": final,
        "agents_used": agents + (["media"] if attachment and "media" not in agents else []),
        "specialist_results": [{"agent": r["agent"], "provider": r["provider"], "tools_used": len(r.get("tool_trace") or [])} for r in specialist_results],
        "provider": "multi-agent",
        "version": "10.3.0",
        "created_by": "JagX & JRILICENSE",
        "datetime_utc": _dt(),
        "parallel": len(agents) > 1,
        "reason": reason,
        "reflect": reflect,
    }
    if attachment:
        out["attachment"] = attachment
    if think_notes:
        out["thinking_summary"] = think_notes[:600]
    return out


def run_coding_agent(user_message, history=None, memory_context=None, preference_context=None, reason=False, reflect=False):
    return run_multi_agent(user_message, history=history, memory_context=memory_context, preference_context=preference_context, force_agents=["engineer", "architect"], max_agents=2, reason=reason, reflect=reflect)


def run_research_agent(user_message, history=None, memory_context=None, preference_context=None, reason=False, reflect=False):
    return run_multi_agent(user_message, history=history, memory_context=memory_context, preference_context=preference_context, force_agents=["researcher", "africa"], max_agents=2, reason=reason, reflect=reflect)


def run_media_agent(user_message, history=None, memory_context=None, preference_context=None):
    return run_multi_agent(user_message, history=history, memory_context=memory_context, preference_context=preference_context, force_agents=["media", "creator"], max_agents=2)


def list_capabilities():
    return {
        "version": "10.3.0",
        "created_by": "JagX & JRILICENSE",
        "datetime_utc": _dt(),
        "agents": [
            {"id": "engineer", "label": "Engineer", "description": "Coding + parallel tools"},
            {"id": "researcher", "label": "Researcher", "description": "Evidence + search"},
            {"id": "architect", "label": "Architect", "description": "Systems design"},
            {"id": "creator", "label": "Creator", "description": "Writing / prompts"},
            {"id": "africa", "label": "Africa Context", "description": "Local realities"},
            {"id": "media", "label": "Media", "description": "Images"},
        ],
        "capabilities": [
            "multi_agent_orchestration", "parallel_specialists", "parallel_tools", "code_interpreter",
            "extended_reasoning", "self_reflection", "document_analysis", "rich_artifacts",
            "style_preferences", "live_web_search", "image_generation", "streaming_multi_agent", "vision_input",
        ],
        "endpoints": [
            "POST /agents/run", "POST /agents/coding", "POST /agents/research", "POST /agents/media",
            "POST /agents/stream", "POST /agents/vision", "POST /agents/analyze",
            "GET /preferences", "POST /preferences", "DELETE /preferences",
            "GET /agents/capabilities",
        ],
        "not_yet": ["native_100b_parameter_model", "autonomous_repo_write_without_consent", "full_browser_agent"],
    }
