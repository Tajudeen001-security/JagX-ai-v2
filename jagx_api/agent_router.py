"""
JagX AI V9 — Agent Router
Intent classification and agent selection.

Created by JagX & JRILICENSE
"""
from __future__ import annotations

import re
from typing import List, Set

# Keywords that strongly indicate each specialist
_ENGINEER_HINTS = (
    "code", "coding", "debug", "bug", "error", "traceback", "function", "class",
    "api", "endpoint", "fastapi", "flask", "django", "flutter", "dart", "react",
    "node", "javascript", "typescript", "python", "sql", "database", "refactor",
    "implement", "build app", "build a", "write code", "fix this", "compile",
    "deploy", "docker", "kubernetes", "git", "github", "test suite", "unit test",
    "backend", "frontend", "mobile app", "android", "ios", "sdk", "library",
)

_RESEARCH_HINTS = (
    "research", "what is", "who is", "explain", "history of", "compare",
    "difference between", "pros and cons", "latest", "current", "news",
    "study", "evidence", "source", "according to", "statistics", "data on",
    "how does", "why does", "overview of", "summary of",
)

_ARCHITECT_HINTS = (
    "architecture", "system design", "scale", "scalability", "infrastructure",
    "microservices", "monolith", "design pattern", "high availability",
    "load balancer", "caching strategy", "database design", "schema",
    "reliability", "fault tolerance", "distributed", "event driven",
    "how should we structure", "best way to organize", "tech stack",
)

_CREATOR_HINTS = (
    "write", "rewrite", "name", "naming", "brand", "slogan", "tagline",
    "copy", "headline", "description", "story", "narrative", "marketing",
    "landing page", "about us", "product description", "tweet", "post",
    "email", "subject line", "creative", "storytelling",
)

_AFRICA_HINTS = (
    "nigeria", "lagos", "abuja", "africa", "african", "pidgin", "yoruba",
    "igbo", "hausa", "naija", "ghana", "kenya", "south africa", "naira",
    "cedis", "shilling", "local", "in nigeria", "nigerian", "west africa",
    "east africa", "sub-saharan", "mobile money", "mtn", "airtel", "gtbank",
)


def _score(text: str, hints: tuple) -> int:
    t = (text or "").lower()
    return sum(1 for h in hints if h in t)


def classify_intent(message: str) -> Set[str]:
    """
    Return the set of specialist agent names that should run.
    Always returns at least one agent. Lead is implicit.
    """
    msg = (message or "").strip()
    if not msg:
        return {"engineer"}  # safe default

    scores = {
        "engineer": _score(msg, _ENGINEER_HINTS),
        "researcher": _score(msg, _RESEARCH_HINTS),
        "architect": _score(msg, _ARCHITECT_HINTS),
        "creator": _score(msg, _CREATOR_HINTS),
        "africa": _score(msg, _AFRICA_HINTS),
    }

    # Strong coding signal → engineer is primary
    if scores["engineer"] >= 2 or re.search(
        r"\b(write|create|build|fix|debug)\b.*(code|function|app|api|class)",
        msg,
        re.I,
    ):
        selected = {"engineer"}
        if scores["architect"] >= 1:
            selected.add("architect")
        if scores["africa"] >= 1:
            selected.add("africa")
        return selected

    # Research heavy
    if scores["researcher"] >= 2:
        selected = {"researcher"}
        if scores["africa"] >= 1:
            selected.add("africa")
        return selected

    # Architecture / system design
    if scores["architect"] >= 2:
        selected = {"architect"}
        if scores["engineer"] >= 1:
            selected.add("engineer")
        return selected

    # Creative writing / naming
    if scores["creator"] >= 2:
        return {"creator"}

    # Africa context
    if scores["africa"] >= 2:
        selected = {"africa"}
        if scores["researcher"] >= 1:
            selected.add("researcher")
        return selected

    # Default: balanced general path (engineer + researcher)
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top[0][1] > 0:
        return {top[0][0]}
    return {"engineer", "researcher"}


def select_agents(message: str, max_agents: int = 3) -> List[str]:
    """
    Public API used by the multi-agent engine.
    Returns ordered list of agent names (max 3).
    """
    selected = classify_intent(message)
    # Stable priority order
    priority = ["engineer", "architect", "researcher", "creator", "africa"]
    ordered = [a for a in priority if a in selected]
    return ordered[:max_agents]
