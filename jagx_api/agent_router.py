"""
JagX AI V9.1 — Agent Router
Intent classification and agent selection (includes Media).

Created by JagX & JRILICENSE
"""
from __future__ import annotations

import re
from typing import List, Set

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
    "how does", "why does", "overview of", "summary of", "today", "this week",
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

_MEDIA_HINTS = (
    "image", "picture", "draw", "illustration", "logo", "render", "photo",
    "generate image", "create image", "visual", "artwork", "thumbnail",
    "banner", "icon", "poster",
)


def _score(text: str, hints: tuple) -> int:
    t = (text or "").lower()
    return sum(1 for h in hints if h in t)


def classify_intent(message: str) -> Set[str]:
    msg = (message or "").strip()
    if not msg:
        return {"engineer"}

    scores = {
        "engineer": _score(msg, _ENGINEER_HINTS),
        "researcher": _score(msg, _RESEARCH_HINTS),
        "architect": _score(msg, _ARCHITECT_HINTS),
        "creator": _score(msg, _CREATOR_HINTS),
        "africa": _score(msg, _AFRICA_HINTS),
        "media": _score(msg, _MEDIA_HINTS),
    }

    if scores["media"] >= 1 or re.search(
        r"\b(generate|create|draw|make)\b.+\b(image|picture|logo|illustration)\b",
        msg,
        re.I,
    ):
        selected = {"media"}
        if scores["creator"] >= 1:
            selected.add("creator")
        return selected

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

    if scores["researcher"] >= 2:
        selected = {"researcher"}
        if scores["africa"] >= 1:
            selected.add("africa")
        return selected

    if scores["architect"] >= 2:
        selected = {"architect"}
        if scores["engineer"] >= 1:
            selected.add("engineer")
        return selected

    if scores["creator"] >= 2:
        return {"creator"}

    if scores["africa"] >= 2:
        selected = {"africa"}
        if scores["researcher"] >= 1:
            selected.add("researcher")
        return selected

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top[0][1] > 0:
        return {top[0][0]}
    return {"engineer", "researcher"}


def select_agents(message: str, max_agents: int = 3) -> List[str]:
    selected = classify_intent(message)
    priority = ["engineer", "architect", "researcher", "creator", "media", "africa"]
    ordered = [a for a in priority if a in selected]
    return ordered[:max_agents]
