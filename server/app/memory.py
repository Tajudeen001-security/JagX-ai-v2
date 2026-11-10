"""JagX v13 — four-layer memory with consolidation.

Layers
------
working     the current run's messages and observations (in-context, not stored here)
episodic    what happened: run summaries, tool outcomes, failures
semantic    durable facts about the user and their world
procedural  how to do it: successful tool sequences, learned skills

The mechanism that matters is **consolidation**: episodes on a topic accumulate,
then collapse into a semantic fact (or a procedural skill) and the raw episodes
decay. Retrieval is hybrid — local embedding cosine blended with keyword
overlap — so it works with zero external services.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from . import db, embeddings

logger = logging.getLogger("jagx.memory")

LAYERS = ("episodic", "semantic", "procedural")


def remember(
    user_id: str,
    content: str,
    *,
    layer: str = "semantic",
    tags: str = "",
    source_run: str = "",
    strength: float = 1.0,
) -> dict:
    content = (content or "").strip()
    if len(content) < 3:
        return {}
    layer = layer if layer in LAYERS else "semantic"
    return db.add_memory(
        user_id,
        content,
        layer=layer,
        tags=tags,
        vector=embeddings.embed(content),
        source_run=source_run,
        strength=strength,
    )


def retrieve(user_id: str, query: str, *, limit: int = 6, layers: Optional[tuple[str, ...]] = None) -> list[dict]:
    """Hybrid retrieval across the stored layers."""
    query = (query or "").strip()
    if not query:
        return []

    rows = db.all_memories_for_search(user_id)
    if not rows:
        return []

    qv = embeddings.embed(query)
    scored: list[tuple[float, dict]] = []
    for r in rows:
        if layers and r.get("layer") not in layers:
            continue
        base = embeddings.hybrid_score(query, r["content"], r.get("vector") or [], qv)
        # strength is a slow-moving multiplier: memories that keep proving useful
        # rank higher, faded episodes rank lower.
        score = base * (0.6 + 0.4 * float(r.get("strength") or 1.0))
        if score > 0.02:
            scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, r in scored[:limit]:
        r = dict(r)
        r["score"] = round(score, 4)
        out.append(r)
        try:
            db.touch_memory(r["id"], min(1.0, float(r.get("strength") or 1.0) + 0.05))
        except Exception:  # noqa: BLE001
            pass
    return out


def context_block(user_id: str, query: str, *, limit: int = 6) -> str:
    """A compact memory block to inject into the system context."""
    hits = retrieve(user_id, query, limit=limit)
    if not hits:
        return ""
    lines = [f"- ({h['layer']}) {h['content']}" for h in hits]
    return "\n".join(lines)[:4000]


def lessons_block(user_id: str, limit: int = 5) -> str:
    """Recent self-improvement lessons, injected as guidance."""
    rows = db.list_lessons(user_id, limit=limit)
    if not rows:
        return ""
    return "\n".join(f"- {r['text']}" for r in rows)[:2000]


# ---------------------------------------------------------------------------
# consolidation
# ---------------------------------------------------------------------------
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on", "with",
    "that", "this", "as", "be", "are", "was", "were", "by", "at", "from", "i", "you",
}


def _topic_key(text: str, k: int = 4) -> str:
    toks = [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP and len(t) > 2]
    return " ".join(sorted(set(toks))[:k])


def consolidate(user_id: str, *, min_cluster: int = 3, decay: bool = True) -> dict:
    """Collapse clusters of episodes into semantic facts; decay the rest.

    This is deliberately simple and deterministic — no model call, so it is
    free, fast and auditable. A model-assisted summariser can be layered on top
    later without changing this contract.
    """
    episodes = db.list_memories(user_id, layer="episodic", limit=500)
    clusters: dict[str, list[dict]] = {}
    for e in episodes:
        clusters.setdefault(_topic_key(e["content"]), []).append(e)

    promoted = 0
    for key, group in clusters.items():
        if len(group) < min_cluster or not key:
            continue
        # Already consolidated? Skip.
        existing = db.list_memories(user_id, layer="semantic", limit=500)
        if any(_topic_key(s["content"]) == key for s in existing):
            continue
        group.sort(key=lambda m: m["created_at"])
        summary = " ".join(g["content"].strip().rstrip(".") for g in group[:4])
        summary = f"Recurring theme: {summary[:600]}"
        remember(user_id, summary, layer="semantic", tags=f"consolidated,{key}", strength=0.9)
        promoted += 1

    decayed = db.decay_memories(user_id) if decay else 0
    pruned = db.prune_weak_memories(user_id) if decay else 0

    return {
        "episodes_scanned": len(episodes),
        "clusters": len(clusters),
        "promoted": promoted,
        "decayed": decayed,
        "pruned": pruned,
    }


def stats(user_id: str) -> dict:
    rows = db.list_memories(user_id, limit=5000)
    by_layer: dict[str, int] = {l: 0 for l in LAYERS}
    for r in rows:
        by_layer[r["layer"]] = by_layer.get(r["layer"], 0) + 1
    return {
        "total": len(rows),
        "by_layer": by_layer,
        "embedding_backend": embeddings.backend(),
    }
