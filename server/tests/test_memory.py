"""Four-layer memory, hybrid retrieval and consolidation."""

from __future__ import annotations

from app import db, embeddings, memory


def test_embedding_backend_is_local():
    assert embeddings.backend() in ("hashed", "sentence")


def test_embed_is_deterministic_and_normalised():
    a = embeddings.embed("the quick brown fox")
    b = embeddings.embed("the quick brown fox")
    assert a == b
    assert len(a) == embeddings.DIM
    norm = sum(x * x for x in a) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_cosine_bounds():
    a = embeddings.embed("machine learning agents")
    b = embeddings.embed("machine learning agents")
    assert abs(embeddings.cosine(a, b) - 1.0) < 1e-6
    assert embeddings.cosine([], []) == 0.0


def test_keyword_overlap():
    assert embeddings.keyword_overlap("dark mode theme", "the user prefers dark mode") > 0.5
    assert embeddings.keyword_overlap("quantum physics", "baking sourdough bread") == 0.0


def test_remember_and_retrieve(user):
    memory.remember(user["id"], "The user is building an autonomous AI agent called JagX.", layer="semantic")
    memory.remember(user["id"], "The user prefers TypeScript for frontend work.", layer="semantic")
    hits = memory.retrieve(user["id"], "what is the user building", limit=3)
    assert hits
    assert any("JagX" in h["content"] for h in hits)


def test_retrieve_respects_layer_filter(user):
    memory.remember(user["id"], "Episodic note about a run.", layer="episodic")
    memory.remember(user["id"], "Semantic fact about the project.", layer="semantic")
    hits = memory.retrieve(user["id"], "project", limit=10, layers=("semantic",))
    assert all(h["layer"] == "semantic" for h in hits)


def test_retrieve_empty_query(user):
    assert memory.retrieve(user["id"], "") == []


def test_short_memory_rejected(user):
    assert memory.remember(user["id"], "x") == {}


def test_context_block(user):
    memory.remember(user["id"], "The user's company is called JagX.", layer="semantic")
    block = memory.context_block(user["id"], "what is the company called")
    assert "JagX" in block


def test_consolidation_promotes_clusters(user):
    for i in range(4):
        memory.remember(
            user["id"],
            f"Deployment pipeline note number {i} about the release process.",
            layer="episodic",
        )
    result = memory.consolidate(user["id"], min_cluster=3)
    assert result["episodes_scanned"] >= 4
    assert result["promoted"] >= 1
    semantic = db.list_memories(user["id"], layer="semantic", limit=200)
    assert any("Recurring theme" in m["content"] for m in semantic)


def test_decay_and_prune(user):
    m = memory.remember(user["id"], "A weak episodic memory that should fade away.", layer="episodic", strength=0.1)
    assert m
    db.decay_memories(user["id"], factor=0.5, floor=0.05)
    pruned = db.prune_weak_memories(user["id"], floor=0.2)
    assert pruned >= 1


def test_stats(user):
    memory.remember(user["id"], "A fact for stats.", layer="semantic")
    st = memory.stats(user["id"])
    assert st["total"] >= 1
    assert "semantic" in st["by_layer"]
    assert st["embedding_backend"] in ("hashed", "sentence")


def test_lessons_block(user):
    db.add_lesson(user["id"], "Always gather evidence before answering.", score=0.9)
    block = memory.lessons_block(user["id"])
    assert "evidence" in block
