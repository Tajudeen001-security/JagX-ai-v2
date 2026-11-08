"""JagX v12 — local embeddings.

Two backends, both free and offline:

* ``sentence`` — sentence-transformers, if the user installed it. Real semantic
  vectors, still fully local.
* ``hashed``   — the built-in fallback: a deterministic hashed bag-of-ngrams
  projection. Zero dependencies, zero downloads, works on a plane.

The interface is deliberately tiny so a real vector store can be dropped in
behind it later without touching callers.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

from .config import settings

DIM = 384
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on", "with",
    "that", "this", "as", "be", "are", "was", "were", "by", "at", "from", "we", "you",
    "i", "he", "she", "they", "but", "not", "have", "has", "had", "do", "does", "did",
}

_backend: str = ""
_model = None


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def _load_sentence_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _model = SentenceTransformer(settings.embedding_model)
    except Exception:  # noqa: BLE001 - optional dependency, fall back silently
        _model = False
    return _model


def backend() -> str:
    """Which backend is actually in use."""
    global _backend
    if _backend:
        return _backend
    want = (settings.embedding_backend or "auto").lower()
    if want in ("auto", "sentence"):
        if _load_sentence_model():
            _backend = "sentence"
            return _backend
        if want == "sentence":
            _backend = "hashed"
            return _backend
    _backend = "hashed"
    return _backend


def _hashed_embed(text: str) -> list[float]:
    """Deterministic hashed n-gram projection, L2-normalised.

    Not as good as a trained encoder, but it is genuinely free, instant, and
    good enough for the hybrid retrieval JagX uses (it is always blended with
    keyword overlap).
    """
    vec = [0.0] * DIM
    toks = _tokens(text)
    if not toks:
        return vec

    grams: list[str] = list(toks)
    grams += [f"{toks[i]}_{toks[i + 1]}" for i in range(len(toks) - 1)]

    for g in grams:
        h = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "big") % DIM
        sign = 1.0 if h[4] & 1 else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def embed(text: str) -> list[float]:
    if backend() == "sentence":
        model = _load_sentence_model()
        if model:
            try:
                v = model.encode([text or ""], normalize_embeddings=True)[0]
                return [float(x) for x in v]
            except Exception:  # noqa: BLE001
                pass
    return _hashed_embed(text)


def embed_many(texts: Iterable[str]) -> list[list[float]]:
    return [embed(t) for t in texts]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def keyword_overlap(query: str, text: str) -> float:
    """Jaccard-ish overlap over content tokens — the BM25 stand-in."""
    q = set(_tokens(query))
    t = set(_tokens(text))
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


def hybrid_score(query: str, text: str, vector: list[float], query_vector: list[float]) -> float:
    """Blend semantic similarity with keyword overlap.

    Weighted toward the vector when a real encoder is present, toward keywords
    when we are on the hashed fallback.
    """
    sem = cosine(query_vector, vector)
    kw = keyword_overlap(query, text)
    if backend() == "sentence":
        return 0.7 * sem + 0.3 * kw
    return 0.45 * sem + 0.55 * kw
