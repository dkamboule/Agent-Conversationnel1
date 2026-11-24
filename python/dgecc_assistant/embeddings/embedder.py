"""Fournisseur d'embeddings Gemini avec cache LRU et complétion du corpus publié."""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import List

from ..core.config import RAG_CONFIG
from ..core.gemini_client import get_gemini_client, with_gemini_retry
from ..database.db import get_db
from ..search.corpus_cache import invalidate_corpus_cache, normalize_french_token, to_unit_vector


class GeminiEmbeddingProvider:
    def __init__(self, model_name: str = RAG_CONFIG.embedding_model) -> None:
        self.model_name = model_name

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = get_gemini_client()
        vectors: List[List[float]] = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            slice_texts = texts[i : i + batch_size]
            response = with_gemini_retry(
                lambda s=slice_texts: client.models.embed_content(
                    model=self.model_name,
                    contents=s,
                ),
                max_retries=2,
                base_delay_ms=1000,
            )
            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) == len(slice_texts):
                for emb in embeddings:
                    values = list(emb.values) if emb and emb.values else []
                    if not values:
                        raise RuntimeError("Vecteur d'embedding vide retourné par le modèle.")
                    vectors.append([float(v) for v in values])
            else:
                for text in slice_texts:
                    single_res = with_gemini_retry(
                        lambda t=text: client.models.embed_content(
                            model=self.model_name,
                            contents=t,
                        ),
                        max_retries=2,
                        base_delay_ms=1000,
                    )
                    single_embs = getattr(single_res, "embeddings", None) or []
                    values = list(single_embs[0].values) if single_embs and single_embs[0].values else []
                    if not values:
                        raise RuntimeError("Vecteur d'embedding vide retourné par le modèle.")
                    vectors.append([float(v) for v in values])
        return vectors


def get_embedding_provider() -> GeminiEmbeddingProvider:
    return GeminiEmbeddingProvider(RAG_CONFIG.embedding_model)


_MAX_QUERY_EMBED_CACHE = 250
_query_cache_lock = threading.Lock()
_query_embedding_cache: "OrderedDict[str, List[float]]" = OrderedDict()


def embed_query_unit_vector(query: str) -> List[float]:
    key = normalize_french_token(query.strip())
    with _query_cache_lock:
        cached = _query_embedding_cache.get(key)
        if cached is not None:
            _query_embedding_cache.move_to_end(key)
            return cached

    provider = get_embedding_provider()
    vec = provider.embed_texts([query])[0]
    unit = to_unit_vector(vec)

    with _query_cache_lock:
        if len(_query_embedding_cache) >= _MAX_QUERY_EMBED_CACHE:
            _query_embedding_cache.popitem(last=False)
        _query_embedding_cache[key] = unit
    return unit


def cosine_similarity(a: List[float], b: List[float]) -> float:
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(length))
    norm_a = math.sqrt(sum(a[i] * a[i] for i in range(length)))
    norm_b = math.sqrt(sum(b[i] * b[i] for i in range(length)))
    denom = norm_a * norm_b
    return 0.0 if denom == 0.0 else dot / denom


_embed_lock = threading.Lock()
_embedding_quota_paused_until = 0.0


def is_embedding_quota_paused() -> bool:
    import time

    return time.time() < _embedding_quota_paused_until


def mark_embedding_quota_exhausted(err: Exception) -> None:
    import time

    msg = str(err).lower()
    daily = "perday" in msg or "per day" in msg or "requestsperday" in msg
    _pause = 3600.0 if daily else 600.0
    global _embedding_quota_paused_until
    _embedding_quota_paused_until = time.time() + _pause


def ensure_published_chunks_embedded() -> int:
    if is_embedding_quota_paused():
        return 0
    with _embed_lock:
        if is_embedding_quota_paused():
            return 0
        db = get_db()
        rows = db.execute(
            """
            SELECT c.id, c.content
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id AND d.current_version = c.version_number
            WHERE d.status = 'PUBLIE' AND (c.embedding_json IS NULL OR c.embedding_json = '')
            """
        ).fetchall()

        if not rows:
            return 0

        provider = get_embedding_provider()
        embedded_count = 0
        try:
            for i in range(0, len(rows), 64):
                batch = rows[i : i + 64]
                vectors = provider.embed_texts([str(r["content"]) for r in batch])
                for idx, row in enumerate(batch):
                    db.execute(
                        "UPDATE document_chunks SET embedding_json = ?, embedding_model = ? WHERE id = ?",
                        (json.dumps(vectors[idx]), provider.model_name, row["id"]),
                    )
                db.commit()
                embedded_count += len(batch)
        except Exception as err:
            msg = str(err).lower()
            if "429" in msg or "resource_exhausted" in msg or "quota" in msg or "unavailable" in msg:
                mark_embedding_quota_exhausted(err)
                if embedded_count > 0:
                    invalidate_corpus_cache()
                return embedded_count
            raise
        invalidate_corpus_cache()
        return embedded_count
