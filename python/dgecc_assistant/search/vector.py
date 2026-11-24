"""Recherche sémantique vectorielle par produit scalaire unitaire."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from ..core.config import RAG_CONFIG
from ..embeddings.embedder import (
    embed_query_unit_vector,
    ensure_published_chunks_embedded,
    is_embedding_quota_paused,
    mark_embedding_quota_exhausted,
)
from .bm25 import RetrievedPassage
from .corpus_cache import dot_product_unit, get_published_corpus_index


def vector_search(
    query: str,
    category: Optional[str] = None,
    published_after: Optional[str] = None,
    version_number: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    limit: int = 12,
) -> List[RetrievedPassage]:
    threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else RAG_CONFIG.default_similarity_threshold
    )

    try:
        ensure_published_chunks_embedded()
    except Exception as err:
        msg = str(err).lower()
        if "429" in msg or "resource_exhausted" in msg or "quota" in msg or "unavailable" in msg:
            mark_embedding_quota_exhausted(err)
        else:
            raise

    index = get_published_corpus_index()
    has_vectors = any(chunk.unit_vector is not None for chunk in index.chunks)
    if not has_vectors or is_embedding_quota_paused():
        return []

    try:
        query_unit = embed_query_unit_vector(query)
    except Exception as err:
        msg = str(err).lower()
        if "429" in msg or "resource_exhausted" in msg or "quota" in msg or "unavailable" in msg:
            mark_embedding_quota_exhausted(err)
            return []
        raise

    scored: List[RetrievedPassage] = []

    for chunk in index.chunks:
        if chunk.unit_vector is None:
            continue
        if category and category != "ALL" and chunk.category != category:
            continue
        if published_after and chunk.published_date < published_after:
            continue
        if version_number is not None and chunk.version_number != version_number:
            continue

        sim = dot_product_unit(query_unit, chunk.unit_vector)
        if sim >= threshold:
            scored.append(
                RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    version_number=chunk.version_number,
                    chunk_index=chunk.chunk_index,
                    title=chunk.title,
                    doc_type=chunk.doc_type,
                    category=chunk.category,
                    service_name=chunk.service_name,
                    reference_code=chunk.reference_code,
                    published_date=chunk.published_date,
                    valid_until=chunk.valid_until,
                    content=chunk.content,
                    vector_score=round(sim, 4),
                )
            )

    scored.sort(key=lambda p: p.vector_score or 0.0, reverse=True)
    top = scored[:limit]
    for idx, item in enumerate(top):
        item.vector_rank = idx + 1
    return top
