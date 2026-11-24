"""Recherche lexicale BM25 Okapi sur le corpus publié."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .corpus_cache import (
    get_published_corpus_index,
    normalize_french_token,
    tokenize_french,
)


@dataclass
class RetrievedPassage:
    chunk_id: str
    document_id: str
    version_number: int
    chunk_index: int
    title: str
    doc_type: str
    category: str
    service_name: str
    reference_code: str
    published_date: str
    valid_until: Optional[str]
    content: str
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    vector_score: Optional[float] = None
    vector_rank: Optional[int] = None


def bm25_search(
    query: str,
    category: Optional[str] = None,
    published_after: Optional[str] = None,
    version_number: Optional[int] = None,
    limit: int = 12,
) -> List[RetrievedPassage]:
    index = get_published_corpus_index()
    if not index.chunks:
        return []

    query_tokens = tokenize_french(query)
    if not query_tokens:
        return []

    k1 = 1.5
    b = 0.75
    n_docs = len(index.chunks)
    normalized_query = normalize_french_token(query)
    scored: List[RetrievedPassage] = []

    for chunk in index.chunks:
        if category and category != "ALL" and chunk.category != category:
            continue
        if published_after and chunk.published_date < published_after:
            continue
        if version_number is not None and chunk.version_number != version_number:
            continue

        score = 0.0
        for qt in query_tokens:
            freq = chunk.tf.get(qt, 0)
            if freq == 0:
                continue
            df = index.doc_freq.get(qt, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            num = freq * (k1 + 1.0)
            den = freq + k1 * (1.0 - b + b * (chunk.token_length / max(1.0, index.avg_doc_len)))
            score += idf * (num / den)

        if chunk.reference_norm and chunk.reference_norm in normalized_query:
            score += 4.5

        if score > 0.0:
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
                    bm25_score=round(score, 4),
                )
            )

    scored.sort(key=lambda p: p.bm25_score or 0.0, reverse=True)
    top = scored[:limit]
    for idx, item in enumerate(top):
        item.bm25_rank = idx + 1
    return top
