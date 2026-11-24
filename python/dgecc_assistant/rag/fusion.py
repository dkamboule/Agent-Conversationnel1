"""Fusion RRF (Reciprocal Rank Fusion) et re-classement des passages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..core.config import RAG_CONFIG
from ..search.bm25 import RetrievedPassage
from ..search.corpus_cache import normalize_french_token, tokenize_french


@dataclass
class FusedPassage(RetrievedPassage):
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    is_expired: bool = False


def reciprocal_rank_fusion(
    bm25_results: List[RetrievedPassage],
    vector_results: List[RetrievedPassage],
    rrf_k: Optional[int] = None,
    bm25_weight: Optional[float] = None,
    vector_weight: Optional[float] = None,
    top_k: Optional[int] = None,
    query_entities: Optional[List[str]] = None,
    query_keywords: Optional[List[str]] = None,
    enable_reranking: bool = True,
    min_rerank_score: Optional[float] = None,
    min_bm25_score: Optional[float] = None,
) -> List[FusedPassage]:
    k = rrf_k if rrf_k is not None else RAG_CONFIG.default_rrf_k
    w_bm25 = bm25_weight if bm25_weight is not None else RAG_CONFIG.bm25_weight
    w_vec = vector_weight if vector_weight is not None else RAG_CONFIG.vector_weight
    limit = top_k if top_k is not None else RAG_CONFIG.default_top_k

    merged: Dict[str, FusedPassage] = {}

    for item in bm25_results:
        rank = item.bm25_rank or 1
        contrib = w_bm25 / (k + rank)
        existing = merged.get(item.chunk_id)
        if existing:
            existing.bm25_score = item.bm25_score
            existing.bm25_rank = item.bm25_rank
            existing.rrf_score += contrib
        else:
            merged[item.chunk_id] = FusedPassage(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                version_number=item.version_number,
                chunk_index=item.chunk_index,
                title=item.title,
                doc_type=item.doc_type,
                category=item.category,
                service_name=item.service_name,
                reference_code=item.reference_code,
                published_date=item.published_date,
                valid_until=item.valid_until,
                content=item.content,
                bm25_score=item.bm25_score,
                bm25_rank=item.bm25_rank,
                vector_score=item.vector_score,
                vector_rank=item.vector_rank,
                rrf_score=contrib,
            )

    for item in vector_results:
        rank = item.vector_rank or 1
        contrib = w_vec / (k + rank)
        existing = merged.get(item.chunk_id)
        if existing:
            existing.vector_score = item.vector_score
            existing.vector_rank = item.vector_rank
            existing.rrf_score += contrib
        else:
            merged[item.chunk_id] = FusedPassage(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                version_number=item.version_number,
                chunk_index=item.chunk_index,
                title=item.title,
                doc_type=item.doc_type,
                category=item.category,
                service_name=item.service_name,
                reference_code=item.reference_code,
                published_date=item.published_date,
                valid_until=item.valid_until,
                content=item.content,
                bm25_score=item.bm25_score,
                bm25_rank=item.bm25_rank,
                vector_score=item.vector_score,
                vector_rank=item.vector_rank,
                rrf_score=contrib,
            )

    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target_terms = set()
    if enable_reranking:
        for ent in query_entities or []:
            target_terms.update(tokenize_french(ent))
        for kw in query_keywords or []:
            target_terms.update(tokenize_french(kw))

    reranked: List[FusedPassage] = []
    for item in merged.values():
        is_expired = bool(item.valid_until and item.valid_until < today_iso)
        item.is_expired = is_expired
        if not enable_reranking:
            pure_score = round(item.rrf_score, 5)
            item.rrf_score = pure_score
            item.rerank_score = pure_score
        else:
            norm_content = normalize_french_token(f"{item.title} {item.content}")
            entity_boost = sum(0.004 for term in target_terms if term in norm_content)
            freshness_factor = 0.7 if is_expired else 1.0
            dual_bonus = 0.008 if (item.bm25_rank and item.vector_rank) else 0.0

            item.rerank_score = round(
                (item.rrf_score + entity_boost + dual_bonus) * freshness_factor, 5
            )
            item.rrf_score = round(item.rrf_score, 5)

        if min_rerank_score is not None and item.rerank_score < min_rerank_score:
            continue
        if (
            min_bm25_score is not None
            and not item.vector_rank
            and (item.bm25_score or 0.0) < min_bm25_score
        ):
            continue
        reranked.append(item)

    reranked.sort(key=lambda p: p.rerank_score, reverse=True)
    return reranked[:limit]
