"""Orchestrateur du pipeline RAG hybride (NLP -> BM25+Vecteur -> RRF -> LLM -> Garde-fous)."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ..core.config import RAG_CONFIG
from ..core.security import detect_prompt_injection
from ..database.db import get_db, log_activity
from ..guardrails.verifier import (
    GuardrailCheckItem,
    GuardrailsReport,
    build_refusal_message,
    verify_generated_response,
)
from ..llm.generator import get_llm_provider
from ..memory.memory_manager import load_conversation_memory
from ..nlp.analyzer import NlpAnalysisResult, analyze_user_question
from ..search.bm25 import bm25_search
from ..search.corpus_cache import normalize_french_token, on_corpus_invalidated
from ..search.vector import vector_search
from .context_builder import build_rag_prompt_context
from .fusion import FusedPassage, reciprocal_rank_fusion

_MAX_RESPONSE_CACHE = 100
_standalone_response_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

on_corpus_invalidated(lambda: _standalone_response_cache.clear())


@dataclass
class RagQueryRequest:
    question: str
    conversation_id: Optional[str] = None
    category_filter: Optional[str] = None
    published_after: Optional[str] = None
    version_number: Optional[int] = None
    top_k: Optional[int] = None
    rrf_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    guest_session_id: Optional[str] = None
    on_stage: Optional[Callable[[str, Any], None]] = None
    on_token: Optional[Callable[[str], None]] = None


def _format_sources(passages: List[FusedPassage]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for p in passages:
        excerpt = re.sub(r"^\[[^\]]+\]\n", "", p.content)[:380]
        formatted.append(
            {
                "chunkId": p.chunk_id,
                "documentId": p.document_id,
                "title": p.title,
                "referenceCode": p.reference_code,
                "docType": p.doc_type,
                "category": p.category,
                "serviceName": p.service_name,
                "publishedDate": p.published_date,
                "validUntil": p.valid_until,
                "versionNumber": p.version_number,
                "excerpt": excerpt,
                "bm25Score": p.bm25_score,
                "bm25Rank": p.bm25_rank,
                "vectorScore": p.vector_score,
                "vectorRank": p.vector_rank,
                "rrfScore": p.rrf_score,
                "rerankScore": p.rerank_score,
                "isExpired": p.is_expired,
            }
        )
    return formatted


def run_hybrid_rag_pipeline(req: RagQueryRequest) -> Dict[str, Any]:
    total_start = time.time()
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()

    conversation_id = req.conversation_id
    if conversation_id:
        existing = db.execute(
            "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if existing is None:
            conversation_id = None

    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        title = req.question.strip()[:72] or "Nouvelle demande DGECC"
        db.execute(
            """
            INSERT INTO conversations (id, user_id, guest_session_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                req.user_id,
                None if req.user_id else (req.guest_session_id or "anonymous"),
                title,
                now,
                now,
            ),
        )
    else:
        db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )

    user_msg_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, 'user', ?, ?)",
        (user_msg_id, conversation_id, req.question.strip(), now),
    )
    db.commit()

    flagged, reason = detect_prompt_injection(req.question)
    if flagged:
        refusal = build_refusal_message(
            "Direction Générale — Secrétariat du Directeur Général",
            "demande contenant des instructions incompatibles avec le cadre réglementaire",
        )
        guardrails = GuardrailsReport(
            passed=False,
            refused=True,
            confidence_level="FAIBLE",
            has_expired_sources=False,
            oriented_service="Direction Générale — Secrétariat du Directeur Général",
            checks=[
                GuardrailCheckItem(
                    name="Sécurité / Anti-injection de prompt",
                    passed=False,
                    detail=reason or "Motif bloqué.",
                )
            ],
        )
        analysis = NlpAnalysisResult(
            intent="HORS_PERIMETRE",
            needs_retrieval=False,
            reformulated_query=req.question,
            suggested_service="Direction Générale — Secrétariat du Directeur Général",
        )
        assistant_msg_id = str(uuid.uuid4())
        metrics = {
            "nlpMs": 0,
            "retrievalMs": 0,
            "generationMs": 0,
            "guardrailsMs": 1,
            "totalMs": int((time.time() - total_start) * 1000),
            "bm25Hits": 0,
            "vectorHits": 0,
            "fusedCount": 0,
        }
        db.execute(
            """
            INSERT INTO messages (
              id, conversation_id, role, content, nlp_analysis_json, sources_json, guardrails_json, metrics_json, created_at
            ) VALUES (?, ?, 'assistant', ?, ?, '[]', ?, ?, ?)
            """,
            (
                assistant_msg_id,
                conversation_id,
                refusal,
                json.dumps(analysis.to_dict(), ensure_ascii=False),
                json.dumps(guardrails.to_dict(), ensure_ascii=False),
                json.dumps(metrics),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        db.commit()
        if req.on_token:
            req.on_token(refusal)
        log_activity(
            action="GUARDRAIL_BLOCKED_INJECTION",
            resource_type="conversation",
            user_id=req.user_id,
            user_email=req.user_email,
            resource_id=conversation_id,
            details={"reason": reason},
        )
        return {
            "conversationId": conversation_id,
            "messageId": assistant_msg_id,
            "answer": refusal,
            "analysis": analysis.to_dict(),
            "sources": [],
            "guardrails": guardrails.to_dict(),
            "metrics": metrics,
        }

    memory = load_conversation_memory(
        conversation_id, user_id=req.user_id, guest_session_id=req.guest_session_id
    )

    cache_key = (
        f"{req.category_filter or 'ALL'}|"
        f"{req.top_k if req.top_k is not None else RAG_CONFIG.default_top_k}|"
        f"{normalize_french_token(req.question.strip())}"
    )
    if not memory.recent_turns and not req.published_after and req.version_number is None:
        cached = _standalone_response_cache.get(cache_key)
        if cached:
            _standalone_response_cache.move_to_end(cache_key)
            if req.on_token:
                req.on_token(cached["answer"])
            total_ms = max(1, int((time.time() - total_start) * 1000))
            metrics = {
                "nlpMs": 0,
                "retrievalMs": 0,
                "generationMs": 0,
                "guardrailsMs": 0,
                "totalMs": total_ms,
                "bm25Hits": cached["bm25Hits"],
                "vectorHits": cached["vectorHits"],
                "fusedCount": len(cached["sources"]),
            }
            assistant_msg_id = str(uuid.uuid4())
            db.execute(
                """
                INSERT INTO messages (
                  id, conversation_id, role, content, nlp_analysis_json, sources_json, guardrails_json, metrics_json, created_at
                ) VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?)
                """,
                (
                    assistant_msg_id,
                    conversation_id,
                    cached["answer"],
                    json.dumps(cached["analysis"], ensure_ascii=False),
                    json.dumps(cached["sources"], ensure_ascii=False),
                    json.dumps(cached["guardrails"], ensure_ascii=False),
                    json.dumps(metrics),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            db.commit()
            return {
                "conversationId": conversation_id,
                "messageId": assistant_msg_id,
                "answer": cached["answer"],
                "analysis": cached["analysis"],
                "sources": cached["sources"],
                "guardrails": cached["guardrails"],
                "metrics": metrics,
            }

    if req.on_stage:
        req.on_stage("nlp", "Analyse et recherche hybride...")
    nlp_start = time.time()
    analysis = analyze_user_question(req.question, memory)
    nlp_ms = int((time.time() - nlp_start) * 1000)

    bm25_hits = []
    vector_hits = []
    fused_passages: List[FusedPassage] = []
    retrieval_ms = 0

    if analysis.needs_retrieval:
        ret_start = time.time()
        search_query = analysis.reformulated_query or req.question
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_bm25 = pool.submit(
                bm25_search,
                search_query,
                req.category_filter,
                req.published_after,
                req.version_number,
                12,
            )
            f_vec = pool.submit(
                vector_search,
                search_query,
                req.category_filter,
                req.published_after,
                req.version_number,
                req.similarity_threshold,
                12,
            )
            bm25_hits = f_bm25.result()
            vector_hits = f_vec.result()

        fused_passages = reciprocal_rank_fusion(
            bm25_hits,
            vector_hits,
            rrf_k=req.rrf_k,
            top_k=req.top_k,
            query_entities=analysis.entities,
            query_keywords=analysis.keywords,
        )
        retrieval_ms = int((time.time() - ret_start) * 1000)

    if req.on_stage:
        req.on_stage("generation", "Rédaction de la réponse sourcée...")
    gen_start = time.time()
    llm = get_llm_provider()

    if analysis.needs_retrieval and not fused_passages:
        answer = build_refusal_message(
            analysis.suggested_service,
            "aucun document institutionnel publié et validé ne correspond suffisamment à cette question",
        )
        if req.on_token:
            req.on_token(answer)
    elif not analysis.needs_retrieval and analysis.intent == "HORS_PERIMETRE":
        answer = build_refusal_message(
            "Direction Générale — Secrétariat du Directeur Général",
            "la question sort du périmètre de compétences de la Direction Générale des Examens, Concours et de la Certification",
        )
        if req.on_token:
            req.on_token(answer)
    elif not analysis.needs_retrieval and analysis.intent == "SALUTATION":
        answer = (
            "Bonjour. Je suis l'assistant d'orientation de la **DGECC**. "
            "Posez votre question sur les examens, concours ou certifications. "
            "En cas de doute, seule l'administration fait foi."
        )
        if req.on_token:
            req.on_token(answer)
    else:
        prompt = build_rag_prompt_context(
            question=req.question,
            analysis=analysis,
            memory=memory,
            passages=fused_passages,
        )
        answer = (
            llm.generate_stream(prompt, req.on_token)
            if req.on_token
            else llm.generate(prompt)
        )
    generation_ms = int((time.time() - gen_start) * 1000)

    guard_start = time.time()
    guardrails = verify_generated_response(
        question=req.question,
        answer=answer,
        analysis=analysis,
        passages=fused_passages,
    )

    if guardrails.refused and fused_passages:
        strict_prompt = (
            build_rag_prompt_context(
                question=req.question,
                analysis=analysis,
                memory=memory,
                passages=fused_passages,
            )
            + "\n\nAVERTISSEMENT CONTRÔLE : Reformule uniquement avec les faits littéraux "
            "présents dans les extraits ci-dessus, en citant les références exactes."
        )
        regenerated = llm.generate(strict_prompt)
        recheck = verify_generated_response(
            question=req.question,
            answer=regenerated,
            analysis=analysis,
            passages=fused_passages,
        )
        if not recheck.refused:
            answer = regenerated
            guardrails = recheck
        else:
            answer = build_refusal_message(
                guardrails.oriented_service,
                "les vérifications de conformité documentaire n’ont pas pu confirmer chaque affirmation",
            )
            guardrails = recheck

    guardrails_ms = int((time.time() - guard_start) * 1000)
    total_ms = int((time.time() - total_start) * 1000)
    formatted_sources = _format_sources(fused_passages)

    metrics = {
        "nlpMs": nlp_ms,
        "retrievalMs": retrieval_ms,
        "generationMs": generation_ms,
        "guardrailsMs": guardrails_ms,
        "totalMs": total_ms,
        "bm25Hits": len(bm25_hits),
        "vectorHits": len(vector_hits),
        "fusedCount": len(fused_passages),
    }

    assistant_msg_id = str(uuid.uuid4())
    db.execute(
        """
        INSERT INTO messages (
          id, conversation_id, role, content, nlp_analysis_json, sources_json, guardrails_json, metrics_json, created_at
        ) VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?)
        """,
        (
            assistant_msg_id,
            conversation_id,
            answer,
            json.dumps(analysis.to_dict(), ensure_ascii=False),
            json.dumps(formatted_sources, ensure_ascii=False),
            json.dumps(guardrails.to_dict(), ensure_ascii=False),
            json.dumps(metrics),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    db.commit()

    if (
        not memory.recent_turns
        and not req.published_after
        and req.version_number is None
        and not guardrails.refused
        and analysis.needs_retrieval
    ):
        if len(_standalone_response_cache) >= _MAX_RESPONSE_CACHE:
            _standalone_response_cache.popitem(last=False)
        _standalone_response_cache[cache_key] = {
            "answer": answer,
            "analysis": analysis.to_dict(),
            "sources": formatted_sources,
            "guardrails": guardrails.to_dict(),
            "bm25Hits": len(bm25_hits),
            "vectorHits": len(vector_hits),
        }

    return {
        "conversationId": conversation_id,
        "messageId": assistant_msg_id,
        "answer": answer,
        "analysis": analysis.to_dict(),
        "sources": formatted_sources,
        "guardrails": guardrails.to_dict(),
        "metrics": metrics,
    }
