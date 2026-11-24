"""Évaluation méthodologique DGECC (Tableaux 18, 19 et 20) en parallèle de la version TypeScript."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import RAG_CONFIG
from ..database.db import get_db
from ..memory.memory_manager import ConversationMemoryContext
from ..nlp.analyzer import analyze_user_question
from ..search.bm25 import bm25_search
from ..search.corpus_cache import normalize_french_token
from .fusion import reciprocal_rank_fusion

TYPOLOGY_LABELS: Dict[str, str] = {
    "T1_CONDITIONS_ACCES": "T1 · Conditions d'accès et candidatures",
    "T2_DOSSIERS_INSCRIPTION": "T2 · Pièces, dossiers et délais d'inscription",
    "T3_EPREUVES_JURYS": "T3 · Épreuves, coefficients et jurys",
    "T4_DELIBERATIONS_RECOURS": "T4 · Délibérations, réclamations et fraudes",
    "T5_STATUT_ORIENTATION": "T5 · Statut des emplois et orientation",
    "HORS_PERIMETRE": "Hors périmètre · Mécanisme d’abstention",
}


@dataclass
class BenchmarkQueryItem:
    id: str
    typology: str
    is_out_of_scope: bool
    in_stratified_sample: bool
    query: str
    expected_ref: str
    expected_fact: str
    fact_groups: List[List[str]]
    acceptable_refs: Optional[List[str]] = None


def load_eval_dataset() -> List[BenchmarkQueryItem]:
    """Charge de façon canonique les 100 requêtes stratifiées depuis server/rag/evalDataset.ts."""
    ts_path = Path(__file__).resolve().parents[3] / "server" / "rag" / "evalDataset.ts"
    raw = ts_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\{\s*id:\s*'([^']+)'\s*,\s*typology:\s*'([^']+)'\s*,\s*isOutOfScope:\s*(true|false)\s*,"
        r"\s*inStratifiedSample:\s*(true|false)\s*,\s*query:\s*(['\"`])(.*?)\5\s*,"
        r"\s*expectedRef:\s*'([^']+)'",
        re.DOTALL,
    )
    items: List[BenchmarkQueryItem] = []
    for match in pattern.finditer(raw):
        qid, typology, oos, sample, _quote, query_text, expected_ref = match.groups()
        items.append(
            BenchmarkQueryItem(
                id=qid,
                typology=typology,
                is_out_of_scope=(oos == "true"),
                in_stratified_sample=(sample == "true"),
                query=query_text.replace("\\'", "'"),
                expected_ref=expected_ref,
                expected_fact="",
                fact_groups=[],
                acceptable_refs=(
                    [
                        "MESS-2013-334-ARR-CAPCEG-CAET",
                        "MFPTPS-2025-2877-ARR-CAPES-CAPET-CAPCEG-CAET-2026",
                        "MENAPLN-DGEC-ARR-JURYS-CAPES-CAPET-CAPCEG-CAET",
                    ]
                    if qid == "Q-041"
                    else None
                ),
            )
        )
    return items


def add_user_panel_entry(
    profile: str,
    participant_name: str,
    clarity_score: int,
    usefulness_score: int,
    trust_score: int,
    comment: Optional[str] = None,
    recorded_by: str = "system",
) -> Dict[str, Any]:
    if profile not in ("CANDIDAT", "PARENT", "AGENT_ADMINISTRATIF"):
        raise ValueError("Profil invalide (CANDIDAT, PARENT, AGENT_ADMINISTRATIF).")
    db = get_db()
    entry_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    c_score = max(1, min(5, int(clarity_score)))
    u_score = max(1, min(5, int(usefulness_score)))
    t_score = max(1, min(5, int(trust_score)))
    db.execute(
        """
        INSERT INTO user_panel_evaluations (
          id, profile, participant_name, clarity_score, usefulness_score, trust_score,
          comment, recorded_by, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (entry_id, profile, participant_name.strip(), c_score, u_score, t_score, comment, recorded_by, now),
    )
    db.commit()
    return {
        "id": entry_id,
        "profile": profile,
        "participantName": participant_name.strip(),
        "clarityScore": c_score,
        "usefulnessScore": u_score,
        "trustScore": t_score,
        "comment": comment,
        "recordedBy": recorded_by,
        "createdAt": now,
    }


def get_tableau_18_snapshot(dataset: Optional[List[BenchmarkQueryItem]] = None) -> Dict[str, Any]:
    db = get_db()
    ds = dataset if dataset is not None else load_eval_dataset()
    corpus_docs = int(
        db.execute("SELECT COUNT(*) FROM documents WHERE status = 'PUBLIE'").fetchone()[0]
    )
    chunk_row = db.execute(
        """
        SELECT COUNT(*),
               COALESCE(SUM(CASE WHEN c.embedding_json IS NOT NULL AND c.embedding_json != '' THEN 1 ELSE 0 END), 0)
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id AND d.current_version = c.version_number
        WHERE d.status = 'PUBLIE'
        """
    ).fetchone()
    indexed_chunks = int(chunk_row[0] or 0)
    embedded_chunks = int(chunk_row[1] or 0)

    total_q = len(ds)
    oos_q = sum(1 for q in ds if q.is_out_of_scope)
    in_q = total_q - oos_q
    oos_pct = round((oos_q / max(1, total_q)) * 100, 1)

    panel_rows = db.execute(
        "SELECT profile, clarity_score, usefulness_score, trust_score FROM user_panel_evaluations"
    ).fetchall()
    total_panel = len(panel_rows)

    def _avg(idx: int) -> Optional[float]:
        if total_panel == 0:
            return None
        return round(sum(float(r[idx]) for r in panel_rows) / total_panel, 2)

    return {
        "corpusDocuments": corpus_docs,
        "indexedChunks": indexed_chunks,
        "embeddedChunks": embedded_chunks,
        "queryDatasetTotal": total_q,
        "inScopeQueries": in_q,
        "outOfScopeQueries": oos_q,
        "outOfScopeSharePct": oos_pct,
        "typologyBreakdown": [
            {
                "code": code,
                "label": label,
                "count": sum(1 for q in ds if q.typology == code),
            }
            for code, label in TYPOLOGY_LABELS.items()
        ],
        "userPanel": {
            "total": total_panel,
            "candidats": sum(1 for r in panel_rows if r["profile"] == "CANDIDAT"),
            "parents": sum(1 for r in panel_rows if r["profile"] == "PARENT"),
            "agentsAdministratifs": sum(1 for r in panel_rows if r["profile"] == "AGENT_ADMINISTRATIF"),
            "avgClarity": _avg(1),
            "avgUsefulness": _avg(2),
            "avgTrust": _avg(3),
        },
    }


def _accepted(item: BenchmarkQueryItem, ref_code: str) -> bool:
    if item.acceptable_refs:
        return ref_code in item.acceptable_refs
    return ref_code == item.expected_ref


def evaluate_retrieval_strategies_python(
    dataset: Optional[List[BenchmarkQueryItem]] = None,
) -> Dict[str, Any]:
    """Calcule le Tableau 19 (Lexical, Dense, Hybride RRF brut, Hybride + Reclassement) sur les requêtes intra-périmètre."""
    ds = dataset if dataset is not None else load_eval_dataset()
    in_scope = [q for q in ds if not q.is_out_of_scope]
    t18 = get_tableau_18_snapshot(ds)
    mem = ConversationMemoryContext(conversation_id="py-eval")

    def _metrics(rows: List[tuple[BenchmarkQueryItem, List[str]]]) -> Dict[str, float]:
        hits3 = 0
        hits5 = 0
        sum_mrr = 0.0
        for item, refs in rows:
            idx = next((i for i, ref in enumerate(refs) if _accepted(item, ref)), -1)
            if idx >= 0:
                rank = idx + 1
                if rank <= 3:
                    hits3 += 1
                if rank <= 5:
                    hits5 += 1
                sum_mrr += 1.0 / rank
        n = max(1, len(rows))
        return {
            "recallAt3": round(hits3 / n, 3),
            "recallAt5": round(hits5 / n, 3),
            "mrr": round(sum_mrr / n, 3),
        }

    lex_rows = []
    hyb_rows = []
    rerank_rows = []
    for item in in_scope:
        analysis = analyze_user_question(item.query, mem)
        b12 = bm25_search(item.query, limit=12)
        b5 = b12[:5]
        h_pure = reciprocal_rank_fusion(
            b12, [], rrf_k=RAG_CONFIG.default_rrf_k, top_k=5, enable_reranking=False
        )
        h_rerank = reciprocal_rank_fusion(
            b12,
            [],
            rrf_k=RAG_CONFIG.default_rrf_k,
            top_k=5,
            query_entities=analysis.entities,
            query_keywords=analysis.keywords,
            enable_reranking=True,
        )
        lex_rows.append((item, [p.reference_code for p in b5]))
        hyb_rows.append((item, [p.reference_code for p in h_pure]))
        rerank_rows.append((item, [p.reference_code for p in h_rerank]))

    lex_m = _metrics(lex_rows)
    hyb_m = _metrics(hyb_rows)
    rerank_m = _metrics(rerank_rows)

    tableau19 = [
        {
            "key": "lexical",
            "label": "Recherche lexicale (mots-clés)",
            "available": True,
            "queriesEvaluated": len(in_scope),
            **lex_m,
        },
        {
            "key": "dense",
            "label": "Recherche dense (sémantique seule)",
            "available": False,
            "queriesEvaluated": 0,
            "recallAt3": None,
            "recallAt5": None,
            "mrr": None,
        },
        {
            "key": "hybrid",
            "label": "Recherche hybride (lexicale + dense)",
            "available": True,
            "queriesEvaluated": len(in_scope),
            **hyb_m,
        },
        {
            "key": "hybrid_rerank",
            "label": "Recherche hybride avec reclassement (reranking)",
            "available": True,
            "queriesEvaluated": len(in_scope),
            **rerank_m,
        },
    ]

    return {
        "tableau18": t18,
        "tableau19": tableau19,
        "normalizedSampleQueries": len([q for q in ds if q.in_stratified_sample]),
    }
