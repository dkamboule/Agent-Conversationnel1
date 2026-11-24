"""Utilitaire d'export de la base documentaire (vecteurs + métadonnées) et du squelette RAG hybride DGECC.

Usage:
    python3 python/scripts/export_rag_bundle.py \
        --db-path data/dgecc.db \
        --output-dir python/exports/dgecc_rag_bundle \
        --zip
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Permet l'exécution directe via `python3 python/scripts/export_rag_bundle.py`
_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from dgecc_assistant.core.config import APP_CONFIG, RAG_CONFIG  # noqa: E402
from dgecc_assistant.core.dgecc_org import (  # noqa: E402
    DGECC_ORG_STRUCTURE,
    DGECC_REGULATORY_DOMAINS,
    DGECC_SERVICES_LIST,
)
from dgecc_assistant.rag.context_builder import DGECC_SYSTEM_INSTRUCTION  # noqa: E402

_SKELETON_TEMPLATE = '''"""Squelette autonome du moteur RAG hybride configuré pour la DGECC.

Généré automatiquement par `export_rag_bundle.py`.
Charge la configuration (`config/rag_config.json`), les métadonnées et les vecteurs
(`corpus/chunks_vectors.jsonl`) pour exécuter une recherche hybride BM25 + Vecteurs + RRF
suivie d'une génération encadrée par Gemini et les garde-fous DGECC.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FRENCH_STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en", "au", "aux",
    "pour", "par", "sur", "dans", "avec", "sans", "sous", "vers", "chez", "entre",
    "ce", "cet", "cette", "ces", "mon", "ton", "son", "notre", "votre", "leur",
    "qui", "que", "quoi", "dont", "ou", "il", "elle", "ils", "elles", "nous", "vous",
    "est", "sont", "etre", "avoir", "fait", "faire", "peut", "doit", "plus", "moins",
    "tout", "tous", "toute", "toutes", "comme", "mais", "donc", "car", "si", "ni",
    "se", "ne", "pas", "qu", "d", "l", "n", "s", "c", "j", "m", "t", "y", "a",
}

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}(?:-[a-z0-9]+)*")


def normalize_french_token(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def tokenize_french(text: str) -> List[str]:
    norm = normalize_french_token(text)
    return [tok for tok in _TOKEN_RE.findall(norm) if tok not in FRENCH_STOPWORDS]


def to_unit_vector(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        return [0.0 for _ in vec]
    return [v / norm for v in vec]


def dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    reference_code: str
    title: str
    doc_type: str
    category: str
    service_name: str
    published_date: str
    valid_until: Optional[str]
    version_number: int
    chunk_index: int
    content: str
    unit_vector: Optional[List[float]] = None
    tf: Dict[str, int] = field(default_factory=dict)
    token_length: int = 0


@dataclass
class ScoredPassage:
    record: ChunkRecord
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    vector_score: Optional[float] = None
    vector_rank: Optional[int] = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    is_expired: bool = False


class DgeccOrganigrammeRouter:
    """Routage NLP déterministe vers le service compétent de l'organigramme DGECC."""

    @staticmethod
    def route_question(question: str) -> Dict[str, Any]:
        norm = normalize_french_token(question.strip())
        mentions_bac = bool(re.search(r"\\b(baccalaureat|bac)\\b", norm))
        mentions_tech_pro = bool(re.search(r"\\b(technique|professionnel|cap|bt|bep)\\b", norm))
        mentions_bepc = bool(re.search(r"\\bbepc\\b", norm))

        if re.search(r"\\b(tiers-temps|handicap|amenagement|braille)\\b", norm):
            service = "DEPE — Service des études, de l'évaluation et de la prospection"
            intent = "AMENAGEMENT_HANDICAP"
        elif re.search(r"\\b(reclamation|recours|contester|copie|jury)\\b", norm):
            service = "Direction Générale — Contrôle Interne (CI)"
            intent = "RESULTATS_CONTESTATION"
        elif re.search(r"\\b(duplicata|attestation|releve\\s+de\\s+notes)\\b", norm):
            service = (
                "DEB — Service des attestations et diplômes"
                if mentions_bac
                else "DECEPS — Service des attestations et diplômes"
            )
            intent = "EQUIVALENCE_CERTIFICATION"
        elif re.search(r"\\b(equivalence|certification\\s+professionnelle|dcp|vae)\\b", norm):
            service = "Direction des Certifications Professionnelles (DCP)"
            intent = "EQUIVALENCE_CERTIFICATION"
        elif re.search(r"\\b(frais|paiement|quittance|recepisse|cout|coute|tarif)\\b", norm):
            service = "Direction Générale — Services Financier (SF)"
            intent = "INSCRIPTION_PROCEDURE"
        elif mentions_bac:
            service = (
                "DEB — Service du baccalauréat de l'enseignement technique professionnel"
                if mentions_tech_pro
                else "DEB — Service du baccalauréat de l'enseignement général"
            )
            intent = "CALENDRIER_EXAMEN"
        elif mentions_bepc:
            service = "DECEPS — Service des examens de l'enseignement post-primaire général"
            intent = "INSCRIPTION_PROCEDURE"
        elif mentions_tech_pro:
            service = "DECEPS — Service des examens de l'enseignement technique professionnel"
            intent = "INSCRIPTION_PROCEDURE"
        else:
            service = "Direction Générale — Secrétariat du Directeur Général"
            intent = "ORIENTATION_SERVICE"

        return {
            "intent": intent,
            "suggestedService": service,
            "keywords": tokenize_french(question),
        }


class DgeccHybridRagEngine:
    """Squelette complet du moteur RAG hybride (BM25 + Vecteurs + RRF + Garde-fous)."""

    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()
        config_path = self.bundle_dir / "config" / "rag_config.json"
        chunks_path = self.bundle_dir / "corpus" / "chunks_vectors.jsonl"

        self.config: Dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        self.rag_params: Dict[str, Any] = self.config["ragConfig"]
        self.system_instruction: str = self.config["systemInstruction"]

        self.chunks: List[ChunkRecord] = []
        self.doc_freq: Dict[str, int] = {}
        self.avg_doc_len: float = 1.0
        self._load_corpus(chunks_path)

    def _load_corpus(self, chunks_path: Path) -> None:
        total_len = 0
        with chunks_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                toks = tokenize_french(
                    f"{row['title']} {row['referenceCode']} {row['content']}"
                )
                total_len += len(toks)
                tf: Dict[str, int] = {}
                seen = set()
                for tok in toks:
                    tf[tok] = tf.get(tok, 0) + 1
                    if tok not in seen:
                        seen.add(tok)
                        self.doc_freq[tok] = self.doc_freq.get(tok, 0) + 1

                raw_vec = row.get("embedding")
                unit_vec = (
                    to_unit_vector([float(x) for x in raw_vec])
                    if isinstance(raw_vec, list) and raw_vec
                    else None
                )

                self.chunks.append(
                    ChunkRecord(
                        chunk_id=row["chunkId"],
                        document_id=row["documentId"],
                        reference_code=row["referenceCode"],
                        title=row["title"],
                        doc_type=row["docType"],
                        category=row["category"],
                        service_name=row["serviceName"],
                        published_date=row["publishedDate"],
                        valid_until=row.get("validUntil"),
                        version_number=int(row["versionNumber"]),
                        chunk_index=int(row["chunkIndex"]),
                        content=row["content"],
                        unit_vector=unit_vec,
                        tf=tf,
                        token_length=len(toks),
                    )
                )
        self.avg_doc_len = total_len / max(1, len(self.chunks))

    def bm25_search(
        self,
        query: str,
        category_filter: Optional[str] = None,
        limit: int = 12,
    ) -> List[ScoredPassage]:
        if not self.chunks:
            return []
        q_tokens = tokenize_french(query)
        if not q_tokens:
            return []

        k1, b = 1.5, 0.75
        n_docs = len(self.chunks)
        norm_q = normalize_french_token(query)
        hits: List[ScoredPassage] = []

        for rec in self.chunks:
            if category_filter and category_filter != "ALL" and rec.category != category_filter:
                continue
            score = 0.0
            for qt in q_tokens:
                freq = rec.tf.get(qt, 0)
                if freq == 0:
                    continue
                df = self.doc_freq.get(qt, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                den = freq + k1 * (1.0 - b + b * (rec.token_length / max(1.0, self.avg_doc_len)))
                score += idf * ((freq * (k1 + 1.0)) / den)
            if normalize_french_token(rec.reference_code) in norm_q:
                score += 4.5
            if score > 0.0:
                hits.append(ScoredPassage(record=rec, bm25_score=round(score, 4)))

        hits.sort(key=lambda h: h.bm25_score or 0.0, reverse=True)
        top = hits[:limit]
        for idx, item in enumerate(top, start=1):
            item.bm25_rank = idx
        return top

    def vector_search(
        self,
        query_vector: List[float],
        category_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        limit: int = 12,
    ) -> List[ScoredPassage]:
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else float(self.rag_params["defaultSimilarityThreshold"])
        )
        q_unit = to_unit_vector(query_vector)
        hits: List[ScoredPassage] = []
        for rec in self.chunks:
            if rec.unit_vector is None:
                continue
            if category_filter and category_filter != "ALL" and rec.category != category_filter:
                continue
            sim = dot_product(q_unit, rec.unit_vector)
            if sim >= threshold:
                hits.append(ScoredPassage(record=rec, vector_score=round(sim, 4)))

        hits.sort(key=lambda h: h.vector_score or 0.0, reverse=True)
        top = hits[:limit]
        for idx, item in enumerate(top, start=1):
            item.vector_rank = idx
        return top

    def fuse_rrf(
        self,
        bm25_hits: List[ScoredPassage],
        vector_hits: List[ScoredPassage],
        query_keywords: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[ScoredPassage]:
        rrf_k = int(self.rag_params["defaultRrfK"])
        w_bm25 = float(self.rag_params["bm25Weight"])
        w_vec = float(self.rag_params["vectorWeight"])
        limit = top_k if top_k is not None else int(self.rag_params["defaultTopK"])

        merged: Dict[str, ScoredPassage] = {}
        for item in bm25_hits:
            rank = item.bm25_rank or 1
            cid = item.record.chunk_id
            merged[cid] = ScoredPassage(
                record=item.record,
                bm25_score=item.bm25_score,
                bm25_rank=item.bm25_rank,
                rrf_score=w_bm25 / (rrf_k + rank),
            )
        for item in vector_hits:
            rank = item.vector_rank or 1
            cid = item.record.chunk_id
            contrib = w_vec / (rrf_k + rank)
            if cid in merged:
                merged[cid].vector_score = item.vector_score
                merged[cid].vector_rank = item.vector_rank
                merged[cid].rrf_score += contrib
            else:
                merged[cid] = ScoredPassage(
                    record=item.record,
                    vector_score=item.vector_score,
                    vector_rank=item.vector_rank,
                    rrf_score=contrib,
                )

        today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        kw_set = set(query_keywords or [])
        ranked: List[ScoredPassage] = []
        for item in merged.values():
            norm_text = normalize_french_token(f"{item.record.title} {item.record.content}")
            boost = sum(0.004 for kw in kw_set if kw in norm_text)
            expired = bool(item.record.valid_until and item.record.valid_until < today_iso)
            freshness = 0.7 if expired else 1.0
            dual = 0.008 if (item.bm25_rank and item.vector_rank) else 0.0
            item.is_expired = expired
            item.rerank_score = round((item.rrf_score + boost + dual) * freshness, 5)
            item.rrf_score = round(item.rrf_score, 5)
            ranked.append(item)

        ranked.sort(key=lambda p: p.rerank_score, reverse=True)
        return ranked[:limit]

    def build_prompt(
        self,
        question: str,
        routing: Dict[str, Any],
        passages: List[ScoredPassage],
    ) -> str:
        blocks = []
        for idx, p in enumerate(passages, start=1):
            r = p.record
            blocks.append(
                "\\n".join(
                    [
                        f"### EXTRAIT {idx}",
                        f"Document : {r.title} ({r.reference_code})",
                        f"Service : {r.service_name} | Date : {r.published_date}",
                        "Contenu :",
                        r.content,
                    ]
                )
            )
        passages_str = "\\n\\n---\\n\\n".join(blocks) if blocks else "AUCUN EXTRAIT PERTINENT."
        return (
            f"Service concerné pressenti : {routing['suggestedService']}\\n\\n"
            f"EXTRAITS FOURNIS :\\n{passages_str}\\n\\n"
            f"QUESTION DE L'USAGER :\\n{question}"
        )

    def retrieve(
        self,
        question: str,
        category_filter: Optional[str] = None,
        use_gemini_embeddings: bool = True,
    ) -> Tuple[Dict[str, Any], List[ScoredPassage]]:
        routing = DgeccOrganigrammeRouter.route_question(question)
        bm25_hits = self.bm25_search(question, category_filter=category_filter)
        vector_hits: List[ScoredPassage] = []

        has_vectors = any(rec.unit_vector is not None for rec in self.chunks)
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if use_gemini_embeddings and has_vectors and api_key:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(headers={"User-Agent": "aistudio-build"}),
                )
                emb_res = client.models.embed_content(
                    model=self.rag_params["embeddingModel"],
                    contents=question,
                )
                q_vec = [float(v) for v in emb_res.embeddings[0].values]
                vector_hits = self.vector_search(q_vec, category_filter=category_filter)
            except Exception:
                vector_hits = []

        fused = self.fuse_rrf(
            bm25_hits,
            vector_hits,
            query_keywords=routing["keywords"],
        )
        return routing, fused

    def answer(self, question: str, category_filter: Optional[str] = None) -> Dict[str, Any]:
        routing, passages = self.retrieve(question, category_filter=category_filter)
        if not passages:
            return {
                "answer": (
                    "Les extraits fournis ne contiennent pas la réponse.\\n"
                    f"Veuillez contacter le service concerné : {routing['suggestedService']}.\\n"
                    "En cas de doute, seule l'administration fait foi."
                ),
                "routing": routing,
                "sources": [],
            }

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY absente de l'environnement serveur. "
                "Définissez GEMINI_API_KEY pour activer la génération."
            )

        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(headers={"User-Agent": "aistudio-build"}),
        )
        prompt = self.build_prompt(question, routing, passages)
        model_name = self.rag_params["llmModel"]
        thinking = (
            types.ThinkingLevel.MINIMAL
            if model_name == "gemini-3.1-flash-lite"
            else types.ThinkingLevel.LOW
        )
        res = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                thinking_config=types.ThinkingConfig(thinking_level=thinking),
            ),
        )
        return {
            "answer": (res.text or "").strip(),
            "routing": routing,
            "sources": [
                {
                    "referenceCode": p.record.reference_code,
                    "title": p.record.title,
                    "publishedDate": p.record.published_date,
                    "rerankScore": p.rerank_score,
                }
                for p in passages
            ],
        }


if __name__ == "__main__":
    import sys

    bundle = Path(__file__).resolve().parents[1]
    engine = DgeccHybridRagEngine(bundle)
    q = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Quelles sont les dates des épreuves écrites du Baccalauréat 2026 ?"
    )
    routing_info, top_passages = engine.retrieve(q, use_gemini_embeddings=bool(os.environ.get("GEMINI_API_KEY")))
    print(f"Passages chargés : {len(engine.chunks)}")
    print(f"Service pressenti : {routing_info['suggestedService']}")
    for rank, item in enumerate(top_passages, start=1):
        print(
            f"  #{rank} [{item.record.reference_code}] "
            f"score={item.rerank_score} (bm25={item.bm25_score}, vec={item.vector_score})"
        )
    if os.environ.get("GEMINI_API_KEY"):
        out = engine.answer(q)
        print("\\nRéponse générée :\\n" + out["answer"])
'''

_SKELETON_README = """# Squelette autonome du moteur RAG hybride DGECC

Ce répertoire est généré automatiquement par `export_rag_bundle.py`.

## Contenu
- `../config/rag_config.json` : Paramètres du moteur RAG, modèles Gemini, organigramme DGECC et consigne système stricte.
- `../corpus/documents_metadata.json` : Métadonnées documentaires exportées (sans données personnelles ni secrets).
- `../corpus/chunks_vectors.jsonl` : Segments et vecteurs d'embeddings (format JSONL ligne par ligne).
- `../corpus/manifest.json` : Sommaire statistique et empreintes SHA-256 d'intégrité.
- `dgecc_rag_skeleton.py` : Moteur RAG hybride autonome (BM25 + Vecteurs + RRF + routage organigramme DGECC).

## Exécution rapide
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="votre_cle_api"
python3 skeleton/dgecc_rag_skeleton.py "Quelles sont les dates des épreuves écrites du Baccalauréat 2026 ?"
```
"""

_SKELETON_REQUIREMENTS = "google-genai>=1.10.0\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def export_dgecc_rag_bundle(
    db_path: Path,
    output_dir: Path,
    include_unpublished: bool = False,
    create_zip: bool = False,
) -> Dict[str, Any]:
    resolved_db = Path(db_path).resolve()
    if not resolved_db.exists():
        raise FileNotFoundError(f"Base SQLite introuvable : {resolved_db}")

    out_root = Path(output_dir).resolve()
    config_dir = out_root / "config"
    corpus_dir = out_root / "corpus"
    skeleton_dir = out_root / "skeleton"
    for d in (config_dir, corpus_dir, skeleton_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Ouverture en lecture seule stricte ; aucune donnée utilisateur/secret n'est lue.
    conn = sqlite3.connect(f"file:{resolved_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        status_rows = conn.execute(
            "SELECT status, COUNT(*) AS count FROM documents GROUP BY status"
        ).fetchall()
        status_counts = {str(r["status"]): int(r["count"]) for r in status_rows}

        if include_unpublished:
            docs_rows = conn.execute(
                """
                SELECT id, title, doc_type, category, service_name, reference_code,
                       published_date, valid_until, current_version, status,
                       file_name, file_format, created_at, updated_at
                FROM documents
                ORDER BY published_date DESC, title ASC
                """
            ).fetchall()
        else:
            docs_rows = conn.execute(
                """
                SELECT id, title, doc_type, category, service_name, reference_code,
                       published_date, valid_until, current_version, status,
                       file_name, file_format, created_at, updated_at
                FROM documents
                WHERE status = ?
                ORDER BY published_date DESC, title ASC
                """,
                ("PUBLIE",),
            ).fetchall()

        documents_metadata: List[Dict[str, Any]] = []
        exported_chunks_count = 0
        embedded_chunks_count = 0
        vector_dimensions: Optional[int] = None
        embedding_models = set()

        chunks_jsonl_path = corpus_dir / "chunks_vectors.jsonl"
        with chunks_jsonl_path.open("w", encoding="utf-8") as chunks_fh:
            for doc in docs_rows:
                doc_id = str(doc["id"])
                cur_ver = int(doc["current_version"])

                version_row = conn.execute(
                    """
                    SELECT chunk_size, chunk_overlap, length(normalized_text) AS text_length, created_at
                    FROM document_versions
                    WHERE document_id = ? AND version_number = ?
                    """,
                    (doc_id, cur_ver),
                ).fetchone()

                history_rows = conn.execute(
                    """
                    SELECT version_number, chunk_count, embedding_model, bm25_terms_count, duration_ms, status, created_at
                    FROM indexation_history
                    WHERE document_id = ?
                    ORDER BY created_at DESC
                    """,
                    (doc_id,),
                ).fetchall()

                chunk_rows = conn.execute(
                    """
                    SELECT id, version_number, chunk_index, content, token_count, char_count,
                           metadata_json, embedding_json, embedding_model, created_at
                    FROM document_chunks
                    WHERE document_id = ? AND version_number = ?
                    ORDER BY chunk_index ASC
                    """,
                    (doc_id, cur_ver),
                ).fetchall()

                doc_embedded = 0
                for ch in chunk_rows:
                    raw_emb = ch["embedding_json"]
                    embedding_vec: Optional[List[float]] = None
                    if raw_emb:
                        try:
                            parsed_vec = json.loads(raw_emb)
                            if isinstance(parsed_vec, list) and parsed_vec:
                                embedding_vec = [float(x) for x in parsed_vec]
                                doc_embedded += 1
                                embedded_chunks_count += 1
                                if vector_dimensions is None:
                                    vector_dimensions = len(embedding_vec)
                        except Exception:
                            embedding_vec = None

                    if ch["embedding_model"]:
                        embedding_models.add(str(ch["embedding_model"]))

                    try:
                        meta_obj = json.loads(ch["metadata_json"]) if ch["metadata_json"] else {}
                    except Exception:
                        meta_obj = {}

                    record = {
                        "chunkId": str(ch["id"]),
                        "documentId": doc_id,
                        "referenceCode": str(doc["reference_code"]),
                        "title": str(doc["title"]),
                        "docType": str(doc["doc_type"]),
                        "category": str(doc["category"]),
                        "serviceName": str(doc["service_name"]),
                        "publishedDate": str(doc["published_date"]),
                        "validUntil": str(doc["valid_until"]) if doc["valid_until"] else None,
                        "status": str(doc["status"]),
                        "versionNumber": int(ch["version_number"]),
                        "chunkIndex": int(ch["chunk_index"]),
                        "content": str(ch["content"]),
                        "tokenCount": int(ch["token_count"]),
                        "charCount": int(ch["char_count"]),
                        "metadata": meta_obj,
                        "embeddingModel": str(ch["embedding_model"]) if ch["embedding_model"] else None,
                        "vectorDimension": len(embedding_vec) if embedding_vec else 0,
                        "embedding": embedding_vec,
                    }
                    chunks_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    exported_chunks_count += 1

                documents_metadata.append(
                    {
                        "id": doc_id,
                        "title": str(doc["title"]),
                        "docType": str(doc["doc_type"]),
                        "category": str(doc["category"]),
                        "serviceName": str(doc["service_name"]),
                        "referenceCode": str(doc["reference_code"]),
                        "publishedDate": str(doc["published_date"]),
                        "validUntil": str(doc["valid_until"]) if doc["valid_until"] else None,
                        "currentVersion": cur_ver,
                        "status": str(doc["status"]),
                        "fileName": str(doc["file_name"]),
                        "fileFormat": str(doc["file_format"]),
                        "chunkCount": len(chunk_rows),
                        "embeddedCount": doc_embedded,
                        "versionConfig": (
                            {
                                "chunkSize": int(version_row["chunk_size"]),
                                "chunkOverlap": int(version_row["chunk_overlap"]),
                                "normalizedTextLength": int(version_row["text_length"]),
                                "createdAt": str(version_row["created_at"]),
                            }
                            if version_row
                            else None
                        ),
                        "indexationHistory": [dict(h) for h in history_rows],
                        "createdAt": str(doc["created_at"]),
                        "updatedAt": str(doc["updated_at"]),
                    }
                )
    finally:
        conn.close()

    exported_at = datetime.now(timezone.utc).isoformat()
    rag_config_payload = {
        "schemaVersion": "1.0.0",
        "exportedAt": exported_at,
        "institution": "DGECC — Direction Générale des Examens, Concours et de la Certification",
        "ragConfig": {
            "embeddingModel": RAG_CONFIG.embedding_model,
            "llmModel": RAG_CONFIG.llm_model,
            "evalModel": RAG_CONFIG.eval_model,
            "defaultChunkSize": RAG_CONFIG.default_chunk_size,
            "defaultChunkOverlap": RAG_CONFIG.default_chunk_overlap,
            "defaultTopK": RAG_CONFIG.default_top_k,
            "defaultRrfK": RAG_CONFIG.default_rrf_k,
            "defaultSimilarityThreshold": RAG_CONFIG.default_similarity_threshold,
            "bm25Weight": RAG_CONFIG.bm25_weight,
            "vectorWeight": RAG_CONFIG.vector_weight,
            "maxHistoryMessages": RAG_CONFIG.max_history_messages,
        },
        "systemInstruction": DGECC_SYSTEM_INSTRUCTION,
        "organigramme": {
            "domains": DGECC_REGULATORY_DOMAINS,
            "services": DGECC_SERVICES_LIST,
            "directions": [dataclasses.asdict(d) for d in DGECC_ORG_STRUCTURE],
        },
        "corpusSummary": {
            "documentsByStatus": status_counts,
            "exportedDocuments": len(documents_metadata),
            "exportedChunks": exported_chunks_count,
            "embeddedChunks": embedded_chunks_count,
            "vectorDimension": vector_dimensions or 0,
            "embeddingModelsPresent": sorted(embedding_models),
            "includeUnpublished": include_unpublished,
        },
    }

    rag_config_path = config_dir / "rag_config.json"
    rag_config_path.write_text(
        json.dumps(rag_config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    docs_meta_path = corpus_dir / "documents_metadata.json"
    docs_meta_path.write_text(
        json.dumps(documents_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    skeleton_py_path = skeleton_dir / "dgecc_rag_skeleton.py"
    skeleton_py_path.write_text(_SKELETON_TEMPLATE, encoding="utf-8")
    (skeleton_dir / "README.md").write_text(_SKELETON_README, encoding="utf-8")
    (skeleton_dir / "requirements.txt").write_text(_SKELETON_REQUIREMENTS, encoding="utf-8")

    manifest_payload = {
        "schemaVersion": "1.0.0",
        "exportedAt": exported_at,
        "sourceDatabase": resolved_db.name,
        "includeUnpublished": include_unpublished,
        "counts": {
            "documents": len(documents_metadata),
            "chunks": exported_chunks_count,
            "embeddedChunks": embedded_chunks_count,
            "vectorDimension": vector_dimensions or 0,
        },
        "files": {
            "config/rag_config.json": {"sha256": _sha256_file(rag_config_path)},
            "corpus/documents_metadata.json": {"sha256": _sha256_file(docs_meta_path)},
            "corpus/chunks_vectors.jsonl": {"sha256": _sha256_file(chunks_jsonl_path)},
            "skeleton/dgecc_rag_skeleton.py": {"sha256": _sha256_file(skeleton_py_path)},
        },
    }
    manifest_path = corpus_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    zip_path: Optional[Path] = None
    if create_zip:
        zip_path = out_root.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(out_root.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.relative_to(out_root))

    return {
        "outputDir": str(out_root),
        "zipPath": str(zip_path) if zip_path else None,
        "exportedDocuments": len(documents_metadata),
        "exportedChunks": exported_chunks_count,
        "embeddedChunks": embedded_chunks_count,
        "vectorDimension": vector_dimensions or 0,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exporte la configuration de la base documentaire DGECC (vecteurs + métadonnées) "
            "et génère un squelette Python autonome du moteur RAG hybride."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=APP_CONFIG.db_path,
        help="Chemin vers la base SQLite source (défaut : data/dgecc.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PYTHON_ROOT / "exports" / "dgecc_rag_bundle",
        help="Répertoire de destination du bundle exporté",
    )
    parser.add_argument(
        "--include-unpublished",
        action="store_true",
        help="Inclure aussi les documents BROUILLON / EN_ATTENTE (par défaut : PUBLIE uniquement)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Créer également une archive ZIP à côté du dossier exporté",
    )

    args = parser.parse_args(argv)
    summary = export_dgecc_rag_bundle(
        db_path=args.db_path,
        output_dir=args.output_dir,
        include_unpublished=args.include_unpublished,
        create_zip=args.zip,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
