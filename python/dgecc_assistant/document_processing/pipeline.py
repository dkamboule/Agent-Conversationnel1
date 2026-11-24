"""Pipeline d'ingestion, chunking et indexation hybride (BM25 + vecteurs)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from ..core.config import RAG_CONFIG
from ..database.db import get_db, log_activity
from ..embeddings.embedder import get_embedding_provider
from ..search.corpus_cache import invalidate_corpus_cache, tokenize_french
from .chunker import ChunkMetadata, chunk_document_text
from .extractor import ExtractionResult, OcrMode, SupportedFormat, extract_document_with_ocr
from .normalizer import normalize_text

DocumentStatus = Literal["BROUILLON", "EN_ATTENTE", "VALIDE", "PUBLIE", "ARCHIVE", "REJETE"]


@dataclass
class IngestDocumentInput:
    title: str
    doc_type: str
    category: str
    service_name: str
    reference_code: str
    published_date: str
    valid_until: Optional[str]
    file_name: str
    fmt: SupportedFormat
    data: bytes
    uploaded_by: str
    uploaded_by_email: str
    mime_type: Optional[str] = None
    ocr_mode: OcrMode = "auto"
    reviewed_text: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    initial_status: DocumentStatus = "EN_ATTENTE"


def ingest_document(inp: IngestDocumentInput) -> Dict[str, Any]:
    db = get_db()

    if inp.reviewed_text and len(inp.reviewed_text.strip()) >= 20:
        extraction = ExtractionResult(
            text=inp.reviewed_text,
            ocr_used=(inp.fmt == "IMAGE" or inp.ocr_mode in ("force", "auto")),
            ocr_reason="image_scan" if inp.fmt == "IMAGE" else "forced",
            native_char_count=len(inp.reviewed_text),
        )
    else:
        extraction = extract_document_with_ocr(
            inp.data,
            inp.fmt,
            file_name=inp.file_name,
            mime_type=inp.mime_type,
            ocr_mode=inp.ocr_mode,
        )

    raw_text = extraction.text
    normalized_text = normalize_text(raw_text)
    if len(normalized_text) < 20:
        raise ValueError(
            "Le contenu extrait du document (même après reconnaissance OCR) est insuffisant "
            "ou illisible (moins de 20 caractères exploitables)."
        )

    doc_id = str(uuid.uuid4())
    version_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    chunk_size = inp.chunk_size or RAG_CONFIG.default_chunk_size
    chunk_overlap = (
        inp.chunk_overlap if inp.chunk_overlap is not None else RAG_CONFIG.default_chunk_overlap
    )

    db.execute(
        """
        INSERT INTO documents (
          id, title, doc_type, category, service_name, reference_code,
          published_date, valid_until, current_version, status,
          file_name, file_format, uploaded_by, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            inp.title.strip(),
            inp.doc_type.strip(),
            inp.category.strip(),
            inp.service_name.strip(),
            inp.reference_code.strip(),
            inp.published_date,
            inp.valid_until,
            inp.initial_status,
            inp.file_name,
            inp.fmt,
            inp.uploaded_by,
            now,
            now,
        ),
    )

    db.execute(
        """
        INSERT INTO document_versions (
          id, document_id, version_number, raw_text, normalized_text,
          chunk_size, chunk_overlap, created_by, created_at
        ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)
        """,
        (
            version_id,
            doc_id,
            raw_text,
            normalized_text,
            chunk_size,
            chunk_overlap,
            inp.uploaded_by,
            now,
        ),
    )

    meta = ChunkMetadata(
        document_id=doc_id,
        title=inp.title.strip(),
        doc_type=inp.doc_type.strip(),
        category=inp.category.strip(),
        service_name=inp.service_name.strip(),
        reference_code=inp.reference_code.strip(),
        published_date=inp.published_date,
        valid_until=inp.valid_until,
        version_number=1,
    )
    chunks = chunk_document_text(normalized_text, meta, chunk_size, chunk_overlap)

    for chunk in chunks:
        db.execute(
            """
            INSERT INTO document_chunks (
              id, document_id, version_number, chunk_index, content,
              token_count, char_count, metadata_json, created_at
            ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                doc_id,
                chunk.chunk_index,
                chunk.content,
                chunk.token_count,
                chunk.char_count,
                json.dumps(chunk.metadata.to_dict(), ensure_ascii=False),
                now,
            ),
        )
    db.commit()

    log_activity(
        action="DOCUMENT_UPLOAD_OCR" if extraction.ocr_used else "DOCUMENT_UPLOAD",
        resource_type="document",
        user_id=inp.uploaded_by,
        user_email=inp.uploaded_by_email,
        resource_id=doc_id,
        details={
            "title": inp.title,
            "format": inp.fmt,
            "chunksCreated": len(chunks),
            "status": inp.initial_status,
            "ocrUsed": extraction.ocr_used,
            "ocrReason": extraction.ocr_reason,
            "ocrModel": extraction.ocr_model,
            "pageCount": extraction.page_count,
        },
    )

    return {
        "id": doc_id,
        "title": inp.title,
        "status": inp.initial_status,
        "version": 1,
        "chunksCount": len(chunks),
        "normalizedLength": len(normalized_text),
        "ocrUsed": extraction.ocr_used,
        "ocrReason": extraction.ocr_reason,
        "ocrModel": extraction.ocr_model,
        "pageCount": extraction.page_count,
    }


def index_document_version(
    document_id: str,
    triggered_by_user_id: str,
    triggered_by_email: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    bump_version: bool = False,
) -> Dict[str, Any]:
    start_ms = int(time.time() * 1000)
    db = get_db()

    doc = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    if doc is None:
        raise ValueError("Document introuvable.")

    current_version_num = int(doc["current_version"])
    version_row = db.execute(
        "SELECT * FROM document_versions WHERE document_id = ? AND version_number = ?",
        (document_id, current_version_num),
    ).fetchone()
    if version_row is None:
        raise ValueError("Version documentaire introuvable.")

    eff_chunk_size = chunk_size if chunk_size is not None else int(version_row["chunk_size"])
    eff_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None else int(version_row["chunk_overlap"])
    )
    target_version = current_version_num + 1 if bump_version else current_version_num
    now = datetime.now(timezone.utc).isoformat()

    if bump_version:
        db.execute(
            """
            INSERT INTO document_versions (
              id, document_id, version_number, raw_text, normalized_text,
              chunk_size, chunk_overlap, created_by, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                document_id,
                target_version,
                str(version_row["raw_text"]),
                str(version_row["normalized_text"]),
                eff_chunk_size,
                eff_chunk_overlap,
                triggered_by_user_id,
                now,
            ),
        )
        db.execute(
            "UPDATE documents SET current_version = ?, updated_at = ? WHERE id = ?",
            (target_version, now, document_id),
        )
    else:
        db.execute(
            "DELETE FROM document_chunks WHERE document_id = ? AND version_number = ?",
            (document_id, target_version),
        )
        db.execute(
            "UPDATE document_versions SET chunk_size = ?, chunk_overlap = ? "
            "WHERE document_id = ? AND version_number = ?",
            (eff_chunk_size, eff_chunk_overlap, document_id, target_version),
        )

    meta = ChunkMetadata(
        document_id=document_id,
        title=str(doc["title"]),
        doc_type=str(doc["doc_type"]),
        category=str(doc["category"]),
        service_name=str(doc["service_name"]),
        reference_code=str(doc["reference_code"]),
        published_date=str(doc["published_date"]),
        valid_until=str(doc["valid_until"]) if doc["valid_until"] else None,
        version_number=target_version,
    )
    chunks = chunk_document_text(
        str(version_row["normalized_text"]), meta, eff_chunk_size, eff_chunk_overlap
    )

    provider = get_embedding_provider()
    vectors = provider.embed_texts([c.content for c in chunks])

    bm25_terms_count = 0
    for idx, c in enumerate(chunks):
        bm25_terms_count += len(tokenize_french(c.content))
        db.execute(
            """
            INSERT INTO document_chunks (
              id, document_id, version_number, chunk_index, content,
              token_count, char_count, metadata_json, embedding_json, embedding_model, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                document_id,
                target_version,
                c.chunk_index,
                c.content,
                c.token_count,
                c.char_count,
                json.dumps(c.metadata.to_dict(), ensure_ascii=False),
                json.dumps(vectors[idx]),
                provider.model_name,
                now,
            ),
        )

    duration_ms = int(time.time() * 1000) - start_ms
    db.execute(
        """
        INSERT INTO indexation_history (
          id, document_id, version_number, chunk_count, embedding_model,
          bm25_terms_count, duration_ms, status, triggered_by, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'SUCCES', ?, ?)
        """,
        (
            str(uuid.uuid4()),
            document_id,
            target_version,
            len(chunks),
            provider.model_name,
            bm25_terms_count,
            duration_ms,
            triggered_by_user_id,
            now,
        ),
    )
    db.commit()
    invalidate_corpus_cache()

    log_activity(
        action="DOCUMENT_INDEXED",
        resource_type="document",
        user_id=triggered_by_user_id,
        user_email=triggered_by_email,
        resource_id=document_id,
        details={
            "version": target_version,
            "chunkCount": len(chunks),
            "bm25TermsCount": bm25_terms_count,
            "embeddingModel": provider.model_name,
            "durationMs": duration_ms,
        },
    )

    return {
        "documentId": document_id,
        "versionNumber": target_version,
        "chunkCount": len(chunks),
        "bm25TermsCount": bm25_terms_count,
        "embeddingModel": provider.model_name,
        "durationMs": duration_ms,
    }
