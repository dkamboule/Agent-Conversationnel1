"""Serveur REST & SSE FastAPI pour l'assistant conversationnel DGECC (version Python)."""

from __future__ import annotations

import base64
import hmac
import json
import queue
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .core.config import APP_CONFIG, RAG_CONFIG
from .core.gemini_client import format_gemini_user_error
from .core.security import (
    AuthPayload,
    RoleCode,
    hash_password,
    rate_limiter,
    sign_token,
    verify_password,
    verify_token,
)
from .database.db import get_db, log_activity
from .document_processing.extractor import detect_format, extract_document_with_ocr
from .document_processing.normalizer import normalize_text
from .document_processing.pipeline import (
    IngestDocumentInput,
    index_document_version,
    ingest_document,
)
from .rag.orchestrator import RagQueryRequest, run_hybrid_rag_pipeline
from .search.corpus_cache import invalidate_corpus_cache

app = FastAPI(
    title="API DGECC — Agent Conversationnel RAG Hybride (Python)",
    version="1.0.0",
    description=(
        "API REST et SSE officielle de l'assistant d'orientation DGECC "
        "(BM25 + embeddings vectoriels + RRF + OCR + garde-fous)."
    ),
)


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    conversationId: Optional[str] = None
    categoryFilter: Optional[str] = None
    publishedAfter: Optional[str] = None
    topK: Optional[int] = None
    rrfK: Optional[int] = None
    similarityThreshold: Optional[float] = None


class ValidateDocumentRequest(BaseModel):
    status: str
    rejectionReason: Optional[str] = None


def _get_optional_auth(
    authorization: Optional[str] = Header(default=None),
) -> Optional[AuthPayload]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:].strip()
    return verify_token(token)


def _require_auth(
    user: Optional[AuthPayload] = Depends(_get_optional_auth),
) -> AuthPayload:
    if user is None:
        raise HTTPException(status_code=401, detail="Authentification requise.")
    return user


def _require_roles(*allowed: RoleCode):
    def _checker(user: AuthPayload = Depends(_require_auth)) -> AuthPayload:
        if user.role not in allowed:
            raise HTTPException(
                status_code=403,
                detail="Permissions insuffisantes pour cette opération.",
            )
        return user

    return _checker


def _resolve_owner(
    user: Optional[AuthPayload],
    x_dgecc_session: Optional[str],
) -> Dict[str, Optional[str]]:
    if user is not None:
        return {"user_id": user.sub, "user_email": user.email, "guest_session_id": None}
    guest_id = (x_dgecc_session or "guest-default").strip()[:120]
    return {"user_id": None, "user_email": None, "guest_session_id": guest_id}


def _check_rate_limit(request: Request, max_req: int, window_s: float) -> None:
    client_host = request.client.host if request.client else "local"
    key = f"{client_host}:{request.url.path}"
    if not rate_limiter.allow(key, max_req, window_s):
        raise HTTPException(
            status_code=429,
            detail="Trop de requêtes effectuées. Veuillez patienter quelques instants avant de réessayer.",
        )


@app.get("/api/v1/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "runtime": "python-fastapi"}


@app.post("/api/v1/auth/login")
def auth_login(payload: LoginRequest, request: Request) -> Dict[str, Any]:
    _check_rate_limit(request, 20, 60.0)
    clean_email = payload.email.strip().lower()
    if not clean_email or not payload.password:
        raise HTTPException(status_code=400, detail="Identifiants requis.")

    if APP_CONFIG.admin_passcode_override and hmac.compare_digest(
        payload.password, APP_CONFIG.admin_passcode_override
    ):
        token = sign_token("env-admin", clean_email, "Administrateur DGECC", "ADMIN")
        log_activity(
            action="AUTH_LOGIN_ADMIN_OVERRIDE",
            resource_type="session",
            user_id="env-admin",
            user_email=clean_email,
            ip_address=request.client.host if request.client else None,
        )
        return {
            "token": token,
            "user": {
                "id": "env-admin",
                "email": clean_email,
                "fullName": "Administrateur DGECC",
                "role": "ADMIN",
                "service": "Direction Générale DGECC",
            },
        }

    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE email = ? AND is_active = 1", (clean_email,)
    ).fetchone()
    if user is None or not verify_password(
        payload.password, str(user["password_hash"]), str(user["salt"])
    ):
        raise HTTPException(status_code=401, detail="Courriel ou mot de passe incorrect.")

    token = sign_token(
        str(user["id"]),
        str(user["email"]),
        str(user["full_name"]),
        user["role_code"],
    )
    log_activity(
        action="AUTH_LOGIN",
        resource_type="session",
        user_id=str(user["id"]),
        user_email=str(user["email"]),
        ip_address=request.client.host if request.client else None,
    )
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "fullName": user["full_name"],
            "role": user["role_code"],
            "service": user["service"],
        },
    }


@app.get("/api/v1/auth/me")
def auth_me(user: AuthPayload = Depends(_require_auth)) -> Dict[str, Any]:
    db = get_db()
    role_row = db.execute(
        "SELECT label, permissions_json FROM roles WHERE code = ?", (user.role,)
    ).fetchone()
    return {
        "user": {
            "id": user.sub,
            "email": user.email,
            "fullName": user.full_name,
            "role": user.role,
            "roleLabel": role_row["label"] if role_row else user.role,
            "permissions": json.loads(role_row["permissions_json"]) if role_row else [],
        }
    }


@app.post("/api/v1/chat/message")
def chat_message(
    payload: ChatRequest,
    request: Request,
    user: Optional[AuthPayload] = Depends(_get_optional_auth),
    x_dgecc_session: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _check_rate_limit(request, 30, 60.0)
    owner = _resolve_owner(user, x_dgecc_session)
    try:
        return run_hybrid_rag_pipeline(
            RagQueryRequest(
                question=payload.question.strip(),
                conversation_id=payload.conversationId,
                category_filter=payload.categoryFilter,
                published_after=payload.publishedAfter,
                top_k=payload.topK,
                rrf_k=payload.rrfK,
                similarity_threshold=payload.similarityThreshold,
                user_id=owner["user_id"],
                user_email=owner["user_email"],
                guest_session_id=owner["guest_session_id"],
            )
        )
    except Exception as err:
        status_code, msg = format_gemini_user_error(err)
        raise HTTPException(status_code=status_code, detail=msg)


@app.post("/api/v1/chat/stream")
def chat_stream(
    payload: ChatRequest,
    request: Request,
    user: Optional[AuthPayload] = Depends(_get_optional_auth),
    x_dgecc_session: Optional[str] = Header(default=None),
):
    _check_rate_limit(request, 30, 60.0)
    owner = _resolve_owner(user, x_dgecc_session)
    events: "queue.Queue[Optional[str]]" = queue.Queue()

    def emit(event: str, data: Any) -> None:
        events.put(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n")

    def worker() -> None:
        try:
            result = run_hybrid_rag_pipeline(
                RagQueryRequest(
                    question=payload.question.strip(),
                    conversation_id=payload.conversationId,
                    category_filter=payload.categoryFilter,
                    published_after=payload.publishedAfter,
                    top_k=payload.topK,
                    rrf_k=payload.rrfK,
                    similarity_threshold=payload.similarityThreshold,
                    user_id=owner["user_id"],
                    user_email=owner["user_email"],
                    guest_session_id=owner["guest_session_id"],
                    on_stage=lambda stage, detail: emit("stage", {"stage": stage, "detail": detail}),
                    on_token=lambda delta: emit("token", {"delta": delta}),
                )
            )
            emit("complete", result)
        except Exception as err:
            _, msg = format_gemini_user_error(err)
            emit("error", {"error": msg})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream_generator():
        while True:
            item = events.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/v1/chat/conversations")
def list_conversations(
    user: Optional[AuthPayload] = Depends(_get_optional_auth),
    x_dgecc_session: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    db = get_db()
    owner = _resolve_owner(user, x_dgecc_session)
    if owner["user_id"]:
        rows = db.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE user_id = ? ORDER BY updated_at DESC LIMIT 40",
            (owner["user_id"],),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE guest_session_id = ? ORDER BY updated_at DESC LIMIT 40",
            (owner["guest_session_id"],),
        ).fetchall()
    return {"conversations": [dict(r) for r in rows]}


@app.get("/api/v1/documents")
def list_documents(
    user: Optional[AuthPayload] = Depends(_get_optional_auth),
) -> Dict[str, Any]:
    db = get_db()
    is_staff = user is not None and user.role in ("ADMIN", "VALIDATOR", "EDITOR")
    if is_staff:
        rows = db.execute(
            """
            SELECT d.*,
              (SELECT COUNT(*) FROM document_chunks c WHERE c.document_id = d.id AND c.version_number = d.current_version) AS chunk_count,
              (SELECT COUNT(*) FROM document_chunks c WHERE c.document_id = d.id AND c.version_number = d.current_version AND c.embedding_json IS NOT NULL) AS embedded_count
            FROM documents d ORDER BY d.updated_at DESC
            """
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT d.id, d.title, d.doc_type, d.category, d.service_name, d.reference_code,
              d.published_date, d.valid_until, d.current_version, d.status, d.file_format, d.updated_at,
              (SELECT COUNT(*) FROM document_chunks c WHERE c.document_id = d.id AND c.version_number = d.current_version) AS chunk_count
            FROM documents d WHERE d.status = 'PUBLIE' ORDER BY d.published_date DESC
            """
        ).fetchall()
    return {"documents": [dict(r) for r in rows]}


@app.post("/api/v1/documents/ocr-preview")
async def ocr_preview(
    file: UploadFile = File(...),
    ocrMode: str = Form(default="auto"),
    user: AuthPayload = Depends(_require_roles("ADMIN", "EDITOR")),
) -> Dict[str, Any]:
    _ = user
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Aucun fichier valide n’a été transmis.")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Le fichier dépasse la limite autorisée de 15 Mo.")

    fmt = detect_format(file.filename or "document", file.content_type)
    if fmt is None:
        raise HTTPException(
            status_code=400,
            detail="Format non supporté. Formats acceptés : PDF, PNG, JPG, WEBP, DOCX, HTML, TXT.",
        )

    mode = ocrMode if ocrMode in ("force", "off") else "auto"
    try:
        extraction = extract_document_with_ocr(
            data,
            fmt,
            file_name=file.filename or "document",
            mime_type=file.content_type,
            ocr_mode=mode,  # type: ignore[arg-type]
        )
        normalized = normalize_text(extraction.text)
        return {
            "text": normalized,
            "ocrUsed": extraction.ocr_used,
            "ocrReason": extraction.ocr_reason,
            "ocrModel": extraction.ocr_model,
            "nativeCharCount": extraction.native_char_count,
            "pageCount": extraction.page_count,
            "normalizedLength": len(normalized),
        }
    except Exception as err:
        status_code, msg = format_gemini_user_error(err)
        raise HTTPException(status_code=max(status_code, 400), detail=msg)


@app.post("/api/v1/documents/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    referenceCode: str = Form(...),
    docType: str = Form(default="Note officielle"),
    category: str = Form(default="Direction Générale & Services Rattachés"),
    serviceName: str = Form(default="Direction Générale — Secrétariat du Directeur Général"),
    publishedDate: str = Form(default=""),
    validUntil: Optional[str] = Form(default=None),
    ocrMode: str = Form(default="auto"),
    reviewedText: Optional[str] = Form(default=None),
    initialStatus: str = Form(default="EN_ATTENTE"),
    user: AuthPayload = Depends(_require_roles("ADMIN", "EDITOR")),
) -> Dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Aucun fichier valide n’a été transmis.")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Le fichier dépasse la limite autorisée de 15 Mo.")

    fmt = detect_format(file.filename or "document", file.content_type)
    if fmt is None:
        raise HTTPException(
            status_code=400,
            detail="Format non supporté. Formats acceptés : PDF, PNG, JPG, WEBP, DOCX, HTML, TXT.",
        )

    if not title.strip() or not referenceCode.strip():
        raise HTTPException(
            status_code=400,
            detail="Le titre et la référence réglementaire sont obligatoires.",
        )

    eff_date = publishedDate.strip() or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    mode = ocrMode if ocrMode in ("force", "off") else "auto"
    status = "BROUILLON" if initialStatus == "BROUILLON" else "EN_ATTENTE"

    try:
        return ingest_document(
            IngestDocumentInput(
                title=title,
                doc_type=docType.strip() or "Note officielle",
                category=category.strip() or "Direction Générale & Services Rattachés",
                service_name=serviceName.strip()
                or "Direction Générale — Secrétariat du Directeur Général",
                reference_code=referenceCode,
                published_date=eff_date,
                valid_until=validUntil.strip() if validUntil and validUntil.strip() else None,
                file_name=file.filename or "document",
                mime_type=file.content_type,
                fmt=fmt,
                data=data,
                ocr_mode=mode,  # type: ignore[arg-type]
                reviewed_text=reviewedText,
                uploaded_by=user.sub,
                uploaded_by_email=user.email,
                initial_status=status,  # type: ignore[arg-type]
            )
        )
    except Exception as err:
        status_code, msg = format_gemini_user_error(err)
        raise HTTPException(status_code=max(status_code, 400), detail=msg)


@app.put("/api/v1/documents/{doc_id}/validate")
def validate_document(
    doc_id: str,
    payload: ValidateDocumentRequest,
    request: Request,
    user: AuthPayload = Depends(_require_roles("ADMIN", "VALIDATOR")),
) -> Dict[str, Any]:
    allowed = {"BROUILLON", "EN_ATTENTE", "VALIDE", "REJETE", "ARCHIVE"}
    if payload.status not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Statut cible invalide. Autorisés : {', '.join(sorted(allowed))}",
        )

    db = get_db()
    doc = db.execute("SELECT id, title FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document introuvable.")

    now = datetime.now(timezone.utc).isoformat()
    reason = (
        payload.rejectionReason.strip()
        if payload.status == "REJETE" and payload.rejectionReason
        else None
    )
    db.execute(
        "UPDATE documents SET status = ?, validated_by = ?, rejection_reason = ?, updated_at = ? WHERE id = ?",
        (payload.status, user.sub, reason, now, doc_id),
    )
    db.commit()
    invalidate_corpus_cache()

    log_activity(
        action=f"DOCUMENT_STATUS_{payload.status}",
        resource_type="document",
        user_id=user.sub,
        user_email=user.email,
        resource_id=doc_id,
        details={"title": doc["title"], "rejectionReason": reason},
        ip_address=request.client.host if request.client else None,
    )
    return {"id": doc_id, "status": payload.status}


@app.post("/api/v1/documents/{doc_id}/publish")
def publish_document(
    doc_id: str,
    request: Request,
    user: AuthPayload = Depends(_require_roles("ADMIN", "VALIDATOR")),
) -> Dict[str, Any]:
    db = get_db()
    doc = db.execute("SELECT id, status, title FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document introuvable.")

    try:
        stats = index_document_version(doc_id, user.sub, user.email)
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            "UPDATE documents SET status = 'PUBLIE', validated_by = ?, updated_at = ? WHERE id = ?",
            (user.sub, now, doc_id),
        )
        db.commit()
        invalidate_corpus_cache()

        log_activity(
            action="DOCUMENT_PUBLISHED",
            resource_type="document",
            user_id=user.sub,
            user_email=user.email,
            resource_id=doc_id,
            details={"title": doc["title"], **stats},
            ip_address=request.client.host if request.client else None,
        )
        return {"id": doc_id, "status": "PUBLIE", "indexation": stats}
    except Exception as err:
        status_code, msg = format_gemini_user_error(err)
        raise HTTPException(status_code=status_code, detail=msg)


@app.get("/api/v1/admin/statistics")
def admin_statistics(
    user: AuthPayload = Depends(_require_roles("ADMIN", "VALIDATOR", "EDITOR")),
) -> Dict[str, Any]:
    _ = user
    db = get_db()
    doc_stats = [
        dict(r)
        for r in db.execute(
            "SELECT status, COUNT(*) AS count FROM documents GROUP BY status"
        ).fetchall()
    ]
    total_chunks = db.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
    embedded_chunks = db.execute(
        "SELECT COUNT(*) FROM document_chunks WHERE embedding_json IS NOT NULL"
    ).fetchone()[0]
    total_conversations = db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    total_messages = db.execute(
        "SELECT COUNT(*) FROM messages WHERE role = 'assistant'"
    ).fetchone()[0]
    return {
        "documentsByStatus": doc_stats,
        "totalChunks": total_chunks,
        "embeddedChunks": embedded_chunks,
        "totalConversations": total_conversations,
        "totalAssistantResponses": total_messages,
        "ragConfig": {
            "embeddingModel": RAG_CONFIG.embedding_model,
            "llmModel": RAG_CONFIG.llm_model,
            "evalModel": RAG_CONFIG.eval_model,
        },
    }
