"""Chargement isolé de l'historique conversationnel."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from ..core.config import RAG_CONFIG
from ..database.db import get_db


@dataclass
class ConversationTurn:
    role: Literal["user", "assistant"]
    content: str
    created_at: str


@dataclass
class ConversationMemoryContext:
    conversation_id: str
    recent_turns: List[ConversationTurn] = field(default_factory=list)
    consulted_references: List[str] = field(default_factory=list)


def load_conversation_memory(
    conversation_id: str,
    user_id: Optional[str] = None,
    guest_session_id: Optional[str] = None,
) -> ConversationMemoryContext:
    db = get_db()
    conv = db.execute(
        "SELECT id, user_id, guest_session_id FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()

    if conv is None:
        return ConversationMemoryContext(conversation_id=conversation_id)

    authorized = (user_id and conv["user_id"] == user_id) or (
        not user_id and guest_session_id and conv["guest_session_id"] == guest_session_id
    )
    if not authorized:
        raise PermissionError(
            "Accès refusé à cette conversation : isolation de session requise."
        )

    rows = db.execute(
        """
        SELECT role, content, sources_json, created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (conversation_id, RAG_CONFIG.max_history_messages),
    ).fetchall()

    ordered = list(reversed(rows))
    refs: List[str] = []
    seen_refs = set()

    for r in ordered:
        raw_sources = r["sources_json"]
        if raw_sources:
            try:
                parsed = json.loads(raw_sources)
                if isinstance(parsed, list):
                    for s in parsed:
                        ref = s.get("referenceCode")
                        if ref:
                            label = f"{ref} ({s.get('title', '')})"
                            if label not in seen_refs:
                                seen_refs.add(label)
                                refs.append(label)
            except Exception:
                pass

    return ConversationMemoryContext(
        conversation_id=conversation_id,
        recent_turns=[
            ConversationTurn(
                role=r["role"],
                content=str(r["content"]),
                created_at=str(r["created_at"]),
            )
            for r in ordered
        ],
        consulted_references=refs,
    )
