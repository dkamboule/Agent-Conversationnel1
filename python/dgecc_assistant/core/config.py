"""Configuration applicative et paramètres du moteur RAG DGECC."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

_EPHEMERAL_SESSION_SECRET = secrets.token_hex(32)
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class AppConfig:
    port: int = int(os.environ.get("PYTHON_PORT", os.environ.get("PORT", "8000")))
    db_path: Path = Path(
        os.environ.get("DGECC_DB_PATH", str(_DEFAULT_ROOT / "data" / "dgecc.db"))
    ).resolve()
    session_secret: str = os.environ.get("SESSION_SECRET") or _EPHEMERAL_SESSION_SECRET
    admin_passcode_override: str = os.environ.get("DGECC_ADMIN_PASSCODE", "")
    token_ttl_seconds: int = 60 * 60 * 12  # 12 heures


@dataclass(frozen=True)
class RagConfig:
    embedding_model: str = "gemini-embedding-2"
    llm_model: str = "gemini-3.1-flash-lite"
    eval_model: str = "gemini-4-hydrogen"
    default_chunk_size: int = 850
    default_chunk_overlap: int = 160
    default_top_k: int = 5
    default_rrf_k: int = 60
    default_similarity_threshold: float = 0.25
    bm25_weight: float = 1.0
    vector_weight: float = 1.0
    max_history_messages: int = 6


APP_CONFIG = AppConfig()
RAG_CONFIG = RagConfig()
