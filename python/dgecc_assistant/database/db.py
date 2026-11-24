"""Connexion SQLite WAL compatible avec data/dgecc.db et journal d'audit."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.config import APP_CONFIG

_lock = threading.Lock()
_connection: Optional[sqlite3.Connection] = None
_active_db_path: Optional[Path] = None


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS roles (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          code TEXT UNIQUE NOT NULL,
          label TEXT NOT NULL,
          permissions_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS users (
          id TEXT PRIMARY KEY,
          email TEXT UNIQUE NOT NULL,
          full_name TEXT NOT NULL,
          password_hash TEXT NOT NULL,
          salt TEXT NOT NULL,
          role_code TEXT NOT NULL REFERENCES roles(code),
          service TEXT,
          is_active INTEGER NOT NULL DEFAULT 1,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS activity_logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT,
          user_email TEXT,
          action TEXT NOT NULL,
          resource_type TEXT NOT NULL,
          resource_id TEXT,
          details_json TEXT NOT NULL DEFAULT '{}',
          ip_address TEXT,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          doc_type TEXT NOT NULL,
          category TEXT NOT NULL,
          service_name TEXT NOT NULL,
          reference_code TEXT NOT NULL,
          published_date TEXT NOT NULL,
          valid_until TEXT,
          current_version INTEGER NOT NULL DEFAULT 1,
          status TEXT NOT NULL DEFAULT 'BROUILLON',
          file_name TEXT NOT NULL,
          file_format TEXT NOT NULL,
          uploaded_by TEXT,
          validated_by TEXT,
          rejection_reason TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS document_versions (
          id TEXT PRIMARY KEY,
          document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          version_number INTEGER NOT NULL,
          raw_text TEXT NOT NULL,
          normalized_text TEXT NOT NULL,
          chunk_size INTEGER NOT NULL DEFAULT 850,
          chunk_overlap INTEGER NOT NULL DEFAULT 160,
          created_by TEXT,
          created_at TEXT NOT NULL,
          UNIQUE (document_id, version_number)
        );

        CREATE TABLE IF NOT EXISTS document_chunks (
          id TEXT PRIMARY KEY,
          document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          version_number INTEGER NOT NULL,
          chunk_index INTEGER NOT NULL,
          content TEXT NOT NULL,
          token_count INTEGER NOT NULL,
          char_count INTEGER NOT NULL,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          embedding_json TEXT,
          embedding_model TEXT,
          created_at TEXT NOT NULL,
          UNIQUE (document_id, version_number, chunk_index)
        );

        CREATE TABLE IF NOT EXISTS indexation_history (
          id TEXT PRIMARY KEY,
          document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          version_number INTEGER NOT NULL,
          chunk_count INTEGER NOT NULL,
          embedding_model TEXT NOT NULL,
          bm25_terms_count INTEGER NOT NULL,
          duration_ms INTEGER NOT NULL,
          status TEXT NOT NULL,
          triggered_by TEXT,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversations (
          id TEXT PRIMARY KEY,
          user_id TEXT,
          guest_session_id TEXT,
          title TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
          id TEXT PRIMARY KEY,
          conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          nlp_analysis_json TEXT,
          sources_json TEXT,
          guardrails_json TEXT,
          metrics_json TEXT,
          feedback TEXT,
          feedback_comment TEXT,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS evaluation_runs (
          id TEXT PRIMARY KEY,
          run_name TEXT NOT NULL,
          corpus_size INTEGER NOT NULL,
          recall_at_k REAL NOT NULL,
          precision_at_k REAL NOT NULL,
          mrr REAL NOT NULL,
          ndcg_at_k REAL NOT NULL,
          faithfulness REAL NOT NULL,
          answer_relevancy REAL NOT NULL,
          context_precision REAL NOT NULL,
          context_recall REAL NOT NULL,
          avg_latency_ms REAL NOT NULL,
          details_json TEXT NOT NULL,
          methodology_json TEXT,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS user_panel_evaluations (
          id TEXT PRIMARY KEY,
          profile TEXT NOT NULL CHECK (profile IN ('CANDIDAT', 'PARENT', 'AGENT_ADMINISTRATIF')),
          participant_name TEXT NOT NULL,
          clarity_score INTEGER NOT NULL CHECK (clarity_score BETWEEN 1 AND 5),
          usefulness_score INTEGER NOT NULL CHECK (usefulness_score BETWEEN 1 AND 5),
          trust_score INTEGER NOT NULL CHECK (trust_score BETWEEN 1 AND 5),
          comment TEXT,
          recorded_by TEXT,
          created_at TEXT NOT NULL
        );
        """
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(evaluation_runs)").fetchall()}
    if "methodology_json" not in cols:
        conn.execute("ALTER TABLE evaluation_runs ADD COLUMN methodology_json TEXT")
        conn.commit()


def get_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    global _connection, _active_db_path
    target = (db_path or _active_db_path or APP_CONFIG.db_path).resolve()
    with _lock:
        if _connection is not None and _active_db_path == target:
            return _connection
        if _connection is not None:
            _connection.close()
            _connection = None
        target.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(target), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _init_schema(conn)
        _connection = conn
        _active_db_path = target
        return conn


def set_db_path(db_path: Path) -> sqlite3.Connection:
    global _active_db_path, _connection
    with _lock:
        if _connection is not None:
            _connection.close()
            _connection = None
        _active_db_path = db_path.resolve()
    return get_db(_active_db_path)


def log_activity(
    action: str,
    resource_type: str,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> None:
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        """
        INSERT INTO activity_logs
          (user_id, user_email, action, resource_type, resource_id, details_json, ip_address, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            user_email,
            action,
            resource_type,
            resource_id,
            json.dumps(details or {}, ensure_ascii=False),
            ip_address,
            now,
        ),
    )
    db.commit()
