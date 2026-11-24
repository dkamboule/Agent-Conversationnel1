"""Index corpus publié en mémoire, tokenisation française et vecteurs normalisés."""

from __future__ import annotations

import json
import math
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ..database.db import get_db

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
    normalized = normalize_french_token(text)
    return [tok for tok in _TOKEN_RE.findall(normalized) if tok not in FRENCH_STOPWORDS]


def to_unit_vector(vec: List[float]) -> List[float]:
    norm_sq = sum(v * v for v in vec)
    norm = math.sqrt(norm_sq)
    if norm <= 0.0:
        return [0.0 for _ in vec]
    return [v / norm for v in vec]


def dot_product_unit(a: List[float], b: List[float]) -> float:
    length = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(length))


@dataclass
class CachedPublishedChunk:
    chunk_id: str
    document_id: str
    version_number: int
    chunk_index: int
    content: str
    title: str
    doc_type: str
    category: str
    service_name: str
    reference_code: str
    reference_norm: str
    published_date: str
    valid_until: Optional[str]
    tf: Dict[str, int]
    token_length: int
    unit_vector: Optional[List[float]]


@dataclass
class PublishedCorpusIndex:
    chunks: List[CachedPublishedChunk]
    avg_doc_len: float
    doc_freq: Dict[str, int]


_cache_lock = threading.Lock()
_cached_corpus: Optional[PublishedCorpusIndex] = None
_invalidation_listeners: List[Callable[[], None]] = []


def on_corpus_invalidated(cb: Callable[[], None]) -> None:
    _invalidation_listeners.append(cb)


def invalidate_corpus_cache() -> None:
    global _cached_corpus
    with _cache_lock:
        _cached_corpus = None
    for cb in _invalidation_listeners:
        cb()


def get_published_corpus_index() -> PublishedCorpusIndex:
    global _cached_corpus
    with _cache_lock:
        if _cached_corpus is not None:
            return _cached_corpus

        db = get_db()
        rows = db.execute(
            """
            SELECT
              c.id AS chunk_id,
              c.document_id AS document_id,
              c.version_number AS version_number,
              c.chunk_index AS chunk_index,
              c.content AS content,
              c.embedding_json AS embedding_json,
              d.title AS title,
              d.doc_type AS doc_type,
              d.category AS category,
              d.service_name AS service_name,
              d.reference_code AS reference_code,
              d.published_date AS published_date,
              d.valid_until AS valid_until
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id AND c.version_number = d.current_version
            WHERE d.status = 'PUBLIE'
            """
        ).fetchall()

        doc_freq: Dict[str, int] = {}
        total_len = 0
        chunks: List[CachedPublishedChunk] = []

        for r in rows:
            toks = tokenize_french(f"{r['title']} {r['reference_code']} {r['content']}")
            total_len += len(toks)
            tf: Dict[str, int] = {}
            seen = set()
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
                if tok not in seen:
                    seen.add(tok)
                    doc_freq[tok] = doc_freq.get(tok, 0) + 1

            unit_vec: Optional[List[float]] = None
            raw_emb = r["embedding_json"]
            if raw_emb:
                try:
                    parsed = json.loads(raw_emb)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        unit_vec = to_unit_vector([float(x) for x in parsed])
                except Exception:
                    unit_vec = None

            chunks.append(
                CachedPublishedChunk(
                    chunk_id=str(r["chunk_id"]),
                    document_id=str(r["document_id"]),
                    version_number=int(r["version_number"]),
                    chunk_index=int(r["chunk_index"]),
                    content=str(r["content"]),
                    title=str(r["title"]),
                    doc_type=str(r["doc_type"]),
                    category=str(r["category"]),
                    service_name=str(r["service_name"]),
                    reference_code=str(r["reference_code"]),
                    reference_norm=normalize_french_token(str(r["reference_code"])),
                    published_date=str(r["published_date"]),
                    valid_until=str(r["valid_until"]) if r["valid_until"] else None,
                    tf=tf,
                    token_length=len(toks),
                    unit_vector=unit_vec,
                )
            )

        _cached_corpus = PublishedCorpusIndex(
            chunks=chunks,
            avg_doc_len=total_len / max(1, len(chunks)),
            doc_freq=doc_freq,
        )
        return _cached_corpus
