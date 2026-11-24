"""Découpage (chunking) intelligent avec chevauchement et métadonnées."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChunkMetadata:
    document_id: str
    title: str
    doc_type: str
    category: str
    service_name: str
    reference_code: str
    published_date: str
    valid_until: Optional[str]
    version_number: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documentId": self.document_id,
            "title": self.title,
            "docType": self.doc_type,
            "category": self.category,
            "serviceName": self.service_name,
            "referenceCode": self.reference_code,
            "publishedDate": self.published_date,
            "validUntil": self.valid_until,
            "versionNumber": self.version_number,
        }


@dataclass
class DocumentChunkItem:
    chunk_index: int
    content: str
    token_count: int
    char_count: int
    metadata: ChunkMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunkIndex": self.chunk_index,
            "content": self.content,
            "tokenCount": self.token_count,
            "charCount": self.char_count,
            "metadata": asdict(self.metadata),
        }


def estimate_token_count(text: str) -> int:
    words = [w for w in text.strip().split() if w]
    return max(1, round(len(words) * 1.35))


def chunk_document_text(
    normalized_text: str,
    metadata: ChunkMetadata,
    chunk_size: int = 850,
    chunk_overlap: int = 160,
) -> List[DocumentChunkItem]:
    safe_size = max(250, min(2400, chunk_size))
    safe_overlap = max(0, min(int(safe_size * 0.45), chunk_overlap))

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", normalized_text) if p.strip()]
    chunks: List[str] = []
    current = ""

    def push_chunk(body: str) -> None:
        trimmed = body.strip()
        if not trimmed:
            return
        prefix = f"[{metadata.reference_code} — {metadata.title} | {metadata.category}]\n"
        chunks.append(f"{prefix}{trimmed}")

    for para in paragraphs:
        if len(para) > safe_size:
            if current:
                push_chunk(current)
                current = ""
            sentences = re.split(r"(?<=[.!?])\s+", para)
            seg = ""
            for sentence in sentences:
                candidate = f"{seg} {sentence}".strip()
                if len(candidate) <= safe_size:
                    seg = candidate
                else:
                    if seg:
                        push_chunk(seg)
                    if len(sentence) > safe_size:
                        start = 0
                        step = max(100, safe_size - safe_overlap)
                        while start < len(sentence):
                            push_chunk(sentence[start : start + safe_size])
                            start += step
                        seg = ""
                    else:
                        tail = seg[max(0, len(seg) - safe_overlap) :]
                        seg = f"{tail} {sentence}".strip()
            if seg:
                current = seg
            continue

        combined = f"{current}\n\n{para}".strip() if current else para
        if len(combined) <= safe_size:
            current = combined
        else:
            push_chunk(current)
            tail = current[max(0, len(current) - safe_overlap) :]
            clean_tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            current = f"{clean_tail}\n\n{para}" if clean_tail else para
            if len(current) > safe_size:
                current = para

    if current:
        push_chunk(current)

    return [
        DocumentChunkItem(
            chunk_index=idx,
            content=content,
            token_count=estimate_token_count(content),
            char_count=len(content),
            metadata=metadata,
        )
        for idx, content in enumerate(chunks)
    ]
