"""Normalisation et nettoyage des textes extraits ou scannés."""

from __future__ import annotations

import re

from ..core.security import sanitize_document_text

_PAGE_MARKER_RE = re.compile(r"^(page\s+\d+(\s*/\s*\d+)?|-\s*\d+\s*-)$", re.IGNORECASE)


def normalize_text(raw: str) -> str:
    sanitized = sanitize_document_text(raw)
    text = (
        sanitized.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00A0", " ")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines = []
    for line in text.split("\n"):
        trimmed = line.strip()
        if _PAGE_MARKER_RE.match(trimmed):
            continue
        if trimmed == "" and cleaned_lines and cleaned_lines[-1] == "":
            continue
        cleaned_lines.append(trimmed)

    return "\n".join(cleaned_lines).strip()
