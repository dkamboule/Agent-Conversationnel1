"""Extraction multi-formats (PDF, Images scannées, DOCX, HTML, TXT) avec OCR automatique."""

from __future__ import annotations

import html as html_lib
import io
import re
from dataclasses import dataclass
from typing import Literal, Optional

from .ocr import perform_document_ocr

SupportedFormat = Literal["PDF", "DOCX", "HTML", "TXT", "IMAGE"]
OcrMode = Literal["auto", "force", "off"]
OcrReason = Literal["image_scan", "pdf_scanned_auto", "forced", "none"]


@dataclass
class ExtractionResult:
    text: str
    ocr_used: bool
    ocr_reason: OcrReason
    ocr_model: Optional[str] = None
    native_char_count: int = 0
    page_count: Optional[int] = None


def detect_format(file_name: str, mime_type: Optional[str] = None) -> Optional[SupportedFormat]:
    lower = file_name.lower()
    mime = (mime_type or "").lower()

    if lower.endswith(".pdf") or mime == "application/pdf":
        return "PDF"
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp")) or mime in {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
    }:
        return "IMAGE"
    if (
        lower.endswith(".docx")
        or mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        return "DOCX"
    if lower.endswith((".html", ".htm")) or mime == "text/html":
        return "HTML"
    if lower.endswith(".txt") or mime == "text/plain":
        return "TXT"
    return None


def resolve_ocr_mime_type(
    file_name: str,
    fmt: SupportedFormat,
    mime_type: Optional[str] = None,
) -> str:
    if fmt == "PDF":
        return "application/pdf"
    lower = file_name.lower()
    mime = (mime_type or "").lower()
    if lower.endswith(".png") or mime == "image/png":
        return "image/png"
    if lower.endswith(".webp") or mime == "image/webp":
        return "image/webp"
    return "image/jpeg"


def _count_meaningful_chars(text: str) -> int:
    return len(re.findall(r"[a-zA-ZÀ-ÿ0-9]", text))


def extract_document_with_ocr(
    data: bytes,
    fmt: SupportedFormat,
    file_name: str = "document",
    mime_type: Optional[str] = None,
    ocr_mode: OcrMode = "auto",
) -> ExtractionResult:
    if fmt == "TXT":
        text = data.decode("utf-8", errors="replace")
        return ExtractionResult(
            text=text,
            ocr_used=False,
            ocr_reason="none",
            native_char_count=_count_meaningful_chars(text),
        )

    if fmt == "HTML":
        raw_html = data.decode("utf-8", errors="replace")
        cleaned = re.sub(
            r"<script\b[^>]*>.*?</script>", " ", raw_html, flags=re.IGNORECASE | re.DOTALL
        )
        cleaned = re.sub(
            r"<style\b[^>]*>.*?</style>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL
        )
        cleaned = re.sub(
            r"</(p|div|li|h[1-6]|tr|section|article|br)>", "\n", cleaned, flags=re.IGNORECASE
        )
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        text = html_lib.unescape(cleaned)
        return ExtractionResult(
            text=text,
            ocr_used=False,
            ocr_reason="none",
            native_char_count=_count_meaningful_chars(text),
        )

    if fmt == "DOCX":
        try:
            import docx  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Le paquet 'python-docx' est requis pour extraire les fichiers DOCX."
            ) from exc
        document = docx.Document(io.BytesIO(data))
        text = "\n\n".join(p.text for p in document.paragraphs if p.text.strip())
        return ExtractionResult(
            text=text,
            ocr_used=False,
            ocr_reason="none",
            native_char_count=_count_meaningful_chars(text),
        )

    if fmt == "IMAGE":
        if ocr_mode == "off":
            raise ValueError(
                "L'extraction d'une image scannée nécessite l'activation de la reconnaissance OCR."
            )
        resolved_mime = resolve_ocr_mime_type(file_name, fmt, mime_type)
        ocr_res = perform_document_ocr(data, resolved_mime)
        return ExtractionResult(
            text=ocr_res.text,
            ocr_used=True,
            ocr_reason="forced" if ocr_mode == "force" else "image_scan",
            ocr_model=ocr_res.model_used,
            native_char_count=0,
            page_count=1,
        )

    # PDF: tentative d'extraction texte native puis OCR automatique si PDF scanné
    native_text = ""
    page_count = 1
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(data))
        page_count = max(1, len(reader.pages))
        native_text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        native_text = ""

    native_chars = _count_meaningful_chars(native_text)
    min_expected_chars = max(40, page_count * 35)
    needs_auto_ocr = native_chars < min_expected_chars

    if ocr_mode == "force" or (ocr_mode == "auto" and needs_auto_ocr):
        ocr_res = perform_document_ocr(data, "application/pdf")
        ocr_chars = _count_meaningful_chars(ocr_res.text)
        if ocr_chars >= native_chars or ocr_mode == "force":
            return ExtractionResult(
                text=ocr_res.text,
                ocr_used=True,
                ocr_reason="forced" if ocr_mode == "force" else "pdf_scanned_auto",
                ocr_model=ocr_res.model_used,
                native_char_count=native_chars,
                page_count=page_count,
            )

    return ExtractionResult(
        text=native_text,
        ocr_used=False,
        ocr_reason="none",
        native_char_count=native_chars,
        page_count=page_count,
    )
