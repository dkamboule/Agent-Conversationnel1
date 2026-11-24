"""Moteur OCR multimodal Gemini Vision pour documents administratifs scannés."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.gemini_client import (
    get_gemini_client,
    is_transient_gemini_error,
    with_gemini_retry,
)

OCR_SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
}

OCR_PROMPT = (
    "Tu es un moteur OCR haute fidélité spécialisé dans la numérisation de documents "
    "administratifs officiels en français (DGECC).\n"
    "Transcris fidèlement et intégralement TOUT le texte visible sur ce document scanné "
    "(image ou PDF), page par page.\n\n"
    "RÈGLES STRICTES :\n"
    "1. Respecte exactement les mots, sigles (DGECC, DEB, DECEPS, DCP, DEPE, BAC, BEPC, CAP, BT), "
    "numéros de référence, dates, montants et accents français.\n"
    "2. Préserve la structure logique (titres, articles, sections, listes à puces, tableaux présentés "
    "proprement ligne par ligne).\n"
    "3. Ignore les artefacts de numérisation (bordures de scanner, tampons décoratifs illisibles, "
    "numéros de page isolés) sans inventer ni résumer de texte.\n"
    "4. Si un passage est partiellement flou, ne devine jamais de chiffre ou de date : transcris "
    "uniquement ce qui est lisible.\n"
    "5. Retourne UNIQUEMENT le texte brut transcrit, sans commentaire introductif, sans bloc markdown "
    "(```) et sans conclusion."
)


def _strip_code_fences(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:text|txt|markdown)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()


@dataclass
class OcrExtractionOutput:
    text: str
    model_used: str


def perform_document_ocr(data: bytes, mime_type: str) -> OcrExtractionOutput:
    if mime_type not in OCR_SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Type MIME non pris en charge pour l'OCR ({mime_type}). "
            "Formats OCR acceptés : PDF, PNG, JPEG, WEBP."
        )

    from google.genai import types

    client = get_gemini_client()

    def call_ocr_model(model_name: str) -> str:
        thinking_level = (
            types.ThinkingLevel.MINIMAL
            if model_name == "gemini-3.1-flash-lite"
            else types.ThinkingLevel.LOW
        )
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=data, mime_type=mime_type),
                OCR_PROMPT,
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
            ),
        )
        return _strip_code_fences(response.text or "")

    try:
        text = with_gemini_retry(
            lambda: call_ocr_model("gemini-4-hydrogen"),
            max_retries=2,
            base_delay_ms=400,
        )
        return OcrExtractionOutput(text=text, model_used="gemini-4-hydrogen")
    except BaseException as err:
        if is_transient_gemini_error(err):
            text = with_gemini_retry(
                lambda: call_ocr_model("gemini-3.1-flash-lite"),
                max_retries=1,
                base_delay_ms=400,
            )
            return OcrExtractionOutput(text=text, model_used="gemini-3.1-flash-lite")
        raise
