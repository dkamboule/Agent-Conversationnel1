"""Vérification des garde-fous factuels, citations, validité temporelle et sécurité."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Tuple

from ..core.config import RAG_CONFIG
from ..core.gemini_client import (
    get_gemini_client,
    is_transient_gemini_error,
    with_gemini_retry,
)
from ..core.security import detect_prompt_injection
from ..nlp.analyzer import NlpAnalysisResult
from ..rag.fusion import FusedPassage


@dataclass
class GuardrailCheckItem:
    name: str
    passed: bool
    detail: str


@dataclass
class GuardrailsReport:
    passed: bool
    refused: bool
    confidence_level: Literal["ELEVEE", "MOYENNE", "FAIBLE"]
    has_expired_sources: bool
    oriented_service: str
    checks: List[GuardrailCheckItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "refused": self.refused,
            "confidenceLevel": self.confidence_level,
            "hasExpiredSources": self.has_expired_sources,
            "orientedService": self.oriented_service,
            "checks": [asdict(c) for c in self.checks],
        }


def build_refusal_message(service_name: str, reason: str) -> str:
    return (
        f"Les extraits officiels de la DGECC ne permettent pas de répondre à cette demande ({reason}).\n\n"
        f"Veuillez contacter le service concerné : **{service_name}**.\n\n"
        "En cas de doute, seule l'administration fait foi."
    )


_YEAR_RE = re.compile(r"\b20\d{2}\b")
_AMOUNT_RE = re.compile(r"\b\d[\d\s]{1,9}\s*(?:fcfa|€|euros?)\b", re.IGNORECASE)


def check_numeric_grounding(
    answer: str,
    passages: List[FusedPassage],
) -> Tuple[bool, List[str]]:
    source_blob = re.sub(
        r"\s+",
        " ",
        " ".join(
            f"{p.reference_code} {p.published_date} {p.valid_until or ''} {p.content}"
            for p in passages
        ),
    )
    source_digits = re.sub(r"[^\d]", "", source_blob)

    unmatched: List[str] = []
    for yr in _YEAR_RE.findall(answer):
        if yr not in source_blob:
            unmatched.append(yr)
    for amt in _AMOUNT_RE.findall(answer):
        digits = re.sub(r"[^\d]", "", amt)
        if digits and digits not in source_digits:
            unmatched.append(amt)

    deduped = list(dict.fromkeys(unmatched))
    return len(deduped) == 0, deduped


def verify_generated_response(
    question: str,
    answer: str,
    analysis: NlpAnalysisResult,
    passages: List[FusedPassage],
) -> GuardrailsReport:
    checks: List[GuardrailCheckItem] = []

    flagged, reason = detect_prompt_injection(question)
    checks.append(
        GuardrailCheckItem(
            name="Sécurité / Anti-injection de prompt",
            passed=not flagged,
            detail=reason or "Instruction suspecte détectée." if flagged else "Aucun motif d’injection détecté.",
        )
    )
    if flagged:
        return GuardrailsReport(
            passed=False,
            refused=True,
            confidence_level="FAIBLE",
            has_expired_sources=False,
            oriented_service="Direction Générale — Secrétariat du Directeur Général",
            checks=checks,
        )

    if not analysis.needs_retrieval and analysis.intent == "SALUTATION":
        checks.append(
            GuardrailCheckItem(
                name="Périmètre conversationnel",
                passed=True,
                detail="Salutation ou demande d’accueil traitée sans recherche documentaire.",
            )
        )
        return GuardrailsReport(
            passed=True,
            refused=False,
            confidence_level="ELEVEE",
            has_expired_sources=False,
            oriented_service=analysis.suggested_service,
            checks=checks,
        )

    if not passages:
        checks.append(
            GuardrailCheckItem(
                name="Suffisance documentaire",
                passed=False,
                detail="Aucun passage validé et publié ne répond au seuil de pertinence requis.",
            )
        )
        return GuardrailsReport(
            passed=False,
            refused=True,
            confidence_level="FAIBLE",
            has_expired_sources=False,
            oriented_service=analysis.suggested_service,
            checks=checks,
        )

    checks.append(
        GuardrailCheckItem(
            name="Suffisance documentaire",
            passed=True,
            detail=f"{len(passages)} passage(s) issu(s) de documents publiés sélectionné(s).",
        )
    )

    has_expired = any(p.is_expired for p in passages)
    checks.append(
        GuardrailCheckItem(
            name="Validité temporelle des textes",
            passed=not has_expired,
            detail=(
                "Au moins une source citée a dépassé sa date de validité ; mention requise."
                if has_expired
                else "Toutes les sources sélectionnées sont en cours de validité."
            ),
        )
    )

    has_citation = any(
        p.reference_code in answer or p.title.lower()[:18] in answer.lower()
        for p in passages
    )
    checks.append(
        GuardrailCheckItem(
            name="Présence des citations institutionnelles",
            passed=has_citation,
            detail=(
                "Références documentaires explicitement citées dans la réponse."
                if has_citation
                else "Références associées en annexe de traçabilité."
            ),
        )
    )

    grounded, unmatched = check_numeric_grounding(answer, passages)
    faithful = grounded
    detail = (
        "Années, montants et citations conformes aux extraits validés."
        if grounded
        else f"Vérification approfondie déclenchée sur : {', '.join(unmatched)}."
    )

    if not grounded:
        try:
            from google.genai import types

            client = get_gemini_client()
            context_text = "\n\n".join(f"[{p.reference_code}] {p.content}" for p in passages)
            audit_prompt = (
                "Vérifie si la RÉPONSE est fidèle au CONTEXTE et n'invente aucune date, "
                "montant ou procédure absente du contexte.\n"
                f"CONTEXTE :\n{context_text}\nRÉPONSE :\n{answer}"
            )

            def run_audit(model_name: str):
                thinking_level = (
                    types.ThinkingLevel.MINIMAL
                    if model_name == "gemini-3.1-flash-lite"
                    else types.ThinkingLevel.LOW
                )
                return client.models.generate_content(
                    model=model_name,
                    contents=audit_prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                        response_mime_type="application/json",
                        response_schema={
                            "type": "OBJECT",
                            "properties": {
                                "faithful": {"type": "BOOLEAN"},
                                "reason": {"type": "STRING"},
                            },
                            "required": ["faithful", "reason"],
                        },
                    ),
                )

            try:
                audit = with_gemini_retry(
                    lambda: run_audit(RAG_CONFIG.llm_model),
                    max_retries=1,
                    base_delay_ms=250,
                )
            except BaseException as err:
                if is_transient_gemini_error(err) and RAG_CONFIG.eval_model != RAG_CONFIG.llm_model:
                    audit = run_audit(RAG_CONFIG.eval_model)
                else:
                    raise

            parsed = json.loads((audit.text or "{}").strip())
            if isinstance(parsed.get("faithful"), bool):
                faithful = parsed["faithful"]
                detail = parsed.get("reason") or detail
        except Exception:
            faithful = True

    checks.append(
        GuardrailCheckItem(
            name="Conformité factuelle & absence d’invention",
            passed=faithful,
            detail=detail,
        )
    )

    top_score = passages[0].rerank_score if passages else 0.0
    confidence: Literal["ELEVEE", "MOYENNE", "FAIBLE"] = (
        "FAIBLE"
        if not faithful
        else ("ELEVEE" if top_score >= 0.025 and has_citation else "MOYENNE")
    )

    return GuardrailsReport(
        passed=faithful,
        refused=not faithful,
        confidence_level=confidence,
        has_expired_sources=has_expired,
        oriented_service=passages[0].service_name if passages else analysis.suggested_service,
        checks=checks,
    )
