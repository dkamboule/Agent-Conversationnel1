"""Analyse linguistique et routage vers les services de l'organigramme DGECC."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Literal, Tuple

from ..core.config import RAG_CONFIG
from ..core.gemini_client import (
    get_gemini_client,
    is_transient_gemini_error,
    with_gemini_retry,
)
from ..memory.memory_manager import ConversationMemoryContext
from ..search.corpus_cache import normalize_french_token, tokenize_french

DgeccIntent = Literal[
    "CALENDRIER_EXAMEN",
    "CONDITIONS_CANDIDATURE",
    "PIECES_JUSTIFICATIVES",
    "INSCRIPTION_PROCEDURE",
    "RESULTATS_CONTESTATION",
    "EQUIVALENCE_CERTIFICATION",
    "AMENAGEMENT_HANDICAP",
    "ORIENTATION_SERVICE",
    "SALUTATION",
    "HORS_PERIMETRE",
]


@dataclass
class NlpAnalysisResult:
    intent: DgeccIntent
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    temporal_references: List[str] = field(default_factory=list)
    needs_retrieval: bool = True
    reformulated_query: str = ""
    suggested_service: str = "Direction Générale — Secrétariat du Directeur Général"

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "entities": self.entities,
            "keywords": self.keywords,
            "temporalReferences": self.temporal_references,
            "needsRetrieval": self.needs_retrieval,
            "reformulatedQuery": self.reformulated_query,
            "suggestedService": self.suggested_service,
        }


_GREETING_RE = re.compile(
    r"^(bonjour|bonsoir|salut|coucou|hello|hi|merci(\s+beaucoup)?|"
    r"au\s+revoir|bonne\s+(journee|soiree))\s*[!?.]*$",
    re.IGNORECASE,
)

_OUT_OF_SCOPE_RE = re.compile(
    r"\b(meteo|quel\s+temps|recette|cuisine|football|match|bourse|crypto|bitcoin|blague|poeme|film|musique)\b",
    re.IGNORECASE,
)

_ANAPHORA_RE = re.compile(
    r"\b(celui-ci|celle-ci|ceux-ci|celles-ci|ce\s+dernier|cette\s+derniere|le\s+meme|"
    r"la\s+meme|y\s+inscrire|pour\s+ca|a\s+ce\s+sujet|qu'en\s+est-il|et\s+les\s+|et\s+pour)\b",
    re.IGNORECASE,
)

_TEMPORAL_RE = re.compile(
    r"\b(202[4-9]|janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|"
    r"novembre|decembre|\d+\s*(?:heures?|jours?|mois|ans?))\b"
)

_REF_CODE_RE = re.compile(r"\bDGECC-\d{4}-[A-Z]+-\d+\b")

_KNOWN_ENTITIES: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(baccalaureat|bac)\b", re.I), "Baccalauréat"),
    (re.compile(r"\bbepc\b", re.I), "BEPC"),
    (re.compile(r"\bcap\b", re.I), "CAP"),
    (re.compile(r"\b(bt|brevet\s+de\s+technicien)\b", re.I), "Brevet de Technicien"),
    (
        re.compile(r"\b(concours\s+direct|concours\s+professionnel|fonction\s+publique)\b", re.I),
        "Concours Fonction Publique",
    ),
    (re.compile(r"\bcategorie\s+[abc]\b", re.I), "Catégorie concours"),
    (re.compile(r"\b(duplicata|diplome\s+perdu)\b", re.I), "Duplicata de diplôme"),
    (re.compile(r"\bequivalence\b", re.I), "Équivalence de diplôme"),
    (re.compile(r"\battestation\b", re.I), "Attestation de réussite"),
    (re.compile(r"\b(tiers-temps|handicap|amenagement)\b", re.I), "Aménagement handicap / Tiers-temps"),
    (re.compile(r"\b(reclamation|recours|copie|jury)\b", re.I), "Recours et consultation de copies"),
    (re.compile(r"\b(frais|paiement|quittance|recepisse|exoneration)\b", re.I), "Frais et paiement"),
]


def classify_deterministic(question: str) -> NlpAnalysisResult:
    norm = normalize_french_token(question.strip())
    keywords = tokenize_french(question)
    temporal_refs = list(dict.fromkeys(_TEMPORAL_RE.findall(norm)))
    ref_matches = _REF_CODE_RE.findall(question.upper())

    entities: List[str] = list(ref_matches)
    for pattern, label in _KNOWN_ENTITIES:
        if pattern.search(norm):
            entities.append(label)
    entities = list(dict.fromkeys(entities))

    if _GREETING_RE.match(norm):
        return NlpAnalysisResult(
            intent="SALUTATION",
            needs_retrieval=False,
            reformulated_query=question,
        )

    if _OUT_OF_SCOPE_RE.search(norm) and not entities:
        return NlpAnalysisResult(
            intent="HORS_PERIMETRE",
            keywords=keywords,
            temporal_references=temporal_refs,
            needs_retrieval=False,
            reformulated_query=question,
        )

    intent: DgeccIntent = "ORIENTATION_SERVICE"
    suggested_service = "Direction Générale — Secrétariat du Directeur Général"

    mentions_bac = bool(re.search(r"\b(baccalaureat|bac)\b", norm))
    mentions_tech_pro = bool(re.search(r"\b(technique|professionnel|cap|bt|bep)\b", norm))
    mentions_bepc = bool(re.search(r"\bbepc\b", norm))

    if re.search(r"\b(tiers-temps|handicap|amenagement|medical|braille|secretaire\s+d'examen)\b", norm):
        intent = "AMENAGEMENT_HANDICAP"
        suggested_service = "DEPE — Service des études, de l'évaluation et de la prospection"
    elif re.search(
        r"\b(reclamation|recours|contester|copie|copies|erreur\s+materielle|jury|controle\s+interne)\b",
        norm,
    ):
        intent = "RESULTATS_CONTESTATION"
        suggested_service = "Direction Générale — Contrôle Interne (CI)"
    elif re.search(r"\b(duplicata|attestation|releve\s+de\s+notes|diplome\s+perdu)\b", norm):
        intent = "EQUIVALENCE_CERTIFICATION"
        if mentions_bac:
            suggested_service = "DEB — Service des attestations et diplômes"
        elif re.search(r"\barchive\b", norm):
            suggested_service = "Direction Générale — Service des Archives et de la Documentation (SADo)"
        else:
            suggested_service = "DECEPS — Service des attestations et diplômes"
    elif re.search(
        r"\b(equivalence|certification\s+professionnelle|dcp|qualification|diplome\s+etranger|vae)\b",
        norm,
    ):
        intent = "EQUIVALENCE_CERTIFICATION"
        suggested_service = "Direction des Certifications Professionnelles (DCP)"
    elif re.search(r"\b(statistique|statistiques|taux\s+de\s+reussite|prospective|etude|evaluation)\b", norm):
        intent = "ORIENTATION_SERVICE"
        suggested_service = (
            "DEPE — Service des statistiques"
            if re.search(r"\bstatistique", norm)
            else "DEPE — Service des études, de l'évaluation et de la prospection"
        )
    elif re.search(
        r"\b(frais|paiement|quittance|recepisse|exoneration|timbre\s+fiscal|cout|coute|tarif)\b",
        norm,
    ):
        intent = "INSCRIPTION_PROCEDURE"
        suggested_service = "Direction Générale — Services Financier (SF)"
    elif re.search(
        r"\b(piece|pieces|dossier|justificatif|casier|extrait|certificat\s+medical|acte\s+de\s+naissance|cni)\b",
        norm,
    ):
        intent = "PIECES_JUSTIFICATIVES"
        if mentions_bac:
            suggested_service = (
                "DEB — Service du baccalauréat de l'enseignement technique professionnel"
                if mentions_tech_pro
                else "DEB — Service du baccalauréat de l'enseignement général"
            )
        elif re.search(r"\bconcours\b", norm):
            suggested_service = "DECEPS — Service des examens professionnels"
        elif mentions_tech_pro:
            suggested_service = "DECEPS — Service des examens de l'enseignement technique professionnel"
        else:
            suggested_service = "DECEPS — Service des examens de l'enseignement post-primaire général"
    elif re.search(
        r"\b(condition|age|diplome\s+requis|categorie\s+[abc]|eligibilite|peut-on|droit\s+civique|concours)\b",
        norm,
    ):
        intent = "CONDITIONS_CANDIDATURE"
        suggested_service = "DECEPS — Service des examens professionnels"
    elif re.search(r"\b(inscrire|inscription|convocation|centre\s+d'examen|en\s+ligne)\b", norm):
        intent = "INSCRIPTION_PROCEDURE"
        if mentions_bac:
            suggested_service = (
                "DEB — Service du baccalauréat de l'enseignement technique professionnel"
                if mentions_tech_pro
                else "DEB — Service du baccalauréat de l'enseignement général"
            )
        elif mentions_bepc:
            suggested_service = "DECEPS — Service des examens de l'enseignement post-primaire général"
        elif mentions_tech_pro:
            suggested_service = "DECEPS — Service des examens de l'enseignement technique professionnel"
        else:
            suggested_service = "DECEPS — Secrétariat"
    elif re.search(
        r"\b(date|dates|calendrier|quand|epreuve|epreuves|resultat|resultats|session|rattrapage|cloture|ouverture)\b",
        norm,
    ):
        intent = "CALENDRIER_EXAMEN"
        if mentions_bac:
            suggested_service = (
                "DEB — Service du baccalauréat de l'enseignement technique professionnel"
                if mentions_tech_pro
                else "DEB — Service du baccalauréat de l'enseignement général"
            )
        elif mentions_bepc:
            suggested_service = "DECEPS — Service des examens de l'enseignement post-primaire général"
        elif mentions_tech_pro:
            suggested_service = "DECEPS — Service des examens de l'enseignement technique professionnel"
        else:
            suggested_service = "DECEPS — Secrétariat"

    return NlpAnalysisResult(
        intent=intent,
        entities=entities,
        keywords=keywords,
        temporal_references=temporal_refs,
        needs_retrieval=True,
        reformulated_query=question,
        suggested_service=suggested_service,
    )


def analyze_user_question(
    question: str,
    memory: ConversationMemoryContext,
) -> NlpAnalysisResult:
    fast_result = classify_deterministic(question)
    norm = normalize_french_token(question.strip())
    word_count = len(norm.split())

    needs_reformulation = (
        fast_result.needs_retrieval
        and len(memory.recent_turns) > 0
        and (_ANAPHORA_RE.search(norm) is not None or (word_count <= 5 and not fast_result.entities))
    )

    if not needs_reformulation:
        return fast_result

    try:
        from google.genai import types

        client = get_gemini_client()
        history_summary = "\n".join(
            f"{'Usager' if t.role == 'user' else 'Assistant'}: {t.content[:220]}"
            for t in memory.recent_turns[-4:]
        )
        prompt_text = (
            "Reformule brièvement cette question de suivi en français sous forme de requête "
            "autonome claire pour rechercher dans les textes DGECC.\n"
            f"Historique:\n{history_summary}\n"
            f'Question de suivi: "{question}"'
        )

        def run_reformulation(model_name: str):
            thinking_level = (
                types.ThinkingLevel.MINIMAL
                if model_name == "gemini-3.1-flash-lite"
                else types.ThinkingLevel.LOW
            )
            return client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {"reformulatedQuery": {"type": "STRING"}},
                        "required": ["reformulatedQuery"],
                    },
                ),
            )

        try:
            response = with_gemini_retry(
                lambda: run_reformulation(RAG_CONFIG.llm_model),
                max_retries=1,
                base_delay_ms=250,
            )
        except BaseException as err:
            if is_transient_gemini_error(err) and RAG_CONFIG.eval_model != RAG_CONFIG.llm_model:
                response = run_reformulation(RAG_CONFIG.eval_model)
            else:
                raise

        parsed = json.loads((response.text or "{}").strip())
        rewritten = (parsed.get("reformulatedQuery") or "").strip()
        if rewritten:
            merged = classify_deterministic(rewritten)
            merged.reformulated_query = rewritten
            return merged
    except Exception:
        pass

    return fast_result
