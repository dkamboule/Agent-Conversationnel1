"""Structure officielle de l'organigramme de la DGECC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class DgeccDirectionSpec:
    id: str
    domain: str
    label: str
    acronym: str
    description: str
    services: List[str]


DGECC_ORG_STRUCTURE: List[DgeccDirectionSpec] = [
    DgeccDirectionSpec(
        id="DG",
        domain="Direction Générale & Services Rattachés",
        label="Direction Générale",
        acronym="DG",
        description=(
            "Pilotage stratégique, appui technique, gestion financière, ressources humaines, "
            "archives, documentation et contrôle interne."
        ),
        services=[
            "Direction Générale — Secrétariat du Directeur Général",
            "Direction Générale — Chargés d'Appui Technique (CAT)",
            "Direction Générale — Bureau Comptable Matières Secondaires (BCMS)",
            "Direction Générale — Services Financier (SF)",
            "Direction Générale — Service de Gestion des Ressources Humaines (SGRH)",
            "Direction Générale — Service des Archives et de la Documentation (SADo)",
            "Direction Générale — Contrôle Interne (CI)",
        ],
    ),
    DgeccDirectionSpec(
        id="DECEPS",
        domain="Examens & Concours Post-primaire et Secondaire (DECEPS)",
        label=(
            "Direction des Examens et Concours de l'Enseignement "
            "Post-primaire et Secondaire"
        ),
        acronym="DECEPS",
        description=(
            "Organisation des examens de l'enseignement post-primaire général, technique "
            "professionnel, des examens professionnels et délivrance des attestations et diplômes."
        ),
        services=[
            "DECEPS — Secrétariat",
            "DECEPS — Service des examens de l'enseignement post-primaire général",
            "DECEPS — Service des examens de l'enseignement technique professionnel",
            "DECEPS — Service des examens professionnels",
            "DECEPS — Service des attestations et diplômes",
        ],
    ),
    DgeccDirectionSpec(
        id="DCP",
        domain="Certifications Professionnelles (DCP)",
        label="Direction des Certifications Professionnelles",
        acronym="DCP",
        description=(
            "Gestion des certifications professionnelles, titres de qualification, "
            "validation des acquis et équivalences professionnelles."
        ),
        services=["Direction des Certifications Professionnelles (DCP)"],
    ),
    DgeccDirectionSpec(
        id="DEPE",
        domain="Études, Prospective et Évaluation (DEPE)",
        label="Direction des Études, de la Prospective et de l'Évaluation",
        acronym="DEPE",
        description=(
            "Études méthodologiques, évaluation des dispositifs d'examen, prospective "
            "pédagogique et production des statistiques officielles."
        ),
        services=[
            "DEPE — Secrétariat",
            "DEPE — Service des études, de l'évaluation et de la prospection",
            "DEPE — Service des statistiques",
        ],
    ),
    DgeccDirectionSpec(
        id="DEB",
        domain="Examens du Baccalauréat (DEB)",
        label="Direction des Examens du Baccalauréat",
        acronym="DEB",
        description=(
            "Organisation du baccalauréat de l'enseignement général et de l'enseignement "
            "technique professionnel, et délivrance des attestations et diplômes du baccalauréat."
        ),
        services=[
            "DEB — Secrétariat",
            "DEB — Service du baccalauréat de l'enseignement général",
            "DEB — Service du baccalauréat de l'enseignement technique professionnel",
            "DEB — Service des attestations et diplômes",
        ],
    ),
]

DGECC_REGULATORY_DOMAINS: List[str] = [d.domain for d in DGECC_ORG_STRUCTURE]
DGECC_SERVICES_LIST: List[str] = [s for d in DGECC_ORG_STRUCTURE for s in d.services]
DEFAULT_SERVICE_BY_DOMAIN: Dict[str, str] = {
    d.domain: d.services[0] for d in DGECC_ORG_STRUCTURE
}
