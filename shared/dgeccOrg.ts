export interface DgeccDirectionSpec {
  id: string;
  domain: string;
  label: string;
  acronym: string;
  description: string;
  services: string[];
}

export const DGECC_ORG_STRUCTURE: DgeccDirectionSpec[] = [
  {
    id: 'DG',
    domain: 'Direction Générale & Services Rattachés',
    label: 'Direction Générale',
    acronym: 'DG',
    description:
      'Pilotage stratégique, appui technique, gestion financière, ressources humaines, archives, documentation et contrôle interne.',
    services: [
      'Direction Générale — Secrétariat du Directeur Général',
      "Direction Générale — Chargés d'Appui Technique (CAT)",
      'Direction Générale — Bureau Comptable Matières Secondaires (BCMS)',
      'Direction Générale — Services Financier (SF)',
      'Direction Générale — Service de Gestion des Ressources Humaines (SGRH)',
      'Direction Générale — Service des Archives et de la Documentation (SADo)',
      'Direction Générale — Contrôle Interne (CI)',
    ],
  },
  {
    id: 'DECEPS',
    domain: 'Examens & Concours Post-primaire et Secondaire (DECEPS)',
    label:
      "Direction des Examens et Concours de l'Enseignement Post-primaire et Secondaire",
    acronym: 'DECEPS',
    description:
      "Organisation des examens de l'enseignement post-primaire général, technique professionnel, des examens professionnels et délivrance des attestations et diplômes.",
    services: [
      'DECEPS — Secrétariat',
      "DECEPS — Service des examens de l'enseignement post-primaire général",
      "DECEPS — Service des examens de l'enseignement technique professionnel",
      'DECEPS — Service des examens professionnels',
      'DECEPS — Service des attestations et diplômes',
    ],
  },
  {
    id: 'DCP',
    domain: 'Certifications Professionnelles (DCP)',
    label: 'Direction des Certifications Professionnelles',
    acronym: 'DCP',
    description:
      'Gestion des certifications professionnelles, titres de qualification, validation des acquis et équivalences professionnelles.',
    services: ['Direction des Certifications Professionnelles (DCP)'],
  },
  {
    id: 'DEPE',
    domain: 'Études, Prospective et Évaluation (DEPE)',
    label: "Direction des Études, de la Prospective et de l'Évaluation",
    acronym: 'DEPE',
    description:
      "Études méthodologiques, évaluation des dispositifs d'examen, prospective pédagogique et production des statistiques officielles.",
    services: [
      'DEPE — Secrétariat',
      "DEPE — Service des études, de l'évaluation et de la prospection",
      'DEPE — Service des statistiques',
    ],
  },
  {
    id: 'DEB',
    domain: 'Examens du Baccalauréat (DEB)',
    label: 'Direction des Examens du Baccalauréat',
    acronym: 'DEB',
    description:
      "Organisation du baccalauréat de l'enseignement général et de l'enseignement technique professionnel, et délivrance des attestations et diplômes du baccalauréat.",
    services: [
      'DEB — Secrétariat',
      "DEB — Service du baccalauréat de l'enseignement général",
      "DEB — Service du baccalauréat de l'enseignement technique professionnel",
      'DEB — Service des attestations et diplômes',
    ],
  },
];

export const DGECC_REGULATORY_DOMAINS: string[] = DGECC_ORG_STRUCTURE.map((d) => d.domain);

export const DGECC_SERVICES_LIST: string[] = DGECC_ORG_STRUCTURE.flatMap((d) => d.services);

export const DEFAULT_SERVICE_BY_DOMAIN: Record<string, string> = Object.fromEntries(
  DGECC_ORG_STRUCTURE.map((d) => [d.domain, d.services[0]]),
);
