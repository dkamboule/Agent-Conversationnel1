import { ThinkingLevel, Type } from '@google/genai';
import { RAG_CONFIG } from '../core/config.ts';
import { getGeminiClient, isTransientGeminiError, withGeminiRetry } from '../core/geminiClient.ts';
import type { ConversationMemoryContext } from '../memory/memoryManager.ts';
import { normalizeFrenchToken, tokenizeFrench } from '../search/corpusCache.ts';

export type DgeccIntent =
  | 'CALENDRIER_EXAMEN'
  | 'CONDITIONS_CANDIDATURE'
  | 'PIECES_JUSTIFICATIVES'
  | 'INSCRIPTION_PROCEDURE'
  | 'RESULTATS_CONTESTATION'
  | 'EQUIVALENCE_CERTIFICATION'
  | 'AMENAGEMENT_HANDICAP'
  | 'ORIENTATION_SERVICE'
  | 'SALUTATION'
  | 'HORS_PERIMETRE';

export interface NlpAnalysisResult {
  intent: DgeccIntent;
  entities: string[];
  keywords: string[];
  temporalReferences: string[];
  needsRetrieval: boolean;
  reformulatedQuery: string;
  suggestedService: string;
}

const GREETING_REGEX =
  /^(bonjour|bonsoir|salut|coucou|hello|hi|merci(\s+beaucoup)?|au\s+revoir|bonne\s+(journee|soiree))\s*[!?.]*$/i;

const OUT_OF_SCOPE_REGEX =
  /\b(meteo|quel\s+temps|recette|cuisine|football|match|bourse|crypto|bitcoin|blague|poeme|film|musique)\b/i;

const ANAPHORA_REGEX =
  /\b(celui-ci|celle-ci|ceux-ci|celles-ci|ce\s+dernier|cette\s+derniere|le\s+meme|la\s+meme|y\s+inscrire|pour\s+ca|a\s+ce\s+sujet|qu'en\s+est-il|et\s+les\s+|et\s+pour)\b/i;

const KNOWN_ENTITIES: Array<{ pattern: RegExp; label: string }> = [
  { pattern: /\b(baccalaureat|bac)\b/i, label: 'Baccalauréat' },
  { pattern: /\bbepc\b/i, label: 'BEPC' },
  { pattern: /\bcap-?ceg\b/i, label: 'CAP-CEG' },
  { pattern: /\bcaet\b/i, label: 'CAET' },
  { pattern: /\bcapes\b/i, label: 'CAPES' },
  { pattern: /\bcapet\b/i, label: 'CAPET' },
  { pattern: /\bcap\b/i, label: 'CAP' },
  { pattern: /\b(bt|brevet\s+de\s+technicien)\b/i, label: 'Brevet de Technicien' },
  { pattern: /\b(concours\s+direct|concours\s+professionnel|examen\s+professionnel|fonction\s+publique)\b/i, label: 'Examens et Concours Professionnels' },
  { pattern: /\bcategorie\s+[abc]\b/i, label: 'Catégorie concours' },
  { pattern: /\b(duplicata|diplome\s+perdu)\b/i, label: 'Duplicata de diplôme' },
  { pattern: /\bequivalence\b/i, label: 'Équivalence de diplôme' },
  { pattern: /\battestation\b/i, label: 'Attestation de réussite' },
  { pattern: /\b(tiers-temps|handicap|amenagement)\b/i, label: 'Aménagement handicap / Tiers-temps' },
  { pattern: /\b(reclamation|recours|copie|jury)\b/i, label: 'Recours et consultation de copies' },
  { pattern: /\b(frais|paiement|quittance|recepisse|exoneration)\b/i, label: 'Frais et paiement' },
];

function classifyDeterministic(question: string): NlpAnalysisResult {
  const norm = normalizeFrenchToken(question.trim());
  const keywords = tokenizeFrench(question);

  // Temporal extraction
  const temporalMatches =
    norm.match(/\b(202[4-9]|janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre|\d+\s*(heures?|jours?|mois|ans?))\b/g) ||
    [];
  const temporalReferences = Array.from(new Set(temporalMatches));

  // Reference code extraction
  const refMatches = question.toUpperCase().match(/\bDGECC-\d{4}-[A-Z]+-\d+\b/g) || [];

  // Entities
  const entities: string[] = [...refMatches];
  for (const item of KNOWN_ENTITIES) {
    if (item.pattern.test(norm)) entities.push(item.label);
  }

  // Greetings
  if (GREETING_REGEX.test(norm)) {
    return {
      intent: 'SALUTATION',
      entities: [],
      keywords: [],
      temporalReferences: [],
      needsRetrieval: false,
      reformulatedQuery: question,
      suggestedService: 'Direction Générale — Secrétariat du Directeur Général',
    };
  }

  // Clearly out-of-scope without any DGECC keyword
  if (OUT_OF_SCOPE_REGEX.test(norm) && entities.length === 0) {
    return {
      intent: 'HORS_PERIMETRE',
      entities: [],
      keywords,
      temporalReferences,
      needsRetrieval: false,
      reformulatedQuery: question,
      suggestedService: 'Direction Générale — Secrétariat du Directeur Général',
    };
  }

  // Fast intent + organigramme service routing
  let intent: DgeccIntent = 'ORIENTATION_SERVICE';
  let suggestedService = 'Direction Générale — Secrétariat du Directeur Général';

  const mentionsBac = /\b(baccalaureat|bac)\b/.test(norm);
  const mentionsExamPro =
    /\b(capes|capet|cap-?ceg|caet|examen\s+professionnel|examens\s+professionnels)\b/.test(norm);
  const mentionsTechPro = /\b(technique|professionnel|cap|bt|bep)\b/.test(norm);
  const mentionsBepc = /\bbepc\b/.test(norm);

  if (/\b(tiers-temps|handicap|amenagement|medical|braille|secretaire\s+d'examen)\b/.test(norm)) {
    intent = 'AMENAGEMENT_HANDICAP';
    suggestedService = "DEPE — Service des études, de l'évaluation et de la prospection";
  } else if (/\b(reclamation|recours|contester|copie|copies|erreur\s+materielle|jury|controle\s+interne)\b/.test(norm)) {
    intent = 'RESULTATS_CONTESTATION';
    suggestedService = mentionsExamPro
      ? 'DECEPS — Service des examens professionnels'
      : 'Direction Générale — Contrôle Interne (CI)';
  } else if (/\b(duplicata|attestation|releve\s+de\s+notes|diplome\s+perdu)\b/.test(norm)) {
    intent = 'EQUIVALENCE_CERTIFICATION';
    suggestedService = mentionsBac
      ? 'DEB — Service des attestations et diplômes'
      : /\barchive\b/.test(norm)
        ? 'Direction Générale — Service des Archives et de la Documentation (SADo)'
        : 'DECEPS — Service des attestations et diplômes';
  } else if (/\b(equivalence|certification\s+professionnelle|dcp|qualification|diplome\s+etranger|vae)\b/.test(norm)) {
    intent = 'EQUIVALENCE_CERTIFICATION';
    suggestedService = 'Direction des Certifications Professionnelles (DCP)';
  } else if (/\b(statistique|statistiques|taux\s+de\s+reussite|prospective|etude|evaluation)\b/.test(norm)) {
    intent = 'ORIENTATION_SERVICE';
    suggestedService = /\bstatistique/.test(norm)
      ? 'DEPE — Service des statistiques'
      : "DEPE — Service des études, de l'évaluation et de la prospection";
  } else if (/\b(frais|paiement|quittance|recepisse|exoneration|timbre\s+fiscal|cout|coute|tarif)\b/.test(norm)) {
    intent = 'INSCRIPTION_PROCEDURE';
    suggestedService = 'Direction Générale — Services Financier (SF)';
  } else if (/\b(piece|pieces|dossier|justificatif|casier|extrait|certificat\s+medical|acte\s+de\s+naissance|cni)\b/.test(norm)) {
    intent = 'PIECES_JUSTIFICATIVES';
    suggestedService = mentionsExamPro
      ? 'DECEPS — Service des examens professionnels'
      : mentionsBac
        ? mentionsTechPro
          ? "DEB — Service du baccalauréat de l'enseignement technique professionnel"
          : "DEB — Service du baccalauréat de l'enseignement général"
        : /\bconcours\b/.test(norm)
          ? 'DECEPS — Service des examens professionnels'
          : mentionsTechPro
            ? "DECEPS — Service des examens de l'enseignement technique professionnel"
            : "DECEPS — Service des examens de l'enseignement post-primaire général";
  } else if (/\b(condition|age|diplome\s+requis|categorie\s+[abc]|eligibilite|peut-on|droit\s+civique|concours)\b/.test(norm) || mentionsExamPro) {
    intent = 'CONDITIONS_CANDIDATURE';
    suggestedService = 'DECEPS — Service des examens professionnels';
  } else if (/\b(inscrire|inscription|convocation|centre\s+d'examen|en\s+ligne)\b/.test(norm)) {
    intent = 'INSCRIPTION_PROCEDURE';
    suggestedService = mentionsExamPro
      ? 'DECEPS — Service des examens professionnels'
      : mentionsBac
        ? mentionsTechPro
          ? "DEB — Service du baccalauréat de l'enseignement technique professionnel"
          : "DEB — Service du baccalauréat de l'enseignement général"
        : mentionsBepc
          ? "DECEPS — Service des examens de l'enseignement post-primaire général"
          : mentionsTechPro
            ? "DECEPS — Service des examens de l'enseignement technique professionnel"
            : 'DECEPS — Secrétariat';
  } else if (/\b(date|dates|calendrier|quand|epreuve|epreuves|resultat|resultats|session|rattrapage|cloture|ouverture)\b/.test(norm)) {
    intent = 'CALENDRIER_EXAMEN';
    suggestedService = mentionsExamPro
      ? 'DECEPS — Service des examens professionnels'
      : mentionsBac
        ? mentionsTechPro
          ? "DEB — Service du baccalauréat de l'enseignement technique professionnel"
          : "DEB — Service du baccalauréat de l'enseignement général"
        : mentionsBepc
          ? "DECEPS — Service des examens de l'enseignement post-primaire général"
          : mentionsTechPro
            ? "DECEPS — Service des examens de l'enseignement technique professionnel"
            : 'DECEPS — Secrétariat';
  }

  return {
    intent,
    entities: Array.from(new Set(entities)),
    keywords,
    temporalReferences,
    needsRetrieval: true,
    reformulatedQuery: question,
    suggestedService,
  };
}

export async function analyzeUserQuestion(
  question: string,
  memory: ConversationMemoryContext,
): Promise<NlpAnalysisResult> {
  const fastResult = classifyDeterministic(question);

  // Only invoke LLM disambiguation when there is prior context AND ambiguous follow-up wording
  const norm = normalizeFrenchToken(question.trim());
  const wordCount = norm.split(/\s+/).filter(Boolean).length;
  const needsContextualReformulation =
    fastResult.needsRetrieval &&
    memory.recentTurns.length > 0 &&
    (ANAPHORA_REGEX.test(norm) || (wordCount <= 5 && fastResult.entities.length === 0));

  if (!needsContextualReformulation) {
    return fastResult;
  }

  try {
    const ai = getGeminiClient();
    const historySummary = memory.recentTurns
      .slice(-4)
      .map((t) => `${t.role === 'user' ? 'Usager' : 'Assistant'}: ${t.content.slice(0, 220)}`)
      .join('\n');

    const promptText = `Reformule brièvement cette question de suivi en français sous forme de requête autonome claire pour rechercher dans les textes DGECC.\nHistorique:\n${historySummary}\nQuestion de suivi: "${question}"`;
    const runReformulation = (model: string) =>
      ai.models.generateContent({
        model,
        contents: promptText,
        config: {
          thinkingConfig: {
            thinkingLevel:
              model === 'gemini-3.1-flash-lite' ? ThinkingLevel.MINIMAL : ThinkingLevel.LOW,
          },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              reformulatedQuery: { type: Type.STRING },
            },
            required: ['reformulatedQuery'],
          },
        },
      });

    let response;
    try {
      response = await withGeminiRetry(() => runReformulation(RAG_CONFIG.llmModel), {
        maxRetries: 1,
        baseDelayMs: 250,
      });
    } catch (err) {
      if (isTransientGeminiError(err) && RAG_CONFIG.evalModel !== RAG_CONFIG.llmModel) {
        response = await runReformulation(RAG_CONFIG.evalModel);
      } else {
        throw err;
      }
    }
    const parsed = JSON.parse((response.text || '{}').trim()) as {
      reformulatedQuery?: string;
    };
    if (parsed.reformulatedQuery?.trim()) {
      const merged = classifyDeterministic(parsed.reformulatedQuery.trim());
      return {
        ...merged,
        reformulatedQuery: parsed.reformulatedQuery.trim(),
      };
    }
  } catch {
    // Fall back to deterministic analysis seamlessly
  }

  return fastResult;
}
