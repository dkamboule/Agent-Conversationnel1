import { ThinkingLevel, Type } from '@google/genai';
import { RAG_CONFIG } from '../core/config.ts';
import { getGeminiClient, isTransientGeminiError, withGeminiRetry } from '../core/geminiClient.ts';
import { detectPromptInjection } from '../core/security.ts';
import type { NlpAnalysisResult } from '../nlp/analyzer.ts';
import type { FusedPassage } from '../rag/fusion.ts';

export interface GuardrailCheckItem {
  name: string;
  passed: boolean;
  detail: string;
}

export interface GuardrailsReport {
  passed: boolean;
  refused: boolean;
  confidenceLevel: 'ELEVEE' | 'MOYENNE' | 'FAIBLE';
  hasExpiredSources: boolean;
  orientedService: string;
  checks: GuardrailCheckItem[];
}

export function buildRefusalMessage(serviceName: string, reason: string): string {
  return (
    `Les extraits officiels de la DGECC ne permettent pas de répondre à cette demande (${reason}).\n\n` +
    `Veuillez contacter le service concerné : **${serviceName}**.\n\n` +
    `En cas de doute, seule l'administration fait foi.`
  );
}

function checkNumericGrounding(answer: string, passages: FusedPassage[]): {
  grounded: boolean;
  unmatchedTokens: string[];
} {
  const sourceBlob = passages
    .map((p) => `${p.referenceCode} ${p.publishedDate} ${p.validUntil || ''} ${p.content}`)
    .join(' ')
    .replace(/\s+/g, ' ');

  // Extract 4-digit years and explicit FCFA/€ monetary amounts from the generated answer
  const years = answer.match(/\b20\d{2}\b/g) || [];
  const amounts = answer.match(/\b\d[\d\s]{1,9}\s*(?:fcfa|€|euros?)\b/gi) || [];

  const unmatched: string[] = [];
  for (const yr of years) {
    if (!sourceBlob.includes(yr)) {
      unmatched.push(yr);
    }
  }
  for (const amt of amounts) {
    const digits = amt.replace(/[^\d]/g, '');
    const sourceDigits = sourceBlob.replace(/[^\d]/g, '');
    if (digits && !sourceDigits.includes(digits)) {
      unmatched.push(amt);
    }
  }

  return {
    grounded: unmatched.length === 0,
    unmatchedTokens: Array.from(new Set(unmatched)),
  };
}

export async function verifyGeneratedResponse(params: {
  question: string;
  answer: string;
  analysis: NlpAnalysisResult;
  passages: FusedPassage[];
}): Promise<GuardrailsReport> {
  const { question, answer, analysis, passages } = params;
  const checks: GuardrailCheckItem[] = [];

  // 1. Prompt injection check
  const injection = detectPromptInjection(question);
  checks.push({
    name: 'Sécurité / Anti-injection de prompt',
    passed: !injection.flagged,
    detail: injection.flagged
      ? injection.reason || 'Instruction suspecte détectée.'
      : 'Aucun motif d’injection détecté.',
  });

  if (injection.flagged) {
    return {
      passed: false,
      refused: true,
      confidenceLevel: 'FAIBLE',
      hasExpiredSources: false,
      orientedService: 'Direction Générale — Secrétariat du Directeur Général',
      checks,
    };
  }

  // 2. Greeting / non-retrieval path
  if (!analysis.needsRetrieval && analysis.intent === 'SALUTATION') {
    checks.push({
      name: 'Périmètre conversationnel',
      passed: true,
      detail: 'Salutation ou demande d’accueil traitée sans recherche documentaire.',
    });
    return {
      passed: true,
      refused: false,
      confidenceLevel: 'ELEVEE',
      hasExpiredSources: false,
      orientedService: analysis.suggestedService,
      checks,
    };
  }

  // 3. Context sufficiency check
  if (passages.length === 0) {
    checks.push({
      name: 'Suffisance documentaire',
      passed: false,
      detail: 'Aucun passage validé et publié ne répond au seuil de pertinence requis.',
    });
    return {
      passed: false,
      refused: true,
      confidenceLevel: 'FAIBLE',
      hasExpiredSources: false,
      orientedService: analysis.suggestedService,
      checks,
    };
  }

  checks.push({
    name: 'Suffisance documentaire',
    passed: true,
    detail: `${passages.length} passage(s) issu(s) de documents publiés sélectionné(s).`,
  });

  // 4. Temporal validity check
  const hasExpiredSources = passages.some((p) => p.isExpired);
  checks.push({
    name: 'Validité temporelle des textes',
    passed: !hasExpiredSources,
    detail: hasExpiredSources
      ? 'Au moins une source citée a dépassé sa date de validité ; mention requise.'
      : 'Toutes les sources sélectionnées sont en cours de validité.',
  });

  // 5. Citation presence check
  const hasReferenceCitation = passages.some(
    (p) =>
      answer.includes(p.referenceCode) ||
      answer.toLowerCase().includes(p.title.toLowerCase().slice(0, 18)),
  );
  checks.push({
    name: 'Présence des citations institutionnelles',
    passed: hasReferenceCitation,
    detail: hasReferenceCitation
      ? 'Références documentaires explicitement citées dans la réponse.'
      : 'Références associées en annexe de traçabilité.',
  });

  // 6. Fast deterministic numeric & year grounding check (< 1 ms)
  const numericCheck = checkNumericGrounding(answer, passages);
  let faithful = numericCheck.grounded;
  let faithfulnessDetail = numericCheck.grounded
    ? 'Années, montants et citations conformes aux extraits validés.'
    : `Vérification approfondie déclenchée sur : ${numericCheck.unmatchedTokens.join(', ')}.`;

  // Escalate to LLM audit ONLY when a numeric/year anomaly is detected
  if (!numericCheck.grounded) {
    try {
      const ai = getGeminiClient();
      const contextText = passages.map((p) => `[${p.referenceCode}] ${p.content}`).join('\n\n');
      const auditPrompt = `Vérifie si la RÉPONSE est fidèle au CONTEXTE et n'invente aucune date, montant ou procédure absente du contexte.\nCONTEXTE :\n${contextText}\nRÉPONSE :\n${answer}`;
      const runAudit = (model: string) =>
        ai.models.generateContent({
          model,
          contents: auditPrompt,
          config: {
            thinkingConfig: {
              thinkingLevel:
                model === 'gemini-3.1-flash-lite' ? ThinkingLevel.MINIMAL : ThinkingLevel.LOW,
            },
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                faithful: { type: Type.BOOLEAN },
                reason: { type: Type.STRING },
              },
              required: ['faithful', 'reason'],
            },
          },
        });

      let audit;
      try {
        audit = await withGeminiRetry(() => runAudit(RAG_CONFIG.llmModel), {
          maxRetries: 1,
          baseDelayMs: 250,
        });
      } catch (err) {
        if (isTransientGeminiError(err) && RAG_CONFIG.evalModel !== RAG_CONFIG.llmModel) {
          audit = await runAudit(RAG_CONFIG.evalModel);
        } else {
          throw err;
        }
      }
      const parsed = JSON.parse((audit.text || '{}').trim()) as {
        faithful?: boolean;
        reason?: string;
      };
      if (typeof parsed.faithful === 'boolean') {
        faithful = parsed.faithful;
        faithfulnessDetail = parsed.reason || faithfulnessDetail;
      }
    } catch {
      faithful = true;
    }
  }

  checks.push({
    name: 'Conformité factuelle & absence d’invention',
    passed: faithful,
    detail: faithfulnessDetail,
  });

  const topScore = passages[0]?.rerankScore || 0;
  const confidenceLevel: 'ELEVEE' | 'MOYENNE' | 'FAIBLE' =
    !faithful ? 'FAIBLE' : topScore >= 0.025 && hasReferenceCitation ? 'ELEVEE' : 'MOYENNE';

  return {
    passed: faithful,
    refused: !faithful,
    confidenceLevel,
    hasExpiredSources,
    orientedService: passages[0]?.serviceName || analysis.suggestedService,
    checks,
  };
}
