import { ThinkingLevel } from '@google/genai';
import {
  getGeminiClient,
  isTransientGeminiError,
  withGeminiRetry,
} from '../core/geminiClient.ts';

export const OCR_SUPPORTED_MIME_TYPES = new Set([
  'application/pdf',
  'image/png',
  'image/jpeg',
  'image/webp',
]);

const OCR_PROMPT = `Tu es un moteur OCR haute fidélité spécialisé dans la numérisation de documents administratifs officiels en français (DGECC).
Transcris fidèlement et intégralement TOUT le texte visible sur ce document scanné (image ou PDF), page par page.

RÈGLES STRICTES :
1. Respecte exactement les mots, sigles (DGECC, DEB, DECEPS, DCP, DEPE, BAC, BEPC, CAP, BT), numéros de référence, dates, montants et accents français.
2. Préserve la structure logique (titres, articles, sections, listes à puces, tableaux présentés proprement ligne par ligne).
3. Ignore les artefacts de numérisation (bordures de scanner, tampons décoratifs illisibles, numéros de page isolés) sans inventer ni résumer de texte.
4. Si un passage est partiellement flou, ne devine jamais de chiffre ou de date : transcris uniquement ce qui est lisible.
5. Retourne UNIQUEMENT le texte brut transcrit, sans commentaire introductif, sans bloc markdown (\`\`\`) et sans conclusion.`;

function stripCodeFences(raw: string): string {
  return raw
    .trim()
    .replace(/^```(?:text|txt|markdown)?\s*\n?/i, '')
    .replace(/\n?```\s*$/i, '')
    .trim();
}

export interface OcrExtractionOutput {
  text: string;
  modelUsed: string;
}

let fallbackDailyExhausted = false;

function isDailyQuotaError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  const lower = msg.toLowerCase();
  return lower.includes('perday') || lower.includes('per day') || lower.includes('limit: 20');
}

export async function performDocumentOcr(
  buffer: Buffer,
  mimeType: string,
): Promise<OcrExtractionOutput> {
  return performBatchImagesOcr([buffer], mimeType, 'gemini-3.1-flash-lite');
}

export async function performBatchImagesOcr(
  buffers: Buffer[],
  mimeType: string,
  primaryModel: 'gemini-3.1-flash-lite' | 'gemini-4-hydrogen' = 'gemini-3.1-flash-lite',
): Promise<OcrExtractionOutput> {
  if (!OCR_SUPPORTED_MIME_TYPES.has(mimeType)) {
    throw new Error(
      `Type MIME non pris en charge pour l'OCR (${mimeType}). Formats OCR acceptés : PDF, PNG, JPEG, WEBP.`,
    );
  }
  if (buffers.length === 0) {
    return { text: '', modelUsed: primaryModel };
  }

  const ai = getGeminiClient();
  const imageParts = buffers.map((buf) => ({
    inlineData: {
      mimeType,
      data: buf.toString('base64'),
    },
  }));

  const callOcrModel = async (model: string): Promise<string> => {
    const thinkingLevel =
      model === 'gemini-3.1-flash-lite' ? ThinkingLevel.MINIMAL : ThinkingLevel.LOW;

    const response = await ai.models.generateContent({
      model,
      contents: {
        parts: [...imageParts, { text: OCR_PROMPT }],
      },
      config: {
        thinkingConfig: { thinkingLevel },
        abortSignal: AbortSignal.timeout(30_000),
      },
    });

    return stripCodeFences(response.text || '');
  };

  const fallbackModel =
    primaryModel === 'gemini-4-hydrogen' ? 'gemini-3.1-flash-lite' : 'gemini-4-hydrogen';

  try {
    const text = await withGeminiRetry(() => callOcrModel(primaryModel), {
      maxRetries: 4,
      baseDelayMs: 4000,
    });
    if (!text.trim() && buffers.length > 0 && !(fallbackModel === 'gemini-4-hydrogen' && fallbackDailyExhausted)) {
      const fallbackText = await withGeminiRetry(() => callOcrModel(fallbackModel), {
        maxRetries: 1,
        baseDelayMs: 2000,
      });
      return { text: fallbackText, modelUsed: fallbackModel };
    }
    return { text, modelUsed: primaryModel };
  } catch (err) {
    if (fallbackModel === 'gemini-4-hydrogen' && fallbackDailyExhausted) {
      throw err;
    }
    console.warn(
      `[DGECC OCR] ${primaryModel} fallback -> ${fallbackModel}:`,
      err instanceof Error ? err.message : String(err),
    );
    if (isTransientGeminiError(err)) {
      try {
        const text = await withGeminiRetry(() => callOcrModel(fallbackModel), {
          maxRetries: 1,
          baseDelayMs: 2000,
        });
        return { text, modelUsed: fallbackModel };
      } catch (fbErr) {
        if (fallbackModel === 'gemini-4-hydrogen' && isDailyQuotaError(fbErr)) {
          fallbackDailyExhausted = true;
        }
        throw err;
      }
    }
    throw err;
  }
}
