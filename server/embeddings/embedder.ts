import { RAG_CONFIG } from '../core/config.ts';
import {
  getGeminiClient,
  hasConfiguredGeminiKey,
  isDailyQuotaError,
  isQuotaError,
  isTransientGeminiError,
  withGeminiRetry,
} from '../core/geminiClient.ts';
import { getDb } from '../database/db.ts';
import {
  invalidateCorpusCache,
  normalizeFrenchToken,
  toUnitFloat32,
} from '../search/corpusCache.ts';

export interface EmbeddingProvider {
  modelName: string;
  embedTexts(texts: string[]): Promise<number[][]>;
}

export class GeminiEmbeddingProvider implements EmbeddingProvider {
  public readonly modelName: string;

  constructor(modelName = RAG_CONFIG.embeddingModel) {
    this.modelName = modelName;
  }

  async embedTexts(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const ai = getGeminiClient();
    const vectors: number[][] = new Array(texts.length);
    const batchSize = 32;

    for (let i = 0; i < texts.length; i += batchSize) {
      const slice = texts.slice(i, i + batchSize);
      const response = await withGeminiRetry(
        () =>
          ai.models.embedContent({
            model: this.modelName,
            contents: slice,
          }),
        { maxRetries: 4, baseDelayMs: 4000 },
      );

      const embeddings = response.embeddings || [];
      if (embeddings.length === slice.length) {
        for (let j = 0; j < slice.length; j++) {
          const values = embeddings[j]?.values;
          if (!values || values.length === 0) {
            throw new Error("Vecteur d'embedding vide retourné par le modèle.");
          }
          vectors[i + j] = values;
        }
      } else {
        // Fallback one-by-one if a provider returns fewer vectors than requested
        for (let j = 0; j < slice.length; j++) {
          const singleRes = await withGeminiRetry(
            () =>
              ai.models.embedContent({
                model: this.modelName,
                contents: slice[j],
              }),
            { maxRetries: 4, baseDelayMs: 4000 },
          );
          const values = singleRes.embeddings?.[0]?.values;
          if (!values || values.length === 0) {
            throw new Error("Vecteur d'embedding vide retourné par le modèle.");
          }
          vectors[i + j] = values;
        }
      }

      if (i + batchSize < texts.length) {
        await new Promise((r) => setTimeout(r, 700));
      }
    }
    return vectors;
  }
}

export function getEmbeddingProvider(): EmbeddingProvider {
  return new GeminiEmbeddingProvider(RAG_CONFIG.embeddingModel);
}

// LRU cache for query embeddings to eliminate repeated embedding API round-trips
const MAX_QUERY_EMBED_CACHE = 250;
const queryEmbeddingCache = new Map<string, Float32Array>();

export async function embedQueryUnitVector(query: string): Promise<Float32Array> {
  const key = normalizeFrenchToken(query.trim());
  const cached = queryEmbeddingCache.get(key);
  if (cached) {
    // Refresh LRU order
    queryEmbeddingCache.delete(key);
    queryEmbeddingCache.set(key, cached);
    return cached;
  }

  const provider = getEmbeddingProvider();
  const [vec] = await provider.embedTexts([query]);
  const unit = toUnitFloat32(vec);

  if (queryEmbeddingCache.size >= MAX_QUERY_EMBED_CACHE) {
    const oldestKey = queryEmbeddingCache.keys().next().value;
    if (oldestKey !== undefined) queryEmbeddingCache.delete(oldestKey);
  }
  queryEmbeddingCache.set(key, unit);
  return unit;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  if (len === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < len; i++) {
    const va = a[i];
    const vb = b[i];
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

let embeddingInFlight: Promise<number> | null = null;
let embeddingQuotaPausedUntil = 0;
let lastQuotaWarnAt = 0;

export function isEmbeddingQuotaPaused(): boolean {
  return Date.now() < embeddingQuotaPausedUntil;
}

export function markEmbeddingQuotaExhausted(err: unknown): void {
  const daily = isDailyQuotaError(err);
  const pauseMs = daily ? 60 * 60_000 : 10 * 60_000;
  embeddingQuotaPausedUntil = Date.now() + pauseMs;

  if (Date.now() - lastQuotaWarnAt > 60_000) {
    lastQuotaWarnAt = Date.now();
    console.warn(
      `[DGECC] Quota embeddings (${RAG_CONFIG.embeddingModel}) atteint — recherche BM25 active, vectorisation différée (${daily ? 'quota journalier' : 'pause temporaire'}).`,
    );
  }
}

export async function ensurePublishedChunksEmbedded(): Promise<number> {
  if (!hasConfiguredGeminiKey() || isEmbeddingQuotaPaused()) return 0;
  if (embeddingInFlight) return embeddingInFlight;

  embeddingInFlight = (async () => {
    const db = getDb();
    const rows = db
      .prepare(
        `SELECT c.id, c.content
         FROM document_chunks c
         JOIN documents d ON d.id = c.document_id AND d.current_version = c.version_number
         WHERE d.status = 'PUBLIE' AND (c.embedding_json IS NULL OR c.embedding_json = '')`,
      )
      .all() as Array<{ id: string; content: string }>;

    if (rows.length === 0) return 0;

    const provider = getEmbeddingProvider();
    const updateStmt = db.prepare(
      `UPDATE document_chunks SET embedding_json = ?, embedding_model = ? WHERE id = ?`,
    );

    const persistBatchSize = 64;
    let embeddedCount = 0;
    try {
      for (let i = 0; i < rows.length; i += persistBatchSize) {
        const batchRows = rows.slice(i, i + persistBatchSize);
        const vectors = await provider.embedTexts(batchRows.map((r) => r.content));
        db.exec('BEGIN TRANSACTION;');
        try {
          for (let j = 0; j < batchRows.length; j++) {
            updateStmt.run(JSON.stringify(vectors[j]), provider.modelName, batchRows[j].id);
          }
          db.exec('COMMIT;');
          embeddedCount += batchRows.length;
        } catch (err) {
          db.exec('ROLLBACK;');
          throw err;
        }
      }
    } catch (err) {
      if (isQuotaError(err) || isTransientGeminiError(err)) {
        markEmbeddingQuotaExhausted(err);
        if (embeddedCount > 0) invalidateCorpusCache();
        return embeddedCount;
      }
      throw err;
    }

    invalidateCorpusCache();
    return embeddedCount;
  })();

  try {
    return await embeddingInFlight;
  } finally {
    embeddingInFlight = null;
  }
}
