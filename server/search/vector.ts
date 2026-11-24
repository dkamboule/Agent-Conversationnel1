import { RAG_CONFIG } from '../core/config.ts';
import {
  hasConfiguredGeminiKey,
  isQuotaError,
  isTransientGeminiError,
} from '../core/geminiClient.ts';
import {
  embedQueryUnitVector,
  ensurePublishedChunksEmbedded,
  isEmbeddingQuotaPaused,
  markEmbeddingQuotaExhausted,
} from '../embeddings/embedder.ts';
import type { RetrievedPassage, SearchFilterOptions } from './bm25.ts';
import { dotProductUnit, getPublishedCorpusIndex } from './corpusCache.ts';

export interface VectorSearchOptions extends SearchFilterOptions {
  similarityThreshold?: number;
}

export function scorePublishedChunksWithUnitVector(
  queryUnit: Float32Array,
  filters: VectorSearchOptions = {},
): RetrievedPassage[] {
  const limit = filters.limit ?? 12;
  const threshold = filters.similarityThreshold ?? RAG_CONFIG.defaultSimilarityThreshold;
  const index = getPublishedCorpusIndex();
  const scored: RetrievedPassage[] = [];

  for (const chunk of index.chunks) {
    if (!chunk.unitVector) continue;
    if (filters.category && filters.category !== 'ALL' && chunk.category !== filters.category) {
      continue;
    }
    if (filters.publishedAfter && chunk.publishedDate < filters.publishedAfter) {
      continue;
    }
    if (filters.versionNumber && chunk.versionNumber !== filters.versionNumber) {
      continue;
    }

    const sim = dotProductUnit(queryUnit, chunk.unitVector);
    if (sim >= threshold) {
      scored.push({
        chunkId: chunk.chunkId,
        documentId: chunk.documentId,
        versionNumber: chunk.versionNumber,
        chunkIndex: chunk.chunkIndex,
        title: chunk.title,
        docType: chunk.docType,
        category: chunk.category,
        serviceName: chunk.serviceName,
        referenceCode: chunk.referenceCode,
        publishedDate: chunk.publishedDate,
        validUntil: chunk.validUntil,
        content: chunk.content,
        vectorScore: Number(sim.toFixed(4)),
      });
    }
  }

  scored.sort((a, b) => (b.vectorScore || 0) - (a.vectorScore || 0));
  return scored.slice(0, limit).map((item, idx) => ({
    ...item,
    vectorRank: idx + 1,
  }));
}

export async function vectorSearch(
  query: string,
  filters: VectorSearchOptions = {},
): Promise<RetrievedPassage[]> {
  if (!hasConfiguredGeminiKey()) {
    return [];
  }

  try {
    await ensurePublishedChunksEmbedded();
  } catch (err) {
    if (isQuotaError(err) || isTransientGeminiError(err)) {
      markEmbeddingQuotaExhausted(err);
    } else {
      throw err;
    }
  }

  const index = getPublishedCorpusIndex();
  const hasAnyVectors = index.chunks.some((c) => Boolean(c.unitVector));
  if (!hasAnyVectors || isEmbeddingQuotaPaused()) {
    return [];
  }

  let queryUnit: Float32Array;
  try {
    queryUnit = await embedQueryUnitVector(query);
  } catch (err) {
    if (isQuotaError(err) || isTransientGeminiError(err)) {
      markEmbeddingQuotaExhausted(err);
      return [];
    }
    throw err;
  }

  return scorePublishedChunksWithUnitVector(queryUnit, filters);
}
