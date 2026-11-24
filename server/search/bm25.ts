import {
  getPublishedCorpusIndex,
  normalizeFrenchToken,
  tokenizeFrench,
} from './corpusCache.ts';

export { normalizeFrenchToken, tokenizeFrench };

export interface SearchFilterOptions {
  category?: string;
  publishedAfter?: string;
  versionNumber?: number;
  limit?: number;
}

export interface RetrievedPassage {
  chunkId: string;
  documentId: string;
  versionNumber: number;
  chunkIndex: number;
  title: string;
  docType: string;
  category: string;
  serviceName: string;
  referenceCode: string;
  publishedDate: string;
  validUntil: string | null;
  content: string;
  bm25Score?: number;
  bm25Rank?: number;
  vectorScore?: number;
  vectorRank?: number;
}

export function bm25Search(query: string, filters: SearchFilterOptions = {}): RetrievedPassage[] {
  const limit = filters.limit ?? 12;
  const index = getPublishedCorpusIndex();
  if (index.chunks.length === 0) return [];

  const queryTokens = tokenizeFrench(query);
  if (queryTokens.length === 0) return [];

  const k1 = 1.5;
  const b = 0.75;
  const N = index.chunks.length;
  const normalizedQuery = normalizeFrenchToken(query);

  const scored: RetrievedPassage[] = [];

  for (const chunk of index.chunks) {
    if (filters.category && filters.category !== 'ALL' && chunk.category !== filters.category) {
      continue;
    }
    if (filters.publishedAfter && chunk.publishedDate < filters.publishedAfter) {
      continue;
    }
    if (filters.versionNumber && chunk.versionNumber !== filters.versionNumber) {
      continue;
    }

    let score = 0;
    for (const qt of queryTokens) {
      const f = chunk.tf.get(qt) || 0;
      if (f === 0) continue;
      const df = index.docFreq.get(qt) || 0;
      const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
      const num = f * (k1 + 1);
      const den = f + k1 * (1 - b + b * (chunk.tokenLength / Math.max(1, index.avgDocLen)));
      score += idf * (num / den);
    }

    if (chunk.referenceNorm && normalizedQuery.includes(chunk.referenceNorm)) {
      score += 4.5;
    }

    if (score > 0) {
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
        bm25Score: Number(score.toFixed(4)),
      });
    }
  }

  scored.sort((a, bScore) => (bScore.bm25Score || 0) - (a.bm25Score || 0));
  return scored.slice(0, limit).map((item, idx) => ({
    ...item,
    bm25Rank: idx + 1,
  }));
}
