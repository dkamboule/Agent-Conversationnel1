import { RAG_CONFIG } from '../core/config.ts';
import { normalizeFrenchToken, tokenizeFrench, type RetrievedPassage } from '../search/bm25.ts';

export interface FusedPassage extends RetrievedPassage {
  rrfScore: number;
  rerankScore: number;
  isExpired: boolean;
}

export interface FusionParams {
  rrfK?: number;
  bm25Weight?: number;
  vectorWeight?: number;
  topK?: number;
  queryEntities?: string[];
  queryKeywords?: string[];
  enableReranking?: boolean;
  minRerankScore?: number;
  minBm25Score?: number;
}

export function reciprocalRankFusion(
  bm25Results: RetrievedPassage[],
  vectorResults: RetrievedPassage[],
  params: FusionParams = {},
): FusedPassage[] {
  const k = params.rrfK ?? RAG_CONFIG.defaultRrfK;
  const wBm25 = params.bm25Weight ?? RAG_CONFIG.bm25Weight;
  const wVector = params.vectorWeight ?? RAG_CONFIG.vectorWeight;
  const topK = params.topK ?? RAG_CONFIG.defaultTopK;
  const enableReranking = params.enableReranking ?? true;

  const merged = new Map<string, RetrievedPassage & { rrfScore: number }>();

  for (const item of bm25Results) {
    const rank = item.bm25Rank || 1;
    const contrib = wBm25 / (k + rank);
    const existing = merged.get(item.chunkId);
    if (existing) {
      existing.bm25Score = item.bm25Score;
      existing.bm25Rank = item.bm25Rank;
      existing.rrfScore += contrib;
    } else {
      merged.set(item.chunkId, { ...item, rrfScore: contrib });
    }
  }

  for (const item of vectorResults) {
    const rank = item.vectorRank || 1;
    const contrib = wVector / (k + rank);
    const existing = merged.get(item.chunkId);
    if (existing) {
      existing.vectorScore = item.vectorScore;
      existing.vectorRank = item.vectorRank;
      existing.rrfScore += contrib;
    } else {
      merged.set(item.chunkId, { ...item, rrfScore: contrib });
    }
  }

  const todayIso = new Date().toISOString().slice(0, 10);
  const entityTokens = enableReranking
    ? (params.queryEntities || []).flatMap((e) => tokenizeFrench(e))
    : [];
  const keywordTokens = enableReranking
    ? (params.queryKeywords || []).flatMap((kw) => tokenizeFrench(kw))
    : [];
  const targetTerms = Array.from(new Set([...entityTokens, ...keywordTokens]));

  const reranked: FusedPassage[] = Array.from(merged.values()).map((item) => {
    const isExpired = Boolean(item.validUntil && item.validUntil < todayIso);
    if (!enableReranking) {
      const pureScore = Number(item.rrfScore.toFixed(5));
      return {
        ...item,
        rrfScore: pureScore,
        rerankScore: pureScore,
        isExpired,
      };
    }

    const normalizedContent = normalizeFrenchToken(`${item.title} ${item.content}`);
    let entityBoost = 0;
    for (const term of targetTerms) {
      if (normalizedContent.includes(term)) {
        entityBoost += 0.004;
      }
    }
    const freshnessFactor = isExpired ? 0.7 : 1.0;
    const dualMatchBonus = item.bm25Rank && item.vectorRank ? 0.008 : 0;

    const rerankScore = Number(
      ((item.rrfScore + entityBoost + dualMatchBonus) * freshnessFactor).toFixed(5),
    );

    return {
      ...item,
      rrfScore: Number(item.rrfScore.toFixed(5)),
      rerankScore,
      isExpired,
    };
  });

  const filtered = reranked.filter((item) => {
    if (params.minRerankScore !== undefined && item.rerankScore < params.minRerankScore) {
      return false;
    }
    if (
      params.minBm25Score !== undefined &&
      !item.vectorRank &&
      (item.bm25Score ?? 0) < params.minBm25Score
    ) {
      return false;
    }
    return true;
  });

  filtered.sort((a, b) => b.rerankScore - a.rerankScore);
  return filtered.slice(0, topK);
}
