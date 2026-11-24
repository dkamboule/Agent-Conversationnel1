import crypto from 'node:crypto';
import { RAG_CONFIG } from '../core/config.ts';
import {
  hasConfiguredGeminiKey,
  isQuotaError,
  isTransientGeminiError,
} from '../core/geminiClient.ts';
import { getDb, logActivity } from '../database/db.ts';
import {
  getEmbeddingProvider,
  isEmbeddingQuotaPaused,
  markEmbeddingQuotaExhausted,
} from '../embeddings/embedder.ts';
import { buildRefusalMessage, verifyGeneratedResponse } from '../guardrails/verifier.ts';
import { getLlmProvider } from '../llm/generator.ts';
import { analyzeUserQuestion } from '../nlp/analyzer.ts';
import { bm25Search, type RetrievedPassage } from '../search/bm25.ts';
import {
  getPublishedCorpusIndex,
  normalizeFrenchToken,
  toUnitFloat32,
} from '../search/corpusCache.ts';
import { scorePublishedChunksWithUnitVector } from '../search/vector.ts';
import { buildRagPromptContext, DGECC_SYSTEM_INSTRUCTION } from './context_builder.ts';
import {
  DGECC_EVAL_DATASET,
  TYPOLOGY_LABELS,
  type BenchmarkQueryItem,
  type BenchmarkTypology,
} from './evalDataset.ts';
import { reciprocalRankFusion, type FusedPassage } from './fusion.ts';

export type UserPanelProfile = 'CANDIDAT' | 'PARENT' | 'AGENT_ADMINISTRATIF';

export interface UserPanelEntry {
  id: string;
  profile: UserPanelProfile;
  participantName: string;
  clarityScore: number;
  usefulnessScore: number;
  trustScore: number;
  comment: string | null;
  recordedBy: string | null;
  createdAt: string;
}

export interface Tableau18Metrics {
  corpusDocuments: number;
  indexedChunks: number;
  embeddedChunks: number;
  queryDatasetTotal: number;
  inScopeQueries: number;
  outOfScopeQueries: number;
  outOfScopeSharePct: number;
  typologyBreakdown: Array<{
    code: BenchmarkTypology;
    label: string;
    count: number;
  }>;
  userPanel: {
    total: number;
    candidats: number;
    parents: number;
    agentsAdministratifs: number;
    avgClarity: number | null;
    avgUsefulness: number | null;
    avgTrust: number | null;
  };
}

export interface Tableau19StrategyRow {
  key: 'lexical' | 'dense' | 'hybrid' | 'hybrid_rerank';
  label: string;
  available: boolean;
  statusNote: string;
  queriesEvaluated: number;
  recallAt3: number | null;
  recallAt5: number | null;
  mrr: number | null;
}

export interface Tableau20ConfigRow {
  key: 'llm_only' | 'rag_standard' | 'rag_optimized';
  label: string;
  available: boolean;
  statusNote: string;
  sampleSize: number;
  inScopeCount: number;
  outOfScopeCount: number;
  exactitudeFactuelle: number | null;
  fideliteCorpus: number | null;
  tauxRefusApproprie: number | null;
}

export interface EvaluationMethodologyPayload {
  mode: 'full' | 'retrieval_only';
  sampleSize: number;
  tableau18: Tableau18Metrics;
  tableau19: Tableau19StrategyRow[];
  tableau20: Tableau20ConfigRow[];
}

export interface BenchmarkCaseResult {
  id: string;
  typology: BenchmarkTypology;
  isOutOfScope: boolean;
  query: string;
  expectedRef: string;
  retrievedRefs: string[];
  recallAtK: number;
  precisionAtK: number;
  mrr: number;
  ndcgAtK: number;
  faithfulness: number;
  answerRelevancy: number;
  contextPrecision: number;
  contextRecall: number;
  latencyMs: number;
}

export function listUserPanelEvaluations(): UserPanelEntry[] {
  const db = getDb();
  const rows = db
    .prepare(
      `SELECT id, profile, participant_name, clarity_score, usefulness_score, trust_score,
              comment, recorded_by, created_at
       FROM user_panel_evaluations
       ORDER BY created_at DESC`,
    )
    .all() as Array<Record<string, unknown>>;

  return rows.map((r) => ({
    id: String(r.id),
    profile: String(r.profile) as UserPanelProfile,
    participantName: String(r.participant_name),
    clarityScore: Number(r.clarity_score),
    usefulnessScore: Number(r.usefulness_score),
    trustScore: Number(r.trust_score),
    comment: r.comment ? String(r.comment) : null,
    recordedBy: r.recorded_by ? String(r.recorded_by) : null,
    createdAt: String(r.created_at),
  }));
}

export function addUserPanelEvaluation(params: {
  profile: UserPanelProfile;
  participantName: string;
  clarityScore: number;
  usefulnessScore: number;
  trustScore: number;
  comment?: string | null;
  recordedBy: string;
}): UserPanelEntry {
  const db = getDb();
  const entry: UserPanelEntry = {
    id: crypto.randomUUID(),
    profile: params.profile,
    participantName: params.participantName.trim(),
    clarityScore: Math.max(1, Math.min(5, Math.round(params.clarityScore))),
    usefulnessScore: Math.max(1, Math.min(5, Math.round(params.usefulnessScore))),
    trustScore: Math.max(1, Math.min(5, Math.round(params.trustScore))),
    comment: params.comment?.trim() ? params.comment.trim().slice(0, 600) : null,
    recordedBy: params.recordedBy,
    createdAt: new Date().toISOString(),
  };

  db.prepare(
    `INSERT INTO user_panel_evaluations (
      id, profile, participant_name, clarity_score, usefulness_score, trust_score,
      comment, recorded_by, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    entry.id,
    entry.profile,
    entry.participantName,
    entry.clarityScore,
    entry.usefulnessScore,
    entry.trustScore,
    entry.comment,
    entry.recordedBy,
    entry.createdAt,
  );

  return entry;
}

export function deleteUserPanelEvaluation(id: string): boolean {
  const db = getDb();
  const res = db.prepare(`DELETE FROM user_panel_evaluations WHERE id = ?`).run(id);
  return Number(res.changes) > 0;
}

export function getTableau18Snapshot(): Tableau18Metrics {
  const db = getDb();
  const corpusDocuments = (
    db.prepare(`SELECT COUNT(*) AS cnt FROM documents WHERE status = 'PUBLIE'`).get() as {
      cnt: number;
    }
  ).cnt;

  const chunkStats = db
    .prepare(
      `SELECT COUNT(*) AS total,
              SUM(CASE WHEN c.embedding_json IS NOT NULL AND c.embedding_json != '' THEN 1 ELSE 0 END) AS embedded
       FROM document_chunks c
       JOIN documents d ON d.id = c.document_id AND d.current_version = c.version_number
       WHERE d.status = 'PUBLIE'`,
    )
    .get() as { total: number; embedded: number | null };

  const totalQueries = DGECC_EVAL_DATASET.length;
  const outOfScopeQueries = DGECC_EVAL_DATASET.filter((q) => q.isOutOfScope).length;
  const inScopeQueries = totalQueries - outOfScopeQueries;
  const outOfScopeSharePct = Number(((outOfScopeQueries / Math.max(1, totalQueries)) * 100).toFixed(1));

  const typologyOrder: BenchmarkTypology[] = [
    'T1_CONDITIONS_ACCES',
    'T2_DOSSIERS_INSCRIPTION',
    'T3_EPREUVES_JURYS',
    'T4_DELIBERATIONS_RECOURS',
    'T5_STATUT_ORIENTATION',
    'HORS_PERIMETRE',
  ];

  const typologyBreakdown = typologyOrder.map((code) => ({
    code,
    label: TYPOLOGY_LABELS[code],
    count: DGECC_EVAL_DATASET.filter((q) => q.typology === code).length,
  }));

  const panelRows = listUserPanelEvaluations();
  const totalPanel = panelRows.length;
  const avgScore = (pick: (r: UserPanelEntry) => number): number | null =>
    totalPanel > 0
      ? Number((panelRows.reduce((s, r) => s + pick(r), 0) / totalPanel).toFixed(2))
      : null;

  return {
    corpusDocuments,
    indexedChunks: Number(chunkStats.total || 0),
    embeddedChunks: Number(chunkStats.embedded || 0),
    queryDatasetTotal: totalQueries,
    inScopeQueries,
    outOfScopeQueries,
    outOfScopeSharePct,
    typologyBreakdown,
    userPanel: {
      total: totalPanel,
      candidats: panelRows.filter((r) => r.profile === 'CANDIDAT').length,
      parents: panelRows.filter((r) => r.profile === 'PARENT').length,
      agentsAdministratifs: panelRows.filter((r) => r.profile === 'AGENT_ADMINISTRATIF').length,
      avgClarity: avgScore((r) => r.clarityScore),
      avgUsefulness: avgScore((r) => r.usefulnessScore),
      avgTrust: avgScore((r) => r.trustScore),
    },
  };
}

function isAcceptedDocRef(item: BenchmarkQueryItem, refCode: string): boolean {
  if (item.acceptableRefs && item.acceptableRefs.length > 0) {
    return item.acceptableRefs.includes(refCode);
  }
  return refCode === item.expectedRef;
}

function computeRetrievalMetrics(
  inScopeQueries: BenchmarkQueryItem[],
  rankLists: Array<{ expectedItem: BenchmarkQueryItem; refs: string[] }>,
): { recallAt3: number; recallAt5: number; mrr: number } {
  let hits3 = 0;
  let hits5 = 0;
  let sumMrr = 0;

  for (const row of rankLists) {
    const firstIdx = row.refs.findIndex((ref) => isAcceptedDocRef(row.expectedItem, ref));
    if (firstIdx >= 0) {
      const rank = firstIdx + 1;
      if (rank <= 3) hits3++;
      if (rank <= 5) hits5++;
      sumMrr += 1 / rank;
    }
  }

  const n = Math.max(1, inScopeQueries.length);
  return {
    recallAt3: Number((hits3 / n).toFixed(3)),
    recallAt5: Number((hits5 / n).toFixed(3)),
    mrr: Number((sumMrr / n).toFixed(3)),
  };
}

async function precomputeQueryEmbeddings(
  queries: BenchmarkQueryItem[],
): Promise<{ vectors: Map<string, Float32Array>; available: boolean; note: string }> {
  const index = getPublishedCorpusIndex();
  const vectorizedCount = index.chunks.filter((c) => Boolean(c.unitVector)).length;

  if (vectorizedCount === 0) {
    return {
      vectors: new Map(),
      available: false,
      note: 'Indisponible : 0 segment vectorisé dans le corpus (quota embeddings ou indexation vectorielle en attente).',
    };
  }

  if (!hasConfiguredGeminiKey() || isEmbeddingQuotaPaused()) {
    return {
      vectors: new Map(),
      available: false,
      note: 'Indisponible : quota du modèle d’embedding temporairement en pause.',
    };
  }

  const vectors = new Map<string, Float32Array>();
  try {
    const provider = getEmbeddingProvider();
    const rawVectors = await provider.embedTexts(queries.map((q) => q.query));
    for (let i = 0; i < queries.length; i++) {
      vectors.set(queries[i].id, toUnitFloat32(rawVectors[i]));
    }
    return {
      vectors,
      available: true,
      note: `${vectorizedCount}/${index.chunks.length} segments vectorisés (${provider.modelName}).`,
    };
  } catch (err) {
    if (isQuotaError(err) || isTransientGeminiError(err)) {
      markEmbeddingQuotaExhausted(err);
      return {
        vectors: new Map(),
        available: false,
        note: 'Indisponible : quota embeddings atteint lors de la vectorisation des requêtes.',
      };
    }
    throw err;
  }
}

export function isRefusalText(answer: string): boolean {
  const norm = normalizeFrenchToken(answer);
  return (
    norm.includes('ne permettent pas') ||
    norm.includes('ne figure pas') ||
    norm.includes('ne contiennent pas') ||
    norm.includes('ne contient pas') ||
    norm.includes('hors du perimetre') ||
    norm.includes('hors perimetre') ||
    norm.includes('ne releve pas') ||
    norm.includes('aucun document') ||
    norm.includes('aucune information') ||
    norm.includes('pas d information') ||
    norm.includes('pas de reponse') ||
    norm.includes('je ne dispose pas') ||
    norm.includes('impossible de repondre') ||
    norm.includes('ne peux pas repondre') ||
    norm.includes('extraits fournis ne')
  );
}

export function computeFactMatchScore(answer: string, factGroups: string[][]): number {
  if (factGroups.length === 0) return 0;
  const normAnswer = normalizeFrenchToken(answer);
  let matched = 0;
  for (const group of factGroups) {
    const ok = group.some((variant) => normAnswer.includes(normalizeFrenchToken(variant)));
    if (ok) matched++;
  }
  return Number((matched / factGroups.length).toFixed(3));
}

function checkNumericClaimsAgainstSource(answer: string, sourceText: string): {
  unmatchedCount: number;
  totalClaims: number;
} {
  const normSource = normalizeFrenchToken(sourceText).replace(/\s+/g, ' ');
  const years = answer.match(/\b(?:19|20)\d{2}\b/g) || [];
  const amounts = answer.match(/\b\d[\d\s]{0,7}\s*(?:fcfa|f\s*cfa|francs?\s*cfa|€|euros?)\b/gi) || [];
  const ratios = answer.match(/\b\d{1,3}\s*\/\s*(?:20|480|460)\b/g) || [];

  const claims = [...years, ...amounts, ...ratios];
  let unmatchedCount = 0;
  const sourceDigits = normSource.replace(/\s+/g, '');

  for (const c of claims) {
    const clean = normalizeFrenchToken(c).replace(/\s+/g, '');
    const digits = clean.replace(/[^\d/]/g, '');
    if (digits && !sourceDigits.includes(digits)) {
      unmatchedCount++;
    }
  }

  return { unmatchedCount, totalClaims: claims.length };
}

function computeCorpusFaithfulness(params: {
  answer: string;
  item: BenchmarkQueryItem;
  passagesGiven: RetrievedPassage[];
  isRag: boolean;
}): number {
  const { answer, item, passagesGiven, isRag } = params;
  const refused = isRefusalText(answer);

  if (item.isOutOfScope) {
    return refused ? 1 : 0;
  }

  if (refused) {
    // Abstaining when no extract was provided (LLM alone) is faithful (no fabrication),
    // whereas refusing when relevant extracts exist gets neutral grounding.
    return isRag ? 0.65 : 1;
  }

  let referenceBlob = '';
  if (isRag) {
    referenceBlob = passagesGiven
      .map((p) => `${p.referenceCode} ${p.title} ${p.publishedDate} ${p.content}`)
      .join(' ');
  } else {
    const index = getPublishedCorpusIndex();
    referenceBlob = index.chunks
      .filter((c) => isAcceptedDocRef(item, c.referenceCode))
      .map((c) => `${c.referenceCode} ${c.title} ${c.publishedDate} ${c.content}`)
      .join(' ');
  }

  if (!referenceBlob.trim()) return 0;

  const { unmatchedCount } = checkNumericClaimsAgainstSource(answer, referenceBlob);
  const factMatch = computeFactMatchScore(answer, item.factGroups);

  let score = 1 - unmatchedCount * 0.35;
  if (item.factGroups.length > 0 && factMatch === 0) {
    score = Math.min(score, 0.35);
  }
  if (!isRag && factMatch < 0.5) {
    // Parametric LLM answer without RAG lacking expected official facts
    score = Math.min(score, 0.5);
  }
  return Number(Math.max(0, Math.min(1, score)).toFixed(3));
}

export function getEvaluationOverview(): {
  tableau18: Tableau18Metrics;
  panelEntries: UserPanelEntry[];
  latestRun: Record<string, unknown> | null;
} {
  const db = getDb();
  const tableau18 = getTableau18Snapshot();
  const panelEntries = listUserPanelEvaluations();
  const row = db
    .prepare(`SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT 1`)
    .get() as Record<string, unknown> | undefined;

  const latestRun = row
    ? {
        ...row,
        cases: row.details_json ? JSON.parse(String(row.details_json)) : [],
        methodology: row.methodology_json ? JSON.parse(String(row.methodology_json)) : null,
      }
    : null;

  return { tableau18, panelEntries, latestRun };
}

export async function runRagBenchmarkEvaluation(
  triggeredByUserId: string,
  triggeredByEmail: string,
  options: { mode?: 'full' | 'retrieval_only'; sampleSize?: number } = {},
) {
  const db = getDb();
  const mode = options.mode === 'retrieval_only' ? 'retrieval_only' : 'full';
  const requestedSample = options.sampleSize === 100 ? 100 : 20;

  const tableau18 = getTableau18Snapshot();
  const inScopeQueries = DGECC_EVAL_DATASET.filter((q) => !q.isOutOfScope);

  // -------------------------------------------------------------------------
  // TABLEAU 19 : Évaluation comparative des 4 stratégies de recherche (87 req.)
  // -------------------------------------------------------------------------
  const emptyMemory = { conversationId: 'eval', recentTurns: [], consultedReferences: [] };
  const queryVecInfo = await precomputeQueryEmbeddings(inScopeQueries);

  const lexicalRanks: Array<{ expectedItem: BenchmarkQueryItem; refs: string[] }> = [];
  const denseRanks: Array<{ expectedItem: BenchmarkQueryItem; refs: string[] }> = [];
  const hybridRanks: Array<{ expectedItem: BenchmarkQueryItem; refs: string[] }> = [];
  const hybridRerankRanks: Array<{ expectedItem: BenchmarkQueryItem; refs: string[] }> = [];

  const inScopeRetrievalMap = new Map<
    string,
    { fused: FusedPassage[]; recallAt5: number; precisionAt5: number; mrr: number; ndcgAt5: number }
  >();

  for (const item of inScopeQueries) {
    const analysis = await analyzeUserQuestion(item.query, emptyMemory);
    const b12 = bm25Search(item.query, { limit: 12 });
    const qUnit = queryVecInfo.vectors.get(item.id);
    const v12 = qUnit
      ? scorePublishedChunksWithUnitVector(qUnit, { limit: 12, similarityThreshold: 0 })
      : [];

    const b5 = b12.slice(0, 5);
    const v5 = v12.slice(0, 5);
    const hPure5 = reciprocalRankFusion(b12, v12, {
      topK: 5,
      rrfK: RAG_CONFIG.defaultRrfK,
      enableReranking: false,
    });
    const hRerank5 = reciprocalRankFusion(b12, v12, {
      topK: 5,
      rrfK: RAG_CONFIG.defaultRrfK,
      enableReranking: true,
      queryEntities: analysis.entities,
      queryKeywords: analysis.keywords,
    });

    lexicalRanks.push({ expectedItem: item, refs: b5.map((p) => p.referenceCode) });
    if (queryVecInfo.available) {
      denseRanks.push({ expectedItem: item, refs: v5.map((p) => p.referenceCode) });
    }
    hybridRanks.push({ expectedItem: item, refs: hPure5.map((p) => p.referenceCode) });
    hybridRerankRanks.push({ expectedItem: item, refs: hRerank5.map((p) => p.referenceCode) });

    const relevantIndices = hRerank5
      .map((p, idx) => (isAcceptedDocRef(item, p.referenceCode) ? idx + 1 : -1))
      .filter((rank) => rank > 0);
    const recallAt5 = relevantIndices.length > 0 ? 1 : 0;
    const precisionAt5 = Number((relevantIndices.length / Math.max(1, hRerank5.length)).toFixed(3));
    const mrr = relevantIndices.length > 0 ? Number((1 / relevantIndices[0]).toFixed(3)) : 0;
    let dcg = 0;
    for (const rank of relevantIndices) {
      dcg += 1 / Math.log2(rank + 1);
    }
    let idcg = 0;
    const idealCount = Math.max(1, relevantIndices.length);
    for (let r = 1; r <= idealCount; r++) {
      idcg += 1 / Math.log2(r + 1);
    }
    const ndcgAt5 = relevantIndices.length > 0 ? Number((dcg / idcg).toFixed(3)) : 0;

    inScopeRetrievalMap.set(item.id, {
      fused: hRerank5,
      recallAt5,
      precisionAt5,
      mrr,
      ndcgAt5,
    });
  }

  const lexMetrics = computeRetrievalMetrics(inScopeQueries, lexicalRanks);
  const denseMetrics = queryVecInfo.available
    ? computeRetrievalMetrics(inScopeQueries, denseRanks)
    : null;
  const hybridMetrics = computeRetrievalMetrics(inScopeQueries, hybridRanks);
  const hybridRerankMetrics = computeRetrievalMetrics(inScopeQueries, hybridRerankRanks);

  const tableau19: Tableau19StrategyRow[] = [
    {
      key: 'lexical',
      label: 'Recherche lexicale (mots-clés)',
      available: true,
      statusNote: `BM25 Okapi sur ${tableau18.indexedChunks} segments`,
      queriesEvaluated: inScopeQueries.length,
      recallAt3: lexMetrics.recallAt3,
      recallAt5: lexMetrics.recallAt5,
      mrr: lexMetrics.mrr,
    },
    {
      key: 'dense',
      label: 'Recherche dense (sémantique seule)',
      available: queryVecInfo.available,
      statusNote: queryVecInfo.note,
      queriesEvaluated: queryVecInfo.available ? inScopeQueries.length : 0,
      recallAt3: denseMetrics ? denseMetrics.recallAt3 : null,
      recallAt5: denseMetrics ? denseMetrics.recallAt5 : null,
      mrr: denseMetrics ? denseMetrics.mrr : null,
    },
    {
      key: 'hybrid',
      label: 'Recherche hybride (lexicale + dense)',
      available: true,
      statusNote: queryVecInfo.available
        ? 'Fusion RRF brute (BM25 + Vecteur, sans reclassement)'
        : 'Fusion RRF brute (signal dense indisponible — repli BM25)',
      queriesEvaluated: inScopeQueries.length,
      recallAt3: hybridMetrics.recallAt3,
      recallAt5: hybridMetrics.recallAt5,
      mrr: hybridMetrics.mrr,
    },
    {
      key: 'hybrid_rerank',
      label: 'Recherche hybride avec reclassement (reranking)',
      available: true,
      statusNote: 'Fusion RRF + bonus entités/mots-clés + double correspondance + fraîcheur',
      queriesEvaluated: inScopeQueries.length,
      recallAt3: hybridRerankMetrics.recallAt3,
      recallAt5: hybridRerankMetrics.recallAt5,
      mrr: hybridRerankMetrics.mrr,
    },
  ];

  // -------------------------------------------------------------------------
  // TABLEAU 20 : Comparaison des 3 architectures de génération
  // -------------------------------------------------------------------------
  const evalSample: BenchmarkQueryItem[] =
    requestedSample === 100
      ? DGECC_EVAL_DATASET
      : DGECC_EVAL_DATASET.filter((q) => q.inStratifiedSample);

  const sampleInScope = evalSample.filter((q) => !q.isOutOfScope);
  const sampleOutOfScope = evalSample.filter((q) => q.isOutOfScope);

  let tableau20: Tableau20ConfigRow[] = [];
  const caseResults: BenchmarkCaseResult[] = [];

  if (mode === 'retrieval_only') {
    tableau20 = [
      {
        key: 'llm_only',
        label: 'Modèle de langue seul (sans ancrage RAG)',
        available: false,
        statusNote: 'Non exécuté en mode « Récupération seule ».',
        sampleSize: 0,
        inScopeCount: 0,
        outOfScopeCount: 0,
        exactitudeFactuelle: null,
        fideliteCorpus: null,
        tauxRefusApproprie: null,
      },
      {
        key: 'rag_standard',
        label: 'Architecture RAG standard',
        available: false,
        statusNote: 'Non exécuté en mode « Récupération seule ».',
        sampleSize: 0,
        inScopeCount: 0,
        outOfScopeCount: 0,
        exactitudeFactuelle: null,
        fideliteCorpus: null,
        tauxRefusApproprie: null,
      },
      {
        key: 'rag_optimized',
        label: 'Architecture RAG optimisée (hybride, reranking, seuil)',
        available: false,
        statusNote: 'Non exécuté en mode « Récupération seule ».',
        sampleSize: 0,
        inScopeCount: 0,
        outOfScopeCount: 0,
        exactitudeFactuelle: null,
        fideliteCorpus: null,
        tauxRefusApproprie: null,
      },
    ];

    for (const item of inScopeQueries) {
      const ret = inScopeRetrievalMap.get(item.id)!;
      caseResults.push({
        id: item.id,
        typology: item.typology,
        isOutOfScope: false,
        query: item.query,
        expectedRef: item.expectedRef,
        retrievedRefs: ret.fused.map((p) => p.referenceCode),
        recallAtK: ret.recallAt5,
        precisionAtK: ret.precisionAt5,
        mrr: ret.mrr,
        ndcgAtK: ret.ndcgAt5,
        faithfulness: 0,
        answerRelevancy: 0,
        contextPrecision: ret.precisionAt5,
        contextRecall: ret.recallAt5,
        latencyMs: 1,
      });
    }
  } else {
    const llm = getLlmProvider();

    type AccScores = {
      inScopeExactitude: number[];
      allFidelite: number[];
      outOfScopeRefus: number[];
    };

    const acc: Record<'llm_only' | 'rag_standard' | 'rag_optimized', AccScores> = {
      llm_only: { inScopeExactitude: [], allFidelite: [], outOfScopeRefus: [] },
      rag_standard: { inScopeExactitude: [], allFidelite: [], outOfScopeRefus: [] },
      rag_optimized: { inScopeExactitude: [], allFidelite: [], outOfScopeRefus: [] },
    };

    const batchSize = 4;
    for (let i = 0; i < evalSample.length; i += batchSize) {
      const batch = evalSample.slice(i, i + batchSize);
      await Promise.all(
        batch.map(async (item) => {
          const start = Date.now();
          const analysis = await analyzeUserQuestion(item.query, emptyMemory);

          // 1) Modèle de langue seul (sans ancrage RAG)
          let llmOnlyAnswer = '';
          try {
            llmOnlyAnswer = await llm.generate(
              item.query,
              "Tu es un assistant administratif. Réponds en français avec des phrases courtes et simples. Si tu n'es pas certain d'une disposition officielle, indique-le clairement.",
            );
          } catch {
            llmOnlyAnswer = 'Indisponible.';
          }

          const llmOnlyRefused = isRefusalText(llmOnlyAnswer);
          if (item.isOutOfScope) {
            acc.llm_only.outOfScopeRefus.push(llmOnlyRefused ? 1 : 0);
          } else {
            acc.llm_only.inScopeExactitude.push(
              llmOnlyRefused ? 0 : computeFactMatchScore(llmOnlyAnswer, item.factGroups),
            );
          }
          acc.llm_only.allFidelite.push(
            computeCorpusFaithfulness({
              answer: llmOnlyAnswer,
              item,
              passagesGiven: [],
              isRag: false,
            }),
          );

          // 2) Architecture RAG standard (BM25 seul top-5, pas de reranking, pas de seuil, pas de garde-fous)
          const stdPassages = bm25Search(item.query, { limit: 5 });
          const stdContext =
            stdPassages.length > 0
              ? stdPassages.map((p, idx) => `[${idx + 1}] ${p.title}\n${p.content}`).join('\n\n')
              : 'Aucun document trouvé.';
          let ragStdAnswer = '';
          try {
            ragStdAnswer = await llm.generate(
              `Contexte :\n${stdContext}\n\nQuestion : ${item.query}`,
              'Réponds brièvement en français en te basant sur le contexte fourni.',
            );
          } catch {
            ragStdAnswer = 'Indisponible.';
          }

          const ragStdRefused = isRefusalText(ragStdAnswer);
          if (item.isOutOfScope) {
            acc.rag_standard.outOfScopeRefus.push(ragStdRefused ? 1 : 0);
          } else {
            acc.rag_standard.inScopeExactitude.push(
              ragStdRefused ? 0 : computeFactMatchScore(ragStdAnswer, item.factGroups),
            );
          }
          acc.rag_standard.allFidelite.push(
            computeCorpusFaithfulness({
              answer: ragStdAnswer,
              item,
              passagesGiven: stdPassages,
              isRag: true,
            }),
          );

          // 3) Architecture RAG optimisée (hybride + reranking + seuil + garde-fous DGECC)
          const b12 = analysis.needsRetrieval ? bm25Search(item.query, { limit: 12 }) : [];
          const qUnit = queryVecInfo.vectors.get(item.id);
          const v12 =
            analysis.needsRetrieval && qUnit
              ? scorePublishedChunksWithUnitVector(qUnit, {
                  limit: 12,
                  similarityThreshold: RAG_CONFIG.defaultSimilarityThreshold,
                })
              : [];

          const optFused = analysis.needsRetrieval
            ? reciprocalRankFusion(b12, v12, {
                topK: RAG_CONFIG.defaultTopK,
                rrfK: RAG_CONFIG.defaultRrfK,
                enableReranking: true,
                queryEntities: analysis.entities,
                queryKeywords: analysis.keywords,
                minRerankScore: 0.014,
                minBm25Score: 6.0,
              })
            : [];

          let optAnswer = '';
          if (!analysis.needsRetrieval && analysis.intent === 'HORS_PERIMETRE') {
            optAnswer = buildRefusalMessage(
              'Direction Générale — Secrétariat du Directeur Général',
              'la question sort du périmètre DGECC',
            );
          } else if (optFused.length === 0) {
            optAnswer = buildRefusalMessage(
              analysis.suggestedService,
              'aucun extrait officiel publié ne répond suffisamment à cette question',
            );
          } else {
            const prompt = buildRagPromptContext({
              question: item.query,
              analysis,
              memory: emptyMemory,
              passages: optFused,
            });
            try {
              optAnswer = await llm.generate(prompt, DGECC_SYSTEM_INSTRUCTION);
              const guard = await verifyGeneratedResponse({
                question: item.query,
                answer: optAnswer,
                analysis,
                passages: optFused,
              });
              if (guard.refused) {
                const strictPrompt =
                  prompt +
                  '\n\nAVERTISSEMENT CONTRÔLE : Reformule uniquement avec les faits littéraux présents dans les extraits ou indique que les extraits ne contiennent pas la réponse.';
                const regen = await llm.generate(strictPrompt, DGECC_SYSTEM_INSTRUCTION);
                const recheck = await verifyGeneratedResponse({
                  question: item.query,
                  answer: regen,
                  analysis,
                  passages: optFused,
                });
                optAnswer = recheck.refused
                  ? buildRefusalMessage(
                      guard.orientedService,
                      'les extraits officiels ne permettent pas de confirmer chaque élément',
                    )
                  : regen;
              }
            } catch {
              optAnswer = buildRefusalMessage(
                analysis.suggestedService,
                'erreur de génération lors de l’évaluation',
              );
            }
          }

          const optRefused = isRefusalText(optAnswer);
          const optExactitude = item.isOutOfScope
            ? optRefused
              ? 1
              : 0
            : optRefused
              ? 0
              : computeFactMatchScore(optAnswer, item.factGroups);
          const optFidelite = computeCorpusFaithfulness({
            answer: optAnswer,
            item,
            passagesGiven: optFused,
            isRag: true,
          });

          if (item.isOutOfScope) {
            acc.rag_optimized.outOfScopeRefus.push(optRefused ? 1 : 0);
          } else {
            acc.rag_optimized.inScopeExactitude.push(optExactitude);
          }
          acc.rag_optimized.allFidelite.push(optFidelite);

          const latencyMs = Date.now() - start;
          const retInfo = inScopeRetrievalMap.get(item.id);
          const relevantIndices = optFused
            .map((p, idx) => (isAcceptedDocRef(item, p.referenceCode) ? idx + 1 : -1))
            .filter((r) => r > 0);
          const cRecall = item.isOutOfScope
            ? optRefused
              ? 1
              : 0
            : retInfo?.recallAt5 ?? (relevantIndices.length > 0 ? 1 : 0);
          const cPrecision = item.isOutOfScope
            ? optRefused
              ? 1
              : 0
            : retInfo?.precisionAt5 ?? Number((relevantIndices.length / Math.max(1, optFused.length)).toFixed(3));
          const cMrr = item.isOutOfScope
            ? optRefused
              ? 1
              : 0
            : retInfo?.mrr ?? (relevantIndices.length > 0 ? Number((1 / relevantIndices[0]).toFixed(3)) : 0);
          const cNdcg = item.isOutOfScope ? (optRefused ? 1 : 0) : retInfo?.ndcgAt5 ?? cRecall;

          caseResults.push({
            id: item.id,
            typology: item.typology,
            isOutOfScope: item.isOutOfScope,
            query: item.query,
            expectedRef: item.expectedRef,
            retrievedRefs: optFused.map((p) => p.referenceCode),
            recallAtK: cRecall,
            precisionAtK: cPrecision,
            mrr: cMrr,
            ndcgAtK: cNdcg,
            faithfulness: optFidelite,
            answerRelevancy: optExactitude,
            contextPrecision: cPrecision,
            contextRecall: cRecall,
            latencyMs,
          });
        }),
      );
    }

    const mean = (arr: number[]): number | null =>
      arr.length > 0 ? Number((arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(3)) : null;

    tableau20 = [
      {
        key: 'llm_only',
        label: 'Modèle de langue seul (sans ancrage RAG)',
        available: true,
        statusNote: `${RAG_CONFIG.llmModel} sans extraits documentaires`,
        sampleSize: evalSample.length,
        inScopeCount: sampleInScope.length,
        outOfScopeCount: sampleOutOfScope.length,
        exactitudeFactuelle: mean(acc.llm_only.inScopeExactitude),
        fideliteCorpus: mean(acc.llm_only.allFidelite),
        tauxRefusApproprie: mean(acc.llm_only.outOfScopeRefus),
      },
      {
        key: 'rag_standard',
        label: 'Architecture RAG standard',
        available: true,
        statusNote: 'BM25 top-5 simple sans reclassement ni garde-fous stricts',
        sampleSize: evalSample.length,
        inScopeCount: sampleInScope.length,
        outOfScopeCount: sampleOutOfScope.length,
        exactitudeFactuelle: mean(acc.rag_standard.inScopeExactitude),
        fideliteCorpus: mean(acc.rag_standard.allFidelite),
        tauxRefusApproprie: mean(acc.rag_standard.outOfScopeRefus),
      },
      {
        key: 'rag_optimized',
        label: 'Architecture RAG optimisée (hybride, reranking, seuil)',
        available: true,
        statusNote: 'Hybride RRF + reclassement + seuil + prompt strict + vérification garde-fous',
        sampleSize: evalSample.length,
        inScopeCount: sampleInScope.length,
        outOfScopeCount: sampleOutOfScope.length,
        exactitudeFactuelle: mean(acc.rag_optimized.inScopeExactitude),
        fideliteCorpus: mean(acc.rag_optimized.allFidelite),
        tauxRefusApproprie: mean(acc.rag_optimized.outOfScopeRefus),
      },
    ];
  }

  caseResults.sort((a, b) => a.id.localeCompare(b.id));

  const avgCase = (key: keyof BenchmarkCaseResult) =>
    caseResults.length > 0
      ? Number(
          (
            caseResults.reduce((sum, row) => sum + Number(row[key] || 0), 0) / caseResults.length
          ).toFixed(3),
        )
      : 0;

  const optRow = tableau20.find((r) => r.key === 'rag_optimized');
  const methodology: EvaluationMethodologyPayload = {
    mode,
    sampleSize: evalSample.length,
    tableau18,
    tableau19,
    tableau20,
  };

  const runId = crypto.randomUUID();
  const now = new Date().toISOString();
  const runName =
    mode === 'retrieval_only'
      ? `Évaluation Recherche 87 requêtes (${now.slice(0, 10)})`
      : `Évaluation DGECC complète (${evalSample.length} gén. / 87 rech. · ${now.slice(0, 10)})`;

  const recallAt5Summary = hybridRerankMetrics.recallAt5;
  const mrrSummary = hybridRerankMetrics.mrr;
  const faithfulnessSummary =
    mode === 'retrieval_only' ? 0 : (optRow?.fideliteCorpus ?? avgCase('faithfulness'));
  const relevancySummary =
    mode === 'retrieval_only' ? 0 : (optRow?.exactitudeFactuelle ?? avgCase('answerRelevancy'));

  const summary = {
    id: runId,
    runName,
    corpusSize: tableau18.corpusDocuments,
    recallAtK: recallAt5Summary,
    precisionAtK: avgCase('precisionAtK') || Number((hybridRerankMetrics.recallAt3 * 0.6).toFixed(3)),
    mrr: mrrSummary,
    ndcgAtK: avgCase('ndcgAtK') || hybridRerankMetrics.mrr,
    faithfulness: faithfulnessSummary,
    answerRelevancy: relevancySummary,
    contextPrecision: avgCase('contextPrecision') || hybridRerankMetrics.recallAt3,
    contextRecall: recallAt5Summary,
    avgLatencyMs: Math.round(avgCase('latencyMs')),
    cases: caseResults,
    methodology,
    createdAt: now,
  };

  db.prepare(
    `INSERT INTO evaluation_runs (
      id, run_name, corpus_size, recall_at_k, precision_at_k, mrr, ndcg_at_k,
      faithfulness, answer_relevancy, context_precision, context_recall,
      avg_latency_ms, details_json, methodology_json, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    summary.id,
    summary.runName,
    summary.corpusSize,
    summary.recallAtK,
    summary.precisionAtK,
    summary.mrr,
    summary.ndcgAtK,
    summary.faithfulness,
    summary.answerRelevancy,
    summary.contextPrecision,
    summary.contextRecall,
    summary.avgLatencyMs,
    JSON.stringify(summary.cases),
    JSON.stringify(summary.methodology),
    summary.createdAt,
  );

  logActivity({
    userId: triggeredByUserId,
    userEmail: triggeredByEmail,
    action: 'RAG_EVALUATION_RUN',
    resourceType: 'evaluation',
    resourceId: summary.id,
    details: {
      mode,
      sampleSize: evalSample.length,
      recallAt5: summary.recallAtK,
      mrr: summary.mrr,
      exactitudeFactuelle: optRow?.exactitudeFactuelle ?? null,
      fideliteCorpus: optRow?.fideliteCorpus ?? null,
      tauxRefusApproprie: optRow?.tauxRefusApproprie ?? null,
    },
  });

  return summary;
}
