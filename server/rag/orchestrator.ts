import crypto from 'node:crypto';
import { RAG_CONFIG } from '../core/config.ts';
import { detectPromptInjection } from '../core/security.ts';
import { getDb, logActivity } from '../database/db.ts';
import {
  buildRefusalMessage,
  verifyGeneratedResponse,
  type GuardrailsReport,
} from '../guardrails/verifier.ts';
import { getLlmProvider } from '../llm/generator.ts';
import { loadConversationMemory } from '../memory/memoryManager.ts';
import { analyzeUserQuestion, type NlpAnalysisResult } from '../nlp/analyzer.ts';
import { bm25Search } from '../search/bm25.ts';
import { normalizeFrenchToken, onCorpusInvalidated } from '../search/corpusCache.ts';
import { vectorSearch } from '../search/vector.ts';
import { buildRagPromptContext } from './context_builder.ts';
import { reciprocalRankFusion, type FusedPassage } from './fusion.ts';

export interface RagQueryRequest {
  question: string;
  conversationId?: string;
  categoryFilter?: string;
  publishedAfter?: string;
  versionNumber?: number;
  topK?: number;
  rrfK?: number;
  similarityThreshold?: number;
  owner: {
    userId?: string | null;
    userEmail?: string | null;
    guestSessionId?: string | null;
  };
  onStage?: (stage: string, detail?: unknown) => void;
  onToken?: (delta: string) => void;
}

export interface RagQueryResponse {
  conversationId: string;
  messageId: string;
  answer: string;
  analysis: NlpAnalysisResult;
  sources: Array<{
    chunkId: string;
    documentId: string;
    title: string;
    referenceCode: string;
    docType: string;
    category: string;
    serviceName: string;
    publishedDate: string;
    validUntil: string | null;
    versionNumber: number;
    excerpt: string;
    bm25Score: number | null;
    bm25Rank: number | null;
    vectorScore: number | null;
    vectorRank: number | null;
    rrfScore: number;
    rerankScore: number;
    isExpired: boolean;
  }>;
  guardrails: GuardrailsReport;
  metrics: {
    nlpMs: number;
    retrievalMs: number;
    generationMs: number;
    guardrailsMs: number;
    totalMs: number;
    bm25Hits: number;
    vectorHits: number;
    fusedCount: number;
  };
}

interface CachedAnswerEntry {
  answer: string;
  analysis: NlpAnalysisResult;
  sources: RagQueryResponse['sources'];
  guardrails: GuardrailsReport;
  bm25Hits: number;
  vectorHits: number;
}

const MAX_RESPONSE_CACHE = 100;
const standaloneResponseCache = new Map<string, CachedAnswerEntry>();

onCorpusInvalidated(() => {
  standaloneResponseCache.clear();
});

export async function runHybridRagPipeline(req: RagQueryRequest): Promise<RagQueryResponse> {
  const totalStart = Date.now();
  const db = getDb();

  // 0. Ensure or create isolated conversation
  let conversationId = req.conversationId;
  const now = new Date().toISOString();

  if (conversationId) {
    const existing = db
      .prepare(`SELECT id FROM conversations WHERE id = ?`)
      .get(conversationId) as { id: string } | undefined;
    if (!existing) {
      conversationId = undefined;
    }
  }

  if (!conversationId) {
    conversationId = crypto.randomUUID();
    const title = req.question.trim().slice(0, 72) || 'Nouvelle demande DGECC';
    db.prepare(
      `INSERT INTO conversations (id, user_id, guest_session_id, title, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
    ).run(
      conversationId,
      req.owner.userId ?? null,
      req.owner.userId ? null : req.owner.guestSessionId ?? 'anonymous',
      title,
      now,
      now,
    );
  } else {
    db.prepare(`UPDATE conversations SET updated_at = ? WHERE id = ?`).run(now, conversationId);
  }

  const userMessageId = crypto.randomUUID();
  db.prepare(
    `INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, 'user', ?, ?)`,
  ).run(userMessageId, conversationId, req.question.trim(), now);

  // Pre-screen prompt injection before anything else
  const preInjection = detectPromptInjection(req.question);
  if (preInjection.flagged) {
    const refusal = buildRefusalMessage(
      'Direction Générale — Secrétariat du Directeur Général',
      'demande contenant des instructions incompatibles avec le cadre réglementaire',
    );
    const guardrails: GuardrailsReport = {
      passed: false,
      refused: true,
      confidenceLevel: 'FAIBLE',
      hasExpiredSources: false,
      orientedService: 'Direction Générale — Secrétariat du Directeur Général',
      checks: [
        {
          name: 'Sécurité / Anti-injection de prompt',
          passed: false,
          detail: preInjection.reason || 'Motif bloqué.',
        },
      ],
    };
    const analysis: NlpAnalysisResult = {
      intent: 'HORS_PERIMETRE',
      entities: [],
      keywords: [],
      temporalReferences: [],
      needsRetrieval: false,
      reformulatedQuery: req.question,
      suggestedService: 'Direction Générale — Secrétariat du Directeur Général',
    };
    const assistantMsgId = crypto.randomUUID();
    const metrics = {
      nlpMs: 0,
      retrievalMs: 0,
      generationMs: 0,
      guardrailsMs: 1,
      totalMs: Date.now() - totalStart,
      bm25Hits: 0,
      vectorHits: 0,
      fusedCount: 0,
    };
    db.prepare(
      `INSERT INTO messages (
        id, conversation_id, role, content, nlp_analysis_json, sources_json, guardrails_json, metrics_json, created_at
      ) VALUES (?, ?, 'assistant', ?, ?, '[]', ?, ?, ?)`,
    ).run(
      assistantMsgId,
      conversationId,
      refusal,
      JSON.stringify(analysis),
      JSON.stringify(guardrails),
      JSON.stringify(metrics),
      new Date().toISOString(),
    );
    req.onToken?.(refusal);

    logActivity({
      userId: req.owner.userId,
      userEmail: req.owner.userEmail,
      action: 'GUARDRAIL_BLOCKED_INJECTION',
      resourceType: 'conversation',
      resourceId: conversationId,
      details: { reason: preInjection.reason },
    });

    return {
      conversationId,
      messageId: assistantMsgId,
      answer: refusal,
      analysis,
      sources: [],
      guardrails,
      metrics,
    };
  }

  // 1. Load isolated conversation memory
  const memory = loadConversationMemory(conversationId, req.owner);

  // Instant standalone response cache check (for zero-turn / FAQ questions)
  const cacheKey = `${req.categoryFilter || 'ALL'}|${req.topK ?? RAG_CONFIG.defaultTopK}|${normalizeFrenchToken(req.question.trim())}`;
  if (memory.recentTurns.length === 0 && !req.publishedAfter && !req.versionNumber) {
    const cached = standaloneResponseCache.get(cacheKey);
    if (cached) {
      req.onToken?.(cached.answer);
      const totalMs = Math.max(1, Date.now() - totalStart);
      const metrics = {
        nlpMs: 0,
        retrievalMs: 0,
        generationMs: 0,
        guardrailsMs: 0,
        totalMs,
        bm25Hits: cached.bm25Hits,
        vectorHits: cached.vectorHits,
        fusedCount: cached.sources.length,
      };
      const assistantMsgId = crypto.randomUUID();
      db.prepare(
        `INSERT INTO messages (
          id, conversation_id, role, content, nlp_analysis_json, sources_json, guardrails_json, metrics_json, created_at
        ) VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?)`,
      ).run(
        assistantMsgId,
        conversationId,
        cached.answer,
        JSON.stringify(cached.analysis),
        JSON.stringify(cached.sources),
        JSON.stringify(cached.guardrails),
        JSON.stringify(metrics),
        new Date().toISOString(),
      );
      return {
        conversationId,
        messageId: assistantMsgId,
        answer: cached.answer,
        analysis: cached.analysis,
        sources: cached.sources,
        guardrails: cached.guardrails,
        metrics,
      };
    }
  }

  // 2. Fast NLP Analysis (< 1 ms unless multi-turn anaphora disambiguation needed)
  req.onStage?.('nlp', 'Analyse et recherche hybride...');
  const nlpStart = Date.now();
  const analysis = await analyzeUserQuestion(req.question, memory);
  const nlpMs = Date.now() - nlpStart;

  // 3. Parallel Hybrid Search (BM25 + Vector) in memory
  let bm25Hits: ReturnType<typeof bm25Search> = [];
  let vectorHits: Awaited<ReturnType<typeof vectorSearch>> = [];
  let fusedPassages: FusedPassage[] = [];
  let retrievalMs = 0;

  if (analysis.needsRetrieval) {
    const retStart = Date.now();
    const searchQuery = analysis.reformulatedQuery || req.question;

    const [bRes, vRes] = await Promise.all([
      Promise.resolve(
        bm25Search(searchQuery, {
          category: req.categoryFilter,
          publishedAfter: req.publishedAfter,
          versionNumber: req.versionNumber,
          limit: 12,
        }),
      ),
      vectorSearch(searchQuery, {
        category: req.categoryFilter,
        publishedAfter: req.publishedAfter,
        versionNumber: req.versionNumber,
        similarityThreshold: req.similarityThreshold ?? RAG_CONFIG.defaultSimilarityThreshold,
        limit: 12,
      }),
    ]);

    bm25Hits = bRes;
    vectorHits = vRes;

    fusedPassages = reciprocalRankFusion(bm25Hits, vectorHits, {
      topK: req.topK ?? RAG_CONFIG.defaultTopK,
      rrfK: req.rrfK ?? RAG_CONFIG.defaultRrfK,
      queryEntities: analysis.entities,
      queryKeywords: analysis.keywords,
    });
    retrievalMs = Date.now() - retStart;
  }

  // 4. Context construction & zero-thinking instant streaming generation
  req.onStage?.('generation', 'Rédaction de la réponse sourcée...');
  const genStart = Date.now();
  const llm = getLlmProvider();
  let answer = '';

  if (analysis.needsRetrieval && fusedPassages.length === 0) {
    answer = buildRefusalMessage(
      analysis.suggestedService,
      'aucun document institutionnel publié et validé ne correspond suffisamment à cette question',
    );
    req.onToken?.(answer);
  } else if (!analysis.needsRetrieval && analysis.intent === 'HORS_PERIMETRE') {
    answer = buildRefusalMessage(
      'Direction Générale — Secrétariat du Directeur Général',
      'la question sort du périmètre de compétences de la Direction Générale des Examens, Concours et de la Certification',
    );
    req.onToken?.(answer);
  } else if (!analysis.needsRetrieval && analysis.intent === 'SALUTATION') {
    answer =
      "Bonjour. Je suis l'assistant d'orientation de la **DGECC** (Direction Générale des Examens, Concours et de la Certification). Comment puis-je vous renseigner sur les examens nationaux, les concours publics, les inscriptions, les certifications ou les recours ?";
    req.onToken?.(answer);
  } else {
    const prompt = buildRagPromptContext({
      question: req.question,
      analysis,
      memory,
      passages: fusedPassages,
    });
    answer = req.onToken ? await llm.generateStream(prompt, req.onToken) : await llm.generate(prompt);
  }
  const generationMs = Date.now() - genStart;

  // 5. Instant deterministic guardrails verification
  const guardStart = Date.now();
  let guardrails = await verifyGeneratedResponse({
    question: req.question,
    answer,
    analysis,
    passages: fusedPassages,
  });

  if (guardrails.refused && fusedPassages.length > 0) {
    const strictPrompt =
      buildRagPromptContext({
        question: req.question,
        analysis,
        memory,
        passages: fusedPassages,
      }) +
      '\n\nAVERTISSEMENT CONTRÔLE : Reformule uniquement avec les faits littéraux présents dans les extraits ci-dessus, en citant les références exactes.';
    const regenerated = await llm.generate(strictPrompt);
    const recheck = await verifyGeneratedResponse({
      question: req.question,
      answer: regenerated,
      analysis,
      passages: fusedPassages,
    });
    if (!recheck.refused) {
      answer = regenerated;
      guardrails = recheck;
    } else {
      answer = buildRefusalMessage(
        guardrails.orientedService,
        'les vérifications de conformité documentaire n’ont pas pu confirmer chaque affirmation',
      );
      guardrails = recheck;
    }
  }
  const guardrailsMs = Date.now() - guardStart;
  const totalMs = Date.now() - totalStart;

  const formattedSources = fusedPassages.map((p) => ({
    chunkId: p.chunkId,
    documentId: p.documentId,
    title: p.title,
    referenceCode: p.referenceCode,
    docType: p.docType,
    category: p.category,
    serviceName: p.serviceName,
    publishedDate: p.publishedDate,
    validUntil: p.validUntil,
    versionNumber: p.versionNumber,
    excerpt: p.content.replace(/^\[[^\]]+\]\n/, '').slice(0, 380),
    bm25Score: p.bm25Score ?? null,
    bm25Rank: p.bm25Rank ?? null,
    vectorScore: p.vectorScore ?? null,
    vectorRank: p.vectorRank ?? null,
    rrfScore: p.rrfScore,
    rerankScore: p.rerankScore,
    isExpired: p.isExpired,
  }));

  const metrics = {
    nlpMs,
    retrievalMs,
    generationMs,
    guardrailsMs,
    totalMs,
    bm25Hits: bm25Hits.length,
    vectorHits: vectorHits.length,
    fusedCount: fusedPassages.length,
  };

  const assistantMsgId = crypto.randomUUID();
  db.prepare(
    `INSERT INTO messages (
      id, conversation_id, role, content, nlp_analysis_json, sources_json, guardrails_json, metrics_json, created_at
    ) VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?)`,
  ).run(
    assistantMsgId,
    conversationId,
    answer,
    JSON.stringify(analysis),
    JSON.stringify(formattedSources),
    JSON.stringify(guardrails),
    JSON.stringify(metrics),
    new Date().toISOString(),
  );

  // Store verified standalone answer in cache for instant future FAQ hits
  if (
    memory.recentTurns.length === 0 &&
    !req.publishedAfter &&
    !req.versionNumber &&
    !guardrails.refused &&
    analysis.needsRetrieval
  ) {
    if (standaloneResponseCache.size >= MAX_RESPONSE_CACHE) {
      const oldest = standaloneResponseCache.keys().next().value;
      if (oldest !== undefined) standaloneResponseCache.delete(oldest);
    }
    standaloneResponseCache.set(cacheKey, {
      answer,
      analysis,
      sources: formattedSources,
      guardrails,
      bm25Hits: bm25Hits.length,
      vectorHits: vectorHits.length,
    });
  }

  return {
    conversationId,
    messageId: assistantMsgId,
    answer,
    analysis,
    sources: formattedSources,
    guardrails,
    metrics,
  };
}
