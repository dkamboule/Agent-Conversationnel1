export type RoleCode = 'ADMIN' | 'VALIDATOR' | 'EDITOR' | 'USER';

export interface AuthUser {
  id: string;
  email: string;
  fullName: string;
  role: RoleCode;
  service?: string | null;
  roleLabel?: string;
  permissions?: string[];
}

export type DocumentStatus =
  | 'BROUILLON'
  | 'EN_ATTENTE'
  | 'VALIDE'
  | 'PUBLIE'
  | 'ARCHIVE'
  | 'REJETE';

export interface DocumentItem {
  id: string;
  title: string;
  doc_type: string;
  category: string;
  service_name: string;
  reference_code: string;
  published_date: string;
  valid_until: string | null;
  current_version: number;
  status: DocumentStatus;
  file_name?: string;
  file_format: string;
  rejection_reason?: string | null;
  chunk_count: number;
  embedded_count?: number;
  updated_at: string;
}

export interface DocumentChunkPreview {
  id: string;
  chunk_index: number;
  content: string;
  token_count: number;
  char_count: number;
  embedding_model: string | null;
  has_embedding: number;
}

export interface RagSource {
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
}

export interface GuardrailsReport {
  passed: boolean;
  refused: boolean;
  confidenceLevel: 'ELEVEE' | 'MOYENNE' | 'FAIBLE';
  hasExpiredSources: boolean;
  orientedService: string;
  checks: Array<{
    name: string;
    passed: boolean;
    detail: string;
  }>;
}

export interface NlpAnalysis {
  intent: string;
  entities: string[];
  keywords: string[];
  temporalReferences: string[];
  needsRetrieval: boolean;
  reformulatedQuery: string;
  suggestedService: string;
}

export interface ChatMetrics {
  nlpMs: number;
  retrievalMs: number;
  generationMs: number;
  guardrailsMs: number;
  totalMs: number;
  bm25Hits: number;
  vectorHits: number;
  fusedCount: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  analysis?: NlpAnalysis | null;
  sources?: RagSource[];
  guardrails?: GuardrailsReport | null;
  metrics?: ChatMetrics | null;
  feedback?: 'positive' | 'negative' | null;
  createdAt: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface AdminStatistics {
  documentsByStatus: Array<{ status: string; count: number }>;
  totalChunks: number;
  embeddedChunks: number;
  totalConversations: number;
  totalAssistantResponses: number;
  feedback: { positive: number; negative: number };
  performance: { avgTotalMs: number; avgRetrievalMs: number; refusalRate: number };
  ragConfig: {
    embeddingModel: string;
    llmModel: string;
    defaultChunkSize: number;
    defaultChunkOverlap: number;
    defaultTopK: number;
    defaultRrfK: number;
    defaultSimilarityThreshold: number;
  };
}

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
    code: string;
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

export interface EvaluationRun {
  id: string;
  run_name?: string;
  runName?: string;
  corpus_size?: number;
  corpusSize?: number;
  recall_at_k?: number;
  recallAtK?: number;
  precision_at_k?: number;
  precisionAtK?: number;
  mrr: number;
  ndcg_at_k?: number;
  ndcgAtK?: number;
  faithfulness: number;
  answer_relevancy?: number;
  answerRelevancy?: number;
  context_precision?: number;
  contextPrecision?: number;
  context_recall?: number;
  contextRecall?: number;
  avg_latency_ms?: number;
  avgLatencyMs?: number;
  created_at?: string;
  createdAt?: string;
  methodology?: EvaluationMethodologyPayload | null;
  cases: Array<{
    id: string;
    typology?: string;
    isOutOfScope?: boolean;
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
  }>;
}

export interface EvaluationOverviewResponse {
  tableau18: Tableau18Metrics;
  panelEntries: UserPanelEntry[];
  latestRun: EvaluationRun | null;
}
