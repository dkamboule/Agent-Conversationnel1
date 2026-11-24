import type {
  AdminStatistics,
  AuthUser,
  ChatMessage,
  ConversationSummary,
  DocumentChunkPreview,
  DocumentItem,
  DocumentStatus,
  EvaluationOverviewResponse,
  EvaluationRun,
  UserPanelEntry,
  UserPanelProfile,
} from '../types/dgecc';

// Auth token and guest session ID remain strictly in JS memory (never localStorage/sessionStorage)
let memoryAuthToken: string | null = null;
const guestSessionId = `dgecc_${Math.random().toString(36).slice(2, 12)}_${Date.now().toString(36)}`;

export function setAuthToken(token: string | null) {
  memoryAuthToken = token;
}

export function getAuthToken(): string | null {
  return memoryAuthToken;
}

function buildHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = {
    'X-DGECC-Session': guestSessionId,
    ...extra,
  };
  if (memoryAuthToken) {
    headers.Authorization = `Bearer ${memoryAuthToken}`;
  }
  return headers;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const isFormData = init?.body instanceof FormData;
  const headers = buildHeaders(
    isFormData ? undefined : { 'Content-Type': 'application/json' },
  );

  const response = await fetch(`/api/v1${path}`, {
    ...init,
    headers: {
      ...headers,
      ...(init?.headers as Record<string, string> | undefined),
    },
  });

  const text = await response.text();
  let data: Record<string, unknown> | null = null;
  let parseError = false;
  try {
    data = text ? (JSON.parse(text) as Record<string, unknown>) : null;
  } catch {
    parseError = true;
  }

  if (!response.ok) {
    const errMsg =
      data && typeof data.error === 'string'
        ? data.error
        : `Erreur HTTP ${response.status}${parseError ? ' (réponse non JSON)' : ''}`;
    throw new Error(errMsg);
  }

  if (parseError || !data || typeof data !== 'object') {
    throw new Error(
      'Réponse invalide du serveur (format JSON attendu). Veuillez vérifier votre session et réessayer.',
    );
  }

  return data as T;
}

export const dgeccApi = {
  // Auth
  getBootstrapStatus: () =>
    apiFetch<{ needsAdminSetup: boolean; hasPasscodeOverride: boolean }>('/auth/bootstrap-status'),

  bootstrapAdmin: (payload: {
    email: string;
    fullName: string;
    password: string;
    service?: string;
  }) =>
    apiFetch<{ token: string; user: AuthUser }>('/auth/bootstrap-admin', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  register: (payload: { email: string; fullName: string; password: string }) =>
    apiFetch<{ token: string; user: AuthUser }>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  login: (payload: { email: string; password: string }) =>
    apiFetch<{ token: string; user: AuthUser }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  logout: () => apiFetch<{ ok: boolean }>('/auth/logout', { method: 'POST' }),

  me: () => apiFetch<{ user: AuthUser }>('/auth/me'),

  // Chat & Streaming RAG
  listConversations: () => apiFetch<{ conversations: ConversationSummary[] }>('/chat/conversations'),

  getConversation: (id: string) =>
    apiFetch<{ conversation: { id: string; title: string }; messages: ChatMessage[] }>(
      `/chat/conversations/${encodeURIComponent(id)}`,
    ),

  deleteConversation: (id: string) =>
    apiFetch<{ ok: boolean }>(`/chat/conversations/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),

  sendFeedback: (messageId: string, feedback: 'positive' | 'negative', comment?: string) =>
    apiFetch<{ ok: boolean }>(`/chat/messages/${encodeURIComponent(messageId)}/feedback`, {
      method: 'POST',
      body: JSON.stringify({ feedback, comment }),
    }),

  streamQuestion: async (
    payload: {
      question: string;
      conversationId?: string;
      categoryFilter?: string;
      topK?: number;
      similarityThreshold?: number;
    },
    handlers: {
      onStage: (stage: string, detail?: string) => void;
      onToken: (delta: string) => void;
      onComplete: (result: {
        conversationId: string;
        messageId: string;
        answer: string;
        analysis: ChatMessage['analysis'];
        sources: ChatMessage['sources'];
        guardrails: ChatMessage['guardrails'];
        metrics: ChatMessage['metrics'];
      }) => void;
    },
  ) => {
    const response = await fetch('/api/v1/chat/stream', {
      method: 'POST',
      headers: buildHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(payload),
    });

    if (!response.ok || !response.body) {
      const errData = await response.json().catch(() => ({ error: 'Erreur de connexion au moteur RAG.' }));
      throw new Error(errData.error || 'Erreur lors de la requête RAG.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const processFrame = (frame: string) => {
      const lines = frame.split('\n');
      let eventName = 'message';
      let dataLine = '';
      for (const line of lines) {
        if (line.startsWith('event:')) eventName = line.slice(6).trim();
        if (line.startsWith('data:')) dataLine += line.slice(5).trim();
      }
      if (!dataLine) return;
      let parsed: Record<string, unknown>;
      try {
        parsed = JSON.parse(dataLine) as Record<string, unknown>;
      } catch {
        return;
      }
      if (eventName === 'stage') {
        handlers.onStage(String(parsed.stage || ''), parsed.detail ? String(parsed.detail) : undefined);
      } else if (eventName === 'token') {
        handlers.onToken(String(parsed.delta || ''));
      } else if (eventName === 'complete') {
        handlers.onComplete(parsed as Parameters<typeof handlers.onComplete>[0]);
      } else if (eventName === 'error') {
        throw new Error(
          typeof parsed.error === 'string' && parsed.error
            ? parsed.error
            : 'Erreur de génération RAG.',
        );
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        if (buffer.trim()) {
          processFrame(buffer);
        }
        break;
      }
      buffer += decoder.decode(value, { stream: true });

      const frames = buffer.split('\n\n');
      buffer = frames.pop() || '';

      for (const frame of frames) {
        processFrame(frame);
      }
    }
  },

  // Documents
  listDocuments: () => apiFetch<{ documents: DocumentItem[] }>('/documents'),

  getDocumentDetail: (id: string) =>
    apiFetch<{
      document: DocumentItem;
      versions: Array<{
        id: string;
        version_number: number;
        chunk_size: number;
        chunk_overlap: number;
        created_at: string;
        preview_text: string;
      }>;
      chunks: DocumentChunkPreview[];
    }>(`/documents/${encodeURIComponent(id)}`),

  uploadDocument: (formData: FormData) =>
    apiFetch<{
      id: string;
      title: string;
      status: DocumentStatus;
      version: number;
      chunksCount: number;
    }>('/documents/upload', {
      method: 'POST',
      body: formData,
    }),

  previewDocumentOcr: (payload: {
    fileName: string;
    mimeType?: string;
    fileBase64: string;
    ocrMode?: 'auto' | 'force' | 'off';
  }) =>
    apiFetch<{
      text: string;
      ocrUsed: boolean;
      ocrReason: string;
      ocrModel: string | null;
      nativeCharCount: number;
      pageCount: number | null;
      normalizedLength: number;
    }>('/documents/ocr-preview', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  uploadDocumentJson: (payload: {
    fileName: string;
    mimeType?: string;
    fileBase64: string;
    title: string;
    docType: string;
    category: string;
    serviceName: string;
    referenceCode: string;
    publishedDate: string;
    validUntil?: string | null;
    ocrMode?: 'auto' | 'force' | 'off';
    reviewedText?: string;
    chunkSize?: number;
    chunkOverlap?: number;
    initialStatus?: 'EN_ATTENTE' | 'BROUILLON';
  }) =>
    apiFetch<{
      id: string;
      title: string;
      status: DocumentStatus;
      version: number;
      chunksCount: number;
      ocrUsed?: boolean;
      ocrReason?: string;
      ocrModel?: string | null;
    }>('/documents/upload', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  validateDocument: (id: string, status: DocumentStatus, rejectionReason?: string) =>
    apiFetch<{ id: string; status: DocumentStatus }>(
      `/documents/${encodeURIComponent(id)}/validate`,
      {
        method: 'PUT',
        body: JSON.stringify({ status, rejectionReason }),
      },
    ),

  publishDocument: (id: string) =>
    apiFetch<{ id: string; status: DocumentStatus; indexation: unknown }>(
      `/documents/${encodeURIComponent(id)}/publish`,
      { method: 'POST' },
    ),

  reindexDocument: (
    id: string,
    options?: { chunkSize?: number; chunkOverlap?: number; bumpVersion?: boolean },
  ) =>
    apiFetch<{ chunkCount: number; durationMs: number; versionNumber: number }>(
      `/documents/${encodeURIComponent(id)}/reindex`,
      {
        method: 'POST',
        body: JSON.stringify(options || {}),
      },
    ),

  // Admin
  getStatistics: () => apiFetch<AdminStatistics>('/admin/statistics'),

  listUsers: () =>
    apiFetch<{
      users: Array<{
        id: string;
        email: string;
        full_name: string;
        role_code: string;
        service: string | null;
        is_active: number;
        created_at: string;
      }>;
      roles: Array<{ code: string; label: string; permissions_json: string }>;
    }>('/admin/users'),

  createUser: (payload: {
    email: string;
    fullName: string;
    password: string;
    roleCode: string;
    service: string;
  }) =>
    apiFetch('/admin/users', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  updateUserRole: (id: string, payload: { roleCode?: string; isActive?: boolean }) =>
    apiFetch(`/admin/users/${encodeURIComponent(id)}/role`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  listLogs: () =>
    apiFetch<{
      logs: Array<{
        id: number;
        user_email: string | null;
        action: string;
        resource_type: string;
        resource_id: string | null;
        details_json: string;
        created_at: string;
      }>;
    }>('/admin/logs'),

  listIndexations: () =>
    apiFetch<{
      indexations: Array<{
        id: string;
        document_id: string;
        document_title: string;
        reference_code: string;
        version_number: number;
        chunk_count: number;
        embedding_model: string;
        bm25_terms_count: number;
        duration_ms: number;
        status: string;
        created_at: string;
      }>;
    }>('/admin/indexations'),

  listAdminConversations: () =>
    apiFetch<{
      conversations: Array<{
        id: string;
        title: string;
        user_email: string | null;
        message_count: number;
        created_at: string;
        updated_at: string;
      }>;
    }>('/admin/conversations'),

  listEvaluations: () => apiFetch<{ runs: EvaluationRun[] }>('/admin/evaluations'),

  getEvaluationOverview: () => apiFetch<EvaluationOverviewResponse>('/admin/evaluation-overview'),

  listEvaluationPanel: () => apiFetch<{ entries: UserPanelEntry[] }>('/admin/evaluation-panel'),

  addEvaluationPanelEntry: (payload: {
    profile: UserPanelProfile;
    participantName: string;
    clarityScore: number;
    usefulnessScore: number;
    trustScore: number;
    comment?: string;
  }) =>
    apiFetch<{ entry: UserPanelEntry }>('/admin/evaluation-panel', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  deleteEvaluationPanelEntry: (id: string) =>
    apiFetch<{ ok: boolean }>(`/admin/evaluation-panel/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),

  runEvaluation: (options?: { mode?: 'full' | 'retrieval_only'; sampleSize?: 20 | 100 }) =>
    apiFetch<EvaluationRun>('/admin/evaluate', {
      method: 'POST',
      body: JSON.stringify(options || {}),
    }),
};
