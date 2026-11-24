import { RAG_CONFIG } from '../core/config.ts';
import { getDb } from '../database/db.ts';

export interface ConversationTurn {
  role: 'user' | 'assistant';
  content: string;
  createdAt: string;
}

export interface ConversationMemoryContext {
  conversationId: string;
  recentTurns: ConversationTurn[];
  consultedReferences: string[];
}

export function loadConversationMemory(
  conversationId: string,
  owner: { userId?: string | null; guestSessionId?: string | null },
): ConversationMemoryContext {
  const db = getDb();

  // Enforce strict isolation between users/sessions
  const conv = db
    .prepare(`SELECT id, user_id, guest_session_id FROM conversations WHERE id = ?`)
    .get(conversationId) as
    | { id: string; user_id: string | null; guest_session_id: string | null }
    | undefined;

  if (!conv) {
    return { conversationId, recentTurns: [], consultedReferences: [] };
  }

  const authorized =
    (owner.userId && conv.user_id === owner.userId) ||
    (!owner.userId && owner.guestSessionId && conv.guest_session_id === owner.guestSessionId);

  if (!authorized) {
    throw new Error("Accès refusé à cette conversation : isolation de session requise.");
  }

  const rows = db
    .prepare(
      `SELECT role, content, sources_json, created_at
       FROM messages
       WHERE conversation_id = ?
       ORDER BY created_at DESC
       LIMIT ?`,
    )
    .all(conversationId, RAG_CONFIG.maxHistoryMessages) as Array<{
    role: 'user' | 'assistant';
    content: string;
    sources_json: string | null;
    created_at: string;
  }>;

  const ordered = rows.reverse();
  const refs = new Set<string>();
  for (const r of ordered) {
    if (r.sources_json) {
      try {
        const parsed = JSON.parse(r.sources_json) as Array<{ referenceCode?: string; title?: string }>;
        for (const s of parsed) {
          if (s.referenceCode) refs.add(`${s.referenceCode} (${s.title || ''})`);
        }
      } catch {
        // Ignore malformed JSON
      }
    }
  }

  return {
    conversationId,
    recentTurns: ordered.map((r) => ({
      role: r.role,
      content: r.content,
      createdAt: r.created_at,
    })),
    consultedReferences: Array.from(refs),
  };
}
