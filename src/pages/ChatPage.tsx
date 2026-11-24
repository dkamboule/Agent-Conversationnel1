import { useEffect, useRef, useState, type FormEvent } from 'react';
import {
  ChevronDown,
  ChevronUp,
  FileText,
  MessageSquarePlus,
  Send,
  ShieldCheck,
  ThumbsDown,
  ThumbsUp,
  Trash2,
} from 'lucide-react';
import { DGECC_REGULATORY_DOMAINS } from '../../shared/dgeccOrg';
import { ProseRenderer } from '../components/ProseRenderer';
import { dgeccApi } from '../services/api';
import type { ChatMessage, ConversationSummary } from '../types/dgecc';

interface ChatPageProps {
  initialPrompt?: string | null;
  onConsumedPrompt?: () => void;
}

const STARTER_QUESTIONS = [
  'Quelles pièces et dates limites s’appliquent aux examens professionnels CAPES, CAPET, CAP-CEG et CAET 2026 ?',
  'Quelles pièces fournir pour une dispense d’EPS ou pour l’inscription au BEPC ?',
  'Quels sont les seuils d’admission directe et de rachat au BEPC, CAP et BEP ?',
  'Quelles familles d’emplois relèvent du statut particulier du métier Éducation (Décret 2025-1558) ?',
];

const CATEGORIES = [
  { value: 'ALL', label: 'Toutes les directions (Corpus complet)' },
  ...DGECC_REGULATORY_DOMAINS.map((domain) => ({
    value: domain,
    label: domain,
  })),
];

export function ChatPage({ initialPrompt, onConsumedPrompt }: ChatPageProps) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | undefined>(undefined);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('ALL');
  const [topK, setTopK] = useState(5);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingStage, setStreamingStage] = useState<string | null>(null);
  const [streamingText, setStreamingText] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [lastFailedQuestion, setLastFailedQuestion] = useState<string | null>(null);
  const [expandedSourcesMsg, setExpandedSourcesMsg] = useState<Record<string, boolean>>({});
  const [expandedGuardMsg, setExpandedGuardMsg] = useState<Record<string, boolean>>({});
  const feedEndRef = useRef<HTMLDivElement | null>(null);
  const autoSentRef = useRef<string | null>(null);

  const loadConversations = async () => {
    try {
      const res = await dgeccApi.listConversations();
      setConversations(res.conversations);
    } catch {
      // Ignore if session empty
    }
  };

  useEffect(() => {
    loadConversations();
  }, []);

  useEffect(() => {
    feedEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingText, isStreaming]);

  const handleSelectConversation = async (id: string) => {
    setError(null);
    try {
      const res = await dgeccApi.getConversation(id);
      setActiveConversationId(id);
      setMessages(res.messages);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Impossible de charger la conversation.');
    }
  };

  const handleDeleteConversation = async (id: string) => {
    try {
      await dgeccApi.deleteConversation(id);
      if (activeConversationId === id) {
        setActiveConversationId(undefined);
        setMessages([]);
      }
      await loadConversations();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur lors de la suppression.');
    }
  };

  const startNewConversation = () => {
    setActiveConversationId(undefined);
    setMessages([]);
    setError(null);
  };

  const sendQuestion = async (questionText: string) => {
    const clean = questionText.trim();
    if (!clean || isStreaming) return;

    setError(null);
    setLastFailedQuestion(null);
    setInputValue('');
    setIsStreaming(true);
    setStreamingStage('Analyse linguistique et compréhension de la question...');
    setStreamingText('');

    const optimisticUserMsg: ChatMessage = {
      id: `temp-user-${Date.now()}`,
      role: 'user',
      content: clean,
      createdAt: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimisticUserMsg]);

    try {
      await dgeccApi.streamQuestion(
        {
          question: clean,
          conversationId: activeConversationId,
          categoryFilter: categoryFilter === 'ALL' ? undefined : categoryFilter,
          topK,
        },
        {
          onStage: (_stage, detail) => {
            if (detail) setStreamingStage(detail);
          },
          onToken: (delta) => {
            setStreamingStage(null);
            setStreamingText((prev) => prev + delta);
          },
          onComplete: (result) => {
            setActiveConversationId(result.conversationId);
            const assistantMsg: ChatMessage = {
              id: result.messageId,
              role: 'assistant',
              content: result.answer,
              analysis: result.analysis,
              sources: result.sources,
              guardrails: result.guardrails,
              metrics: result.metrics,
              feedback: null,
              createdAt: new Date().toISOString(),
            };
            setMessages((prev) => [...prev, assistantMsg]);
            setStreamingText('');
            setStreamingStage(null);
            loadConversations();
          },
        },
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Une erreur est survenue lors de la communication avec le moteur RAG.',
      );
      setLastFailedQuestion(clean);
      setStreamingText('');
      setStreamingStage(null);
    } finally {
      setIsStreaming(false);
    }
  };

  useEffect(() => {
    if (initialPrompt && autoSentRef.current !== initialPrompt) {
      autoSentRef.current = initialPrompt;
      onConsumedPrompt?.();
      sendQuestion(initialPrompt);
    }
  }, [initialPrompt]);

  const handleFormSubmit = (e: FormEvent) => {
    e.preventDefault();
    sendQuestion(inputValue);
  };

  const handleFeedback = async (messageId: string, feedback: 'positive' | 'negative') => {
    try {
      await dgeccApi.sendFeedback(messageId, feedback);
      setMessages((prev) =>
        prev.map((m) => (m.id === messageId ? { ...m, feedback } : m)),
      );
    } catch {
      // Ignore transient error
    }
  };

  return (
    <div className="max-w-[1360px] mx-auto px-4 sm:px-6 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[calc(100vh-4.5rem)]">
      {/* Colonne latérale : Historique & Filtres de recherche hybride */}
      <aside className="lg:col-span-3 flex flex-col bg-white border border-[#E5E2DC] rounded-lg p-4 h-fit lg:sticky lg:top-20">
        <button
          type="button"
          onClick={startNewConversation}
          className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium text-white bg-[#1E3A8A] rounded-md hover:bg-[#172E6E] transition-colors cursor-pointer"
        >
          <MessageSquarePlus className="w-4 h-4" />
          Nouvelle conversation
        </button>

        <div className="mt-5 pt-4 border-t border-[#E5E2DC] space-y-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-[#6B7280]">
            Filtres du corpus
          </h2>
          <div>
            <label htmlFor="cat-filter" className="block text-xs text-[#4B5563] mb-1">
              Domaine réglementaire
            </label>
            <select
              id="cat-filter"
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="w-full px-2.5 py-2 text-xs bg-[#F8F7F4] border border-[#D6D3CD] rounded-md text-[#111827]"
            >
              {CATEGORIES.map((c) => (
                <option key={c.value} value={c.value}>
                  {c.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <div className="flex items-center justify-between text-xs text-[#4B5563] mb-1">
              <label htmlFor="topk-range">Passages retenus (Top-K)</label>
              <span className="font-mono tabular-nums font-medium text-[#111827]">{topK}</span>
            </div>
            <input
              id="topk-range"
              type="range"
              min={3}
              max={8}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full accent-[#1E3A8A]"
            />
          </div>
        </div>

        <div className="mt-5 pt-4 border-t border-[#E5E2DC]">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-[#6B7280] mb-2.5">
            Historique de session
          </h2>
          {conversations.length === 0 ? (
            <p className="text-xs text-[#6B7280]">
              Aucune conversation enregistrée pour cette session.
            </p>
          ) : (
            <ul className="space-y-1 max-h-72 overflow-y-auto pr-1">
              {conversations.map((conv) => {
                const isSelected = conv.id === activeConversationId;
                return (
                  <li
                    key={conv.id}
                    className={`group flex items-center justify-between gap-2 rounded-md px-2.5 py-2 text-xs transition-colors ${
                      isSelected ? 'bg-[#EEF2FF] text-[#1E3A8A] font-medium' : 'hover:bg-[#F3F1EC] text-[#374151]'
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => handleSelectConversation(conv.id)}
                      className="flex-1 text-left truncate cursor-pointer"
                      title={conv.title}
                    >
                      {conv.title}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDeleteConversation(conv.id)}
                      aria-label="Supprimer cette conversation"
                      className="opacity-70 hover:opacity-100 text-[#6B7280] hover:text-[#B91C1C] p-1 cursor-pointer"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </aside>

      {/* Zone conversationnelle principale */}
      <section className="lg:col-span-9 flex flex-col bg-white border border-[#E5E2DC] rounded-lg overflow-hidden">
        <div className="px-5 py-4 border-b border-[#E5E2DC] flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="font-serif text-lg font-semibold text-[#111827]">
              Assistant conversationnel d’orientation DGECC
            </h1>
            <p className="text-xs text-[#6B7280]">
              Réponses générées exclusivement à partir des documents publiés et vérifiés par la DGECC.
            </p>
          </div>
          <span className="text-xs text-[#6B7280] font-mono tabular-nums">
            BM25 + Vecteur · Fusion RRF
          </span>
        </div>

        {/* Fil des messages */}
        <div className="flex-1 overflow-y-auto p-5 sm:p-6 space-y-6 bg-[#FBF9F6]">
          {messages.length === 0 && !isStreaming && (
            <div className="max-w-2xl mx-auto py-8">
              <h2 className="font-serif text-2xl font-semibold text-[#111827]">
                Comment la DGECC peut-elle vous orienter aujourd’hui ?
              </h2>
              <p className="mt-2 text-sm text-[#4B5563] leading-relaxed">
                Posez votre question concernant les calendriers d’examens, les conditions d’accès aux
                concours, les pièces justificatives, les demandes de duplicata ou les recours.
              </p>

              <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3">
                {STARTER_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => sendQuestion(q)}
                    className="text-left p-4 bg-white border border-[#E5E2DC] rounded-md hover:border-[#1E3A8A] transition-colors cursor-pointer"
                  >
                    <p className="text-sm text-[#111827] leading-snug">{q}</p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => {
            if (msg.role === 'user') {
              return (
                <div key={msg.id} className="flex justify-end">
                  <div className="max-w-[82%] sm:max-w-[70%] bg-[#1E3A8A] text-white px-4 py-3 rounded-lg text-sm leading-relaxed">
                    {msg.content}
                  </div>
                </div>
              );
            }

            const showDetails = Boolean(expandedSourcesMsg[msg.id]);
            const showGuard = Boolean(expandedGuardMsg[msg.id]);
            const sources = msg.sources || [];

            return (
              <article
                key={msg.id}
                className="bg-white border border-[#E5E2DC] rounded-lg p-5 sm:p-6 space-y-5"
              >
                {/* En-tête analytique sobre */}
                {msg.analysis && (
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[#6B7280] pb-3 border-b border-[#E5E2DC]">
                    <span>Intention : {msg.analysis.intent.replaceAll('_', ' ')}</span>
                    {msg.guardrails && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span
                          className={
                            msg.guardrails.refused
                              ? 'text-[#B91C1C] font-medium'
                              : msg.guardrails.confidenceLevel === 'ELEVEE'
                                ? 'text-[#15803D] font-medium'
                                : 'text-[#B45309] font-medium'
                          }
                        >
                          Fiabilité : {msg.guardrails.confidenceLevel}
                        </span>
                      </>
                    )}
                    {msg.metrics && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className="font-mono tabular-nums">
                          {msg.metrics.totalMs} ms ({msg.metrics.bm25Hits} BM25 /{' '}
                          {msg.metrics.vectorHits} Vecteur)
                        </span>
                      </>
                    )}
                  </div>
                )}

                {/* Réponse principale */}
                <ProseRenderer content={msg.content} />

                {/* Bloc distinct : Références documentaires */}
                {sources.length > 0 && (
                  <div className="pt-4 border-t border-[#E5E2DC]">
                    <div className="flex items-center justify-between gap-4">
                      <h3 className="text-xs font-semibold uppercase tracking-wider text-[#374151] flex items-center gap-2">
                        <FileText className="w-3.5 h-3.5 text-[#1E3A8A]" />
                        Sources documentaires utilisées ({sources.length})
                      </h3>
                      <button
                        type="button"
                        onClick={() =>
                          setExpandedSourcesMsg((prev) => ({
                            ...prev,
                            [msg.id]: !prev[msg.id],
                          }))
                        }
                        className="inline-flex items-center gap-1 text-xs font-medium text-[#1E3A8A] hover:underline cursor-pointer"
                      >
                        {showDetails ? 'Masquer les scores et extraits' : 'Voir extraits et scores RRF'}
                        {showDetails ? (
                          <ChevronUp className="w-3.5 h-3.5" />
                        ) : (
                          <ChevronDown className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>

                    <ul className="mt-3 divide-y divide-[#F0EDE6]">
                      {sources.map((src, idx) => (
                        <li key={src.chunkId} className="py-2.5 text-xs">
                          <div className="flex flex-wrap items-baseline justify-between gap-2">
                            <p className="font-medium text-[#111827]">
                              [{idx + 1}] {src.title}
                            </p>
                            <span className="font-mono tabular-nums text-[#6B7280]">
                              {src.referenceCode} · v{src.versionNumber}
                            </span>
                          </div>
                          <p className="mt-0.5 text-[#6B7280]">
                            {src.category} · {src.serviceName} · Publié le {src.publishedDate}
                            {src.validUntil ? ` · Valide jusqu’au ${src.validUntil}` : ''}
                            {src.isExpired ? ' · Texte arrivé à échéance' : ''}
                          </p>
                          {showDetails && (
                            <div className="mt-2 p-3 bg-[#F8F7F4] rounded text-[#374151] space-y-1.5">
                              <p className="leading-relaxed">{src.excerpt}…</p>
                              <p className="font-mono tabular-nums text-[11px] text-[#6B7280]">
                                RRF : {src.rrfScore} · Rerank : {src.rerankScore}
                                {src.bm25Rank ? ` · Rang BM25 #${src.bm25Rank} (${src.bm25Score})` : ''}
                                {src.vectorRank
                                  ? ` · Rang Vecteur #${src.vectorRank} (${src.vectorScore})`
                                  : ''}
                              </p>
                            </div>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Orientation administrative, Garde-fous & Retour usager */}
                <div className="pt-4 border-t border-[#E5E2DC] flex flex-wrap items-center justify-between gap-4 text-xs">
                  <div className="flex flex-wrap items-center gap-3 text-[#4B5563]">
                    {msg.guardrails && (
                      <>
                        <span>
                          Service d’orientation :{' '}
                          <strong className="font-medium text-[#111827]">
                            {msg.guardrails.orientedService}
                          </strong>
                        </span>
                        <button
                          type="button"
                          onClick={() =>
                            setExpandedGuardMsg((prev) => ({
                              ...prev,
                              [msg.id]: !prev[msg.id],
                            }))
                          }
                          className="inline-flex items-center gap-1 text-[#1E3A8A] hover:underline cursor-pointer"
                        >
                          <ShieldCheck className="w-3.5 h-3.5" />
                          Contrôles ({msg.guardrails.checks.filter((c) => c.passed).length}/
                          {msg.guardrails.checks.length})
                        </button>
                      </>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="text-[#6B7280]">Réponse utile ?</span>
                    <button
                      type="button"
                      onClick={() => handleFeedback(msg.id, 'positive')}
                      aria-label="Marquer la réponse comme utile"
                      className={`p-1.5 rounded border transition-colors cursor-pointer ${
                        msg.feedback === 'positive'
                          ? 'bg-[#ECFDF5] border-[#15803D] text-[#15803D]'
                          : 'border-[#E5E2DC] text-[#6B7280] hover:text-[#111827]'
                      }`}
                    >
                      <ThumbsUp className="w-3.5 h-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => handleFeedback(msg.id, 'negative')}
                      aria-label="Signaler une réponse à améliorer"
                      className={`p-1.5 rounded border transition-colors cursor-pointer ${
                        msg.feedback === 'negative'
                          ? 'bg-[#FEF2F2] border-[#B91C1C] text-[#B91C1C]'
                          : 'border-[#E5E2DC] text-[#6B7280] hover:text-[#111827]'
                      }`}
                    >
                      <ThumbsDown className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                {showGuard && msg.guardrails && (
                  <div className="p-3 bg-[#F8F7F4] border border-[#E5E2DC] rounded text-xs space-y-1.5">
                    {msg.guardrails.checks.map((chk) => (
                      <div key={chk.name} className="flex items-start justify-between gap-4">
                        <span className="font-medium text-[#111827]">{chk.name}</span>
                        <span className={chk.passed ? 'text-[#15803D]' : 'text-[#B91C1C]'}>
                          {chk.passed ? 'Validé' : 'Alerte'} — {chk.detail}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </article>
            );
          })}

          {/* Affichage progressif en streaming */}
          {isStreaming && (
            <div className="bg-white border border-[#E5E2DC] rounded-lg p-5 sm:p-6 space-y-3">
              {streamingStage && !streamingText && (
                <p className="text-xs font-medium text-[#1E3A8A]">{streamingStage}</p>
              )}
              {streamingText ? (
                <div className="relative">
                  <ProseRenderer content={streamingText} />
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="h-3 bg-[#EAE6DF] rounded w-3/4 animate-pulse" />
                  <div className="h-3 bg-[#EAE6DF] rounded w-1/2 animate-pulse" />
                </div>
              )}
            </div>
          )}

          {error && (
            <div
              role="alert"
              className="p-4 bg-[#FEF2F2] border border-[#FECACA] rounded-md text-sm text-[#991B1B] flex flex-wrap items-center justify-between gap-3"
            >
              <span>{error}</span>
              {lastFailedQuestion && !isStreaming && (
                <button
                  type="button"
                  onClick={() => sendQuestion(lastFailedQuestion)}
                  className="px-3 py-1.5 text-xs font-medium bg-white border border-[#FCA5A5] text-[#991B1B] rounded hover:bg-[#FEE2E2] transition-colors cursor-pointer"
                >
                  Réessayer
                </button>
              )}
            </div>
          )}

          <div ref={feedEndRef} />
        </div>

        {/* Zone de saisie */}
        <form
          onSubmit={handleFormSubmit}
          className="p-4 bg-white border-t border-[#E5E2DC] flex items-end gap-3"
        >
          <div className="flex-1">
            <label htmlFor="chat-question-input" className="sr-only">
              Votre question pour la DGECC
            </label>
            <textarea
              id="chat-question-input"
              rows={2}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendQuestion(inputValue);
                }
              }}
              maxLength={2000}
              placeholder="Posez votre question (ex. Quelles pièces fournir pour le concours d'attaché d'administration ?)"
              className="w-full px-3.5 py-2.5 text-sm bg-[#F8F7F4] border border-[#D6D3CD] rounded-md resize-none focus:outline-2 focus:outline-[#1E3A8A]"
            />
          </div>
          <button
            type="submit"
            disabled={isStreaming || inputValue.trim().length < 2}
            className="inline-flex items-center gap-2 px-5 py-3 text-sm font-medium text-white bg-[#1E3A8A] rounded-md hover:bg-[#172E6E] disabled:opacity-50 transition-colors shrink-0 cursor-pointer"
          >
            <Send className="w-4 h-4" />
            Envoyer
          </button>
        </form>
      </section>
    </div>
  );
}
