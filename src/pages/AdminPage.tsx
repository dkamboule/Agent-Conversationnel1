import { useEffect, useState, type FormEvent } from 'react';
import {
  Activity,
  BarChart3,
  CheckCircle2,
  Database,
  FilePlus2,
  FileText,
  History,
  Layers,
  RefreshCw,
  ScanText,
  ShieldAlert,
  Users,
  X,
} from 'lucide-react';
import {
  DEFAULT_SERVICE_BY_DOMAIN,
  DGECC_ORG_STRUCTURE,
  DGECC_REGULATORY_DOMAINS,
  DGECC_SERVICES_LIST,
} from '../../shared/dgeccOrg';
import { AdminEvaluationPanel } from '../components/AdminEvaluationPanel';
import { dgeccApi } from '../services/api';
import type {
  AdminStatistics,
  AuthUser,
  DocumentChunkPreview,
  DocumentItem,
  DocumentStatus,
  EvaluationRun,
  Tableau18Metrics,
  UserPanelEntry,
} from '../types/dgecc';

interface AdminPageProps {
  user: AuthUser | null;
  onOpenAuth: (mode: 'login' | 'bootstrap') => void;
}

type AdminTab = 'overview' | 'documents' | 'users' | 'logs' | 'evaluation';

const STATUS_LABELS: Record<DocumentStatus, { label: string; colorClass: string }> = {
  PUBLIE: { label: 'Publié', colorClass: 'text-[#15803D]' },
  VALIDE: { label: 'Validé', colorClass: 'text-[#1E3A8A]' },
  EN_ATTENTE: { label: 'En attente', colorClass: 'text-[#B45309]' },
  BROUILLON: { label: 'Brouillon', colorClass: 'text-[#4B5563]' },
  ARCHIVE: { label: 'Archivé', colorClass: 'text-[#6B7280]' },
  REJETE: { label: 'Rejeté', colorClass: 'text-[#B91C1C]' },
};

export function AdminPage({ user, onOpenAuth }: AdminPageProps) {
  const [activeTab, setActiveTab] = useState<AdminTab>('overview');
  const [needsAdminSetup, setNeedsAdminSetup] = useState(false);

  const [stats, setStats] = useState<AdminStatistics | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [docSearch, setDocSearch] = useState('');

  const [showUploadForm, setShowUploadForm] = useState(false);
  const [uploadTitle, setUploadTitle] = useState('');
  const [uploadType, setUploadType] = useState('Arrêté ministériel');
  const [uploadCategory, setUploadCategory] = useState(DGECC_REGULATORY_DOMAINS[4]);
  const [uploadService, setUploadService] = useState(
    DEFAULT_SERVICE_BY_DOMAIN[DGECC_REGULATORY_DOMAINS[4]] || DGECC_SERVICES_LIST[0],
  );
  const [uploadRef, setUploadRef] = useState('');
  const [uploadPubDate, setUploadPubDate] = useState(new Date().toISOString().slice(0, 10));
  const [uploadValidUntil, setUploadValidUntil] = useState('2026-12-31');
  const [uploadChunkSize, setUploadChunkSize] = useState(850);
  const [uploadChunkOverlap, setUploadChunkOverlap] = useState(160);
  const [uploadInitialStatus, setUploadInitialStatus] = useState<'EN_ATTENTE' | 'BROUILLON'>(
    'EN_ATTENTE',
  );
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadOcrMode, setUploadOcrMode] = useState<'auto' | 'force' | 'off'>('auto');
  const [ocrPreviewLoading, setOcrPreviewLoading] = useState(false);
  const [ocrPreviewMeta, setOcrPreviewMeta] = useState<{
    ocrUsed: boolean;
    ocrModel: string | null;
    pageCount: number | null;
    normalizedLength: number;
  } | null>(null);
  const [ocrReviewedText, setOcrReviewedText] = useState('');
  const [uploading, setUploading] = useState(false);

  const [inspectedDoc, setInspectedDoc] = useState<{
    document: DocumentItem;
    chunks: DocumentChunkPreview[];
  } | null>(null);

  const [usersList, setUsersList] = useState<
    Array<{
      id: string;
      email: string;
      full_name: string;
      role_code: string;
      service: string | null;
      is_active: number;
      created_at: string;
    }>
  >([]);
  const [newUserEmail, setNewUserEmail] = useState('');
  const [newUserName, setNewUserName] = useState('');
  const [newUserPassword, setNewUserPassword] = useState('');
  const [newUserRole, setNewUserRole] = useState('VALIDATOR');
  const [newUserService, setNewUserService] = useState(DGECC_SERVICES_LIST[0]);

  const [logsTab, setLogsTab] = useState<'indexations' | 'activity' | 'conversations'>('indexations');
  const [indexations, setIndexations] = useState<
    Array<{
      id: string;
      document_title: string;
      reference_code: string;
      version_number: number;
      chunk_count: number;
      embedding_model: string;
      bm25_terms_count: number;
      duration_ms: number;
      status: string;
      created_at: string;
    }>
  >([]);
  const [activityLogs, setActivityLogs] = useState<
    Array<{
      id: number;
      user_email: string | null;
      action: string;
      resource_type: string;
      resource_id: string | null;
      created_at: string;
    }>
  >([]);
  const [adminConversations, setAdminConversations] = useState<
    Array<{
      id: string;
      title: string;
      user_email: string | null;
      message_count: number;
      updated_at: string;
    }>
  >([]);

  const [evaluations, setEvaluations] = useState<EvaluationRun[]>([]);
  const [tableau18, setTableau18] = useState<Tableau18Metrics | null>(null);
  const [panelEntries, setPanelEntries] = useState<UserPanelEntry[]>([]);
  const [runningEval, setRunningEval] = useState(false);

  const [actionBusyId, setActionBusyId] = useState<string | null>(null);
  const [feedbackBanner, setFeedbackBanner] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  const isStaff = user && ['ADMIN', 'VALIDATOR', 'EDITOR'].includes(user.role);

  useEffect(() => {
    dgeccApi
      .getBootstrapStatus()
      .then((res) => setNeedsAdminSetup(res.needsAdminSetup))
      .catch(() => {});
  }, [user]);

  const refreshAdminData = async () => {
    if (!isStaff) return;
    try {
      const [statsRes, docsRes] = await Promise.all([
        dgeccApi.getStatistics(),
        dgeccApi.listDocuments(),
      ]);
      setStats(statsRes);
      setDocuments(docsRes.documents);

      if (user?.role === 'ADMIN' || user?.role === 'VALIDATOR') {
        const [idxRes, logsRes, evalRes, overviewRes] = await Promise.all([
          dgeccApi.listIndexations(),
          dgeccApi.listLogs(),
          dgeccApi.listEvaluations(),
          dgeccApi.getEvaluationOverview(),
        ]);
        setIndexations(idxRes.indexations);
        setActivityLogs(logsRes.logs);
        setEvaluations(evalRes.runs);
        setTableau18(overviewRes.tableau18);
        setPanelEntries(overviewRes.panelEntries);
      }

      if (user?.role === 'ADMIN') {
        const [usersRes, convsRes] = await Promise.all([
          dgeccApi.listUsers(),
          dgeccApi.listAdminConversations(),
        ]);
        setUsersList(usersRes.users);
        setAdminConversations(convsRes.conversations);
      }
    } catch (err) {
      setFeedbackBanner({
        type: 'error',
        message: err instanceof Error ? err.message : 'Erreur de chargement des données.',
      });
    }
  };

  useEffect(() => {
    refreshAdminData();
  }, [user]);

  if (!user || !isStaff) {
    return (
      <main className="max-w-xl mx-auto px-4 py-16">
        <div className="bg-white border border-[#E5E2DC] rounded-lg p-8 text-center">
          <ShieldAlert className="w-10 h-10 text-[#1E3A8A] mx-auto" />
          <h1 className="mt-4 font-serif text-2xl font-semibold text-[#111827]">
            Espace d’administration DGECC
          </h1>
          <p className="mt-2 text-sm text-[#4B5563] leading-relaxed">
            {user
              ? `Votre compte (${user.email}) dispose du rôle Usager. Un profil Rédacteur, Validateur ou Administrateur est requis pour gérer le corpus documentaire.`
              : needsAdminSetup
                ? 'Aucun compte administrateur n’a encore été configuré sur cette instance. Initialisez le premier compte administrateur pour accéder à la gouvernance documentaire.'
                : 'Connectez-vous avec un compte habilité (Administrateur, Validateur ou Rédacteur) pour importer, valider, publier et évaluer le moteur RAG.'}
          </p>
          <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
            {needsAdminSetup ? (
              <button
                type="button"
                onClick={() => onOpenAuth('bootstrap')}
                className="px-5 py-2.5 text-sm font-medium text-white bg-[#1E3A8A] rounded-md hover:bg-[#172E6E] cursor-pointer"
              >
                Configurer le compte Administrateur
              </button>
            ) : (
              <button
                type="button"
                onClick={() => onOpenAuth('login')}
                className="px-5 py-2.5 text-sm font-medium text-white bg-[#1E3A8A] rounded-md hover:bg-[#172E6E] cursor-pointer"
              >
                Se connecter
              </button>
            )}
          </div>
        </div>
      </main>
    );
  }

  const readFileAsBase64 = async (file: File): Promise<string> => {
    const arrayBuffer = await file.arrayBuffer();
    const bytes = new Uint8Array(arrayBuffer);
    let binary = '';
    const chunkLen = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkLen) {
      binary += String.fromCharCode(...bytes.subarray(i, i + chunkLen));
    }
    return btoa(binary);
  };

  const handleOcrPreview = async () => {
    if (!uploadFile) {
      setFeedbackBanner({
        type: 'error',
        message: 'Sélectionnez d’abord un document scanné (PDF, PNG, JPG, WEBP) pour lancer l’OCR.',
      });
      return;
    }
    setOcrPreviewLoading(true);
    setFeedbackBanner(null);
    try {
      const fileBase64 = await readFileAsBase64(uploadFile);
      const preview = await dgeccApi.previewDocumentOcr({
        fileName: uploadFile.name,
        mimeType: uploadFile.type,
        fileBase64,
        ocrMode: uploadOcrMode,
      });
      setOcrReviewedText(preview.text);
      setOcrPreviewMeta({
        ocrUsed: preview.ocrUsed,
        ocrModel: preview.ocrModel,
        pageCount: preview.pageCount,
        normalizedLength: preview.normalizedLength,
      });
    } catch (err) {
      setFeedbackBanner({
        type: 'error',
        message:
          err instanceof Error
            ? err.message
            : 'Erreur lors de la reconnaissance OCR du document.',
      });
    } finally {
      setOcrPreviewLoading(false);
    }
  };

  const handleUploadSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!uploadFile) {
      setFeedbackBanner({
        type: 'error',
        message: 'Veuillez sélectionner un fichier PDF, image scannée (PNG/JPG/WEBP), DOCX, HTML ou TXT.',
      });
      return;
    }
    setUploading(true);
    setFeedbackBanner(null);
    try {
      const fileBase64 = await readFileAsBase64(uploadFile);

      const res = await dgeccApi.uploadDocumentJson({
        fileName: uploadFile.name,
        mimeType: uploadFile.type,
        fileBase64,
        title: uploadTitle,
        docType: uploadType,
        category: uploadCategory,
        serviceName: uploadService,
        referenceCode: uploadRef,
        publishedDate: uploadPubDate,
        validUntil: uploadValidUntil || null,
        ocrMode: uploadOcrMode,
        reviewedText: ocrReviewedText.trim().length >= 20 ? ocrReviewedText : undefined,
        chunkSize: uploadChunkSize,
        chunkOverlap: uploadChunkOverlap,
        initialStatus: uploadInitialStatus,
      });

      if (!res.id || !res.title || typeof res.chunksCount !== 'number' || !res.status) {
        throw new Error('Réponse incomplète du serveur lors de l’importation du document.');
      }

      const ocrNote = res.ocrUsed
        ? ` · OCR appliqué${res.ocrModel ? ` (${res.ocrModel})` : ''}`
        : '';

      setFeedbackBanner({
        type: 'success',
        message: `Document « ${res.title} » importé (${res.chunksCount} passages préparés, statut ${res.status}${ocrNote}).`,
      });
      setShowUploadForm(false);
      setUploadTitle('');
      setUploadRef('');
      setUploadFile(null);
      setOcrReviewedText('');
      setOcrPreviewMeta(null);
      await refreshAdminData();
    } catch (err) {
      setFeedbackBanner({
        type: 'error',
        message: err instanceof Error ? err.message : 'Échec de l’importation.',
      });
    } finally {
      setUploading(false);
    }
  };

  const handleStatusChange = async (id: string, status: DocumentStatus) => {
    setActionBusyId(id);
    setFeedbackBanner(null);
    try {
      await dgeccApi.validateDocument(id, status);
      setFeedbackBanner({
        type: 'success',
        message: `Statut du document mis à jour vers « ${STATUS_LABELS[status].label} ».`,
      });
      await refreshAdminData();
    } catch (err) {
      setFeedbackBanner({
        type: 'error',
        message: err instanceof Error ? err.message : 'Erreur de mise à jour du statut.',
      });
    } finally {
      setActionBusyId(null);
    }
  };

  const handlePublish = async (id: string) => {
    setActionBusyId(id);
    setFeedbackBanner(null);
    try {
      await dgeccApi.publishDocument(id);
      setFeedbackBanner({
        type: 'success',
        message: 'Document publié et indexé (BM25 + embeddings vectoriels) avec succès.',
      });
      await refreshAdminData();
    } catch (err) {
      setFeedbackBanner({
        type: 'error',
        message: err instanceof Error ? err.message : 'Erreur lors de la publication.',
      });
    } finally {
      setActionBusyId(null);
    }
  };

  const handleReindex = async (id: string) => {
    setActionBusyId(id);
    setFeedbackBanner(null);
    try {
      const res = await dgeccApi.reindexDocument(id, { bumpVersion: true });
      setFeedbackBanner({
        type: 'success',
        message: `Ré-indexation v${res.versionNumber} terminée (${res.chunkCount} chunks en ${res.durationMs} ms).`,
      });
      await refreshAdminData();
    } catch (err) {
      setFeedbackBanner({
        type: 'error',
        message: err instanceof Error ? err.message : 'Erreur de ré-indexation.',
      });
    } finally {
      setActionBusyId(null);
    }
  };

  const handleInspectChunks = async (id: string) => {
    try {
      const detail = await dgeccApi.getDocumentDetail(id);
      setInspectedDoc({ document: detail.document, chunks: detail.chunks });
    } catch (err) {
      setFeedbackBanner({
        type: 'error',
        message: err instanceof Error ? err.message : 'Impossible de charger les passages.',
      });
    }
  };

  const handleCreateUser = async (e: FormEvent) => {
    e.preventDefault();
    setFeedbackBanner(null);
    try {
      await dgeccApi.createUser({
        email: newUserEmail,
        fullName: newUserName,
        password: newUserPassword,
        roleCode: newUserRole,
        service: newUserService,
      });
      setNewUserEmail('');
      setNewUserName('');
      setNewUserPassword('');
      setFeedbackBanner({ type: 'success', message: 'Compte utilisateur créé avec succès.' });
      await refreshAdminData();
    } catch (err) {
      setFeedbackBanner({
        type: 'error',
        message: err instanceof Error ? err.message : 'Erreur lors de la création.',
      });
    }
  };

  const handleRunEvaluation = async (options: {
    mode: 'full' | 'retrieval_only';
    sampleSize: 20 | 100;
  }) => {
    setRunningEval(true);
    setFeedbackBanner(null);
    try {
      await dgeccApi.runEvaluation(options);
      setFeedbackBanner({
        type: 'success',
        message:
          options.mode === 'retrieval_only'
            ? 'Évaluation comparative des stratégies de recherche (Tableau 19) terminée.'
            : `Évaluation DGECC complète (Tableaux 18, 19 et 20 · ${options.sampleSize} requêtes) exécutée avec succès.`,
      });
      await refreshAdminData();
    } catch (err) {
      setFeedbackBanner({
        type: 'error',
        message: err instanceof Error ? err.message : "Erreur lors de l'exécution du benchmark.",
      });
    } finally {
      setRunningEval(false);
    }
  };

  const filteredDocs = documents.filter((d) => {
    const matchesStatus = statusFilter === 'ALL' || d.status === statusFilter;
    const q = docSearch.trim().toLowerCase();
    const matchesQuery =
      !q || d.title.toLowerCase().includes(q) || d.reference_code.toLowerCase().includes(q);
    return matchesStatus && matchesQuery;
  });

  const publishedCount =
    stats?.documentsByStatus.find((s) => s.status === 'PUBLIE')?.count || 0;
  const pendingCount =
    stats?.documentsByStatus
      .filter((s) => s.status === 'EN_ATTENTE' || s.status === 'BROUILLON')
      .reduce((a, b) => a + b.count, 0) || 0;

  const latestEval = evaluations[0];

  const navTabs: Array<{ id: AdminTab; label: string; icon: typeof BarChart3; adminOnly?: boolean }> =
    [
      { id: 'overview', label: 'Statistiques', icon: BarChart3 },
      { id: 'documents', label: 'Documents & Pipeline', icon: FileText },
      { id: 'users', label: 'Utilisateurs & Rôles', icon: Users, adminOnly: true },
      { id: 'logs', label: 'Journaux & Indexation', icon: History },
      { id: 'evaluation', label: 'Évaluation RAG', icon: Activity },
    ];

  return (
    <div className="max-w-[1440px] mx-auto px-4 sm:px-6 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[calc(100vh-4.5rem)]">
      {/* Sidebar de navigation SaaS */}
      <aside className="lg:col-span-3 xl:col-span-2 bg-white border border-[#E5E2DC] rounded-lg p-3 h-fit">
        <div className="px-3 py-2 border-b border-[#E5E2DC] mb-2">
          <p className="text-xs font-semibold text-[#111827] truncate">{user.fullName}</p>
          <p className="text-[11px] text-[#6B7280] font-mono">{user.role}</p>
        </div>
        <nav aria-label="Menu d'administration" className="space-y-1">
          {navTabs
            .filter((t) => !t.adminOnly || user.role === 'ADMIN')
            .map((tab) => {
              const Icon = tab.icon;
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 text-xs font-medium rounded-md transition-colors cursor-pointer whitespace-nowrap ${
                    active
                      ? 'bg-[#1E3A8A] text-white'
                      : 'text-[#374151] hover:bg-[#F3F1EC] hover:text-[#111827]'
                  }`}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <span className="truncate">{tab.label}</span>
                </button>
              );
            })}
        </nav>
      </aside>

      {/* Viewport principal */}
      <section className="lg:col-span-9 xl:col-span-10 space-y-6">
        {/* Barre supérieure de fil d'Ariane */}
        <div className="bg-white border border-[#E5E2DC] rounded-lg px-5 py-3.5 flex flex-wrap items-center justify-between gap-4">
          <div className="text-xs text-[#6B7280]">
            <span>Administration</span>
            <span className="mx-2">/</span>
            <span className="font-medium text-[#111827]">
              {navTabs.find((t) => t.id === activeTab)?.label}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={refreshAdminData}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#374151] bg-[#F8F7F4] border border-[#D6D3CD] rounded-md hover:bg-[#EFECE6] cursor-pointer whitespace-nowrap"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Actualiser
            </button>
            {activeTab === 'documents' && ['ADMIN', 'EDITOR'].includes(user.role) && (
              <button
                type="button"
                onClick={() => setShowUploadForm((v) => !v)}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-medium text-white bg-[#1E3A8A] rounded-md hover:bg-[#172E6E] cursor-pointer whitespace-nowrap"
              >
                <FilePlus2 className="w-3.5 h-3.5" />
                Importer un document
              </button>
            )}
          </div>
        </div>

        {feedbackBanner && (
          <div
            role="status"
            className={`px-4 py-3 rounded-md border text-xs flex items-center justify-between gap-4 ${
              feedbackBanner.type === 'success'
                ? 'bg-[#ECFDF5] border-[#A7F3D0] text-[#065F46]'
                : 'bg-[#FEF2F2] border-[#FECACA] text-[#991B1B]'
            }`}
          >
            <span>{feedbackBanner.message}</span>
            <button
              type="button"
              onClick={() => setFeedbackBanner(null)}
              className="cursor-pointer p-1"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* ONGLET 1 : STATISTIQUES */}
        {activeTab === 'overview' && stats && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
              <div className="bg-white border border-[#E5E2DC] rounded-lg p-4">
                <p className="text-xs text-[#6B7280]">Documents publiés</p>
                <p className="mt-2 font-mono text-2xl font-semibold text-[#111827] tabular-nums">
                  {publishedCount}
                </p>
              </div>
              <div className="bg-white border border-[#E5E2DC] rounded-lg p-4">
                <p className="text-xs text-[#6B7280]">En attente / Brouillons</p>
                <p className="mt-2 font-mono text-2xl font-semibold text-[#B45309] tabular-nums">
                  {pendingCount}
                </p>
              </div>
              <div className="bg-white border border-[#E5E2DC] rounded-lg p-4">
                <p className="text-xs text-[#6B7280]">Chunks / Vectorisés</p>
                <p className="mt-2 font-mono text-2xl font-semibold text-[#111827] tabular-nums">
                  {stats.totalChunks} / {stats.embeddedChunks}
                </p>
              </div>
              <div className="bg-white border border-[#E5E2DC] rounded-lg p-4">
                <p className="text-xs text-[#6B7280]">Conversations</p>
                <p className="mt-2 font-mono text-2xl font-semibold text-[#111827] tabular-nums">
                  {stats.totalConversations}
                </p>
              </div>
              <div className="bg-white border border-[#E5E2DC] rounded-lg p-4">
                <p className="text-xs text-[#6B7280]">Latence moyenne</p>
                <p className="mt-2 font-mono text-2xl font-semibold text-[#111827] tabular-nums">
                  {stats.performance.avgTotalMs} ms
                </p>
              </div>
              <div className="bg-white border border-[#E5E2DC] rounded-lg p-4">
                <p className="text-xs text-[#6B7280]">Retours positifs / nég.</p>
                <p className="mt-2 font-mono text-2xl font-semibold text-[#15803D] tabular-nums">
                  {stats.feedback.positive} / {stats.feedback.negative}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white border border-[#E5E2DC] rounded-lg p-5">
                <h2 className="font-serif text-base font-semibold text-[#111827]">
                  Répartition du workflow documentaire
                </h2>
                <div className="mt-4 space-y-3">
                  {(['PUBLIE', 'VALIDE', 'EN_ATTENTE', 'BROUILLON', 'ARCHIVE', 'REJETE'] as DocumentStatus[]).map(
                    (st) => {
                      const count =
                        stats.documentsByStatus.find((x) => x.status === st)?.count || 0;
                      const maxVal = Math.max(
                        1,
                        ...stats.documentsByStatus.map((x) => x.count),
                      );
                      const pct = Math.round((count / maxVal) * 100);
                      return (
                        <div key={st} className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className={`font-medium ${STATUS_LABELS[st].colorClass}`}>
                              {STATUS_LABELS[st].label}
                            </span>
                            <span className="font-mono tabular-nums text-[#111827]">{count}</span>
                          </div>
                          <div className="h-2 bg-[#F3F1EC] rounded overflow-hidden">
                            <div
                              className="h-full bg-[#1E3A8A]"
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        </div>
                      );
                    },
                  )}
                </div>
              </div>

              <div className="bg-white border border-[#E5E2DC] rounded-lg p-5">
                <h2 className="font-serif text-base font-semibold text-[#111827]">
                  Paramètres actifs du moteur RAG hybride
                </h2>
                <dl className="mt-4 divide-y divide-[#E5E2DC] text-xs">
                  <div className="py-2.5 flex justify-between">
                    <dt className="text-[#6B7280]">Modèle de langage (LLM)</dt>
                    <dd className="font-mono text-[#111827]">{stats.ragConfig.llmModel}</dd>
                  </div>
                  <div className="py-2.5 flex justify-between">
                    <dt className="text-[#6B7280]">Modèle d’embeddings</dt>
                    <dd className="font-mono text-[#111827]">{stats.ragConfig.embeddingModel}</dd>
                  </div>
                  <div className="py-2.5 flex justify-between">
                    <dt className="text-[#6B7280]">Taille de chunk / Chevauchement</dt>
                    <dd className="font-mono tabular-nums text-[#111827]">
                      {stats.ragConfig.defaultChunkSize} / {stats.ragConfig.defaultChunkOverlap} car.
                    </dd>
                  </div>
                  <div className="py-2.5 flex justify-between">
                    <dt className="text-[#6B7280]">Paramètre RRF (k) & Top-K</dt>
                    <dd className="font-mono tabular-nums text-[#111827]">
                      k={stats.ragConfig.defaultRrfK} · Top-{stats.ragConfig.defaultTopK}
                    </dd>
                  </div>
                  <div className="py-2.5 flex justify-between">
                    <dt className="text-[#6B7280]">Seuil de similarité cosinus</dt>
                    <dd className="font-mono tabular-nums text-[#111827]">
                      {stats.ragConfig.defaultSimilarityThreshold}
                    </dd>
                  </div>
                  <div className="py-2.5 flex justify-between">
                    <dt className="text-[#6B7280]">Taux de refus sécurisé</dt>
                    <dd className="font-mono tabular-nums text-[#111827]">
                      {Math.round(stats.performance.refusalRate * 100)} %
                    </dd>
                  </div>
                </dl>
              </div>
            </div>
          </div>
        )}

        {/* ONGLET 2 : DOCUMENTS & PIPELINE */}
        {activeTab === 'documents' && (
          <div className="space-y-5">
            {showUploadForm && (
              <form
                onSubmit={handleUploadSubmit}
                className="bg-white border border-[#E5E2DC] rounded-lg p-5 space-y-4"
              >
                <div className="flex items-center justify-between pb-3 border-b border-[#E5E2DC]">
                  <h2 className="font-serif text-base font-semibold text-[#111827]">
                    Importer un document officiel ou scanné (PDF, PNG/JPG/WEBP, DOCX, HTML, TXT)
                  </h2>
                  <button
                    type="button"
                    onClick={() => setShowUploadForm(false)}
                    className="text-xs text-[#6B7280] hover:text-[#111827] cursor-pointer"
                  >
                    Fermer
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                  <div>
                    <label className="block font-medium text-[#374151] mb-1">Titre officiel *</label>
                    <input
                      type="text"
                      required
                      value={uploadTitle}
                      onChange={(e) => setUploadTitle(e.target.value)}
                      placeholder="Ex. Circulaire d'inscription concours ENAM 2026"
                      className="w-full px-3 py-2 border border-[#D6D3CD] rounded-md"
                    />
                  </div>
                  <div>
                    <label className="block font-medium text-[#374151] mb-1">
                      Référence réglementaire *
                    </label>
                    <input
                      type="text"
                      required
                      value={uploadRef}
                      onChange={(e) => setUploadRef(e.target.value)}
                      placeholder="Ex. DGECC-2026-CIRC-12"
                      className="w-full px-3 py-2 border border-[#D6D3CD] rounded-md font-mono"
                    />
                  </div>
                  <div>
                    <label className="block font-medium text-[#374151] mb-1">Type de document</label>
                    <select
                      value={uploadType}
                      onChange={(e) => setUploadType(e.target.value)}
                      className="w-full px-3 py-2 border border-[#D6D3CD] rounded-md bg-white"
                    >
                      <option>Arrêté ministériel</option>
                      <option>Notice réglementaire</option>
                      <option>Circulaire</option>
                      <option>Guide de procédure</option>
                      <option>Note de service</option>
                    </select>
                  </div>
                  <div>
                    <label className="block font-medium text-[#374151] mb-1">
                      Domaine réglementaire (Direction)
                    </label>
                    <select
                      value={uploadCategory}
                      onChange={(e) => {
                        const nextDomain = e.target.value;
                        setUploadCategory(nextDomain);
                        const dir = DGECC_ORG_STRUCTURE.find((d) => d.domain === nextDomain);
                        if (dir && !dir.services.includes(uploadService)) {
                          setUploadService(dir.services[0]);
                        }
                      }}
                      className="w-full px-3 py-2 border border-[#D6D3CD] rounded-md bg-white"
                    >
                      {DGECC_REGULATORY_DOMAINS.map((domain) => (
                        <option key={domain} value={domain}>
                          {domain}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block font-medium text-[#374151] mb-1">
                      Service concerné (Organigramme DGECC)
                    </label>
                    <select
                      value={uploadService}
                      onChange={(e) => setUploadService(e.target.value)}
                      className="w-full px-3 py-2 border border-[#D6D3CD] rounded-md bg-white"
                    >
                      {DGECC_ORG_STRUCTURE.map((dir) => (
                        <optgroup key={dir.id} label={dir.domain}>
                          {dir.services.map((srv) => (
                            <option key={srv} value={srv}>
                              {srv}
                            </option>
                          ))}
                        </optgroup>
                      ))}
                    </select>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block font-medium text-[#374151] mb-1">
                        Date de publication
                      </label>
                      <input
                        type="date"
                        value={uploadPubDate}
                        onChange={(e) => setUploadPubDate(e.target.value)}
                        className="w-full px-3 py-2 border border-[#D6D3CD] rounded-md"
                      />
                    </div>
                    <div>
                      <label className="block font-medium text-[#374151] mb-1">
                        Date de validité
                      </label>
                      <input
                        type="date"
                        value={uploadValidUntil}
                        onChange={(e) => setUploadValidUntil(e.target.value)}
                        className="w-full px-3 py-2 border border-[#D6D3CD] rounded-md"
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-3">
                    <div>
                      <label className="block font-medium text-[#374151] mb-1">Chunk (car.)</label>
                      <input
                        type="number"
                        min={300}
                        max={2000}
                        value={uploadChunkSize}
                        onChange={(e) => setUploadChunkSize(Number(e.target.value))}
                        className="w-full px-2.5 py-2 border border-[#D6D3CD] rounded-md font-mono"
                      />
                    </div>
                    <div>
                      <label className="block font-medium text-[#374151] mb-1">Chevauchement</label>
                      <input
                        type="number"
                        min={0}
                        max={500}
                        value={uploadChunkOverlap}
                        onChange={(e) => setUploadChunkOverlap(Number(e.target.value))}
                        className="w-full px-2.5 py-2 border border-[#D6D3CD] rounded-md font-mono"
                      />
                    </div>
                    <div>
                      <label className="block font-medium text-[#374151] mb-1">Statut initial</label>
                      <select
                        value={uploadInitialStatus}
                        onChange={(e) =>
                          setUploadInitialStatus(e.target.value as 'EN_ATTENTE' | 'BROUILLON')
                        }
                        className="w-full px-2 py-2 border border-[#D6D3CD] rounded-md bg-white"
                      >
                        <option value="EN_ATTENTE">En attente</option>
                        <option value="BROUILLON">Brouillon</option>
                      </select>
                    </div>
                  </div>
                  <div>
                    <label className="block font-medium text-[#374151] mb-1">
                      Fichier source (PDF, image scannée PNG/JPG/WEBP, DOCX, HTML, TXT) *
                    </label>
                    <input
                      type="file"
                      required
                      accept=".pdf,.png,.jpg,.jpeg,.webp,.docx,.html,.htm,.txt"
                      onChange={(e) => {
                        setUploadFile(e.target.files?.[0] || null);
                        setOcrReviewedText('');
                        setOcrPreviewMeta(null);
                      }}
                      className="w-full px-3 py-1.5 border border-[#D6D3CD] rounded-md bg-[#F8F7F4]"
                    />
                  </div>
                  <div>
                    <label className="block font-medium text-[#374151] mb-1">
                      Reconnaissance OCR (documents scannés)
                    </label>
                    <div className="flex flex-wrap items-center gap-2">
                      <select
                        value={uploadOcrMode}
                        onChange={(e) =>
                          setUploadOcrMode(e.target.value as 'auto' | 'force' | 'off')
                        }
                        className="flex-1 min-w-[200px] px-3 py-2 border border-[#D6D3CD] rounded-md bg-white"
                      >
                        <option value="auto">Automatique (images & PDF scannés)</option>
                        <option value="force">Forcer l’OCR sur tout le document</option>
                        <option value="off">Désactivé (texte natif uniquement)</option>
                      </select>
                      <button
                        type="button"
                        disabled={!uploadFile || ocrPreviewLoading}
                        onClick={handleOcrPreview}
                        className="inline-flex items-center gap-1.5 px-3 py-2 font-medium text-[#1E3A8A] bg-[#F8F7F4] border border-[#BFDBFE] rounded-md hover:bg-[#EFF6FF] disabled:opacity-50 cursor-pointer whitespace-nowrap"
                      >
                        <ScanText className="w-3.5 h-3.5" />
                        {ocrPreviewLoading ? 'Reconnaissance OCR…' : 'Prévisualiser l’OCR'}
                      </button>
                    </div>
                  </div>
                </div>

                {ocrPreviewMeta && (
                  <div className="p-4 bg-[#F8F7F4] border border-[#E5E2DC] rounded-md space-y-2 text-xs">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="font-medium text-[#111827]">
                        Texte reconnu ({ocrReviewedText.length} caractères
                        {ocrPreviewMeta.ocrUsed
                          ? ` · OCR ${ocrPreviewMeta.ocrModel || 'actif'}`
                          : ' · extraction texte natif'}
                        {ocrPreviewMeta.pageCount
                          ? ` · ${ocrPreviewMeta.pageCount} page(s)`
                          : ''}
                        )
                      </span>
                      <span className="text-[#6B7280]">
                        Vous pouvez relire ou corriger ce texte avant l’indexation.
                      </span>
                    </div>
                    <textarea
                      rows={6}
                      value={ocrReviewedText}
                      onChange={(e) => setOcrReviewedText(e.target.value)}
                      className="w-full px-3 py-2 bg-white border border-[#D6D3CD] rounded-md font-mono text-xs leading-relaxed"
                    />
                  </div>
                )}

                <div className="pt-2 flex justify-end">
                  <button
                    type="submit"
                    disabled={uploading || ocrPreviewLoading}
                    className="px-5 py-2 text-xs font-medium text-white bg-[#1E3A8A] rounded-md hover:bg-[#172E6E] disabled:opacity-50 cursor-pointer"
                  >
                    {uploading
                      ? 'Extraction / OCR et découpage…'
                      : 'Importer et préparer les chunks'}
                  </button>
                </div>
              </form>
            )}

            {/* Filtres & Tableau documentaire */}
            <div className="bg-white border border-[#E5E2DC] rounded-lg overflow-hidden">
              <div className="p-4 border-b border-[#E5E2DC] flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-1">
                  {(['ALL', 'PUBLIE', 'EN_ATTENTE', 'VALIDE', 'BROUILLON', 'ARCHIVE', 'REJETE'] as const).map(
                    (st) => (
                      <button
                        key={st}
                        type="button"
                        onClick={() => setStatusFilter(st)}
                        className={`px-2.5 py-1 text-xs font-medium rounded cursor-pointer whitespace-nowrap ${
                          statusFilter === st
                            ? 'bg-[#1E3A8A] text-white'
                            : 'text-[#4B5563] hover:bg-[#F3F1EC]'
                        }`}
                      >
                        {st === 'ALL' ? 'Tous' : STATUS_LABELS[st].label}
                      </button>
                    ),
                  )}
                </div>
                <input
                  type="search"
                  value={docSearch}
                  onChange={(e) => setDocSearch(e.target.value)}
                  placeholder="Filtrer par titre ou référence…"
                  className="px-3 py-1.5 text-xs border border-[#D6D3CD] rounded-md w-60"
                />
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-[#F8F7F4] border-b border-[#E5E2DC] text-[#6B7280]">
                    <tr>
                      <th className="py-2.5 px-4 font-medium">Référence & Titre</th>
                      <th className="py-2.5 px-3 font-medium">Domaine réglementaire</th>
                      <th className="py-2.5 px-3 font-medium">Statut</th>
                      <th className="py-2.5 px-3 font-medium text-right">Ver.</th>
                      <th className="py-2.5 px-3 font-medium text-right">Chunks / Emb.</th>
                      <th className="py-2.5 px-4 font-medium text-right">Actions du pipeline</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E5E2DC]">
                    {filteredDocs.map((doc) => {
                      const busy = actionBusyId === doc.id;
                      const stMeta = STATUS_LABELS[doc.status] || STATUS_LABELS.BROUILLON;
                      const canValidate = ['ADMIN', 'VALIDATOR'].includes(user.role);
                      return (
                        <tr key={doc.id} className="hover:bg-[#FBF9F6]">
                          <td className="py-3 px-4 max-w-sm">
                            <p className="font-mono text-[11px] text-[#1E3A8A]">
                              {doc.reference_code} · {doc.file_format}
                            </p>
                            <p className="font-medium text-[#111827] truncate" title={doc.title}>
                              {doc.title}
                            </p>
                          </td>
                          <td className="py-3 px-3 text-[#4B5563] whitespace-nowrap">
                            {doc.category}
                          </td>
                          <td className={`py-3 px-3 font-medium whitespace-nowrap ${stMeta.colorClass}`}>
                            {stMeta.label}
                          </td>
                          <td className="py-3 px-3 text-right font-mono tabular-nums">
                            v{doc.current_version}
                          </td>
                          <td className="py-3 px-3 text-right font-mono tabular-nums">
                            {doc.chunk_count} / {doc.embedded_count ?? 0}
                          </td>
                          <td className="py-3 px-4 text-right whitespace-nowrap space-x-1.5">
                            <button
                              type="button"
                              onClick={() => handleInspectChunks(doc.id)}
                              className="px-2 py-1 border border-[#D6D3CD] rounded hover:bg-[#F3F1EC] cursor-pointer"
                            >
                              Chunks
                            </button>
                            {canValidate && doc.status !== 'PUBLIE' && (
                              <button
                                type="button"
                                disabled={busy}
                                onClick={() => handlePublish(doc.id)}
                                className="px-2.5 py-1 bg-[#15803D] text-white rounded hover:bg-[#166534] disabled:opacity-50 cursor-pointer"
                              >
                                {busy ? 'Indexation…' : 'Publier & Indexer'}
                              </button>
                            )}
                            {canValidate && doc.status === 'EN_ATTENTE' && (
                              <button
                                type="button"
                                disabled={busy}
                                onClick={() => handleStatusChange(doc.id, 'REJETE')}
                                className="px-2 py-1 text-[#B91C1C] border border-[#FECACA] rounded hover:bg-[#FEF2F2] cursor-pointer"
                              >
                                Rejeter
                              </button>
                            )}
                            {canValidate && doc.status === 'PUBLIE' && (
                              <>
                                <button
                                  type="button"
                                  disabled={busy}
                                  onClick={() => handleReindex(doc.id)}
                                  className="px-2 py-1 text-[#1E3A8A] border border-[#BFDBFE] rounded hover:bg-[#EFF6FF] cursor-pointer"
                                >
                                  {busy ? 'En cours…' : 'Ré-indexer (v+1)'}
                                </button>
                                <button
                                  type="button"
                                  disabled={busy}
                                  onClick={() => handleStatusChange(doc.id, 'ARCHIVE')}
                                  className="px-2 py-1 text-[#4B5563] border border-[#D6D3CD] rounded hover:bg-[#F3F1EC] cursor-pointer"
                                >
                                  Archiver
                                </button>
                              </>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ONGLET 3 : UTILISATEURS & RÔLES */}
        {activeTab === 'users' && user.role === 'ADMIN' && (
          <div className="space-y-6">
            <form
              onSubmit={handleCreateUser}
              className="bg-white border border-[#E5E2DC] rounded-lg p-5 space-y-4"
            >
              <h2 className="font-serif text-base font-semibold text-[#111827]">
                Créer un compte collaborateur ou validateur
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 text-xs">
                <input
                  type="text"
                  required
                  value={newUserName}
                  onChange={(e) => setNewUserName(e.target.value)}
                  placeholder="Nom complet"
                  className="px-3 py-2 border border-[#D6D3CD] rounded-md"
                />
                <input
                  type="email"
                  required
                  value={newUserEmail}
                  onChange={(e) => setNewUserEmail(e.target.value)}
                  placeholder="Courriel"
                  className="px-3 py-2 border border-[#D6D3CD] rounded-md"
                />
                <input
                  type="password"
                  required
                  minLength={6}
                  value={newUserPassword}
                  onChange={(e) => setNewUserPassword(e.target.value)}
                  placeholder="Mot de passe"
                  className="px-3 py-2 border border-[#D6D3CD] rounded-md"
                />
                <select
                  value={newUserRole}
                  onChange={(e) => setNewUserRole(e.target.value)}
                  className="px-3 py-2 border border-[#D6D3CD] rounded-md bg-white"
                >
                  <option value="VALIDATOR">VALIDATOR</option>
                  <option value="EDITOR">EDITOR</option>
                  <option value="ADMIN">ADMIN</option>
                  <option value="USER">USER</option>
                </select>
                <div>
                  <input
                    type="text"
                    list="dgecc-org-services"
                    value={newUserService}
                    onChange={(e) => setNewUserService(e.target.value)}
                    placeholder="Service DGECC"
                    className="w-full px-3 py-2 border border-[#D6D3CD] rounded-md"
                  />
                  <datalist id="dgecc-org-services">
                    {DGECC_SERVICES_LIST.map((srv) => (
                      <option key={srv} value={srv} />
                    ))}
                  </datalist>
                </div>
              </div>
              <div className="flex justify-end">
                <button
                  type="submit"
                  className="px-4 py-2 text-xs font-medium text-white bg-[#1E3A8A] rounded-md hover:bg-[#172E6E] cursor-pointer"
                >
                  Ajouter l’utilisateur
                </button>
              </div>
            </form>

            <div className="bg-white border border-[#E5E2DC] rounded-lg overflow-hidden">
              <table className="w-full text-left text-xs">
                <thead className="bg-[#F8F7F4] border-b border-[#E5E2DC] text-[#6B7280]">
                  <tr>
                    <th className="py-2.5 px-4 font-medium">Nom & Courriel</th>
                    <th className="py-2.5 px-3 font-medium">Service</th>
                    <th className="py-2.5 px-3 font-medium">Rôle</th>
                    <th className="py-2.5 px-3 font-medium">État</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E5E2DC]">
                  {usersList.map((u) => (
                    <tr key={u.id}>
                      <td className="py-3 px-4">
                        <p className="font-medium text-[#111827]">{u.full_name}</p>
                        <p className="text-[#6B7280]">{u.email}</p>
                      </td>
                      <td className="py-3 px-3 text-[#4B5563]">{u.service || 'DGECC'}</td>
                      <td className="py-3 px-3">
                        <select
                          value={u.role_code}
                          onChange={async (e) => {
                            await dgeccApi.updateUserRole(u.id, { roleCode: e.target.value });
                            await refreshAdminData();
                          }}
                          className="px-2 py-1 border border-[#D6D3CD] rounded bg-white font-mono"
                        >
                          <option value="ADMIN">ADMIN</option>
                          <option value="VALIDATOR">VALIDATOR</option>
                          <option value="EDITOR">EDITOR</option>
                          <option value="USER">USER</option>
                        </select>
                      </td>
                      <td className="py-3 px-3">
                        <button
                          type="button"
                          onClick={async () => {
                            await dgeccApi.updateUserRole(u.id, { isActive: u.is_active !== 1 });
                            await refreshAdminData();
                          }}
                          className={`font-medium cursor-pointer ${
                            u.is_active === 1 ? 'text-[#15803D]' : 'text-[#B91C1C]'
                          }`}
                        >
                          {u.is_active === 1 ? 'Actif' : 'Désactivé'}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ONGLET 4 : JOURNAUX & INDEXATION */}
        {activeTab === 'logs' && (
          <div className="bg-white border border-[#E5E2DC] rounded-lg overflow-hidden">
            <div className="p-3 border-b border-[#E5E2DC] flex items-center gap-2">
              <button
                type="button"
                onClick={() => setLogsTab('indexations')}
                className={`px-3 py-1.5 text-xs font-medium rounded cursor-pointer ${
                  logsTab === 'indexations'
                    ? 'bg-[#1E3A8A] text-white'
                    : 'text-[#4B5563] hover:bg-[#F3F1EC]'
                }`}
              >
                Historique d’indexation
              </button>
              <button
                type="button"
                onClick={() => setLogsTab('activity')}
                className={`px-3 py-1.5 text-xs font-medium rounded cursor-pointer ${
                  logsTab === 'activity'
                    ? 'bg-[#1E3A8A] text-white'
                    : 'text-[#4B5563] hover:bg-[#F3F1EC]'
                }`}
              >
                Journal d’activité
              </button>
              {user.role === 'ADMIN' && (
                <button
                  type="button"
                  onClick={() => setLogsTab('conversations')}
                  className={`px-3 py-1.5 text-xs font-medium rounded cursor-pointer ${
                    logsTab === 'conversations'
                      ? 'bg-[#1E3A8A] text-white'
                      : 'text-[#4B5563] hover:bg-[#F3F1EC]'
                  }`}
                >
                  Conversations suivies
                </button>
              )}
            </div>

            {logsTab === 'indexations' && (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-[#F8F7F4] border-b border-[#E5E2DC] text-[#6B7280]">
                    <tr>
                      <th className="py-2.5 px-4 font-medium">Document</th>
                      <th className="py-2.5 px-3 font-medium text-right">Version</th>
                      <th className="py-2.5 px-3 font-medium text-right">Chunks</th>
                      <th className="py-2.5 px-3 font-medium text-right">Termes BM25</th>
                      <th className="py-2.5 px-3 font-medium">Modèle vecteur</th>
                      <th className="py-2.5 px-3 font-medium text-right">Durée</th>
                      <th className="py-2.5 px-4 font-medium">Horodatage</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E5E2DC]">
                    {indexations.map((idx) => (
                      <tr key={idx.id}>
                        <td className="py-2.5 px-4">
                          <span className="font-mono text-[#1E3A8A] mr-2">{idx.reference_code}</span>
                          <span className="text-[#111827]">{idx.document_title}</span>
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                          v{idx.version_number}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                          {idx.chunk_count}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                          {idx.bm25_terms_count}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-[#4B5563]">
                          {idx.embedding_model}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                          {idx.duration_ms} ms
                        </td>
                        <td className="py-2.5 px-4 font-mono tabular-nums text-[#6B7280]">
                          {idx.created_at.slice(0, 19).replace('T', ' ')}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {logsTab === 'activity' && (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-[#F8F7F4] border-b border-[#E5E2DC] text-[#6B7280]">
                    <tr>
                      <th className="py-2.5 px-4 font-medium">Action</th>
                      <th className="py-2.5 px-3 font-medium">Acteur</th>
                      <th className="py-2.5 px-3 font-medium">Ressource</th>
                      <th className="py-2.5 px-4 font-medium">Date</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E5E2DC]">
                    {activityLogs.map((log) => (
                      <tr key={log.id}>
                        <td className="py-2.5 px-4 font-mono text-[#111827]">{log.action}</td>
                        <td className="py-2.5 px-3 text-[#4B5563]">
                          {log.user_email || 'Système'}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-[#6B7280]">
                          {log.resource_type} {log.resource_id ? `(${log.resource_id.slice(0, 8)})` : ''}
                        </td>
                        <td className="py-2.5 px-4 font-mono tabular-nums text-[#6B7280]">
                          {log.created_at.slice(0, 19).replace('T', ' ')}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {logsTab === 'conversations' && (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-[#F8F7F4] border-b border-[#E5E2DC] text-[#6B7280]">
                    <tr>
                      <th className="py-2.5 px-4 font-medium">Sujet</th>
                      <th className="py-2.5 px-3 font-medium">Usager</th>
                      <th className="py-2.5 px-3 font-medium text-right">Messages</th>
                      <th className="py-2.5 px-4 font-medium">Dernière activité</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E5E2DC]">
                    {adminConversations.map((c) => (
                      <tr key={c.id}>
                        <td className="py-2.5 px-4 text-[#111827]">{c.title}</td>
                        <td className="py-2.5 px-3 text-[#4B5563]">
                          {c.user_email || 'Session usager anonyme'}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                          {c.message_count}
                        </td>
                        <td className="py-2.5 px-4 font-mono tabular-nums text-[#6B7280]">
                          {c.updated_at.slice(0, 19).replace('T', ' ')}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ONGLET 5 : ÉVALUATION RAG */}
        {activeTab === 'evaluation' && (
          <AdminEvaluationPanel
            user={user}
            tableau18={tableau18}
            panelEntries={panelEntries}
            latestEval={latestEval}
            runningEval={runningEval}
            onRunEvaluation={handleRunEvaluation}
            onRefresh={refreshAdminData}
            onBanner={setFeedbackBanner}
          />
        )}
      </section>

      {/* Modale d'inspection des chunks */}
      {inspectedDoc && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 p-4"
        >
          <div className="w-full max-w-3xl max-h-[85vh] flex flex-col bg-white border border-[#E5E2DC] rounded-lg shadow-xl">
            <div className="p-5 border-b border-[#E5E2DC] flex items-start justify-between gap-4">
              <div>
                <p className="font-mono text-xs text-[#1E3A8A]">
                  {inspectedDoc.document.reference_code} · Version {inspectedDoc.document.current_version}
                </p>
                <h2 className="mt-1 font-serif text-lg font-semibold text-[#111827]">
                  Passages découpés ({inspectedDoc.chunks.length} chunks)
                </h2>
              </div>
              <button
                type="button"
                onClick={() => setInspectedDoc(null)}
                className="p-1.5 text-[#6B7280] hover:text-[#111827] cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="p-5 overflow-y-auto space-y-3 bg-[#F8F7F4]">
              {inspectedDoc.chunks.map((c) => (
                <div key={c.id} className="bg-white border border-[#E5E2DC] rounded-md p-4 text-xs">
                  <div className="flex items-center justify-between text-[#6B7280] font-mono pb-2 border-b border-[#E5E2DC] mb-2">
                    <span>
                      Chunk #{c.chunk_index + 1} · {c.token_count} tokens · {c.char_count} car.
                    </span>
                    <span className={c.has_embedding ? 'text-[#15803D]' : 'text-[#B45309]'}>
                      {c.has_embedding
                        ? `Vectorisé (${c.embedding_model || 'gemini-embedding-2'})`
                        : 'Non vectorisé'}
                    </span>
                  </div>
                  <p className="text-[#1F2937] whitespace-pre-line leading-relaxed">{c.content}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
