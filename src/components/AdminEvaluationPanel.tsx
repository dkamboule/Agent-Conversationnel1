import { useState, type FormEvent } from 'react';
import { Play, Plus, Trash2 } from 'lucide-react';
import { dgeccApi } from '../services/api';
import type {
  AuthUser,
  EvaluationRun,
  Tableau18Metrics,
  Tableau19StrategyRow,
  Tableau20ConfigRow,
  UserPanelEntry,
  UserPanelProfile,
} from '../types/dgecc';

interface AdminEvaluationPanelProps {
  user: AuthUser;
  tableau18: Tableau18Metrics | null;
  panelEntries: UserPanelEntry[];
  latestEval: EvaluationRun | undefined;
  runningEval: boolean;
  onRunEvaluation: (options: { mode: 'full' | 'retrieval_only'; sampleSize: 20 | 100 }) => Promise<void>;
  onRefresh: () => Promise<void>;
  onBanner: (banner: { type: 'success' | 'error'; message: string } | null) => void;
}

const PROFILE_LABELS: Record<UserPanelProfile, string> = {
  CANDIDAT: 'Candidat',
  PARENT: 'Parent d’élève',
  AGENT_ADMINISTRATIF: 'Agent administratif',
};

const DEFAULT_TABLEAU_19_ROWS: Tableau19StrategyRow[] = [
  {
    key: 'lexical',
    label: 'Recherche lexicale (mots-clés)',
    available: false,
    statusNote: 'En attente de lancement du benchmark.',
    queriesEvaluated: 0,
    recallAt3: null,
    recallAt5: null,
    mrr: null,
  },
  {
    key: 'dense',
    label: 'Recherche dense (sémantique seule)',
    available: false,
    statusNote: 'En attente de lancement du benchmark.',
    queriesEvaluated: 0,
    recallAt3: null,
    recallAt5: null,
    mrr: null,
  },
  {
    key: 'hybrid',
    label: 'Recherche hybride (lexicale + dense)',
    available: false,
    statusNote: 'En attente de lancement du benchmark.',
    queriesEvaluated: 0,
    recallAt3: null,
    recallAt5: null,
    mrr: null,
  },
  {
    key: 'hybrid_rerank',
    label: 'Recherche hybride avec reclassement (reranking)',
    available: false,
    statusNote: 'En attente de lancement du benchmark.',
    queriesEvaluated: 0,
    recallAt3: null,
    recallAt5: null,
    mrr: null,
  },
];

const DEFAULT_TABLEAU_20_ROWS: Tableau20ConfigRow[] = [
  {
    key: 'llm_only',
    label: 'Modèle de langue seul (sans ancrage RAG)',
    available: false,
    statusNote: 'En attente de lancement du benchmark complet.',
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
    statusNote: 'En attente de lancement du benchmark complet.',
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
    statusNote: 'En attente de lancement du benchmark complet.',
    sampleSize: 0,
    inScopeCount: 0,
    outOfScopeCount: 0,
    exactitudeFactuelle: null,
    fideliteCorpus: null,
    tauxRefusApproprie: null,
  },
];

function formatScoreCell(val: number | null, unavailableLabel = 'À renseigner') {
  if (val === null || Number.isNaN(val)) {
    return <span className="italic text-[#6B7280] font-sans">{unavailableLabel}</span>;
  }
  return (
    <span className="font-mono tabular-nums text-[#111827] font-medium">
      {(val * 100).toFixed(1)} %{' '}
      <span className="text-[11px] text-[#6B7280] font-normal">({val.toFixed(3)})</span>
    </span>
  );
}

export function AdminEvaluationPanel({
  user,
  tableau18,
  panelEntries,
  latestEval,
  runningEval,
  onRunEvaluation,
  onRefresh,
  onBanner,
}: AdminEvaluationPanelProps) {
  const [evalMode, setEvalMode] = useState<'full_20' | 'full_100' | 'retrieval_only'>('full_20');
  const [caseTypologyFilter, setCaseTypologyFilter] = useState<string>('ALL');

  const [showPanelForm, setShowPanelForm] = useState(false);
  const [panelProfile, setPanelProfile] = useState<UserPanelProfile>('CANDIDAT');
  const [participantName, setParticipantName] = useState('');
  const [clarityScore, setClarityScore] = useState(5);
  const [usefulnessScore, setUsefulnessScore] = useState(5);
  const [trustScore, setTrustScore] = useState(5);
  const [panelComment, setPanelComment] = useState('');
  const [panelBusy, setPanelBusy] = useState(false);

  const t18 = latestEval?.methodology?.tableau18 ?? tableau18;
  const t19Rows = latestEval?.methodology?.tableau19 ?? DEFAULT_TABLEAU_19_ROWS;
  const t20Rows = latestEval?.methodology?.tableau20 ?? DEFAULT_TABLEAU_20_ROWS;

  const handleLaunch = async () => {
    if (evalMode === 'retrieval_only') {
      await onRunEvaluation({ mode: 'retrieval_only', sampleSize: 20 });
    } else if (evalMode === 'full_100') {
      await onRunEvaluation({ mode: 'full', sampleSize: 100 });
    } else {
      await onRunEvaluation({ mode: 'full', sampleSize: 20 });
    }
  };

  const handleAddPanelEntry = async (e: FormEvent) => {
    e.preventDefault();
    setPanelBusy(true);
    onBanner(null);
    try {
      await dgeccApi.addEvaluationPanelEntry({
        profile: panelProfile,
        participantName,
        clarityScore,
        usefulnessScore,
        trustScore,
        comment: panelComment || undefined,
      });
      setParticipantName('');
      setPanelComment('');
      setShowPanelForm(false);
      onBanner({ type: 'success', message: 'Évaluation du panel d’usagers enregistrée.' });
      await onRefresh();
    } catch (err) {
      onBanner({
        type: 'error',
        message: err instanceof Error ? err.message : 'Erreur lors de l’enregistrement du panel.',
      });
    } finally {
      setPanelBusy(false);
    }
  };

  const handleDeletePanelEntry = async (id: string) => {
    setPanelBusy(true);
    onBanner(null);
    try {
      await dgeccApi.deleteEvaluationPanelEntry(id);
      await onRefresh();
    } catch (err) {
      onBanner({
        type: 'error',
        message: err instanceof Error ? err.message : 'Erreur lors de la suppression.',
      });
    } finally {
      setPanelBusy(false);
    }
  };

  const filteredCases = (latestEval?.cases || []).filter(
    (c) => caseTypologyFilter === 'ALL' || c.typology === caseTypologyFilter,
  );

  return (
    <div className="space-y-6">
      {/* Barre d'orchestration du protocole expérimental */}
      <div className="bg-white border border-[#E5E2DC] rounded-lg p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="font-serif text-base font-semibold text-[#111827]">
            Protocole d’évaluation scientifique du système DGECC
          </h2>
          <p className="text-xs text-[#6B7280] mt-1">
            Mesure en conditions réelles les paramètres du corpus (Tableau 18), les 4 stratégies de
            recherche sur 87 requêtes stratifiées (Tableau 19) et les 3 architectures de génération
            incluant l’abstention hors périmètre (Tableau 20).
          </p>
        </div>

        {user.role === 'ADMIN' && (
          <div className="flex flex-wrap items-center gap-2.5">
            <label htmlFor="eval-mode-select" className="sr-only">
              Mode de campagne d’évaluation
            </label>
            <select
              id="eval-mode-select"
              value={evalMode}
              onChange={(e) => setEvalMode(e.target.value as typeof evalMode)}
              disabled={runningEval}
              className="px-3 py-2 text-xs bg-[#F8F7F4] border border-[#E5E2DC] rounded-md text-[#111827]"
            >
              <option value="full_20">Campagne complète · Échantillon stratifié (20 gén. / 87 rech.)</option>
              <option value="full_100">Campagne exhaustive · 100 requêtes (génération + recherche)</option>
              <option value="retrieval_only">Recherche seule · 87 requêtes (Tableau 19 instantané)</option>
            </select>
            <button
              type="button"
              disabled={runningEval}
              onClick={handleLaunch}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium text-white bg-[#1E3A8A] rounded-md hover:bg-[#172E6E] disabled:opacity-50 cursor-pointer whitespace-nowrap shrink-0"
            >
              <Play className="w-3.5 h-3.5" />
              {runningEval ? 'Évaluation en cours…' : 'Lancer l’évaluation'}
            </button>
          </div>
        )}
      </div>

      {/* TABLEAU 18 : Paramètres quantitatifs des jeux de données et de l'échantillon d'évaluation */}
      <div className="bg-white border border-[#E5E2DC] rounded-lg overflow-hidden">
        <div className="px-5 py-3.5 border-b border-[#E5E2DC] flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-serif text-sm font-semibold text-[#111827]">
            Tableau 18 : Paramètres quantitatifs des jeux de données et de l’échantillon d’évaluation
          </h3>
          <span className="text-xs text-[#6B7280] font-mono tabular-nums">
            Données en direct SQLite · Corpus publié
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#F8F7F4] border-b border-[#E5E2DC] text-[#4B5563]">
              <tr>
                <th className="py-2.5 px-5 font-semibold">Paramètre</th>
                <th className="py-2.5 px-4 font-semibold">Valeur observée</th>
                <th className="py-2.5 px-5 font-semibold">Spécifications méthodologiques</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E5E2DC]">
              <tr>
                <td className="py-3 px-5 font-medium text-[#111827]">
                  Volume du corpus documentaire
                </td>
                <td className="py-3 px-4 font-mono tabular-nums text-[#111827]">
                  {t18 ? `${t18.corpusDocuments} document(s) officiel(s)` : 'À renseigner'}
                </td>
                <td className="py-3 px-5 text-[#4B5563]">
                  Nombre de documents après apurement et validation officielle
                </td>
              </tr>
              <tr>
                <td className="py-3 px-5 font-medium text-[#111827]">
                  Granularité d’indexation
                </td>
                <td className="py-3 px-4 font-mono tabular-nums text-[#111827]">
                  {t18
                    ? `${t18.indexedChunks} segments (${t18.embeddedChunks} vectorisés)`
                    : 'À renseigner'}
                </td>
                <td className="py-3 px-5 text-[#4B5563]">
                  Nombre total de segments (chunks) indexés après découpage
                </td>
              </tr>
              <tr>
                <td className="py-3 px-5 font-medium text-[#111827]">
                  Taille du jeu de requêtes <span className="font-normal text-[#6B7280]">(cible : ~100)</span>
                </td>
                <td className="py-3 px-4 font-mono tabular-nums text-[#111827]">
                  {t18
                    ? `${t18.queryDatasetTotal} requêtes (${t18.inScopeQueries} intra-périmètre)`
                    : 'À renseigner'}
                </td>
                <td className="py-3 px-5 text-[#4B5563]">
                  Requêtes stratifiées selon les cinq typologies fonctionnelles
                </td>
              </tr>
              <tr>
                <td className="py-3 px-5 font-medium text-[#111827]">
                  Part de requêtes hors périmètre{' '}
                  <span className="font-normal text-[#6B7280]">(cible : 10 à 15 %)</span>
                </td>
                <td className="py-3 px-4 font-mono tabular-nums text-[#111827]">
                  {t18
                    ? `${t18.outOfScopeSharePct.toFixed(1)} % (${t18.outOfScopeQueries} / ${t18.queryDatasetTotal})`
                    : 'À renseigner'}
                </td>
                <td className="py-3 px-5 text-[#4B5563]">
                  Échantillon dédié à l’évaluation du mécanisme d’abstention
                </td>
              </tr>
              <tr>
                <td className="py-3 px-5 font-medium text-[#111827]">
                  Effectif du panel d’usagers
                </td>
                <td className="py-3 px-4 font-mono tabular-nums text-[#111827]">
                  {tableau18 && tableau18.userPanel.total > 0 ? (
                    <span>
                      {tableau18.userPanel.total} participant(s) (
                      {tableau18.userPanel.candidats} cand. · {tableau18.userPanel.parents} par. ·{' '}
                      {tableau18.userPanel.agentsAdministratifs} agents)
                    </span>
                  ) : (
                    <span className="italic text-[#6B7280] font-sans">
                      À renseigner (0 participant enregistré)
                    </span>
                  )}
                </td>
                <td className="py-3 px-5 text-[#4B5563]">
                  Échantillon composé de candidats, parents et agents administratifs
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Stratification des 5 typologies + gestion du panel */}
        {t18 && (
          <div className="px-5 py-3.5 bg-[#F8F7F4] border-t border-[#E5E2DC] flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-[#4B5563]">
              <span className="font-semibold text-[#111827]">Stratification :</span>
              {t18.typologyBreakdown.map((item) => (
                <span key={item.code} className="font-mono tabular-nums">
                  {item.label.split(' · ')[0]} ({item.count})
                </span>
              ))}
            </div>
            <button
              type="button"
              onClick={() => setShowPanelForm((v) => !v)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#1E3A8A] bg-white border border-[#E5E2DC] rounded-md hover:bg-[#F3F1EC] cursor-pointer whitespace-nowrap"
            >
              <Plus className="w-3.5 h-3.5" />
              {showPanelForm ? 'Fermer la saisie panel' : 'Ajouter un évaluateur au panel'}
            </button>
          </div>
        )}

        {showPanelForm && (
          <form
            onSubmit={handleAddPanelEntry}
            className="p-5 border-t border-[#E5E2DC] bg-white grid grid-cols-1 md:grid-cols-6 gap-3"
          >
            <div>
              <label className="block text-[11px] font-medium text-[#374151] mb-1">Profil</label>
              <select
                value={panelProfile}
                onChange={(e) => setPanelProfile(e.target.value as UserPanelProfile)}
                className="w-full px-2.5 py-2 text-xs border border-[#E5E2DC] rounded-md bg-white"
              >
                <option value="CANDIDAT">Candidat</option>
                <option value="PARENT">Parent d’élève</option>
                <option value="AGENT_ADMINISTRATIF">Agent administratif</option>
              </select>
            </div>
            <div className="md:col-span-2">
              <label className="block text-[11px] font-medium text-[#374151] mb-1">
                Identifiant / Participant
              </label>
              <input
                required
                minLength={2}
                value={participantName}
                onChange={(e) => setParticipantName(e.target.value)}
                placeholder="Ex. Candidat BEPC #04 ou Parent Ouaga"
                className="w-full px-2.5 py-2 text-xs border border-[#E5E2DC] rounded-md"
              />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-[#374151] mb-1">Clarté (/5)</label>
              <input
                type="number"
                min={1}
                max={5}
                value={clarityScore}
                onChange={(e) => setClarityScore(Number(e.target.value))}
                className="w-full px-2.5 py-2 text-xs border border-[#E5E2DC] rounded-md font-mono tabular-nums"
              />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-[#374151] mb-1">Utilité (/5)</label>
              <input
                type="number"
                min={1}
                max={5}
                value={usefulnessScore}
                onChange={(e) => setUsefulnessScore(Number(e.target.value))}
                className="w-full px-2.5 py-2 text-xs border border-[#E5E2DC] rounded-md font-mono tabular-nums"
              />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-[#374151] mb-1">Confiance (/5)</label>
              <input
                type="number"
                min={1}
                max={5}
                value={trustScore}
                onChange={(e) => setTrustScore(Number(e.target.value))}
                className="w-full px-2.5 py-2 text-xs border border-[#E5E2DC] rounded-md font-mono tabular-nums"
              />
            </div>
            <div className="md:col-span-5">
              <input
                value={panelComment}
                onChange={(e) => setPanelComment(e.target.value)}
                placeholder="Observation qualitative de l’usager (facultatif)"
                className="w-full px-2.5 py-2 text-xs border border-[#E5E2DC] rounded-md"
              />
            </div>
            <div className="flex justify-end">
              <button
                type="submit"
                disabled={panelBusy}
                className="w-full px-3 py-2 text-xs font-medium text-white bg-[#1E3A8A] rounded-md hover:bg-[#172E6E] disabled:opacity-50 cursor-pointer"
              >
                {panelBusy ? 'Enregistrement…' : 'Enregistrer'}
              </button>
            </div>
          </form>
        )}

        {panelEntries.length > 0 && (
          <div className="border-t border-[#E5E2DC] overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#F8F7F4] border-b border-[#E5E2DC] text-[#6B7280]">
                <tr>
                  <th className="py-2 px-5 font-medium">Participant</th>
                  <th className="py-2 px-3 font-medium">Profil</th>
                  <th className="py-2 px-3 font-medium text-right">Clarté</th>
                  <th className="py-2 px-3 font-medium text-right">Utilité</th>
                  <th className="py-2 px-3 font-medium text-right">Confiance</th>
                  <th className="py-2 px-4 font-medium">Observation</th>
                  {user.role === 'ADMIN' && <th className="py-2 px-4 font-medium text-right">Action</th>}
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E5E2DC]">
                {panelEntries.map((entry) => (
                  <tr key={entry.id}>
                    <td className="py-2 px-5 font-medium text-[#111827]">{entry.participantName}</td>
                    <td className="py-2 px-3 text-[#4B5563]">{PROFILE_LABELS[entry.profile]}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">{entry.clarityScore}/5</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">{entry.usefulnessScore}/5</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">{entry.trustScore}/5</td>
                    <td className="py-2 px-4 text-[#4B5563]">{entry.comment || '—'}</td>
                    {user.role === 'ADMIN' && (
                      <td className="py-2 px-4 text-right">
                        <button
                          type="button"
                          onClick={() => handleDeletePanelEntry(entry.id)}
                          disabled={panelBusy}
                          title="Supprimer cette entrée du panel"
                          className="p-1 text-[#6B7280] hover:text-[#B91C1C] cursor-pointer"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* TABLEAU 19 : Évaluation comparative des stratégies de recherche documentaire */}
      <div className="bg-white border border-[#E5E2DC] rounded-lg overflow-hidden">
        <div className="px-5 py-3.5 border-b border-[#E5E2DC] flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-serif text-sm font-semibold text-[#111827]">
            Tableau 19 : Évaluation comparative des stratégies de recherche documentaire
          </h3>
          <span className="text-xs text-[#6B7280] font-mono tabular-nums">
            {latestEval?.methodology
              ? `${t19Rows[0]?.queriesEvaluated ?? 87} requêtes intra-périmètre`
              : 'Aucune campagne exécutée'}
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#F8F7F4] border-b border-[#E5E2DC] text-[#4B5563]">
              <tr>
                <th className="py-2.5 px-5 font-semibold">Architecture de recherche</th>
                <th className="py-2.5 px-4 font-semibold text-right">Recall@3</th>
                <th className="py-2.5 px-4 font-semibold text-right">Recall@5</th>
                <th className="py-2.5 px-5 font-semibold text-right">MRR</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E5E2DC]">
              {t19Rows.map((row) => (
                <tr key={row.key}>
                  <td className="py-3 px-5">
                    <p className="font-medium text-[#111827]">{row.label}</p>
                    <p className="text-[11px] text-[#6B7280] mt-0.5">{row.statusNote}</p>
                  </td>
                  <td className="py-3 px-4 text-right">
                    {formatScoreCell(row.recallAt3, row.available ? '—' : 'À renseigner')}
                  </td>
                  <td className="py-3 px-4 text-right">
                    {formatScoreCell(row.recallAt5, row.available ? '—' : 'À renseigner')}
                  </td>
                  <td className="py-3 px-5 text-right">
                    {formatScoreCell(row.mrr, row.available ? '—' : 'À renseigner')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* TABLEAU 20 : Comparaison des performances de génération selon l'architecture du système */}
      <div className="bg-white border border-[#E5E2DC] rounded-lg overflow-hidden">
        <div className="px-5 py-3.5 border-b border-[#E5E2DC] flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-serif text-sm font-semibold text-[#111827]">
            Tableau 20 : Comparaison des performances de génération selon l’architecture du système
          </h3>
          <span className="text-xs text-[#6B7280] font-mono tabular-nums">
            {latestEval?.methodology && latestEval.methodology.mode === 'full'
              ? `Échantillon : ${t20Rows[0]?.sampleSize ?? 0} requêtes (${t20Rows[0]?.inScopeCount ?? 0} intra · ${t20Rows[0]?.outOfScopeCount ?? 0} hors périmètre)`
              : 'En attente de campagne complète'}
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#F8F7F4] border-b border-[#E5E2DC] text-[#4B5563]">
              <tr>
                <th className="py-2.5 px-5 font-semibold">Configuration du système</th>
                <th className="py-2.5 px-4 font-semibold text-right">Exactitude factuelle</th>
                <th className="py-2.5 px-4 font-semibold text-right">Fidélité au corpus</th>
                <th className="py-2.5 px-5 font-semibold text-right">Taux de refus approprié</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E5E2DC]">
              {t20Rows.map((row) => (
                <tr key={row.key}>
                  <td className="py-3 px-5">
                    <p className="font-medium text-[#111827]">{row.label}</p>
                    <p className="text-[11px] text-[#6B7280] mt-0.5">{row.statusNote}</p>
                  </td>
                  <td className="py-3 px-4 text-right">
                    {formatScoreCell(row.exactitudeFactuelle)}
                  </td>
                  <td className="py-3 px-4 text-right">
                    {formatScoreCell(row.fideliteCorpus)}
                  </td>
                  <td className="py-3 px-5 text-right">
                    {formatScoreCell(row.tauxRefusApproprie)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Synthèse & Journal détaillé par requête */}
      {latestEval && (
        <div className="bg-white border border-[#E5E2DC] rounded-lg overflow-hidden">
          <div className="px-5 py-3.5 border-b border-[#E5E2DC] flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="font-serif text-sm font-semibold text-[#111827]">
                Audit détaillé de la dernière campagne · {latestEval.run_name || latestEval.runName}
              </h3>
              <p className="text-[11px] text-[#6B7280] mt-0.5 font-mono tabular-nums">
                Recall@5 : {((latestEval.recall_at_k ?? latestEval.recallAtK ?? 0) * 100).toFixed(1)} % · MRR :{' '}
                {(latestEval.mrr * 100).toFixed(1)} % · Faithfulness :{' '}
                {(latestEval.faithfulness * 100).toFixed(1)} % · Latence moyenne :{' '}
                {latestEval.avg_latency_ms ?? latestEval.avgLatencyMs} ms
              </p>
            </div>
            <select
              value={caseTypologyFilter}
              onChange={(e) => setCaseTypologyFilter(e.target.value)}
              className="px-2.5 py-1.5 text-xs border border-[#E5E2DC] rounded-md bg-[#F8F7F4]"
            >
              <option value="ALL">Toutes les typologies ({latestEval.cases.length})</option>
              <option value="T1_CONDITIONS_ACCES">T1 · Conditions d’accès</option>
              <option value="T2_DOSSIERS_INSCRIPTION">T2 · Dossiers & inscription</option>
              <option value="T3_EPREUVES_JURYS">T3 · Épreuves & jurys</option>
              <option value="T4_DELIBERATIONS_RECOURS">T4 · Délibérations & recours</option>
              <option value="T5_STATUT_ORIENTATION">T5 · Statut & orientation</option>
              <option value="HORS_PERIMETRE">Hors périmètre (abstention)</option>
            </select>
          </div>

          <div className="overflow-x-auto max-h-[420px] overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#F8F7F4] border-b border-[#E5E2DC] text-[#6B7280] sticky top-0">
                <tr>
                  <th className="py-2.5 px-4 font-medium">Cas & Requête</th>
                  <th className="py-2.5 px-3 font-medium">Attendu</th>
                  <th className="py-2.5 px-3 font-medium text-right">Recall</th>
                  <th className="py-2.5 px-3 font-medium text-right">MRR</th>
                  <th className="py-2.5 px-3 font-medium text-right">Fidélité</th>
                  <th className="py-2.5 px-4 font-medium text-right">Latence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E5E2DC]">
                {filteredCases.map((c) => (
                  <tr key={c.id}>
                    <td className="py-2.5 px-4">
                      <span className="font-mono text-[#1E3A8A] mr-2">{c.id}</span>
                      {c.typology && (
                        <span className="font-mono text-[10px] text-[#6B7280] mr-2">
                          [{c.typology.split('_')[0]}]
                        </span>
                      )}
                      <span className="text-[#111827]">{c.query}</span>
                    </td>
                    <td className="py-2.5 px-3 font-mono text-[#4B5563] whitespace-nowrap">
                      {c.expectedRef}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums">{c.recallAtK}</td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums">{c.mrr}</td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums">{c.faithfulness}</td>
                    <td className="py-2.5 px-4 text-right font-mono tabular-nums">{c.latencyMs} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
