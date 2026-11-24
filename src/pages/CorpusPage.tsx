import { useEffect, useState } from 'react';
import { FileText, Search, X } from 'lucide-react';
import { DGECC_REGULATORY_DOMAINS } from '../../shared/dgeccOrg';
import { dgeccApi } from '../services/api';
import type { DocumentItem } from '../types/dgecc';

export function CorpusPage() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('ALL');
  const [activeDoc, setActiveDoc] = useState<{
    document: DocumentItem;
    previewText: string;
  } | null>(null);

  useEffect(() => {
    dgeccApi
      .listDocuments()
      .then((res) => {
        setDocuments(res.documents);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : 'Erreur de chargement du corpus.');
      })
      .finally(() => setLoading(false));
  }, []);

  const categories = [
    'ALL',
    ...Array.from(new Set([...DGECC_REGULATORY_DOMAINS, ...documents.map((d) => d.category)])),
  ];

  const filtered = documents.filter((d) => {
    const matchesCat = selectedCategory === 'ALL' || d.category === selectedCategory;
    const q = searchQuery.trim().toLowerCase();
    const matchesSearch =
      !q ||
      d.title.toLowerCase().includes(q) ||
      d.reference_code.toLowerCase().includes(q) ||
      d.service_name.toLowerCase().includes(q);
    return matchesCat && matchesSearch;
  });

  const openDetail = async (id: string) => {
    try {
      const detail = await dgeccApi.getDocumentDetail(id);
      setActiveDoc({
        document: detail.document,
        previewText: detail.versions[0]?.preview_text || '',
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Impossible de consulter le texte.');
    }
  };

  return (
    <main className="max-w-[1360px] mx-auto px-4 sm:px-6 py-10">
      <div className="pb-6 border-b border-[#E5E2DC] flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.12em] text-[#6B7280]">
            Référentiel réglementaire publié
          </p>
          <h1 className="mt-2 font-serif text-3xl font-semibold text-[#111827]">
            Corpus documentaire officiel de la DGECC
          </h1>
          <p className="mt-2 text-sm text-[#4B5563] max-w-[65ch]">
            Seuls les textes ayant fait l’objet d’une validation réglementaire et d’une publication
            officielle figurent dans ce catalogue et alimentent l’agent conversationnel.
          </p>
        </div>

        <div className="relative w-full md:w-80">
          <Search className="w-4 h-4 text-[#6B7280] absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="search"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Rechercher par titre, référence ou service…"
            className="w-full pl-9 pr-3 py-2 text-sm bg-white border border-[#D6D3CD] rounded-md focus:outline-2 focus:outline-[#1E3A8A]"
          />
        </div>
      </div>

      {/* Filtres par domaine réglementaire (Organigramme DGECC) */}
      <div className="mt-5 flex flex-wrap items-center gap-1.5">
        {categories.map((cat) => (
          <button
            key={cat}
            type="button"
            onClick={() => setSelectedCategory(cat)}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors cursor-pointer whitespace-nowrap ${
              selectedCategory === cat
                ? 'bg-[#1E3A8A] text-white'
                : 'bg-white border border-[#E5E2DC] text-[#4B5563] hover:text-[#111827]'
            }`}
          >
            {cat === 'ALL' ? 'Toutes les directions' : cat}
          </button>
        ))}
      </div>

      {error && (
        <div className="mt-6 p-4 bg-[#FEF2F2] border border-[#FECACA] rounded-md text-sm text-[#991B1B]">
          {error}
        </div>
      )}

      {loading ? (
        <div className="mt-8 space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-24 bg-white border border-[#E5E2DC] rounded-lg animate-pulse" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="mt-8 p-10 bg-white border border-[#E5E2DC] rounded-lg text-center">
          <p className="text-sm text-[#4B5563]">
            Aucun document publié ne correspond à vos critères de recherche.
          </p>
        </div>
      ) : (
        <div className="mt-6 bg-white border border-[#E5E2DC] rounded-lg divide-y divide-[#E5E2DC]">
          {filtered.map((doc) => (
            <div
              key={doc.id}
              className="p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:bg-[#FBF9F6] transition-colors"
            >
              <div className="space-y-1">
                <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-[#6B7280]">
                  <span className="font-mono tabular-nums text-[#1E3A8A] font-medium">
                    {doc.reference_code}
                  </span>
                  <span aria-hidden="true">·</span>
                  <span>{doc.doc_type}</span>
                  <span aria-hidden="true">·</span>
                  <span>{doc.category}</span>
                  <span aria-hidden="true">·</span>
                  <span className="font-mono tabular-nums">Publié le {doc.published_date}</span>
                  {doc.valid_until && (
                    <>
                      <span aria-hidden="true">·</span>
                      <span className="font-mono tabular-nums">
                        Valide jusqu’au {doc.valid_until}
                      </span>
                    </>
                  )}
                </div>
                <h2 className="font-serif text-lg font-semibold text-[#111827]">{doc.title}</h2>
                <p className="text-xs text-[#4B5563]">
                  Service émetteur : {doc.service_name} · Format {doc.file_format} · Version v
                  {doc.current_version} ·{' '}
                  <span className="font-mono tabular-nums">{doc.chunk_count}</span> passages indexés
                </p>
              </div>
              <button
                type="button"
                onClick={() => openDetail(doc.id)}
                className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium text-[#1E3A8A] bg-[#F8F7F4] border border-[#D6D3CD] rounded-md hover:bg-[#EEF2FF] whitespace-nowrap shrink-0 cursor-pointer"
              >
                <FileText className="w-3.5 h-3.5" />
                Lire le texte
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Modale de consultation du texte officiel */}
      {activeDoc && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 p-4"
        >
          <div className="w-full max-w-2xl max-h-[85vh] flex flex-col bg-white border border-[#E5E2DC] rounded-lg shadow-xl">
            <div className="p-5 border-b border-[#E5E2DC] flex items-start justify-between gap-4">
              <div>
                <p className="font-mono text-xs text-[#1E3A8A]">
                  {activeDoc.document.reference_code} · Version {activeDoc.document.current_version}
                </p>
                <h2 className="mt-1 font-serif text-xl font-semibold text-[#111827]">
                  {activeDoc.document.title}
                </h2>
                <p className="mt-1 text-xs text-[#6B7280]">
                  {activeDoc.document.service_name} · Publié le {activeDoc.document.published_date}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setActiveDoc(null)}
                aria-label="Fermer la lecture"
                className="p-1.5 text-[#6B7280] hover:text-[#111827] cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="p-6 overflow-y-auto text-sm leading-relaxed text-[#1F2937] whitespace-pre-line">
              {activeDoc.previewText}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
