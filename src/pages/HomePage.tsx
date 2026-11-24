import { useState } from 'react';
import { ArrowRight, BookOpen } from 'lucide-react';
import { DGECC_ORG_STRUCTURE } from '../../shared/dgeccOrg';
import type { AppRoute } from '../components/TopBar';

interface HomePageProps {
  onNavigate: (route: AppRoute) => void;
  onStartQuestion: (question: string) => void;
}

export function HomePage({ onNavigate, onStartQuestion }: HomePageProps) {
  const [heroImgFailed, setHeroImgFailed] = useState(false);
  const [dossierImgFailed, setDossierImgFailed] = useState(false);
  const [orgImgFailed, setOrgImgFailed] = useState(false);

  const scopes = [
    {
      index: '01 · DECEPS',
      title: 'Examens & concours post-primaire et secondaire',
      description:
        "Organisation du BEPC, du CAP, du BEP et des concours d'entrée en classe de 2nde et 1ère année BEP : conditions, dossiers et délibérations.",
      prompt: 'Quelles sont les conditions d’inscription et les pièces du dossier au BEPC ?',
    },
    {
      index: '02 · DG / SGRH',
      title: 'Statut du métier Éducation, formation et emploi',
      description:
        "Familles d'emplois, modalités d'accès aux concours directs et professionnels et règles de carrière issues du décret n°2025-1558.",
      prompt: 'Quelles familles d’emplois composent le métier Éducation, formation et promotion de l’emploi ?',
    },
    {
      index: '03 · Jury & Réclamations',
      title: 'Délibérations, rachats et réclamations',
      description:
        'Barèmes d’admission directe, épreuves écrites de contrôle, règles de rachat, réclamations post-résultats et mesures disciplinaires.',
      prompt: 'Quels sont les seuils de points pour l’admission, l’écrit de contrôle et le rachat au BEPC ?',
    },
    {
      index: '04 · Besoins spécifiques',
      title: 'Aménagements et dispenses réglementaires',
      description:
        'Conditions de dispense d’EPS, tiers-temps et dispositifs adaptés aux candidats présentant un handicap visuel ou auditif.',
      prompt: 'Quelles mesures sont prévues pour les candidats en situation de handicap aux examens ?',
    },
  ];

  return (
    <main className="bg-[#F8F7F4] text-[#111827]">
      {/* Section Héro Institutionnelle — Grille 2 colonnes étirées */}
      <section className="max-w-[1360px] mx-auto px-4 sm:px-6 pt-10 pb-12 lg:pt-14 lg:pb-16">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-stretch">
          <div className="lg:col-span-6 flex flex-col justify-between py-2">
            <div>
              <p className="text-xs font-medium uppercase tracking-[0.12em] text-[#6B7280]">
                Direction Générale des Examens, Concours et de la Certification
              </p>
              <h1 className="mt-4 font-serif text-3xl sm:text-4xl lg:text-[2.85rem] font-semibold leading-[1.12] tracking-tight text-[#111827] text-balance">
                Orientation officielle des usagers fondée sur les textes validés
              </h1>
              <p className="mt-5 text-base leading-[1.7] text-[#374151] max-w-[58ch]">
                Interrogez en langage naturel le corpus réglementaire de la DGECC. Chaque réponse
                combine recherche lexicale exacte et recherche sémantique vectorielle sur les
                arrêtés, notices et circulaires officiellement publiés, avec citation systématique
                des sources.
              </p>
            </div>

            <div className="mt-8 pt-6 border-t border-[#E5E2DC] flex flex-wrap items-center gap-6">
              <button
                type="button"
                onClick={() => onNavigate('chat')}
                className="inline-flex items-center gap-2.5 px-6 py-3 text-sm font-medium text-white bg-[#1E3A8A] rounded-md hover:bg-[#172E6E] transition-colors duration-150 whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#1E3A8A]"
              >
                Ouvrir l’assistant d’orientation
                <ArrowRight className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={() => onNavigate('corpus')}
                className="inline-flex items-center gap-2 text-sm font-medium text-[#1E3A8A] hover:underline underline-offset-4 whitespace-nowrap shrink-0 cursor-pointer"
              >
                <BookOpen className="w-4 h-4" />
                Consulter le corpus officiel
              </button>
            </div>
          </div>

          <div className="lg:col-span-6 flex flex-col">
            <div className="flex-1 min-h-[280px] lg:min-h-[380px] rounded-lg overflow-hidden border border-[#E5E2DC] bg-[#EAE6DF]">
              {heroImgFailed ? (
                <div className="w-full h-full flex flex-col items-center justify-center p-8 text-center text-[#4B5563]">
                  <BookOpen className="w-10 h-10 mb-3 text-[#1E3A8A]" />
                  <p className="font-serif text-lg text-[#111827]">
                    Siège de la Direction Générale des Examens, Concours et de la Certification (DGECC)
                  </p>
                </div>
              ) : (
                <img
                  src="/assets/aistudio/images_0bdeb8f2.jpg"
                  alt="Siège et portail d’entrée de la Direction Générale des Examens et Concours (DGEC / DGECC) du Ministère de l’Éducation Nationale et de l’Alphabétisation"
                  referrerPolicy="no-referrer"
                  onError={() => setHeroImgFailed(true)}
                  className="w-full h-full lg:h-full object-cover object-center"
                />
              )}
            </div>
            <p className="mt-2.5 text-xs text-[#6B7280]">
              Fig. 1 — Siège et entrée officielle de la Direction Générale des Examens et Concours (DGEC / DGECC).
            </p>
          </div>
        </div>

        {/* Bandeau 4 colonnes pleine largeur sous le héro */}
        <div className="mt-12 pt-10 border-t border-[#E5E2DC] grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
          {scopes.map((scope) => (
            <div key={scope.index} className="flex flex-col justify-between">
              <div>
                <p className="font-mono text-xs text-[#1E3A8A] tabular-nums">{scope.index}</p>
                <h2 className="mt-1.5 font-serif text-lg font-semibold text-[#111827]">
                  {scope.title}
                </h2>
                <p className="mt-2 text-sm leading-relaxed text-[#4B5563]">{scope.description}</p>
              </div>
              <button
                type="button"
                onClick={() => onStartQuestion(scope.prompt)}
                className="mt-4 inline-flex items-center gap-1.5 text-xs font-medium text-[#1E3A8A] hover:underline underline-offset-4 self-start cursor-pointer"
              >
                Poser une question sur ce thème
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>
      </section>

      {/* Section Architecture & Fiabilité documentaire */}
      <section className="bg-white border-t border-[#E5E2DC]">
        <div className="max-w-[1360px] mx-auto px-4 sm:px-6 py-14 lg:py-16">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-stretch">
            <div className="lg:col-span-5 flex flex-col">
              <div className="flex-1 min-h-[260px] rounded-lg overflow-hidden border border-[#E5E2DC] bg-[#EAE6DF]">
                {dossierImgFailed ? (
                  <div className="w-full h-full flex items-center justify-center p-8 text-[#4B5563]">
                    <span className="font-serif text-base">Dossiers réglementaires DGECC</span>
                  </div>
                ) : (
                  <img
                    src="/src/assets/images/dgecc_exam_dossier_1795356473665.jpg"
                    alt="Dossiers d'examens et calendrier réglementaire officiel de la DGECC"
                    referrerPolicy="no-referrer"
                    onError={() => setDossierImgFailed(true)}
                    className="w-full h-full lg:h-full object-cover"
                  />
                )}
              </div>
              <p className="mt-2.5 text-xs text-[#6B7280]">
                Fig. 2 — Textes de référence publiés, calendriers de sessions et notices de concours.
              </p>
            </div>

            <div className="lg:col-span-7 flex flex-col justify-center">
              <p className="text-xs font-medium uppercase tracking-[0.12em] text-[#6B7280]">
                Gouvernance & Traçabilité
              </p>
              <h2 className="mt-2 font-serif text-2xl sm:text-3xl font-semibold text-[#111827] text-balance">
                Des réponses strictement ancrées dans le corpus validé
              </h2>
              <p className="mt-3 text-sm sm:text-base leading-relaxed text-[#374151] max-w-[62ch]">
                Aucun texte en cours de rédaction ou en attente de validation n’est exposé aux
                usagers. Le moteur d’orientation applique un contrôle systématique avant restitution.
              </p>

              <dl className="mt-8 divide-y divide-[#E5E2DC] border-t border-b border-[#E5E2DC]">
                <div className="py-4 grid grid-cols-1 sm:grid-cols-3 gap-2">
                  <dt className="text-sm font-semibold text-[#111827]">Recherche hybride RRF</dt>
                  <dd className="sm:col-span-2 text-sm text-[#4B5563]">
                    Combinaison parallèle de l’indexation lexicale BM25 (références exactes, dates,
                    sigles) et de la recherche sémantique vectorielle, fusionnée par Reciprocal Rank
                    Fusion.
                  </dd>
                </div>
                <div className="py-4 grid grid-cols-1 sm:grid-cols-3 gap-2">
                  <dt className="text-sm font-semibold text-[#111827]">Garde-fous automatisés</dt>
                  <dd className="sm:col-span-2 text-sm text-[#4B5563]">
                    Vérification de la présence des citations, contrôle de la date de validité des
                    textes et refus explicite lorsqu’aucune source fiable ne couvre la demande.
                  </dd>
                </div>
                <div className="py-4 grid grid-cols-1 sm:grid-cols-3 gap-2">
                  <dt className="text-sm font-semibold text-[#111827]">Orientation humaine</dt>
                  <dd className="sm:col-span-2 text-sm text-[#4B5563]">
                    Redirection vers la direction ou le service compétent de l’organigramme (DECEPS,
                    DCP, DEPE, DEB, Contrôle Interne, Services Financier, SADo ou CAT) pour les
                    situations particulières ou recours.
                  </dd>
                </div>
              </dl>
            </div>
          </div>
        </div>
      </section>

      {/* Section Organigramme & Domaines réglementaires */}
      <section className="bg-[#F8F7F4] border-t border-[#E5E2DC]">
        <div className="max-w-[1360px] mx-auto px-4 sm:px-6 py-14 lg:py-16">
          <div className="max-w-[68ch]">
            <p className="text-xs font-medium uppercase tracking-[0.12em] text-[#6B7280]">
              Architecture organisationnelle
            </p>
            <h2 className="mt-2 font-serif text-2xl sm:text-3xl font-semibold text-[#111827] text-balance">
              Domaines réglementaires alignés sur l’organigramme de la DGECC
            </h2>
            <p className="mt-3 text-sm sm:text-base leading-relaxed text-[#374151]">
              Chaque texte du corpus et chaque orientation produite par l’assistant est rattachée à
              la Direction Générale, à ses services d’appui ou à l’une des quatre directions
              techniques compétentes.
            </p>
          </div>

          <div className="mt-8 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            <div className="lg:col-span-6 bg-white border border-[#E5E2DC] rounded-lg divide-y divide-[#E5E2DC]">
              {DGECC_ORG_STRUCTURE.map((dir) => (
                <div key={dir.id} className="p-4 sm:p-5">
                  <div className="flex items-baseline justify-between gap-2">
                    <h3 className="font-serif text-base font-semibold text-[#111827]">
                      {dir.domain}
                    </h3>
                    <span className="font-mono text-xs text-[#1E3A8A] shrink-0">
                      {dir.acronym}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-[#4B5563] leading-relaxed">{dir.description}</p>
                  <ul className="mt-2.5 space-y-1 text-xs text-[#374151]">
                    {dir.services.map((srv) => (
                      <li key={srv} className="flex items-start gap-2">
                        <span className="text-[#1E3A8A]" aria-hidden="true">
                          ·
                        </span>
                        <span>{srv}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>

            <div className="lg:col-span-6">
              <div className="bg-white border border-[#E5E2DC] rounded-lg overflow-hidden p-3">
                {orgImgFailed ? (
                  <div className="py-16 text-center text-sm text-[#4B5563]">
                    Organigramme de la Direction Générale des Examens, Concours et de la Certification (DGECC)
                  </div>
                ) : (
                  <img
                    src="/assets/aistudio/Organigramme_DGECC_54c2960f.png"
                    alt="Organigramme officiel de la DGECC présentant la Direction Générale, ses services rattachés et les directions DECEPS, DCP, DEPE et DEB"
                    onError={() => setOrgImgFailed(true)}
                    className="w-full h-auto rounded"
                  />
                )}
              </div>
              <p className="mt-2.5 text-xs text-[#6B7280]">
                Fig. 3 — Organigramme officiel de la DGECC structurant les domaines réglementaires et
                les services d’orientation.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Pied de page institutionnel */}
      <footer className="border-t border-[#E5E2DC] bg-[#F8F7F4]">
        <div className="max-w-[1360px] mx-auto px-4 sm:px-6 py-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-xs text-[#6B7280]">
          <div>
            <span className="font-medium text-[#111827]">
              Direction Générale des Examens, Concours et de la Certification (DGECC)
            </span>
            <span className="mx-2" aria-hidden="true">
              ·
            </span>
            <span>Portail d’orientation des usagers</span>
          </div>
          <div className="flex items-center gap-6">
            <button
              type="button"
              onClick={() => onNavigate('chat')}
              className="hover:text-[#111827] cursor-pointer"
            >
              Assistant
            </button>
            <button
              type="button"
              onClick={() => onNavigate('corpus')}
              className="hover:text-[#111827] cursor-pointer"
            >
              Corpus officiel
            </button>
            <button
              type="button"
              onClick={() => onNavigate('admin')}
              className="hover:text-[#111827] cursor-pointer"
            >
              Espace administration
            </button>
          </div>
        </div>
      </footer>
    </main>
  );
}
