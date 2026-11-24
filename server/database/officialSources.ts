import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { getDb } from './db.ts';
import { chunkDocumentText } from '../document_processing/chunker.ts';
import { extractDocumentWithOcr } from '../document_processing/extractor.ts';
import { normalizeText } from '../document_processing/normalizer.ts';
import { ensurePublishedChunksEmbedded } from '../embeddings/embedder.ts';
import { tokenizeFrench } from '../search/bm25.ts';
import { getPublishedCorpusIndex, invalidateCorpusCache } from '../search/corpusCache.ts';

export interface OfficialSourceSpec {
  title: string;
  docType: string;
  category: string;
  serviceName: string;
  referenceCode: string;
  publishedDate: string;
  validUntil: string | null;
  status: 'PUBLIE' | 'EN_ATTENTE' | 'BROUILLON';
  fileName: string;
  pdfRelativePath: string;
  cacheRelativePath: string;
}

export interface HydratedSeedDoc extends OfficialSourceSpec {
  content: string;
}

export const LEGACY_SAMPLE_REFERENCE_CODES = [
  'DGECC-2026-ARR-01',
  'DGECC-2026-CONC-04',
  'DGECC-2026-PROC-02',
  'DGECC-2026-CERT-07',
  'DGECC-2026-REC-03',
  'DGECC-2026-INCL-05',
  'DGECC-2027-PROJET-09',
  'DGECC-2026-INT-11',
];

export const OFFICIAL_SOURCE_SPECS: OfficialSourceSpec[] = [
  {
    title:
      "Décret n°2025-1558/PF/PRIM/MFPTPS/MEF portant statut particulier du métier « Éducation, formation et promotion de l'emploi »",
    docType: 'Décret réglementaire',
    category: 'Direction Générale & Services Rattachés',
    serviceName: 'Direction Générale — Service de Gestion des Ressources Humaines (SGRH)',
    referenceCode: 'DECRET-2025-1558-PF-PRIM-MFPTPS-MEF',
    publishedDate: '2025-12-05',
    validUntil: null,
    status: 'PUBLIE',
    fileName:
      'DECRET_2025-1558_PF_PRIM_MFPTPS_MEF_portant_statut_particulier_du_metier_Education__formation_et_promotion_de_l_emploi_cee9d4a3.pdf',
    pdfRelativePath:
      'public/assets/aistudio/DECRET_2025-1558_PF_PRIM_MFPTPS_MEF_portant_statut_particulier_du_metier_Education__formation_et_promotion_de_l_emploi_cee9d4a3.pdf',
    cacheRelativePath: 'server/database/seed_sources/decret_2025_1558.txt',
  },
  {
    title:
      "Manuel de procédures de l'organisation des examens et concours scolaires (BEPC, CAP, BEP, Concours d'entrée en 2nde et en 1ère année BEP)",
    docType: 'Manuel de procédures',
    category: 'Examens & Concours Post-primaire et Secondaire (DECEPS)',
    serviceName: 'DECEPS — Secrétariat',
    referenceCode: 'OCECOS-2014-MAN-BEPC-BEP-CAP',
    publishedDate: '2014-04-08',
    validUntil: null,
    status: 'PUBLIE',
    fileName: 'Manuel_de_procedure__bepc_bep_cap__2014__1__f4786aa0.pdf',
    pdfRelativePath:
      'public/assets/aistudio/Manuel_de_procedure__bepc_bep_cap__2014__1__f4786aa0.pdf',
    cacheRelativePath: 'server/database/seed_sources/manuel_procedures_2014.txt',
  },
  {
    title:
      "Arrêté n°2021-MENAPLN/SG/DGEC portant création, attributions et composition des membres des jurys d'organisation des examens professionnels du CAPES et du CAPET, du CAP-CEG et du CAET",
    docType: 'Arrêté ministériel',
    category: 'Examens & Concours Post-primaire et Secondaire (DECEPS)',
    serviceName: 'DECEPS — Service des examens professionnels',
    referenceCode: 'MENAPLN-DGEC-ARR-JURYS-CAPES-CAPET-CAPCEG-CAET',
    publishedDate: '2014-11-30',
    validUntil: null,
    status: 'PUBLIE',
    fileName: 'ARRETE_PORTANT_CREATION_398fe05c.pdf',
    pdfRelativePath: 'public/assets/aistudio/ARRETE_PORTANT_CREATION_398fe05c.pdf',
    cacheRelativePath: 'server/database/seed_sources/arrete_creation_jurys_examens_pro.txt',
  },
  {
    title:
      "Arrêté n°2013-334/MESS/SG/OCECOS/DECSP portant organisation des examens professionnels du CAP-CEG et du CAET",
    docType: 'Arrêté ministériel',
    category: 'Examens & Concours Post-primaire et Secondaire (DECEPS)',
    serviceName: 'DECEPS — Service des examens professionnels',
    referenceCode: 'MESS-2013-334-ARR-CAPCEG-CAET',
    publishedDate: '2013-08-28',
    validUntil: null,
    status: 'PUBLIE',
    fileName: 'ARRETE_334_4e339437.pdf',
    pdfRelativePath: 'public/assets/aistudio/ARRETE_334_4e339437.pdf',
    cacheRelativePath: 'server/database/seed_sources/arrete_334_capceg_caet_2013.txt',
  },
  {
    title:
      "Arrêté n°2025-2877/MFPTPS/SG/AGRE portant ouverture d'examens professionnels du CAP-CEG, du CAET, du CAPES et du CAPET — session de 2026",
    docType: 'Arrêté ministériel',
    category: 'Examens & Concours Post-primaire et Secondaire (DECEPS)',
    serviceName: 'DECEPS — Service des examens professionnels',
    referenceCode: 'MFPTPS-2025-2877-ARR-CAPES-CAPET-CAPCEG-CAET-2026',
    publishedDate: '2025-11-26',
    validUntil: '2026-12-31',
    status: 'PUBLIE',
    fileName:
      'Arret__portant_ouverture_examens_professionnels_CAPES-CAPET__CAPCEG-CAET_2026_e18c42dd.pdf',
    pdfRelativePath:
      'public/assets/aistudio/Arret__portant_ouverture_examens_professionnels_CAPES-CAPET__CAPCEG-CAET_2026_e18c42dd.pdf',
    cacheRelativePath: 'server/database/seed_sources/arrete_2025_2877_ouverture_2026.txt',
  },
  {
    title:
      "Décret n°2021-1123/PRES/PM/MINEFID/MENAPLN/MESRSI portant organisation de l'examen du baccalauréat",
    docType: 'Décret réglementaire',
    category: 'Examens du Baccalauréat (DEB)',
    serviceName: 'DEB — Secrétariat',
    referenceCode: 'PRES-2021-1123-DEC-BACCALAUREAT',
    publishedDate: '2021-11-04',
    validUntil: null,
    status: 'PUBLIE',
    fileName: 'D_cret_portant_organisation_du_Baccalaur_at_a46892f8.pdf',
    pdfRelativePath:
      'public/assets/aistudio/D_cret_portant_organisation_du_Baccalaur_at_a46892f8.pdf',
    cacheRelativePath: 'server/database/seed_sources/decret_2021_1123_baccalaureat.txt',
  },
  {
    title:
      "Décret n°2021-0187/PRES/PM/MATD/MINEFID/MENAPLN portant organisation de l'examen du Brevet d'Études Professionnelles (BEP)",
    docType: 'Décret réglementaire',
    category: 'Examens & Concours Post-primaire et Secondaire (DECEPS)',
    serviceName: "DECEPS — Service des examens de l'enseignement technique professionnel",
    referenceCode: 'PRES-2021-0187-DEC-BEP',
    publishedDate: '2021-03-31',
    validUntil: null,
    status: 'PUBLIE',
    fileName: 'D_cret_BEP_2021_9896aa21.pdf',
    pdfRelativePath: 'public/assets/aistudio/D_cret_BEP_2021_9896aa21.pdf',
    cacheRelativePath: 'server/database/seed_sources/decret_2021_0187_bep.txt',
  },
  {
    title:
      "Décret n°2021-0213/PRES/PM/MATD/MINEFID/MENAPLN portant organisation de l'examen du Brevet d'Études du Premier Cycle (BEPC)",
    docType: 'Décret réglementaire',
    category: 'Examens & Concours Post-primaire et Secondaire (DECEPS)',
    serviceName: "DECEPS — Service des examens de l'enseignement post-primaire général",
    referenceCode: 'PRES-2021-0213-DEC-BEPC',
    publishedDate: '2021-04-06',
    validUntil: null,
    status: 'PUBLIE',
    fileName: 'Decret_BEPC_2021_30a2d232.pdf',
    pdfRelativePath: 'public/assets/aistudio/Decret_BEPC_2021_30a2d232.pdf',
    cacheRelativePath: 'server/database/seed_sources/decret_2021_0213_bepc.txt',
  },
  {
    title:
      "Décret n°2021-0256/PRES/PM/MATD/MINEFID/MENAPLN portant organisation de l'examen du Certificat d'Aptitude Professionnelle (CAP)",
    docType: 'Décret réglementaire',
    category: 'Examens & Concours Post-primaire et Secondaire (DECEPS)',
    serviceName: "DECEPS — Service des examens de l'enseignement technique professionnel",
    referenceCode: 'PRES-2021-0256-DEC-CAP',
    publishedDate: '2021-04-16',
    validUntil: null,
    status: 'PUBLIE',
    fileName: 'D_cret_CAP_2021_6b7013e2.pdf',
    pdfRelativePath: 'public/assets/aistudio/D_cret_CAP_2021_6b7013e2.pdf',
    cacheRelativePath: 'server/database/seed_sources/decret_2021_0256_cap.txt',
  },
  {
    title:
      'Décret n°2021-0362/PRES/PM/MATD/MINEFID/MENAPLN portant institution et organisation de concours scolaires',
    docType: 'Décret réglementaire',
    category: 'Examens & Concours Post-primaire et Secondaire (DECEPS)',
    serviceName: 'DECEPS — Secrétariat',
    referenceCode: 'PRES-2021-0362-DEC-CONCOURS-SCOLAIRES',
    publishedDate: '2021-05-03',
    validUntil: null,
    status: 'PUBLIE',
    fileName: 'd_cret_concours_scolaires_2021_7091696e.pdf',
    pdfRelativePath: 'public/assets/aistudio/d_cret_concours_scolaires_2021_7091696e.pdf',
    cacheRelativePath: 'server/database/seed_sources/decret_2021_0362_concours_scolaires.txt',
  },
];

export function loadSeedDocumentsFromCache(): HydratedSeedDoc[] {
  const hydrated: HydratedSeedDoc[] = [];
  for (const spec of OFFICIAL_SOURCE_SPECS) {
    const absCachePath = path.resolve(process.cwd(), spec.cacheRelativePath);
    if (!fs.existsSync(absCachePath)) continue;
    const text = fs.readFileSync(absCachePath, 'utf8').trim();
    if (text.length < 1000) continue;
    hydrated.push({
      ...spec,
      content: text,
    });
  }
  return hydrated;
}

function insertSingleOfficialDocument(
  db: ReturnType<typeof getDb>,
  spec: HydratedSeedDoc,
  now: string,
): void {
  const insertDoc = db.prepare(`
    INSERT INTO documents (
      id, title, doc_type, category, service_name, reference_code,
      published_date, valid_until, current_version, status,
      file_name, file_format, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
  `);
  const insertVer = db.prepare(`
    INSERT INTO document_versions (
      id, document_id, version_number, raw_text, normalized_text,
      chunk_size, chunk_overlap, created_at
    ) VALUES (?, ?, 1, ?, ?, 850, 160, ?)
  `);
  const insertChunk = db.prepare(`
    INSERT INTO document_chunks (
      id, document_id, version_number, chunk_index, content,
      token_count, char_count, metadata_json, created_at
    ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)
  `);
  const insertIdx = db.prepare(`
    INSERT INTO indexation_history (
      id, document_id, version_number, chunk_count, embedding_model,
      bm25_terms_count, duration_ms, status, created_at
    ) VALUES (?, ?, 1, ?, ?, ?, ?, 'SUCCES', ?)
  `);

  const docId = crypto.randomUUID();
  const format = spec.fileName.split('.').pop()?.toUpperCase() || 'PDF';
  const normalized = normalizeText(spec.content);

  insertDoc.run(
    docId,
    spec.title,
    spec.docType,
    spec.category,
    spec.serviceName,
    spec.referenceCode,
    spec.publishedDate,
    spec.validUntil,
    spec.status,
    spec.fileName,
    format,
    now,
    now,
  );

  insertVer.run(crypto.randomUUID(), docId, spec.content, normalized, now);

  const chunks = chunkDocumentText(
    normalized,
    {
      documentId: docId,
      title: spec.title,
      docType: spec.docType,
      category: spec.category,
      serviceName: spec.serviceName,
      referenceCode: spec.referenceCode,
      publishedDate: spec.publishedDate,
      validUntil: spec.validUntil,
      versionNumber: 1,
    },
    850,
    160,
  );

  let termsCount = 0;
  for (const c of chunks) {
    termsCount += tokenizeFrench(c.content).length;
    insertChunk.run(
      crypto.randomUUID(),
      docId,
      c.chunkIndex,
      c.content,
      c.tokenCount,
      c.charCount,
      JSON.stringify(c.metadata),
      now,
    );
  }

  if (spec.status === 'PUBLIE') {
    insertIdx.run(
      crypto.randomUUID(),
      docId,
      chunks.length,
      'gemini-embedding-2',
      termsCount,
      0,
      now,
    );
  }
}

export function upsertMissingOfficialDocuments(docs: HydratedSeedDoc[]): number {
  if (docs.length === 0) return 0;
  const db = getDb();
  const now = new Date().toISOString();

  const existingRefs = new Set(
    (db.prepare(`SELECT reference_code FROM documents`).all() as Array<{ reference_code: string }>).map(
      (r) => r.reference_code,
    ),
  );

  const missing = docs.filter((doc) => !existingRefs.has(doc.referenceCode));
  if (missing.length === 0) return 0;

  db.exec('BEGIN TRANSACTION;');
  try {
    for (const spec of missing) {
      insertSingleOfficialDocument(db, spec, now);
    }
    db.exec('COMMIT;');
  } catch (err) {
    db.exec('ROLLBACK;');
    throw err;
  }

  invalidateCorpusCache();
  return missing.length;
}

export function replaceCorpusWithOfficialDocuments(docs: HydratedSeedDoc[]): void {
  if (docs.length === 0) return;
  const db = getDb();
  const now = new Date().toISOString();

  db.exec('BEGIN TRANSACTION;');
  try {
    // Remplacer toute source documentaire existante (conserve users, roles, activity_logs)
    db.prepare(`DELETE FROM documents`).run();

    for (const spec of docs) {
      insertSingleOfficialDocument(db, spec, now);
    }

    db.exec('COMMIT;');
  } catch (err) {
    db.exec('ROLLBACK;');
    throw err;
  }

  invalidateCorpusCache();
}

export async function ensureOfficialSourcesExtractedAndSeeded(): Promise<void> {
  const statusLogPath = path.resolve(
    process.cwd(),
    'server/database/seed_sources/_bootstrap_status.log',
  );
  const logStep = (msg: string) => {
    try {
      fs.mkdirSync(path.dirname(statusLogPath), { recursive: true });
      fs.appendFileSync(statusLogPath, `[${new Date().toISOString()}] ${msg}\n`, 'utf8');
    } catch {
      // Ignore logging errors
    }
  };

  try {
    const db = getDb();
    const alreadyInDbStmt = db.prepare(`
      SELECT 1
      FROM documents d
      JOIN document_versions dv ON dv.document_id = d.id AND dv.version_number = d.current_version
      WHERE d.reference_code = ? AND LENGTH(dv.normalized_text) >= 1000
      LIMIT 1
    `);

    for (const spec of OFFICIAL_SOURCE_SPECS) {
      if (alreadyInDbStmt.get(spec.referenceCode)) {
        continue;
      }

      const absCachePath = path.resolve(process.cwd(), spec.cacheRelativePath);
      if (fs.existsSync(absCachePath) && fs.readFileSync(absCachePath, 'utf8').trim().length >= 1000) {
        continue;
      }

      const absPdfPath = path.resolve(process.cwd(), spec.pdfRelativePath);
      if (!fs.existsSync(absPdfPath)) {
        logStep(`WARN: Source PDF introuvable: ${spec.pdfRelativePath}`);
        continue;
      }

      logStep(`START OCR: ${spec.referenceCode}`);
      const pdfBuffer = fs.readFileSync(absPdfPath);
      const extraction = await extractDocumentWithOcr(pdfBuffer, 'PDF', {
        fileName: spec.fileName,
        mimeType: 'application/pdf',
        ocrMode: 'auto',
        onProgress: (msg) => logStep(`${spec.referenceCode}: ${msg}`),
        checkpointPath: `${absCachePath}.checkpoint.json`,
      });
      const normalized = normalizeText(extraction.text);
      if (normalized.length < 1000) {
        throw new Error(
          `Extraction OCR insuffisante pour ${spec.referenceCode} (${normalized.length} car.)`,
        );
      }

      fs.mkdirSync(path.dirname(absCachePath), { recursive: true });
      fs.writeFileSync(absCachePath, normalized, 'utf8');
      logStep(
        `DONE OCR: ${spec.referenceCode} (${normalized.length} chars, model=${extraction.ocrModel ?? 'n/a'})`,
      );
    }

    const cachedDocs = loadSeedDocumentsFromCache();
    if (cachedDocs.length > 0) {
      const db = getDb();
      const placeholders = LEGACY_SAMPLE_REFERENCE_CODES.map(() => '?').join(', ');
      const deletedLegacy = db
        .prepare(`DELETE FROM documents WHERE reference_code IN (${placeholders})`)
        .run(...LEGACY_SAMPLE_REFERENCE_CODES);
      if (deletedLegacy.changes > 0) {
        invalidateCorpusCache();
      }

      const added = upsertMissingOfficialDocuments(cachedDocs);
      if (added > 0) {
        logStep(`ADDED ${added} official doc(s) to corpus.`);
      }
    }

    await ensurePublishedChunksEmbedded();
    getPublishedCorpusIndex();
    logStep('DONE CORPUS INDEX WARMED');

    if (fs.existsSync(statusLogPath)) {
      fs.unlinkSync(statusLogPath);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logStep(`WARN: ${msg}`);
    console.warn(`[DGECC] Initialisation corpus officiel partielle : ${msg}`);
  }
}
