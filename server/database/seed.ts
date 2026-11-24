import crypto from 'node:crypto';
import { getDb } from './db.ts';
import {
  LEGACY_SAMPLE_REFERENCE_CODES,
  loadSeedDocumentsFromCache,
  upsertMissingOfficialDocuments,
} from './officialSources.ts';
import { chunkDocumentText } from '../document_processing/chunker.ts';
import { normalizeText } from '../document_processing/normalizer.ts';
import { invalidateCorpusCache } from '../search/corpusCache.ts';

export function seedInitialDatabase(): void {
  const db = getDb();

  // 1. Seed Roles
  const roleCount = (db.prepare(`SELECT COUNT(*) AS cnt FROM roles`).get() as { cnt: number }).cnt;
  if (roleCount === 0) {
    const insertRole = db.prepare(
      `INSERT INTO roles (code, label, permissions_json) VALUES (?, ?, ?)`,
    );
    insertRole.run(
      'ADMIN',
      'Administrateur Système & Gouvernance',
      JSON.stringify([
        'documents:upload',
        'documents:validate',
        'documents:publish',
        'documents:archive',
        'documents:reindex',
        'users:manage',
        'logs:read',
        'evaluations:run',
        'conversations:audit',
      ]),
    );
    insertRole.run(
      'VALIDATOR',
      'Validateur Habilité (Comité Réglementaire)',
      JSON.stringify([
        'documents:read_all',
        'documents:validate',
        'documents:publish',
        'documents:archive',
        'logs:read',
      ]),
    );
    insertRole.run(
      'EDITOR',
      'Rédacteur Documentaire',
      JSON.stringify(['documents:upload', 'documents:read_all']),
    );
    insertRole.run(
      'USER',
      'Usager / Candidat',
      JSON.stringify(['chat:use', 'documents:read_published']),
    );
  }

  // 2. Purge any legacy mock sample documents
  const placeholders = LEGACY_SAMPLE_REFERENCE_CODES.map(() => '?').join(', ');
  const deletedLegacy = db
    .prepare(`DELETE FROM documents WHERE reference_code IN (${placeholders})`)
    .run(...LEGACY_SAMPLE_REFERENCE_CODES);
  if (deletedLegacy.changes > 0) {
    invalidateCorpusCache();
  }

  // 3. Seed Official Sources additively from cached transcripts if available
  const officialDocs = loadSeedDocumentsFromCache();
  if (officialDocs.length === 0) {
    return;
  }

  upsertMissingOfficialDocuments(officialDocs);

  // 4. Idempotent sync for official documents
  let anyUpdated = false;
  const now = new Date().toISOString();

  for (const spec of officialDocs) {
    const existingDoc = db
      .prepare(
        `SELECT id, title, category, service_name, current_version FROM documents WHERE reference_code = ?`,
      )
      .get(spec.referenceCode) as
      | {
          id: string;
          title: string;
          category: string;
          service_name: string;
          current_version: number;
        }
      | undefined;

    if (!existingDoc) continue;

    const existingVer = db
      .prepare(
        `SELECT id, raw_text FROM document_versions WHERE document_id = ? AND version_number = ?`,
      )
      .get(existingDoc.id, existingDoc.current_version) as
      | { id: string; raw_text: string }
      | undefined;

    const normalized = normalizeText(spec.content);
    const contentChanged = existingVer ? existingVer.raw_text !== spec.content : false;
    const metaChanged =
      existingDoc.category !== spec.category ||
      existingDoc.service_name !== spec.serviceName ||
      existingDoc.title !== spec.title;

    if (!metaChanged && !contentChanged) continue;

    anyUpdated = true;
    db.prepare(
      `UPDATE documents SET title = ?, category = ?, service_name = ?, updated_at = ? WHERE id = ?`,
    ).run(spec.title, spec.category, spec.serviceName, now, existingDoc.id);

    if (contentChanged && existingVer) {
      db.prepare(
        `UPDATE document_versions SET raw_text = ?, normalized_text = ? WHERE id = ?`,
      ).run(spec.content, normalized, existingVer.id);
    }

    db.prepare(
      `DELETE FROM document_chunks WHERE document_id = ? AND version_number = ?`,
    ).run(existingDoc.id, existingDoc.current_version);

    const chunks = chunkDocumentText(
      normalized,
      {
        documentId: existingDoc.id,
        title: spec.title,
        docType: spec.docType,
        category: spec.category,
        serviceName: spec.serviceName,
        referenceCode: spec.referenceCode,
        publishedDate: spec.publishedDate,
        validUntil: spec.validUntil,
        versionNumber: existingDoc.current_version,
      },
      850,
      160,
    );

    const insertChunk = db.prepare(`
      INSERT INTO document_chunks (
        id, document_id, version_number, chunk_index, content,
        token_count, char_count, metadata_json, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    for (const c of chunks) {
      insertChunk.run(
        crypto.randomUUID(),
        existingDoc.id,
        existingDoc.current_version,
        c.chunkIndex,
        c.content,
        c.tokenCount,
        c.charCount,
        JSON.stringify(c.metadata),
        now,
      );
    }
  }

  if (anyUpdated) {
    invalidateCorpusCache();
  }
}
