import crypto from 'node:crypto';
import { RAG_CONFIG } from '../core/config.ts';
import { isQuotaError, isTransientGeminiError } from '../core/geminiClient.ts';
import { getDb, logActivity } from '../database/db.ts';
import {
  getEmbeddingProvider,
  isEmbeddingQuotaPaused,
  markEmbeddingQuotaExhausted,
} from '../embeddings/embedder.ts';
import { invalidateCorpusCache } from '../search/corpusCache.ts';
import { tokenizeFrench } from '../search/bm25.ts';
import { chunkDocumentText } from './chunker.ts';
import {
  extractDocumentWithOcr,
  type ExtractionResult,
  type OcrMode,
  type SupportedFormat,
} from './extractor.ts';
import { normalizeText } from './normalizer.ts';

export type DocumentStatus =
  | 'BROUILLON'
  | 'EN_ATTENTE'
  | 'VALIDE'
  | 'PUBLIE'
  | 'ARCHIVE'
  | 'REJETE';

export interface IngestDocumentInput {
  title: string;
  docType: string;
  category: string;
  serviceName: string;
  referenceCode: string;
  publishedDate: string;
  validUntil: string | null;
  fileName: string;
  mimeType?: string;
  format: SupportedFormat;
  buffer: Buffer;
  ocrMode?: OcrMode;
  reviewedText?: string;
  chunkSize?: number;
  chunkOverlap?: number;
  uploadedBy: string;
  uploadedByEmail: string;
  initialStatus?: DocumentStatus;
}

export async function ingestDocument(input: IngestDocumentInput) {
  const db = getDb();

  let extraction: ExtractionResult;
  if (typeof input.reviewedText === 'string' && input.reviewedText.trim().length >= 20) {
    extraction = {
      text: input.reviewedText,
      ocrUsed: input.format === 'IMAGE' || input.ocrMode === 'force' || input.ocrMode === 'auto',
      ocrReason: input.format === 'IMAGE' ? 'image_scan' : 'forced',
      nativeCharCount: input.reviewedText.length,
    };
  } else {
    extraction = await extractDocumentWithOcr(input.buffer, input.format, {
      fileName: input.fileName,
      mimeType: input.mimeType,
      ocrMode: input.ocrMode || 'auto',
    });
  }

  const rawText = extraction.text;
  const normalizedText = normalizeText(rawText);

  if (normalizedText.length < 20) {
    throw new Error(
      'Le contenu extrait du document (même après reconnaissance OCR) est insuffisant ou illisible (moins de 20 caractères exploitables).',
    );
  }

  const docId = crypto.randomUUID();
  const versionId = crypto.randomUUID();
  const now = new Date().toISOString();
  const status: DocumentStatus = input.initialStatus || 'EN_ATTENTE';
  const chunkSize = input.chunkSize || RAG_CONFIG.defaultChunkSize;
  const chunkOverlap = input.chunkOverlap ?? RAG_CONFIG.defaultChunkOverlap;

  db.prepare(
    `INSERT INTO documents (
      id, title, doc_type, category, service_name, reference_code,
      published_date, valid_until, current_version, status,
      file_name, file_format, uploaded_by, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)`,
  ).run(
    docId,
    input.title.trim(),
    input.docType.trim(),
    input.category.trim(),
    input.serviceName.trim(),
    input.referenceCode.trim(),
    input.publishedDate,
    input.validUntil,
    status,
    input.fileName,
    input.format,
    input.uploadedBy,
    now,
    now,
  );

  db.prepare(
    `INSERT INTO document_versions (
      id, document_id, version_number, raw_text, normalized_text,
      chunk_size, chunk_overlap, created_by, created_at
    ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)`,
  ).run(
    versionId,
    docId,
    rawText,
    normalizedText,
    chunkSize,
    chunkOverlap,
    input.uploadedBy,
    now,
  );

  // Chunk immediately so preview/inspection is available in admin, without exposing to RAG until PUBLIE
  const chunks = chunkDocumentText(
    normalizedText,
    {
      documentId: docId,
      title: input.title.trim(),
      docType: input.docType.trim(),
      category: input.category.trim(),
      serviceName: input.serviceName.trim(),
      referenceCode: input.referenceCode.trim(),
      publishedDate: input.publishedDate,
      validUntil: input.validUntil,
      versionNumber: 1,
    },
    chunkSize,
    chunkOverlap,
  );

  const insertChunk = db.prepare(
    `INSERT INTO document_chunks (
      id, document_id, version_number, chunk_index, content,
      token_count, char_count, metadata_json, created_at
    ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)`,
  );

  for (const chunk of chunks) {
    insertChunk.run(
      crypto.randomUUID(),
      docId,
      chunk.chunkIndex,
      chunk.content,
      chunk.tokenCount,
      chunk.charCount,
      JSON.stringify(chunk.metadata),
      now,
    );
  }

  logActivity({
    userId: input.uploadedBy,
    userEmail: input.uploadedByEmail,
    action: extraction.ocrUsed ? 'DOCUMENT_UPLOAD_OCR' : 'DOCUMENT_UPLOAD',
    resourceType: 'document',
    resourceId: docId,
    details: {
      title: input.title,
      format: input.format,
      chunksCreated: chunks.length,
      status,
      ocrUsed: extraction.ocrUsed,
      ocrReason: extraction.ocrReason,
      ocrModel: extraction.ocrModel || null,
      pageCount: extraction.pageCount || null,
    },
  });

  return {
    id: docId,
    title: input.title,
    status,
    version: 1,
    chunksCount: chunks.length,
    normalizedLength: normalizedText.length,
    ocrUsed: extraction.ocrUsed,
    ocrReason: extraction.ocrReason,
    ocrModel: extraction.ocrModel || null,
    pageCount: extraction.pageCount || null,
  };
}

export async function indexDocumentVersion(
  documentId: string,
  triggeredByUserId: string,
  triggeredByEmail: string,
  options?: { chunkSize?: number; chunkOverlap?: number; bumpVersion?: boolean },
) {
  const startMs = Date.now();
  const db = getDb();

  const doc = db
    .prepare(`SELECT * FROM documents WHERE id = ?`)
    .get(documentId) as Record<string, unknown> | undefined;
  if (!doc) {
    throw new Error('Document introuvable.');
  }

  const currentVersionNum = Number(doc.current_version);
  const versionRow = db
    .prepare(`SELECT * FROM document_versions WHERE document_id = ? AND version_number = ?`)
    .get(documentId, currentVersionNum) as Record<string, unknown> | undefined;

  if (!versionRow) {
    throw new Error('Version documentaire introuvable.');
  }

  const chunkSize = options?.chunkSize ?? Number(versionRow.chunk_size);
  const chunkOverlap = options?.chunkOverlap ?? Number(versionRow.chunk_overlap);
  const targetVersion = options?.bumpVersion ? currentVersionNum + 1 : currentVersionNum;
  const now = new Date().toISOString();

  if (options?.bumpVersion) {
    db.prepare(
      `INSERT INTO document_versions (
        id, document_id, version_number, raw_text, normalized_text,
        chunk_size, chunk_overlap, created_by, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      crypto.randomUUID(),
      documentId,
      targetVersion,
      String(versionRow.raw_text),
      String(versionRow.normalized_text),
      chunkSize,
      chunkOverlap,
      triggeredByUserId,
      now,
    );
    db.prepare(`UPDATE documents SET current_version = ?, updated_at = ? WHERE id = ?`).run(
      targetVersion,
      now,
      documentId,
    );
  } else {
    db.prepare(`DELETE FROM document_chunks WHERE document_id = ? AND version_number = ?`).run(
      documentId,
      targetVersion,
    );
    db.prepare(
      `UPDATE document_versions SET chunk_size = ?, chunk_overlap = ? WHERE document_id = ? AND version_number = ?`,
    ).run(chunkSize, chunkOverlap, documentId, targetVersion);
  }

  const chunks = chunkDocumentText(
    String(versionRow.normalized_text),
    {
      documentId,
      title: String(doc.title),
      docType: String(doc.doc_type),
      category: String(doc.category),
      serviceName: String(doc.service_name),
      referenceCode: String(doc.reference_code),
      publishedDate: String(doc.published_date),
      validUntil: doc.valid_until ? String(doc.valid_until) : null,
      versionNumber: targetVersion,
    },
    chunkSize,
    chunkOverlap,
  );

  const provider = getEmbeddingProvider();
  let vectors: Array<number[] | null> = new Array(chunks.length).fill(null);
  if (!isEmbeddingQuotaPaused()) {
    try {
      vectors = await provider.embedTexts(chunks.map((c) => c.content));
    } catch (err) {
      if (isQuotaError(err) || isTransientGeminiError(err)) {
        markEmbeddingQuotaExhausted(err);
      } else {
        throw err;
      }
    }
  }

  const insertChunk = db.prepare(
    `INSERT INTO document_chunks (
      id, document_id, version_number, chunk_index, content,
      token_count, char_count, metadata_json, embedding_json, embedding_model, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  );

  let bm25TermsCount = 0;
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    bm25TermsCount += tokenizeFrench(c.content).length;
    const vec = vectors[i];
    insertChunk.run(
      crypto.randomUUID(),
      documentId,
      targetVersion,
      c.chunkIndex,
      c.content,
      c.tokenCount,
      c.charCount,
      JSON.stringify(c.metadata),
      vec ? JSON.stringify(vec) : null,
      vec ? provider.modelName : null,
      now,
    );
  }

  const durationMs = Date.now() - startMs;
  invalidateCorpusCache();

  db.prepare(
    `INSERT INTO indexation_history (
      id, document_id, version_number, chunk_count, embedding_model,
      bm25_terms_count, duration_ms, status, triggered_by, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'SUCCES', ?, ?)`,
  ).run(
    crypto.randomUUID(),
    documentId,
    targetVersion,
    chunks.length,
    provider.modelName,
    bm25TermsCount,
    durationMs,
    triggeredByUserId,
    now,
  );

  logActivity({
    userId: triggeredByUserId,
    userEmail: triggeredByEmail,
    action: 'DOCUMENT_INDEXED',
    resourceType: 'document',
    resourceId: documentId,
    details: {
      version: targetVersion,
      chunkCount: chunks.length,
      bm25TermsCount,
      embeddingModel: provider.modelName,
      durationMs,
    },
  });

  return {
    documentId,
    versionNumber: targetVersion,
    chunkCount: chunks.length,
    bm25TermsCount,
    embeddingModel: provider.modelName,
    durationMs,
  };
}
