import { getDb } from '../database/db.ts';

const FRENCH_STOPWORDS = new Set([
  'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'en', 'au', 'aux',
  'pour', 'par', 'sur', 'dans', 'avec', 'sans', 'sous', 'vers', 'chez', 'entre',
  'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'notre', 'votre', 'leur',
  'qui', 'que', 'quoi', 'dont', 'ou', 'il', 'elle', 'ils', 'elles', 'nous', 'vous',
  'est', 'sont', 'etre', 'avoir', 'fait', 'faire', 'peut', 'doit', 'plus', 'moins',
  'tout', 'tous', 'toute', 'toutes', 'comme', 'mais', 'donc', 'car', 'si', 'ni',
  'se', 'ne', 'pas', 'qu', 'd', 'l', 'n', 's', 'c', 'j', 'm', 't', 'y', 'a',
]);

export function normalizeFrenchToken(text: string): string {
  return text
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '');
}

export function tokenizeFrench(text: string): string[] {
  const normalized = normalizeFrenchToken(text);
  const rawTokens = normalized.match(/[a-z0-9]{2,}(?:-[a-z0-9]+)*/g) || [];
  return rawTokens.filter((tok) => !FRENCH_STOPWORDS.has(tok));
}

export interface CachedPublishedChunk {
  chunkId: string;
  documentId: string;
  versionNumber: number;
  chunkIndex: number;
  content: string;
  title: string;
  docType: string;
  category: string;
  serviceName: string;
  referenceCode: string;
  referenceNorm: string;
  publishedDate: string;
  validUntil: string | null;
  tf: Map<string, number>;
  tokenLength: number;
  unitVector: Float32Array | null;
}

export interface PublishedCorpusIndex {
  chunks: CachedPublishedChunk[];
  avgDocLen: number;
  docFreq: Map<string, number>;
}

let cachedCorpus: PublishedCorpusIndex | null = null;
const responseCacheListeners = new Set<() => void>();

export function onCorpusInvalidated(cb: () => void): void {
  responseCacheListeners.add(cb);
}

export function invalidateCorpusCache(): void {
  cachedCorpus = null;
  for (const cb of responseCacheListeners) {
    cb();
  }
}

export function toUnitFloat32(vec: number[]): Float32Array {
  const arr = new Float32Array(vec.length);
  let normSq = 0;
  for (let i = 0; i < vec.length; i++) {
    const v = vec[i];
    arr[i] = v;
    normSq += v * v;
  }
  const norm = Math.sqrt(normSq);
  if (norm > 0) {
    for (let i = 0; i < arr.length; i++) {
      arr[i] /= norm;
    }
  }
  return arr;
}

export function dotProductUnit(a: Float32Array, b: Float32Array): number {
  const len = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < len; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function getPublishedCorpusIndex(): PublishedCorpusIndex {
  if (cachedCorpus) return cachedCorpus;

  const db = getDb();
  const rows = db
    .prepare(
      `SELECT
        c.id AS chunkId,
        c.document_id AS documentId,
        c.version_number AS versionNumber,
        c.chunk_index AS chunkIndex,
        c.content AS content,
        c.embedding_json AS embeddingJson,
        d.title AS title,
        d.doc_type AS docType,
        d.category AS category,
        d.service_name AS serviceName,
        d.reference_code AS referenceCode,
        d.published_date AS publishedDate,
        d.valid_until AS validUntil
      FROM document_chunks c
      JOIN documents d ON d.id = c.document_id AND c.version_number = d.current_version
      WHERE d.status = 'PUBLIE'`,
    )
    .all() as Array<{
    chunkId: string;
    documentId: string;
    versionNumber: number;
    chunkIndex: number;
    content: string;
    embeddingJson: string | null;
    title: string;
    docType: string;
    category: string;
    serviceName: string;
    referenceCode: string;
    publishedDate: string;
    validUntil: string | null;
  }>;

  const docFreq = new Map<string, number>();
  let totalLen = 0;

  const chunks: CachedPublishedChunk[] = rows.map((r) => {
    const toks = tokenizeFrench(`${r.title} ${r.referenceCode} ${r.content}`);
    totalLen += toks.length;
    const tf = new Map<string, number>();
    const seen = new Set<string>();
    for (const tok of toks) {
      tf.set(tok, (tf.get(tok) || 0) + 1);
      if (!seen.has(tok)) {
        seen.add(tok);
        docFreq.set(tok, (docFreq.get(tok) || 0) + 1);
      }
    }

    let unitVector: Float32Array | null = null;
    if (r.embeddingJson) {
      try {
        const parsed = JSON.parse(r.embeddingJson) as number[];
        if (Array.isArray(parsed) && parsed.length > 0) {
          unitVector = toUnitFloat32(parsed);
        }
      } catch {
        unitVector = null;
      }
    }

    return {
      chunkId: r.chunkId,
      documentId: r.documentId,
      versionNumber: r.versionNumber,
      chunkIndex: r.chunkIndex,
      content: r.content,
      title: r.title,
      docType: r.docType,
      category: r.category,
      serviceName: r.serviceName,
      referenceCode: r.referenceCode,
      referenceNorm: normalizeFrenchToken(r.referenceCode),
      publishedDate: r.publishedDate,
      validUntil: r.validUntil,
      tf,
      tokenLength: toks.length,
      unitVector,
    };
  });

  cachedCorpus = {
    chunks,
    avgDocLen: totalLen / Math.max(1, chunks.length),
    docFreq,
  };
  return cachedCorpus;
}
