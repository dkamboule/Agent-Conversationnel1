import crypto from 'node:crypto';
import path from 'node:path';

export interface RagConfig {
  embeddingModel: string;
  llmModel: string;
  evalModel: string;
  defaultChunkSize: number;
  defaultChunkOverlap: number;
  defaultTopK: number;
  defaultRrfK: number;
  defaultSimilarityThreshold: number;
  bm25Weight: number;
  vectorWeight: number;
  maxHistoryMessages: number;
}

const EPHEMERAL_SESSION_SECRET = crypto.randomBytes(32).toString('hex');

const resolvedPort =
  process.env.NGINX_PORT && process.env.PORT === process.env.NGINX_PORT
    ? Number(process.env.DEFAULT_APP_PORT) || 3000
    : Number(process.env.PORT) || 3000;

export const APP_CONFIG = {
  port: resolvedPort,
  dbPath: path.resolve(process.cwd(), 'data', 'dgecc.db'),
  sessionSecret: process.env.SESSION_SECRET || EPHEMERAL_SESSION_SECRET,
  adminPasscodeOverride: process.env.DGECC_ADMIN_PASSCODE || '',
  tokenTtlSeconds: 60 * 60 * 12, // 12 heures
};

export const RAG_CONFIG: RagConfig = {
  embeddingModel: 'gemini-embedding-2',
  llmModel: 'gemini-3.1-flash-lite',
  evalModel: 'gemini-4-hydrogen',
  defaultChunkSize: 850,
  defaultChunkOverlap: 160,
  defaultTopK: 5,
  defaultRrfK: 60,
  defaultSimilarityThreshold: 0.25,
  bm25Weight: 1.0,
  vectorWeight: 1.0,
  maxHistoryMessages: 6,
};
