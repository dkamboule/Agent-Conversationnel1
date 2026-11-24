import fs from 'node:fs';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { APP_CONFIG } from '../core/config.ts';

let dbInstance: DatabaseSync | null = null;

function createSchema(db: DatabaseSync): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS roles (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      code TEXT UNIQUE NOT NULL,
      label TEXT NOT NULL,
      permissions_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      email TEXT UNIQUE NOT NULL,
      full_name TEXT NOT NULL,
      password_hash TEXT NOT NULL,
      salt TEXT NOT NULL,
      role_code TEXT NOT NULL REFERENCES roles(code),
      service TEXT,
      is_active INTEGER NOT NULL DEFAULT 1,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS activity_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT,
      user_email TEXT,
      action TEXT NOT NULL,
      resource_type TEXT NOT NULL,
      resource_id TEXT,
      details_json TEXT NOT NULL DEFAULT '{}',
      ip_address TEXT,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS documents (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      doc_type TEXT NOT NULL,
      category TEXT NOT NULL,
      service_name TEXT NOT NULL,
      reference_code TEXT NOT NULL,
      published_date TEXT NOT NULL,
      valid_until TEXT,
      current_version INTEGER NOT NULL DEFAULT 1,
      status TEXT NOT NULL DEFAULT 'BROUILLON',
      file_name TEXT NOT NULL,
      file_format TEXT NOT NULL,
      uploaded_by TEXT,
      validated_by TEXT,
      rejection_reason TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS document_versions (
      id TEXT PRIMARY KEY,
      document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      version_number INTEGER NOT NULL,
      raw_text TEXT NOT NULL,
      normalized_text TEXT NOT NULL,
      chunk_size INTEGER NOT NULL DEFAULT 850,
      chunk_overlap INTEGER NOT NULL DEFAULT 160,
      created_by TEXT,
      created_at TEXT NOT NULL,
      UNIQUE (document_id, version_number)
    );

    CREATE TABLE IF NOT EXISTS document_chunks (
      id TEXT PRIMARY KEY,
      document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      version_number INTEGER NOT NULL,
      chunk_index INTEGER NOT NULL,
      content TEXT NOT NULL,
      token_count INTEGER NOT NULL,
      char_count INTEGER NOT NULL,
      metadata_json TEXT NOT NULL DEFAULT '{}',
      embedding_json TEXT,
      embedding_model TEXT,
      created_at TEXT NOT NULL,
      UNIQUE (document_id, version_number, chunk_index)
    );

    CREATE TABLE IF NOT EXISTS indexation_history (
      id TEXT PRIMARY KEY,
      document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      version_number INTEGER NOT NULL,
      chunk_count INTEGER NOT NULL,
      embedding_model TEXT NOT NULL,
      bm25_terms_count INTEGER NOT NULL,
      duration_ms INTEGER NOT NULL,
      status TEXT NOT NULL,
      triggered_by TEXT,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS conversations (
      id TEXT PRIMARY KEY,
      user_id TEXT,
      guest_session_id TEXT,
      title TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS messages (
      id TEXT PRIMARY KEY,
      conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      nlp_analysis_json TEXT,
      sources_json TEXT,
      guardrails_json TEXT,
      metrics_json TEXT,
      feedback TEXT,
      feedback_comment TEXT,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS evaluation_runs (
      id TEXT PRIMARY KEY,
      run_name TEXT NOT NULL,
      corpus_size INTEGER NOT NULL,
      recall_at_k REAL NOT NULL,
      precision_at_k REAL NOT NULL,
      mrr REAL NOT NULL,
      ndcg_at_k REAL NOT NULL,
      faithfulness REAL NOT NULL,
      answer_relevancy REAL NOT NULL,
      context_precision REAL NOT NULL,
      context_recall REAL NOT NULL,
      avg_latency_ms REAL NOT NULL,
      details_json TEXT NOT NULL,
      methodology_json TEXT,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS user_panel_evaluations (
      id TEXT PRIMARY KEY,
      profile TEXT NOT NULL CHECK (profile IN ('CANDIDAT', 'PARENT', 'AGENT_ADMINISTRATIF')),
      participant_name TEXT NOT NULL,
      clarity_score INTEGER NOT NULL CHECK (clarity_score BETWEEN 1 AND 5),
      usefulness_score INTEGER NOT NULL CHECK (usefulness_score BETWEEN 1 AND 5),
      trust_score INTEGER NOT NULL CHECK (trust_score BETWEEN 1 AND 5),
      comment TEXT,
      recorded_by TEXT,
      created_at TEXT NOT NULL
    );
  `);

  const evalCols = db.prepare(`PRAGMA table_info(evaluation_runs)`).all() as Array<{ name: string }>;
  if (!evalCols.some((c) => c.name === 'methodology_json')) {
    db.exec(`ALTER TABLE evaluation_runs ADD COLUMN methodology_json TEXT;`);
  }
}

function configurePragmas(db: DatabaseSync): void {
  db.exec('PRAGMA busy_timeout = 5000;');
  db.exec('PRAGMA journal_mode = WAL;');
  db.exec('PRAGMA synchronous = NORMAL;');
  db.exec('PRAGMA foreign_keys = ON;');
}

const SALVAGE_TABLES = [
  'roles',
  'users',
  'activity_logs',
  'conversations',
  'messages',
  'evaluation_runs',
  'user_panel_evaluations',
] as const;

function isDatabaseHealthy(db: DatabaseSync): boolean {
  try {
    const res = db.prepare('PRAGMA quick_check(1)').get() as { quick_check?: string } | undefined;
    return res?.quick_check === 'ok';
  } catch {
    return false;
  }
}

function salvageIntactTables(db: DatabaseSync): Record<string, Array<Record<string, unknown>>> {
  const dump: Record<string, Array<Record<string, unknown>>> = {};
  for (const table of SALVAGE_TABLES) {
    try {
      dump[table] = db.prepare(`SELECT * FROM ${table}`).all() as Array<Record<string, unknown>>;
    } catch {
      dump[table] = [];
    }
  }
  return dump;
}

function restoreSalvagedTables(
  db: DatabaseSync,
  dump: Record<string, Array<Record<string, unknown>>>,
): void {
  db.exec('BEGIN TRANSACTION;');
  try {
    db.exec('PRAGMA foreign_keys = OFF;');
    for (const table of SALVAGE_TABLES) {
      const rows = dump[table] || [];
      if (rows.length === 0) continue;
      const validCols = (
        db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>
      ).map((c) => c.name);
      const cols = Object.keys(rows[0]).filter((col) => validCols.includes(col));
      if (cols.length === 0) continue;
      const placeholders = cols.map(() => '?').join(', ');
      const stmt = db.prepare(
        `INSERT OR IGNORE INTO ${table} (${cols.join(', ')}) VALUES (${placeholders})`,
      );
      for (const row of rows) {
        stmt.run(...(cols.map((c) => row[c] ?? null) as Array<string | number | null>));
      }
    }
    db.exec('COMMIT;');
  } catch (err) {
    db.exec('ROLLBACK;');
    throw err;
  } finally {
    db.exec('PRAGMA foreign_keys = ON;');
  }
}

export function getDb(): DatabaseSync {
  if (dbInstance) return dbInstance;

  const dir = path.dirname(APP_CONFIG.dbPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  let db = new DatabaseSync(APP_CONFIG.dbPath);

  if (!isDatabaseHealthy(db)) {
    console.warn('[DGECC] Base SQLite corrompue détectée au démarrage — reconstruction sécurisée en cours...');
    const salvaged = salvageIntactTables(db);
    try {
      db.close();
    } catch {
      // Ignore close error on malformed handle
    }

    const backupPath = `${APP_CONFIG.dbPath}.corrupt-${Date.now()}`;
    if (fs.existsSync(APP_CONFIG.dbPath)) {
      fs.renameSync(APP_CONFIG.dbPath, backupPath);
    }
    for (const suffix of ['-wal', '-shm']) {
      const sidecar = `${APP_CONFIG.dbPath}${suffix}`;
      if (fs.existsSync(sidecar)) {
        fs.unlinkSync(sidecar);
      }
    }

    db = new DatabaseSync(APP_CONFIG.dbPath);
    configurePragmas(db);
    createSchema(db);
    restoreSalvagedTables(db, salvaged);
    console.info(`[DGECC] Base SQLite reconstruite (archive corrompue : ${path.basename(backupPath)}).`);
  } else {
    configurePragmas(db);
    createSchema(db);
  }

  dbInstance = db;
  return dbInstance;
}

export function logActivity(params: {
  userId?: string | null;
  userEmail?: string | null;
  action: string;
  resourceType: string;
  resourceId?: string | null;
  details?: Record<string, unknown>;
  ipAddress?: string | null;
}): void {
  const db = getDb();
  const stmt = db.prepare(`
    INSERT INTO activity_logs (user_id, user_email, action, resource_type, resource_id, details_json, ip_address, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `);
  stmt.run(
    params.userId ?? null,
    params.userEmail ?? null,
    params.action,
    params.resourceType,
    params.resourceId ?? null,
    JSON.stringify(params.details ?? {}),
    params.ipAddress ?? null,
    new Date().toISOString(),
  );
}
