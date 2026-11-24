import crypto from 'node:crypto';
import express, { type Router } from 'express';
import multer from 'multer';
import { APP_CONFIG, RAG_CONFIG } from '../core/config.ts';
import { formatGeminiUserError } from '../core/geminiClient.ts';
import {
  hashPassword,
  rateLimit,
  signToken,
  verifyPassword,
  type RoleCode,
} from '../core/security.ts';
import { getDb, logActivity } from '../database/db.ts';
import {
  detectFormat,
  extractDocumentWithOcr,
  type OcrMode,
} from '../document_processing/extractor.ts';
import { normalizeText } from '../document_processing/normalizer.ts';
import {
  indexDocumentVersion,
  ingestDocument,
  type DocumentStatus,
} from '../document_processing/pipeline.ts';
import {
  addUserPanelEvaluation,
  deleteUserPanelEvaluation,
  getEvaluationOverview,
  listUserPanelEvaluations,
  runRagBenchmarkEvaluation,
  type UserPanelProfile,
} from '../rag/evaluator.ts';
import { runHybridRagPipeline } from '../rag/orchestrator.ts';
import { invalidateCorpusCache } from '../search/corpusCache.ts';
import {
  attachAuthUser,
  requireAuth,
  requireRole,
  resolveOwner,
  type AuthenticatedRequest,
} from './middleware.ts';

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 15 * 1024 * 1024 },
});

export function createApiRouter(): Router {
  const router = express.Router();
  router.use(attachAuthUser);

  // --------------------------------------------------------------------------
  // OpenAPI 3.0 Documentation
  // --------------------------------------------------------------------------
  router.get('/openapi.json', (_req, res) => {
    res.json({
      openapi: '3.0.3',
      info: {
        title: 'API DGECC — Agent Conversationnel RAG Hybride',
        version: '1.0.0',
        description:
          'API REST officielle du système RAG hybride (BM25 + embeddings vectoriels + RRF + garde-fous) de la DGECC.',
      },
      paths: {
        '/api/v1/auth/login': { post: { summary: 'Authentification utilisateur / administrateur' } },
        '/api/v1/auth/logout': { post: { summary: 'Clôture de session' } },
        '/api/v1/auth/me': { get: { summary: 'Profil utilisateur authentifié' } },
        '/api/v1/chat/message': { post: { summary: 'Envoi d’une question au moteur RAG hybride' } },
        '/api/v1/chat/stream': { post: { summary: 'Streaming SSE progressif du moteur RAG hybride' } },
        '/api/v1/chat/conversations': { get: { summary: 'Historique des conversations isolées' } },
        '/api/v1/chat/conversations/{id}': {
          get: { summary: 'Détail d’une conversation' },
          delete: { summary: 'Suppression d’une conversation' },
        },
        '/api/v1/documents/upload': {
          post: { summary: 'Importation PDF/Image scannée (OCR)/DOCX/HTML/TXT et chunking' },
        },
        '/api/v1/documents/ocr-preview': {
          post: { summary: 'Prévisualisation OCR de documents scannés (PDF ou image)' },
        },
        '/api/v1/documents': { get: { summary: 'Consultation du corpus documentaire' } },
        '/api/v1/documents/{id}/validate': { put: { summary: 'Transition de statut de validation' } },
        '/api/v1/documents/{id}/publish': { post: { summary: 'Publication et indexation BM25+vecteur' } },
        '/api/v1/documents/{id}/reindex': { post: { summary: 'Ré-indexation paramétrable' } },
        '/api/v1/admin/statistics': { get: { summary: 'Statistiques globales & performances' } },
        '/api/v1/admin/evaluate': { post: { summary: 'Exécution du benchmark qualité RAG' } },
      },
    });
  });

  // --------------------------------------------------------------------------
  // Authentification & Bootstrap Administrateur
  // --------------------------------------------------------------------------
  router.get('/auth/bootstrap-status', (_req, res) => {
    const db = getDb();
    const adminRow = db
      .prepare(`SELECT COUNT(*) AS cnt FROM users WHERE role_code = 'ADMIN' AND is_active = 1`)
      .get() as { cnt: number };
    res.json({
      needsAdminSetup: adminRow.cnt === 0 && !APP_CONFIG.adminPasscodeOverride,
      hasPasscodeOverride: Boolean(APP_CONFIG.adminPasscodeOverride),
    });
  });

  router.post('/auth/bootstrap-admin', rateLimit(10, 60_000), (req, res) => {
    const db = getDb();
    const existingAdmin = db
      .prepare(`SELECT COUNT(*) AS cnt FROM users WHERE role_code = 'ADMIN'`)
      .get() as { cnt: number };

    if (existingAdmin.cnt > 0 || APP_CONFIG.adminPasscodeOverride) {
      res.status(403).json({
        error: 'Le compte administrateur initial a déjà été configuré.',
      });
      return;
    }

    const { email, fullName, password, service } = req.body || {};
    if (
      typeof email !== 'string' ||
      !email.includes('@') ||
      typeof password !== 'string' ||
      password.length < 8
    ) {
      res.status(400).json({
        error: 'Veuillez saisir un courriel valide et un mot de passe d’au moins 8 caractères.',
      });
      return;
    }

    const cleanEmail = email.trim().toLowerCase();
    const cleanName = typeof fullName === 'string' && fullName.trim() ? fullName.trim() : 'Administrateur DGECC';
    const cleanService = typeof service === 'string' && service.trim() ? service.trim() : 'Direction Générale DGECC';

    const { hash, salt } = hashPassword(password);
    const userId = crypto.randomUUID();
    const now = new Date().toISOString();

    db.prepare(
      `INSERT INTO users (id, email, full_name, password_hash, salt, role_code, service, is_active, created_at)
       VALUES (?, ?, ?, ?, ?, 'ADMIN', ?, 1, ?)`,
    ).run(userId, cleanEmail, cleanName, hash, salt, cleanService, now);

    const token = signToken({
      sub: userId,
      email: cleanEmail,
      fullName: cleanName,
      role: 'ADMIN',
    });

    logActivity({
      userId,
      userEmail: cleanEmail,
      action: 'ADMIN_BOOTSTRAPPED',
      resourceType: 'user',
      resourceId: userId,
      ipAddress: req.ip,
    });

    res.status(201).json({
      token,
      user: {
        id: userId,
        email: cleanEmail,
        fullName: cleanName,
        role: 'ADMIN',
        service: cleanService,
      },
    });
  });

  router.post('/auth/register', rateLimit(15, 60_000), (req, res) => {
    const db = getDb();
    const { email, fullName, password } = req.body || {};
    if (
      typeof email !== 'string' ||
      !email.includes('@') ||
      typeof password !== 'string' ||
      password.length < 6
    ) {
      res.status(400).json({
        error: 'Courriel valide et mot de passe (6 caractères minimum) requis.',
      });
      return;
    }

    const cleanEmail = email.trim().toLowerCase();
    const exists = db.prepare(`SELECT id FROM users WHERE email = ?`).get(cleanEmail);
    if (exists) {
      res.status(409).json({ error: 'Un compte existe déjà avec cette adresse courriel.' });
      return;
    }

    const cleanName = typeof fullName === 'string' && fullName.trim() ? fullName.trim() : cleanEmail.split('@')[0];
    const { hash, salt } = hashPassword(password);
    const userId = crypto.randomUUID();
    const now = new Date().toISOString();

    db.prepare(
      `INSERT INTO users (id, email, full_name, password_hash, salt, role_code, service, is_active, created_at)
       VALUES (?, ?, ?, ?, ?, 'USER', 'Usager externe', 1, ?)`,
    ).run(userId, cleanEmail, cleanName, hash, salt, now);

    const token = signToken({
      sub: userId,
      email: cleanEmail,
      fullName: cleanName,
      role: 'USER',
    });

    res.status(201).json({
      token,
      user: { id: userId, email: cleanEmail, fullName: cleanName, role: 'USER', service: 'Usager externe' },
    });
  });

  router.post('/auth/login', rateLimit(20, 60_000), (req, res) => {
    const db = getDb();
    const { email, password } = req.body || {};
    if (typeof email !== 'string' || typeof password !== 'string' || !password) {
      res.status(400).json({ error: 'Identifiants requis.' });
      return;
    }

    const cleanEmail = email.trim().toLowerCase();

    // Optional server-side environment passcode override for ADMIN
    if (
      APP_CONFIG.adminPasscodeOverride &&
      password.length === APP_CONFIG.adminPasscodeOverride.length &&
      crypto.timingSafeEqual(
        Buffer.from(password),
        Buffer.from(APP_CONFIG.adminPasscodeOverride),
      )
    ) {
      const token = signToken({
        sub: 'env-admin',
        email: cleanEmail,
        fullName: 'Administrateur DGECC',
        role: 'ADMIN',
      });
      logActivity({
        userId: 'env-admin',
        userEmail: cleanEmail,
        action: 'AUTH_LOGIN_ADMIN_OVERRIDE',
        resourceType: 'session',
        ipAddress: req.ip,
      });
      res.json({
        token,
        user: {
          id: 'env-admin',
          email: cleanEmail,
          fullName: 'Administrateur DGECC',
          role: 'ADMIN',
          service: 'Direction Générale DGECC',
        },
      });
      return;
    }

    const user = db
      .prepare(`SELECT * FROM users WHERE email = ? AND is_active = 1`)
      .get(cleanEmail) as
      | {
          id: string;
          email: string;
          full_name: string;
          password_hash: string;
          salt: string;
          role_code: RoleCode;
          service: string | null;
        }
      | undefined;

    if (!user || !verifyPassword(password, user.password_hash, user.salt)) {
      res.status(401).json({ error: 'Courriel ou mot de passe incorrect.' });
      return;
    }

    const token = signToken({
      sub: user.id,
      email: user.email,
      fullName: user.full_name,
      role: user.role_code,
    });

    logActivity({
      userId: user.id,
      userEmail: user.email,
      action: 'AUTH_LOGIN',
      resourceType: 'session',
      ipAddress: req.ip,
    });

    res.json({
      token,
      user: {
        id: user.id,
        email: user.email,
        fullName: user.full_name,
        role: user.role_code,
        service: user.service,
      },
    });
  });

  router.post('/auth/logout', (req: AuthenticatedRequest, res) => {
    if (req.authUser) {
      logActivity({
        userId: req.authUser.sub,
        userEmail: req.authUser.email,
        action: 'AUTH_LOGOUT',
        resourceType: 'session',
        ipAddress: req.ip,
      });
    }
    res.json({ ok: true });
  });

  router.get('/auth/me', requireAuth, (req: AuthenticatedRequest, res) => {
    const db = getDb();
    const roleRow = db
      .prepare(`SELECT label, permissions_json FROM roles WHERE code = ?`)
      .get(req.authUser!.role) as { label: string; permissions_json: string } | undefined;

    res.json({
      user: {
        id: req.authUser!.sub,
        email: req.authUser!.email,
        fullName: req.authUser!.fullName,
        role: req.authUser!.role,
        roleLabel: roleRow?.label || req.authUser!.role,
        permissions: roleRow ? JSON.parse(roleRow.permissions_json) : [],
      },
    });
  });

  // --------------------------------------------------------------------------
  // Chat & Intelligence Conversationnelle (RAG Hybride)
  // --------------------------------------------------------------------------
  router.post('/chat/message', rateLimit(30, 60_000), async (req: AuthenticatedRequest, res) => {
    try {
      const { question, conversationId, categoryFilter, publishedAfter, topK, rrfK, similarityThreshold } =
        req.body || {};

      if (typeof question !== 'string' || question.trim().length < 2 || question.length > 2000) {
        res.status(400).json({
          error: 'La question doit contenir entre 2 et 2000 caractères.',
        });
        return;
      }

      const owner = resolveOwner(req);
      const result = await runHybridRagPipeline({
        question: question.trim(),
        conversationId: typeof conversationId === 'string' ? conversationId : undefined,
        categoryFilter: typeof categoryFilter === 'string' ? categoryFilter : undefined,
        publishedAfter: typeof publishedAfter === 'string' ? publishedAfter : undefined,
        topK: typeof topK === 'number' ? topK : undefined,
        rrfK: typeof rrfK === 'number' ? rrfK : undefined,
        similarityThreshold: typeof similarityThreshold === 'number' ? similarityThreshold : undefined,
        owner,
      });

      res.json(result);
    } catch (err) {
      const formatted = formatGeminiUserError(err);
      res.status(formatted.status).json({ error: formatted.message });
    }
  });

  router.post('/chat/stream', rateLimit(30, 60_000), async (req: AuthenticatedRequest, res) => {
    const { question, conversationId, categoryFilter, publishedAfter, topK, rrfK, similarityThreshold } =
      req.body || {};

    if (typeof question !== 'string' || question.trim().length < 2 || question.length > 2000) {
      res.status(400).json({
        error: 'La question doit contenir entre 2 et 2000 caractères.',
      });
      return;
    }

    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const sendEvent = (event: string, data: unknown) => {
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    try {
      const owner = resolveOwner(req);
      const result = await runHybridRagPipeline({
        question: question.trim(),
        conversationId: typeof conversationId === 'string' ? conversationId : undefined,
        categoryFilter: typeof categoryFilter === 'string' ? categoryFilter : undefined,
        publishedAfter: typeof publishedAfter === 'string' ? publishedAfter : undefined,
        topK: typeof topK === 'number' ? topK : undefined,
        rrfK: typeof rrfK === 'number' ? rrfK : undefined,
        similarityThreshold: typeof similarityThreshold === 'number' ? similarityThreshold : undefined,
        owner,
        onStage: (stage, detail) => sendEvent('stage', { stage, detail }),
        onToken: (delta) => sendEvent('token', { delta }),
      });
      sendEvent('complete', result);
      res.end();
    } catch (err) {
      const formatted = formatGeminiUserError(err);
      sendEvent('error', { error: formatted.message });
      res.end();
    }
  });

  router.get('/chat/conversations', (req: AuthenticatedRequest, res) => {
    const db = getDb();
    const owner = resolveOwner(req);
    const rows = owner.userId
      ? db
          .prepare(
            `SELECT id, title, created_at, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT 40`,
          )
          .all(owner.userId)
      : db
          .prepare(
            `SELECT id, title, created_at, updated_at FROM conversations WHERE guest_session_id = ? ORDER BY updated_at DESC LIMIT 40`,
          )
          .all(owner.guestSessionId);
    res.json({ conversations: rows });
  });

  router.get('/chat/conversations/:id', (req: AuthenticatedRequest, res) => {
    const db = getDb();
    const owner = resolveOwner(req);
    const conv = db
      .prepare(`SELECT * FROM conversations WHERE id = ?`)
      .get(req.params.id) as
      | { id: string; user_id: string | null; guest_session_id: string | null; title: string }
      | undefined;

    if (!conv) {
      res.status(404).json({ error: 'Conversation introuvable.' });
      return;
    }

    const authorized =
      (owner.userId && conv.user_id === owner.userId) ||
      (!owner.userId && owner.guestSessionId && conv.guest_session_id === owner.guestSessionId);

    if (!authorized) {
      res.status(403).json({ error: 'Accès non autorisé à cette conversation.' });
      return;
    }

    const msgs = db
      .prepare(`SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC`)
      .all(req.params.id) as Array<Record<string, unknown>>;

    res.json({
      conversation: { id: conv.id, title: conv.title },
      messages: msgs.map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        analysis: m.nlp_analysis_json ? JSON.parse(String(m.nlp_analysis_json)) : null,
        sources: m.sources_json ? JSON.parse(String(m.sources_json)) : [],
        guardrails: m.guardrails_json ? JSON.parse(String(m.guardrails_json)) : null,
        metrics: m.metrics_json ? JSON.parse(String(m.metrics_json)) : null,
        feedback: m.feedback,
        createdAt: m.created_at,
      })),
    });
  });

  router.delete('/chat/conversations/:id', (req: AuthenticatedRequest, res) => {
    const db = getDb();
    const owner = resolveOwner(req);
    const conv = db
      .prepare(`SELECT id, user_id, guest_session_id FROM conversations WHERE id = ?`)
      .get(req.params.id) as
      | { id: string; user_id: string | null; guest_session_id: string | null }
      | undefined;

    if (!conv) {
      res.status(404).json({ error: 'Conversation introuvable.' });
      return;
    }

    const authorized =
      (owner.userId && conv.user_id === owner.userId) ||
      (!owner.userId && owner.guestSessionId && conv.guest_session_id === owner.guestSessionId);

    if (!authorized) {
      res.status(403).json({ error: 'Suppression refusée.' });
      return;
    }

    db.prepare(`DELETE FROM conversations WHERE id = ?`).run(req.params.id);
    res.json({ ok: true });
  });

  router.post('/chat/messages/:id/feedback', (req: AuthenticatedRequest, res) => {
    const db = getDb();
    const { feedback, comment } = req.body || {};
    if (feedback !== 'positive' && feedback !== 'negative') {
      res.status(400).json({ error: 'Type de retour invalide (positive ou negative).' });
      return;
    }
    db.prepare(`UPDATE messages SET feedback = ?, feedback_comment = ? WHERE id = ?`).run(
      feedback,
      typeof comment === 'string' ? comment.slice(0, 500) : null,
      req.params.id,
    );
    res.json({ ok: true });
  });

  // --------------------------------------------------------------------------
  // Pipeline Documentaire & Corpus (Module 1)
  // --------------------------------------------------------------------------
  router.get('/documents', (req: AuthenticatedRequest, res) => {
    const db = getDb();
    const isStaff =
      req.authUser && ['ADMIN', 'VALIDATOR', 'EDITOR'].includes(req.authUser.role);

    const sql = isStaff
      ? `SELECT d.*,
          (SELECT COUNT(*) FROM document_chunks c WHERE c.document_id = d.id AND c.version_number = d.current_version) AS chunk_count,
          (SELECT COUNT(*) FROM document_chunks c WHERE c.document_id = d.id AND c.version_number = d.current_version AND c.embedding_json IS NOT NULL) AS embedded_count
         FROM documents d ORDER BY d.updated_at DESC`
      : `SELECT d.id, d.title, d.doc_type, d.category, d.service_name, d.reference_code,
          d.published_date, d.valid_until, d.current_version, d.status, d.file_format, d.updated_at,
          (SELECT COUNT(*) FROM document_chunks c WHERE c.document_id = d.id AND c.version_number = d.current_version) AS chunk_count
         FROM documents d WHERE d.status = 'PUBLIE' ORDER BY d.published_date DESC`;

    const documents = db.prepare(sql).all();
    res.json({ documents });
  });

  router.get('/documents/:id', (req: AuthenticatedRequest, res) => {
    const db = getDb();
    const doc = db.prepare(`SELECT * FROM documents WHERE id = ?`).get(req.params.id) as
      | Record<string, unknown>
      | undefined;

    if (!doc) {
      res.status(404).json({ error: 'Document introuvable.' });
      return;
    }

    const isStaff =
      req.authUser && ['ADMIN', 'VALIDATOR', 'EDITOR'].includes(req.authUser.role);
    if (!isStaff && doc.status !== 'PUBLIE') {
      res.status(403).json({ error: 'Ce document n’est pas encore publié.' });
      return;
    }

    const versions = db
      .prepare(
        `SELECT id, version_number, chunk_size, chunk_overlap, created_at,
         substr(normalized_text, 1, 1800) AS preview_text
         FROM document_versions WHERE document_id = ? ORDER BY version_number DESC`,
      )
      .all(req.params.id);

    const chunks = db
      .prepare(
        `SELECT id, chunk_index, content, token_count, char_count, embedding_model,
         CASE WHEN embedding_json IS NOT NULL THEN 1 ELSE 0 END AS has_embedding
         FROM document_chunks WHERE document_id = ? AND version_number = ? ORDER BY chunk_index ASC`,
      )
      .all(req.params.id, Number(doc.current_version));

    res.json({ document: doc, versions, chunks });
  });

  const handleOptionalMultipart = (
    req: express.Request,
    res: express.Response,
    next: express.NextFunction,
  ) => {
    const ct = req.headers['content-type'] || '';
    if (ct.includes('multipart/form-data')) {
      upload.single('file')(req, res, (err) => {
        if (err) {
          res.status(400).json({
            error: err instanceof Error ? err.message : 'Erreur lors de la réception du fichier.',
          });
          return;
        }
        next();
      });
    } else {
      next();
    }
  };

  const extractUploadedFileFromRequest = (req: AuthenticatedRequest) => {
    const body = req.body || {};
    let fileBuffer: Buffer | null = null;
    let fileName = '';
    let mimeType: string | undefined;

    if (req.file) {
      fileBuffer = req.file.buffer;
      fileName = req.file.originalname;
      mimeType = req.file.mimetype;
    } else if (typeof body.fileBase64 === 'string' && typeof body.fileName === 'string') {
      const base64Clean = body.fileBase64.includes(',')
        ? body.fileBase64.split(',').pop() || ''
        : body.fileBase64;
      fileBuffer = Buffer.from(base64Clean, 'base64');
      fileName = body.fileName;
      mimeType = typeof body.mimeType === 'string' ? body.mimeType : undefined;
    }

    return { body, fileBuffer, fileName, mimeType };
  };

  router.post(
    '/documents/ocr-preview',
    requireRole('ADMIN', 'EDITOR'),
    handleOptionalMultipart,
    async (req: AuthenticatedRequest, res) => {
      try {
        const { body, fileBuffer, fileName, mimeType } = extractUploadedFileFromRequest(req);
        if (!fileBuffer || fileBuffer.length === 0 || !fileName) {
          res.status(400).json({ error: 'Aucun fichier valide n’a été transmis.' });
          return;
        }
        if (fileBuffer.length > 15 * 1024 * 1024) {
          res.status(413).json({ error: 'Le fichier dépasse la limite autorisée de 15 Mo.' });
          return;
        }

        const format = detectFormat(fileName, mimeType);
        if (!format) {
          res.status(400).json({
            error:
              'Format non supporté. Formats acceptés : PDF, PNG, JPG, WEBP, DOCX, HTML, TXT.',
          });
          return;
        }

        const ocrMode: OcrMode =
          body.ocrMode === 'force' || body.ocrMode === 'off' ? body.ocrMode : 'auto';

        const extraction = await extractDocumentWithOcr(fileBuffer, format, {
          fileName,
          mimeType,
          ocrMode,
        });
        const normalizedText = normalizeText(extraction.text);

        res.json({
          text: normalizedText,
          ocrUsed: extraction.ocrUsed,
          ocrReason: extraction.ocrReason,
          ocrModel: extraction.ocrModel || null,
          nativeCharCount: extraction.nativeCharCount,
          pageCount: extraction.pageCount || null,
          normalizedLength: normalizedText.length,
        });
      } catch (err) {
        const formatted = formatGeminiUserError(err);
        res.status(formatted.status >= 400 ? formatted.status : 400).json({
          error: formatted.message,
        });
      }
    },
  );

  router.post(
    '/documents/upload',
    requireRole('ADMIN', 'EDITOR'),
    handleOptionalMultipart,
    async (req: AuthenticatedRequest, res) => {
      try {
        const { body, fileBuffer, fileName, mimeType } = extractUploadedFileFromRequest(req);

        if (!fileBuffer || fileBuffer.length === 0 || !fileName) {
          res.status(400).json({ error: 'Aucun fichier valide n’a été transmis.' });
          return;
        }

        if (fileBuffer.length > 15 * 1024 * 1024) {
          res.status(413).json({ error: 'Le fichier dépasse la limite autorisée de 15 Mo.' });
          return;
        }

        const {
          title,
          docType,
          category,
          serviceName,
          referenceCode,
          publishedDate,
          validUntil,
          chunkSize,
          chunkOverlap,
          initialStatus,
          ocrMode,
          reviewedText,
        } = body;

        const format = detectFormat(fileName, mimeType);
        if (!format) {
          res.status(400).json({
            error:
              'Format non supporté. Formats acceptés : PDF, PNG, JPG, WEBP, DOCX, HTML, TXT.',
          });
          return;
        }

        if (
          typeof title !== 'string' ||
          !title.trim() ||
          typeof referenceCode !== 'string' ||
          !referenceCode.trim()
        ) {
          res.status(400).json({
            error: 'Le titre et la référence réglementaire sont obligatoires.',
          });
          return;
        }

        const allowedInitial: DocumentStatus =
          initialStatus === 'BROUILLON' ? 'BROUILLON' : 'EN_ATTENTE';
        const allowedOcrMode: OcrMode =
          ocrMode === 'force' || ocrMode === 'off' ? ocrMode : 'auto';

        const result = await ingestDocument({
          title,
          docType: typeof docType === 'string' && docType.trim() ? docType : 'Note officielle',
          category:
            typeof category === 'string' && category.trim()
              ? category
              : 'Direction Générale & Services Rattachés',
          serviceName:
            typeof serviceName === 'string' && serviceName.trim()
              ? serviceName
              : 'Direction Générale — Secrétariat du Directeur Général',
          referenceCode,
          publishedDate:
            typeof publishedDate === 'string' && publishedDate.trim()
              ? publishedDate
              : new Date().toISOString().slice(0, 10),
          validUntil: typeof validUntil === 'string' && validUntil.trim() ? validUntil.trim() : null,
          fileName,
          mimeType,
          format,
          buffer: fileBuffer,
          ocrMode: allowedOcrMode,
          reviewedText: typeof reviewedText === 'string' ? reviewedText : undefined,
          chunkSize: chunkSize ? Number(chunkSize) : RAG_CONFIG.defaultChunkSize,
          chunkOverlap: chunkOverlap ? Number(chunkOverlap) : RAG_CONFIG.defaultChunkOverlap,
          uploadedBy: req.authUser!.sub,
          uploadedByEmail: req.authUser!.email,
          initialStatus: allowedInitial,
        });

        res.status(201).json(result);
      } catch (err) {
        const formatted = formatGeminiUserError(err);
        res.status(formatted.status >= 400 ? formatted.status : 400).json({
          error: formatted.message,
        });
      }
    },
  );

  router.put(
    '/documents/:id/validate',
    requireRole('ADMIN', 'VALIDATOR'),
    (req: AuthenticatedRequest, res) => {
      const db = getDb();
      const { status, rejectionReason } = req.body || {};
      const allowed: DocumentStatus[] = ['BROUILLON', 'EN_ATTENTE', 'VALIDE', 'REJETE', 'ARCHIVE'];
      if (!allowed.includes(status)) {
        res.status(400).json({ error: `Statut cible invalide. Autorisés : ${allowed.join(', ')}` });
        return;
      }

      const doc = db.prepare(`SELECT id, title FROM documents WHERE id = ?`).get(req.params.id) as
        | { id: string; title: string }
        | undefined;
      if (!doc) {
        res.status(404).json({ error: 'Document introuvable.' });
        return;
      }

      const now = new Date().toISOString();
      db.prepare(
        `UPDATE documents SET status = ?, validated_by = ?, rejection_reason = ?, updated_at = ? WHERE id = ?`,
      ).run(
        status,
        req.authUser!.sub,
        status === 'REJETE' && typeof rejectionReason === 'string' ? rejectionReason.trim() : null,
        now,
        req.params.id,
      );

      logActivity({
        userId: req.authUser!.sub,
        userEmail: req.authUser!.email,
        action: `DOCUMENT_STATUS_${status}`,
        resourceType: 'document',
        resourceId: req.params.id,
        details: { title: doc.title, rejectionReason: rejectionReason || null },
        ipAddress: req.ip,
      });

      invalidateCorpusCache();

      res.json({ id: req.params.id, status });
    },
  );

  router.post(
    '/documents/:id/publish',
    requireRole('ADMIN', 'VALIDATOR'),
    async (req: AuthenticatedRequest, res) => {
      try {
        const db = getDb();
        const doc = db.prepare(`SELECT id, status, title FROM documents WHERE id = ?`).get(req.params.id) as
          | { id: string; status: DocumentStatus; title: string }
          | undefined;

        if (!doc) {
          res.status(404).json({ error: 'Document introuvable.' });
          return;
        }

        const indexStats = await indexDocumentVersion(
          req.params.id,
          req.authUser!.sub,
          req.authUser!.email,
        );

        const now = new Date().toISOString();
        db.prepare(`UPDATE documents SET status = 'PUBLIE', validated_by = ?, updated_at = ? WHERE id = ?`).run(
          req.authUser!.sub,
          now,
          req.params.id,
        );
        invalidateCorpusCache();

        logActivity({
          userId: req.authUser!.sub,
          userEmail: req.authUser!.email,
          action: 'DOCUMENT_PUBLISHED',
          resourceType: 'document',
          resourceId: req.params.id,
          details: { title: doc.title, ...indexStats },
          ipAddress: req.ip,
        });

        res.json({ id: req.params.id, status: 'PUBLIE', indexation: indexStats });
      } catch (err) {
        const formatted = formatGeminiUserError(err);
        res.status(formatted.status).json({ error: formatted.message });
      }
    },
  );

  router.post(
    '/documents/:id/reindex',
    requireRole('ADMIN', 'VALIDATOR'),
    async (req: AuthenticatedRequest, res) => {
      try {
        const { chunkSize, chunkOverlap, bumpVersion } = req.body || {};
        const stats = await indexDocumentVersion(
          req.params.id,
          req.authUser!.sub,
          req.authUser!.email,
          {
            chunkSize: typeof chunkSize === 'number' ? chunkSize : undefined,
            chunkOverlap: typeof chunkOverlap === 'number' ? chunkOverlap : undefined,
            bumpVersion: Boolean(bumpVersion),
          },
        );
        res.json(stats);
      } catch (err) {
        const formatted = formatGeminiUserError(err);
        res.status(formatted.status).json({ error: formatted.message });
      }
    },
  );

  // --------------------------------------------------------------------------
  // Administration, Statistiques, RBAC, Journaux & Évaluation RAG
  // --------------------------------------------------------------------------
  router.get(
    '/admin/statistics',
    requireRole('ADMIN', 'VALIDATOR', 'EDITOR'),
    (_req, res) => {
      const db = getDb();

      const docStats = db
        .prepare(`SELECT status, COUNT(*) AS count FROM documents GROUP BY status`)
        .all() as Array<{ status: string; count: number }>;

      const totalChunks = (
        db.prepare(`SELECT COUNT(*) AS cnt FROM document_chunks`).get() as { cnt: number }
      ).cnt;
      const embeddedChunks = (
        db
          .prepare(`SELECT COUNT(*) AS cnt FROM document_chunks WHERE embedding_json IS NOT NULL`)
          .get() as { cnt: number }
      ).cnt;
      const totalConversations = (
        db.prepare(`SELECT COUNT(*) AS cnt FROM conversations`).get() as { cnt: number }
      ).cnt;
      const totalMessages = (
        db.prepare(`SELECT COUNT(*) AS cnt FROM messages WHERE role = 'assistant'`).get() as {
          cnt: number;
        }
      ).cnt;
      const positiveFeedback = (
        db
          .prepare(`SELECT COUNT(*) AS cnt FROM messages WHERE feedback = 'positive'`)
          .get() as { cnt: number }
      ).cnt;
      const negativeFeedback = (
        db
          .prepare(`SELECT COUNT(*) AS cnt FROM messages WHERE feedback = 'negative'`)
          .get() as { cnt: number }
      ).cnt;

      const assistantMetricsRows = db
        .prepare(`SELECT metrics_json, guardrails_json FROM messages WHERE role = 'assistant' AND metrics_json IS NOT NULL ORDER BY created_at DESC LIMIT 50`)
        .all() as Array<{ metrics_json: string; guardrails_json: string | null }>;

      let avgTotalMs = 0;
      let avgRetrievalMs = 0;
      let refusedCount = 0;
      if (assistantMetricsRows.length > 0) {
        let sumTotal = 0;
        let sumRet = 0;
        for (const r of assistantMetricsRows) {
          try {
            const m = JSON.parse(r.metrics_json) as { totalMs?: number; retrievalMs?: number };
            sumTotal += m.totalMs || 0;
            sumRet += m.retrievalMs || 0;
            if (r.guardrails_json) {
              const g = JSON.parse(r.guardrails_json) as { refused?: boolean };
              if (g.refused) refusedCount++;
            }
          } catch {
            // ignore
          }
        }
        avgTotalMs = Math.round(sumTotal / assistantMetricsRows.length);
        avgRetrievalMs = Math.round(sumRet / assistantMetricsRows.length);
      }

      res.json({
        documentsByStatus: docStats,
        totalChunks,
        embeddedChunks,
        totalConversations,
        totalAssistantResponses: totalMessages,
        feedback: {
          positive: positiveFeedback,
          negative: negativeFeedback,
        },
        performance: {
          avgTotalMs,
          avgRetrievalMs,
          refusalRate:
            assistantMetricsRows.length > 0
              ? Number((refusedCount / assistantMetricsRows.length).toFixed(3))
              : 0,
        },
        ragConfig: RAG_CONFIG,
      });
    },
  );

  router.get('/admin/users', requireRole('ADMIN'), (_req, res) => {
    const db = getDb();
    const users = db
      .prepare(
        `SELECT id, email, full_name, role_code, service, is_active, created_at FROM users ORDER BY created_at DESC`,
      )
      .all();
    const roles = db.prepare(`SELECT code, label, permissions_json FROM roles`).all();
    res.json({ users, roles });
  });

  router.post('/admin/users', requireRole('ADMIN'), (req: AuthenticatedRequest, res) => {
    const db = getDb();
    const { email, fullName, password, roleCode, service } = req.body || {};
    const validRoles: RoleCode[] = ['ADMIN', 'VALIDATOR', 'EDITOR', 'USER'];
    if (
      typeof email !== 'string' ||
      !email.includes('@') ||
      typeof password !== 'string' ||
      password.length < 6 ||
      !validRoles.includes(roleCode)
    ) {
      res.status(400).json({ error: 'Données utilisateur invalides.' });
      return;
    }

    const cleanEmail = email.trim().toLowerCase();
    const exists = db.prepare(`SELECT id FROM users WHERE email = ?`).get(cleanEmail);
    if (exists) {
      res.status(409).json({ error: 'Cette adresse courriel est déjà utilisée.' });
      return;
    }

    const { hash, salt } = hashPassword(password);
    const id = crypto.randomUUID();
    const now = new Date().toISOString();
    const cleanName = typeof fullName === 'string' && fullName.trim() ? fullName.trim() : cleanEmail;

    db.prepare(
      `INSERT INTO users (id, email, full_name, password_hash, salt, role_code, service, is_active, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)`,
    ).run(id, cleanEmail, cleanName, hash, salt, roleCode, service || 'DGECC', now);

    logActivity({
      userId: req.authUser!.sub,
      userEmail: req.authUser!.email,
      action: 'USER_CREATED',
      resourceType: 'user',
      resourceId: id,
      details: { email: cleanEmail, roleCode },
      ipAddress: req.ip,
    });

    res.status(201).json({ id, email: cleanEmail, fullName: cleanName, roleCode });
  });

  router.put('/admin/users/:id/role', requireRole('ADMIN'), (req: AuthenticatedRequest, res) => {
    const db = getDb();
    const { roleCode, isActive } = req.body || {};
    const validRoles: RoleCode[] = ['ADMIN', 'VALIDATOR', 'EDITOR', 'USER'];
    if (roleCode && !validRoles.includes(roleCode)) {
      res.status(400).json({ error: 'Rôle invalide.' });
      return;
    }
    if (roleCode) {
      db.prepare(`UPDATE users SET role_code = ? WHERE id = ?`).run(roleCode, req.params.id);
    }
    if (typeof isActive === 'boolean') {
      db.prepare(`UPDATE users SET is_active = ? WHERE id = ?`).run(isActive ? 1 : 0, req.params.id);
    }
    logActivity({
      userId: req.authUser!.sub,
      userEmail: req.authUser!.email,
      action: 'USER_ROLE_UPDATED',
      resourceType: 'user',
      resourceId: req.params.id,
      details: { roleCode, isActive },
      ipAddress: req.ip,
    });
    res.json({ ok: true });
  });

  router.get('/admin/logs', requireRole('ADMIN', 'VALIDATOR'), (_req, res) => {
    const db = getDb();
    const logs = db
      .prepare(`SELECT * FROM activity_logs ORDER BY created_at DESC LIMIT 100`)
      .all();
    res.json({ logs });
  });

  router.get('/admin/indexations', requireRole('ADMIN', 'VALIDATOR', 'EDITOR'), (_req, res) => {
    const db = getDb();
    const indexations = db
      .prepare(
        `SELECT ih.*, d.title AS document_title, d.reference_code
         FROM indexation_history ih
         JOIN documents d ON d.id = ih.document_id
         ORDER BY ih.created_at DESC LIMIT 60`,
      )
      .all();
    res.json({ indexations });
  });

  router.get('/admin/conversations', requireRole('ADMIN'), (_req, res) => {
    const db = getDb();
    const conversations = db
      .prepare(
        `SELECT c.id, c.title, c.user_id, c.created_at, c.updated_at,
         u.email AS user_email,
         (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
         FROM conversations c
         LEFT JOIN users u ON u.id = c.user_id
         ORDER BY c.updated_at DESC LIMIT 50`,
      )
      .all();
    res.json({ conversations });
  });

  router.get('/admin/evaluations', requireRole('ADMIN', 'VALIDATOR'), (_req, res) => {
    const db = getDb();
    const runs = db
      .prepare(`SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT 20`)
      .all() as Array<Record<string, unknown>>;
    res.json({
      runs: runs.map((r) => ({
        ...r,
        cases: r.details_json ? JSON.parse(String(r.details_json)) : [],
        methodology: r.methodology_json ? JSON.parse(String(r.methodology_json)) : null,
      })),
    });
  });

  router.get('/admin/evaluation-overview', requireRole('ADMIN', 'VALIDATOR'), (_req, res) => {
    res.json(getEvaluationOverview());
  });

  router.get('/admin/evaluation-panel', requireRole('ADMIN', 'VALIDATOR'), (_req, res) => {
    res.json({ entries: listUserPanelEvaluations() });
  });

  router.post(
    '/admin/evaluation-panel',
    requireRole('ADMIN', 'VALIDATOR'),
    (req: AuthenticatedRequest, res) => {
      const { profile, participantName, clarityScore, usefulnessScore, trustScore, comment } =
        req.body || {};
      const validProfiles: UserPanelProfile[] = ['CANDIDAT', 'PARENT', 'AGENT_ADMINISTRATIF'];
      if (
        !validProfiles.includes(profile) ||
        typeof participantName !== 'string' ||
        participantName.trim().length < 2
      ) {
        res.status(400).json({
          error: 'Profil (CANDIDAT, PARENT, AGENT_ADMINISTRATIF) et nom du participant requis.',
        });
        return;
      }

      const clarity = Number(clarityScore);
      const usefulness = Number(usefulnessScore);
      const trust = Number(trustScore);
      if (
        !Number.isFinite(clarity) ||
        !Number.isFinite(usefulness) ||
        !Number.isFinite(trust) ||
        clarity < 1 ||
        clarity > 5 ||
        usefulness < 1 ||
        usefulness > 5 ||
        trust < 1 ||
        trust > 5
      ) {
        res.status(400).json({ error: 'Les notes doivent être comprises entre 1 et 5.' });
        return;
      }

      const entry = addUserPanelEvaluation({
        profile,
        participantName,
        clarityScore: clarity,
        usefulnessScore: usefulness,
        trustScore: trust,
        comment: typeof comment === 'string' ? comment : null,
        recordedBy: req.authUser!.email,
      });

      logActivity({
        userId: req.authUser!.sub,
        userEmail: req.authUser!.email,
        action: 'PANEL_EVALUATION_ADDED',
        resourceType: 'evaluation_panel',
        resourceId: entry.id,
        details: { profile: entry.profile, participantName: entry.participantName },
        ipAddress: req.ip,
      });

      res.status(201).json({ entry });
    },
  );

  router.delete(
    '/admin/evaluation-panel/:id',
    requireRole('ADMIN'),
    (req: AuthenticatedRequest, res) => {
      const deleted = deleteUserPanelEvaluation(req.params.id);
      if (!deleted) {
        res.status(404).json({ error: 'Entrée du panel introuvable.' });
        return;
      }
      logActivity({
        userId: req.authUser!.sub,
        userEmail: req.authUser!.email,
        action: 'PANEL_EVALUATION_DELETED',
        resourceType: 'evaluation_panel',
        resourceId: req.params.id,
        ipAddress: req.ip,
      });
      res.json({ ok: true });
    },
  );

  router.post(
    '/admin/evaluate',
    requireRole('ADMIN'),
    async (req: AuthenticatedRequest, res) => {
      try {
        const body = req.body || {};
        const mode = body.mode === 'retrieval_only' ? 'retrieval_only' : 'full';
        const sampleSize = Number(body.sampleSize) === 100 ? 100 : 20;
        const result = await runRagBenchmarkEvaluation(
          req.authUser!.sub,
          req.authUser!.email,
          { mode, sampleSize },
        );
        res.json(result);
      } catch (err) {
        const formatted = formatGeminiUserError(err);
        res.status(formatted.status).json({ error: formatted.message });
      }
    },
  );

  router.use((_req, res) => {
    res.status(404).json({ error: 'Endpoint API introuvable.' });
  });

  router.use(
    (
      err: unknown,
      _req: express.Request,
      res: express.Response,
      _next: express.NextFunction,
    ) => {
      const status =
        typeof (err as { status?: number })?.status === 'number'
          ? (err as { status: number }).status
          : 500;
      const message =
        err instanceof Error ? err.message : 'Erreur interne du serveur API.';
      res.status(status).json({ error: message });
    },
  );

  return router;
}
