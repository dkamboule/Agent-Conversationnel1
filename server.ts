import 'dotenv/config';
import express from 'express';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { createServer as createViteServer, type ViteDevServer } from 'vite';
import { APP_CONFIG } from './server/core/config.ts';
import { createApiRouter } from './server/api/routes.ts';
import { getDb } from './server/database/db.ts';
import { ensureOfficialSourcesExtractedAndSeeded } from './server/database/officialSources.ts';
import { seedInitialDatabase } from './server/database/seed.ts';

async function bootstrapServer() {
  // Initialize SQLite storage and seed institutional DGECC corpus
  getDb();
  seedInitialDatabase();

  const app = express();
  app.set('trust proxy', 1);
  app.use(express.json({ limit: '25mb' }));

  // Mount versioned REST API
  app.use('/api/v1', createApiRouter());

  const httpServer = http.createServer(app);
  const isProd = process.env.NODE_ENV === 'production';
  let vite: ViteDevServer | null = null;

  if (!isProd) {
    process.env.DISABLE_HMR ??= 'true';
    vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: false,
        ws: false,
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    if (fs.existsSync(distDir)) {
      app.use(express.static(distDir));
      app.get('*', (_req, res) => {
        res.sendFile(path.join(distDir, 'index.html'));
      });
    }
  }

  let shuttingDown = false;
  const shutdown = async (signal: string) => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`[DGECC] Arrêt propre du serveur (${signal})...`);
    try {
      if (vite) {
        await vite.close();
      }
    } catch {
      // Ignore Vite close errors during shutdown
    }
    httpServer.close(() => {
      process.exit(0);
    });
    setTimeout(() => process.exit(0), 1500).unref();
  };

  process.on('SIGTERM', () => void shutdown('SIGTERM'));
  process.on('SIGINT', () => void shutdown('SIGINT'));

  httpServer.on('error', (err: NodeJS.ErrnoException) => {
    console.error(`Failed to bind DGECC server on port ${APP_CONFIG.port}:`, err.message);
    process.exit(1);
  });

  httpServer.listen(APP_CONFIG.port, '0.0.0.0', () => {
    console.log(`DGECC RAG server listening on http://0.0.0.0:${APP_CONFIG.port}`);
    // Extract official PDF sources (if not cached yet), embed chunks, and pre-warm BM25 + vector index
    ensureOfficialSourcesExtractedAndSeeded().catch((err) => {
      console.error('[DGECC] Initialisation corpus officiel différée:', err);
    });
  });
}

bootstrapServer().catch((err) => {
  console.error('Failed to start DGECC server:', err);
  process.exit(1);
});
