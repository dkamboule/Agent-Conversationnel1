import crypto from 'node:crypto';
import type { Request, Response, NextFunction } from 'express';
import { APP_CONFIG } from './config.ts';

export type RoleCode = 'ADMIN' | 'VALIDATOR' | 'EDITOR' | 'USER';

export interface AuthPayload {
  sub: string;
  email: string;
  fullName: string;
  role: RoleCode;
  iat: number;
  exp: number;
}

export function hashPassword(password: string, existingSalt?: string): { hash: string; salt: string } {
  const salt = existingSalt || crypto.randomBytes(16).toString('hex');
  const derived = crypto.scryptSync(password, salt, 64).toString('hex');
  return { hash: derived, salt };
}

export function verifyPassword(password: string, storedHash: string, salt: string): boolean {
  if (!password || !storedHash || !salt) return false;
  const derived = crypto.scryptSync(password, salt, 64);
  const target = Buffer.from(storedHash, 'hex');
  if (derived.length !== target.length) return false;
  return crypto.timingSafeEqual(derived, target);
}

function base64UrlEncode(input: string | Buffer): string {
  return Buffer.from(input)
    .toString('base64')
    .replaceAll('+', '-')
    .replaceAll('/', '_')
    .replaceAll('=', '');
}

function base64UrlDecode(input: string): string {
  const padded = input.replaceAll('-', '+').replaceAll('_', '/') + '==='.slice((input.length + 3) % 4);
  return Buffer.from(padded, 'base64').toString('utf8');
}

export function signToken(payload: Omit<AuthPayload, 'iat' | 'exp'>): string {
  const now = Math.floor(Date.now() / 1000);
  const fullPayload: AuthPayload = {
    ...payload,
    iat: now,
    exp: now + APP_CONFIG.tokenTtlSeconds,
  };
  const header = base64UrlEncode(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
  const body = base64UrlEncode(JSON.stringify(fullPayload));
  const signature = base64UrlEncode(
    crypto.createHmac('sha256', APP_CONFIG.sessionSecret).update(`${header}.${body}`).digest(),
  );
  return `${header}.${body}.${signature}`;
}

export function verifyToken(token: string): AuthPayload | null {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    const [header, body, sig] = parts;
    const expectedSig = base64UrlEncode(
      crypto.createHmac('sha256', APP_CONFIG.sessionSecret).update(`${header}.${body}`).digest(),
    );
    const sigBuf = Buffer.from(sig);
    const expectedBuf = Buffer.from(expectedSig);
    if (sigBuf.length !== expectedBuf.length || !crypto.timingSafeEqual(sigBuf, expectedBuf)) {
      return null;
    }
    const parsed = JSON.parse(base64UrlDecode(body)) as AuthPayload;
    if (!parsed.exp || parsed.exp < Math.floor(Date.now() / 1000)) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

const INJECTION_PATTERNS: RegExp[] = [
  /ignore\s+(all\s+)?(previous|above|prior)\s+instructions/i,
  /ignore[zr]?\s+(toutes?\s+)?les?\s+instructions?\s+(précédentes?|antérieures?)/i,
  /oublie[zr]?\s+(toutes?\s+)?les?\s+règles/i,
  /system\s*prompt/i,
  /jailbreak/i,
  /you\s+are\s+now\s+/i,
  /tu\s+es\s+maintenant\s+/i,
  /révèle[zr]?\s+(ton|le)\s+prompt/i,
  /<script\b[^>]*>/i,
];

export function detectPromptInjection(text: string): { flagged: boolean; reason?: string } {
  for (const pattern of INJECTION_PATTERNS) {
    if (pattern.test(text)) {
      return {
        flagged: true,
        reason: 'Tentative de contournement des instructions système détectée.',
      };
    }
  }
  return { flagged: false };
}

export function sanitizeDocumentText(text: string): string {
  // Neutralize embedded prompt-like directives inside uploaded documents
  return text
    .replace(/\[INST\]|\[\/INST\]|<\|im_start\|>|<\|im_end\|>/gi, '')
    .replace(/ignore\s+previous\s+instructions/gi, '[contenu filtré]');
}

// Sliding-window in-memory rate limiter
const rateBuckets = new Map<string, number[]>();

export function rateLimit(maxRequests: number, windowMs: number) {
  return (req: Request, res: Response, next: NextFunction) => {
    const key = `${req.ip || 'local'}:${req.baseUrl}${req.path}`;
    const now = Date.now();
    const timestamps = (rateBuckets.get(key) || []).filter((ts) => now - ts < windowMs);
    if (timestamps.length >= maxRequests) {
      res.status(429).json({
        error: 'Trop de requêtes effectuées. Veuillez patienter quelques instants avant de réessayer.',
      });
      return;
    }
    timestamps.push(now);
    rateBuckets.set(key, timestamps);
    next();
  };
}
