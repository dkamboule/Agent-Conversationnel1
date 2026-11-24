import { GoogleGenAI } from '@google/genai';

let cachedClient: GoogleGenAI | null = null;
let cachedApiKey: string | undefined;

function readConfiguredGeminiKey(): string | undefined {
  const raw = process.env.GEMINI_API_KEY?.trim().replace(/^["']|["']$/g, '');
  if (!raw || raw === 'MY_GEMINI_API_KEY') {
    return undefined;
  }
  return raw;
}

export function hasConfiguredGeminiKey(): boolean {
  return Boolean(readConfiguredGeminiKey());
}

export function getGeminiClient(): GoogleGenAI {
  const apiKey = readConfiguredGeminiKey();
  if (!apiKey) {
    throw new Error(
      'Clé API Gemini absente ou non configurée. Veuillez renseigner GEMINI_API_KEY dans les secrets serveur.',
    );
  }
  if (!cachedClient || cachedApiKey !== apiKey) {
    cachedClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
    cachedApiKey = apiKey;
  }
  return cachedClient;
}

interface ParsedUpstreamError {
  code?: number;
  status?: string;
  message: string;
}

function tryParseJson(input: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(input);
    return parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

function extractUpstreamError(err: unknown): ParsedUpstreamError {
  if (!err) {
    return { message: '' };
  }

  let code: number | undefined;
  let status: string | undefined;
  let message = '';

  if (err instanceof Error) {
    message = err.message || '';
  } else if (typeof err === 'string') {
    message = err;
  } else if (typeof err === 'object') {
    const obj = err as Record<string, unknown>;
    if (typeof obj.status === 'number') code = obj.status;
    if (typeof obj.code === 'number') code = obj.code;
    if (typeof obj.statusText === 'string') status = obj.statusText;
    if (typeof obj.message === 'string') message = obj.message;
  }

  // Unwrap nested JSON error envelopes emitted by @google/genai
  // Example: {"error":{"message":"{\n \"error\": {\n \"code\": 503, ...}}","code":503,"status":"Service Unavailable"}}
  let current = message.trim();
  for (let depth = 0; depth < 3; depth++) {
    const jsonStart = current.indexOf('{');
    if (jsonStart === -1) break;
    const candidate = current.slice(jsonStart);
    const parsed = tryParseJson(candidate);
    if (!parsed) break;

    const inner =
      parsed.error && typeof parsed.error === 'object'
        ? (parsed.error as Record<string, unknown>)
        : parsed;

    if (typeof inner.code === 'number') code = inner.code;
    if (typeof inner.status === 'string') status = inner.status;
    if (typeof inner.message === 'string') {
      current = inner.message.trim();
      message = current;
      continue;
    }
    break;
  }

  if (err && typeof err === 'object') {
    const obj = err as Record<string, unknown>;
    if (code === undefined && typeof obj.status === 'number') code = obj.status;
    if (code === undefined && typeof obj.code === 'number') code = obj.code;
    if (!status && typeof obj.status === 'string') status = obj.status;
  }

  return { code, status, message: message || current };
}

export function isTransientGeminiError(err: unknown): boolean {
  const { code, status, message } = extractUpstreamError(err);
  const combined = `${status || ''} ${message}`.toLowerCase();

  // Never retry explicit client/auth configuration errors
  if (code === 400 || code === 401 || code === 403 || code === 404) {
    return false;
  }
  if (
    combined.includes('api_key_invalid') ||
    combined.includes('permission_denied') ||
    combined.includes('clé api gemini absente') ||
    combined.includes('perday') ||
    combined.includes('per day')
  ) {
    return false;
  }

  if (code === 429 || code === 500 || code === 502 || code === 503 || code === 504) {
    return true;
  }

  return (
    combined.includes('unavailable') ||
    combined.includes('service unavailable') ||
    combined.includes('high demand') ||
    combined.includes('resource_exhausted') ||
    combined.includes('overloaded') ||
    combined.includes('try again later') ||
    combined.includes('deadline_exceeded') ||
    combined.includes('etimedout') ||
    combined.includes('econnreset') ||
    combined.includes('fetch failed') ||
    combined.includes('timeout') ||
    combined.includes('aborted')
  );
}

export function isDailyQuotaError(err: unknown): boolean {
  const { code, status, message } = extractUpstreamError(err);
  const combined = `${status || ''} ${message}`.toLowerCase();
  const hasDailySignal =
    combined.includes('perday') ||
    combined.includes('per day') ||
    combined.includes('requestsperday');
  return (
    hasDailySignal &&
    (code === 429 ||
      combined.includes('resource_exhausted') ||
      combined.includes('quota'))
  );
}

export function isQuotaError(err: unknown): boolean {
  const { code, status, message } = extractUpstreamError(err);
  const combined = `${status || ''} ${message}`.toLowerCase();
  return (
    code === 429 ||
    combined.includes('resource_exhausted') ||
    combined.includes('quota')
  );
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function withGeminiRetry<T>(
  fn: () => Promise<T>,
  options: { maxRetries?: number; baseDelayMs?: number } = {},
): Promise<T> {
  const maxRetries = options.maxRetries ?? 2;
  const baseDelayMs = options.baseDelayMs ?? 350;

  let lastError: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (attempt >= maxRetries || !isTransientGeminiError(err)) {
        throw err;
      }
      const jitter = Math.floor(Math.random() * 120);
      const delayMs = baseDelayMs * Math.pow(2, attempt) + jitter;
      await sleep(delayMs);
    }
  }

  throw lastError;
}

export function formatGeminiUserError(err: unknown): { status: number; message: string } {
  const { code, status, message } = extractUpstreamError(err);
  const combined = `${status || ''} ${message}`.toLowerCase();

  if (message.includes('Accès refusé')) {
    return { status: 403, message };
  }

  if (
    combined.includes('clé api gemini absente') ||
    combined.includes('api_key_invalid') ||
    combined.includes('api key not valid') ||
    code === 401 ||
    code === 403
  ) {
    return {
      status: 503,
      message:
        'Clé API Gemini absente ou invalide. Veuillez vérifier GEMINI_API_KEY dans Settings > Secrets.',
    };
  }

  if (
    code === 503 ||
    combined.includes('unavailable') ||
    combined.includes('high demand') ||
    combined.includes('overloaded')
  ) {
    return {
      status: 503,
      message:
        "Le service d'intelligence artificielle connaît une forte affluence momentanée (503). Veuillez réessayer dans quelques secondes.",
    };
  }

  if (code === 429 || combined.includes('resource_exhausted') || combined.includes('quota')) {
    return {
      status: 429,
      message:
        "Le quota temporaire du service d'intelligence artificielle est atteint (429). Veuillez patienter quelques instants avant de réessayer.",
    };
  }

  if (
    code === 500 ||
    code === 502 ||
    code === 504 ||
    combined.includes('deadline_exceeded') ||
    combined.includes('etimedout') ||
    combined.includes('econnreset') ||
    combined.includes('fetch failed')
  ) {
    return {
      status: 503,
      message:
        "Le service d'intelligence artificielle est momentanément indisponible. Veuillez réessayer dans un instant.",
    };
  }

  // Avoid leaking raw JSON blobs or verbose stack strings to the user
  if (!message || message.trim().startsWith('{') || message.trim().startsWith('[')) {
    return {
      status: 502,
      message:
        "Une erreur technique est survenue lors de l'appel au service d'intelligence artificielle. Veuillez réessayer.",
    };
  }

  return {
    status: code && code >= 400 && code < 600 ? code : 500,
    message,
  };
}
