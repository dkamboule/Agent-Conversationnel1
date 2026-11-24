import type { NextFunction, Request, Response } from 'express';
import { verifyToken, type AuthPayload, type RoleCode } from '../core/security.ts';

export interface AuthenticatedRequest extends Request {
  authUser?: AuthPayload | null;
}

export function attachAuthUser(req: AuthenticatedRequest, _res: Response, next: NextFunction): void {
  const header = req.headers.authorization;
  if (header && header.startsWith('Bearer ')) {
    const token = header.slice(7).trim();
    req.authUser = verifyToken(token);
  } else {
    req.authUser = null;
  }
  next();
}

export function requireAuth(req: AuthenticatedRequest, res: Response, next: NextFunction): void {
  if (!req.authUser) {
    res.status(401).json({ error: 'Authentification requise pour accéder à cette ressource.' });
    return;
  }
  next();
}

export function requireRole(...allowedRoles: RoleCode[]) {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    if (!req.authUser) {
      res.status(401).json({ error: 'Authentification requise.' });
      return;
    }
    if (!allowedRoles.includes(req.authUser.role)) {
      res.status(403).json({
        error: `Privilèges insuffisants. Rôle requis : ${allowedRoles.join(' ou ')}.`,
      });
      return;
    }
    next();
  };
}

export function resolveOwner(req: AuthenticatedRequest): {
  userId: string | null;
  userEmail: string | null;
  guestSessionId: string | null;
} {
  if (req.authUser) {
    return {
      userId: req.authUser.sub,
      userEmail: req.authUser.email,
      guestSessionId: null,
    };
  }
  const sessionHeader = req.headers['x-dgecc-session'];
  const guestSessionId =
    typeof sessionHeader === 'string' && /^[a-zA-Z0-9_-]{8,128}$/.test(sessionHeader)
      ? sessionHeader
      : 'guest-default';

  return {
    userId: null,
    userEmail: null,
    guestSessionId,
  };
}
