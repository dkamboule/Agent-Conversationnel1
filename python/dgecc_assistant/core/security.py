"""Hachage scrypt, jetons JWT HMAC-SHA256, protection anti-injection et rate-limiting."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from .config import APP_CONFIG

RoleCode = Literal["ADMIN", "VALIDATOR", "EDITOR", "USER"]


@dataclass
class AuthPayload:
    sub: str
    email: str
    full_name: str
    role: RoleCode
    iat: int
    exp: int


def hash_password(password: str, existing_salt: Optional[str] = None) -> Tuple[str, str]:
    salt = existing_salt or secrets.token_hex(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt.encode("utf-8"),
        n=16384,
        r=8,
        p=1,
        dklen=64,
    )
    return derived.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    if not password or not stored_hash or not salt:
        return False
    try:
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt.encode("utf-8"),
            n=16384,
            r=8,
            p=1,
            dklen=64,
        )
        target = bytes.fromhex(stored_hash)
    except ValueError:
        return False
    if len(derived) != len(target):
        return False
    return hmac.compare_digest(derived, target)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def sign_token(sub: str, email: str, full_name: str, role: RoleCode) -> str:
    now = int(time.time())
    payload = {
        "sub": sub,
        "email": email,
        "fullName": full_name,
        "role": role,
        "iat": now,
        "exp": now + APP_CONFIG.token_ttl_seconds,
    }
    header_b64 = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8"))
    body_b64 = _b64url_encode(json.dumps(payload).encode("utf-8"))
    signing_input = f"{header_b64}.{body_b64}".encode("ascii")
    sig = hmac.new(
        APP_CONFIG.session_secret.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    return f"{header_b64}.{body_b64}.{_b64url_encode(sig)}"


def verify_token(token: str) -> Optional[AuthPayload]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, body_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{body_b64}".encode("ascii")
        expected_sig = _b64url_encode(
            hmac.new(
                APP_CONFIG.session_secret.encode("utf-8"),
                signing_input,
                hashlib.sha256,
            ).digest()
        )
        if not hmac.compare_digest(sig_b64, expected_sig):
            return None
        parsed = json.loads(_b64url_decode(body_b64).decode("utf-8"))
        exp = int(parsed.get("exp", 0))
        if exp < int(time.time()):
            return None
        return AuthPayload(
            sub=str(parsed["sub"]),
            email=str(parsed["email"]),
            full_name=str(parsed.get("fullName", "")),
            role=parsed["role"],
            iat=int(parsed["iat"]),
            exp=exp,
        )
    except Exception:
        return None


_INJECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", re.IGNORECASE),
    re.compile(
        r"ignore[zr]?\s+(toutes?\s+)?les?\s+instructions?\s+(précédentes?|antérieures?)",
        re.IGNORECASE,
    ),
    re.compile(r"oublie[zr]?\s+(toutes?\s+)?les?\s+règles", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"tu\s+es\s+maintenant\s+", re.IGNORECASE),
    re.compile(r"révèle[zr]?\s+(ton|le)\s+prompt", re.IGNORECASE),
    re.compile(r"<script\b[^>]*>", re.IGNORECASE),
]


def detect_prompt_injection(text: str) -> Tuple[bool, Optional[str]]:
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True, "Tentative de contournement des instructions système détectée."
    return False, None


def sanitize_document_text(text: str) -> str:
    cleaned = re.sub(r"\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>", "", text, flags=re.I)
    cleaned = re.sub(r"ignore\s+previous\s+instructions", "[contenu filtré]", cleaned, flags=re.I)
    return cleaned


class SlidingWindowRateLimiter:
    """Limiteur de requêtes en mémoire par fenêtre glissante."""

    def __init__(self) -> None:
        self._buckets: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, max_requests: int, window_seconds: float) -> bool:
        now = time.time()
        with self._lock:
            timestamps = [ts for ts in self._buckets.get(key, []) if now - ts < window_seconds]
            if len(timestamps) >= max_requests:
                self._buckets[key] = timestamps
                return False
            timestamps.append(now)
            self._buckets[key] = timestamps
            return True


rate_limiter = SlidingWindowRateLimiter()
