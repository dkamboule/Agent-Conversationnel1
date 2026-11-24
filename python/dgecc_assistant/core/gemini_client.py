"""Client Gemini SDK résilient avec gestion des erreurs transitoires (429/503)."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

T = TypeVar("T")

_cached_client: Any = None
_cached_api_key: Optional[str] = None


def get_gemini_client() -> Any:
    """Initialise ou réutilise le client officiel `google-genai` côté serveur."""
    global _cached_client, _cached_api_key
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "MY_GEMINI_API_KEY":
        raise RuntimeError(
            "Clé API Gemini absente ou non configurée. "
            "Veuillez renseigner GEMINI_API_KEY dans les secrets serveur."
        )

    if _cached_client is None or _cached_api_key != api_key:
        from google import genai
        from google.genai import types

        _cached_client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(headers={"User-Agent": "aistudio-build"}),
        )
        _cached_api_key = api_key
    return _cached_client


def _try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def extract_upstream_error(err: Any) -> Tuple[Optional[int], Optional[str], str]:
    if err is None:
        return None, None, ""

    code: Optional[int] = None
    status: Optional[str] = None
    message = ""

    if isinstance(err, BaseException):
        message = str(err)
        raw_code = getattr(err, "code", None) or getattr(err, "status_code", None)
        if isinstance(raw_code, int):
            code = raw_code
        raw_status = getattr(err, "status", None)
        if isinstance(raw_status, str):
            status = raw_status
    elif isinstance(err, str):
        message = err
    elif isinstance(err, dict):
        if isinstance(err.get("status"), int):
            code = err["status"]
        if isinstance(err.get("code"), int):
            code = err["code"]
        if isinstance(err.get("status"), str):
            status = err["status"]
        if isinstance(err.get("message"), str):
            message = err["message"]

    current = message.strip()
    for _ in range(3):
        idx = current.find("{")
        if idx == -1:
            break
        parsed = _try_parse_json(current[idx:])
        if not parsed:
            break
        inner = parsed["error"] if isinstance(parsed.get("error"), dict) else parsed
        if isinstance(inner.get("code"), int):
            code = inner["code"]
        if isinstance(inner.get("status"), str):
            status = inner["status"]
        if isinstance(inner.get("message"), str):
            current = inner["message"].strip()
            message = current
            continue
        break

    return code, status, (message or current)


def is_transient_gemini_error(err: Any) -> bool:
    code, status, message = extract_upstream_error(err)
    combined = f"{status or ''} {message}".lower()

    if code in (400, 401, 403, 404):
        return False
    if (
        "api_key_invalid" in combined
        or "permission_denied" in combined
        or "clé api gemini absente" in combined
    ):
        return False

    if code in (429, 500, 502, 503, 504):
        return True

    transient_markers = (
        "unavailable",
        "service unavailable",
        "high demand",
        "resource_exhausted",
        "overloaded",
        "try again later",
        "deadline_exceeded",
        "timed out",
        "connection reset",
    )
    return any(marker in combined for marker in transient_markers)


def with_gemini_retry(
    fn: Callable[[], T],
    max_retries: int = 2,
    base_delay_ms: int = 350,
) -> T:
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except BaseException as err:
            last_error = err
            if attempt >= max_retries or not is_transient_gemini_error(err):
                raise
            jitter = random.randint(0, 120)
            delay_s = (base_delay_ms * (2**attempt) + jitter) / 1000.0
            time.sleep(delay_s)
    assert last_error is not None
    raise last_error


def format_gemini_user_error(err: Any) -> Tuple[int, str]:
    code, status, message = extract_upstream_error(err)
    combined = f"{status or ''} {message}".lower()

    if "Accès refusé" in message:
        return 403, message

    if (
        "clé api gemini absente" in combined
        or "api_key_invalid" in combined
        or "api key not valid" in combined
        or code in (401, 403)
    ):
        return (
            503,
            "Clé API Gemini absente ou invalide. Veuillez vérifier GEMINI_API_KEY dans les secrets serveur.",
        )

    if (
        code == 503
        or "unavailable" in combined
        or "high demand" in combined
        or "overloaded" in combined
    ):
        return (
            503,
            "Le service d'intelligence artificielle connaît une forte affluence momentanée (503). Veuillez réessayer dans quelques secondes.",
        )

    if code == 429 or "resource_exhausted" in combined or "quota" in combined:
        return (
            429,
            "Le quota temporaire du service d'intelligence artificielle est atteint (429). Veuillez patienter quelques instants avant de réessayer.",
        )

    if (
        code in (500, 502, 504)
        or "deadline_exceeded" in combined
        or "timed out" in combined
        or "connection reset" in combined
    ):
        return (
            503,
            "Le service d'intelligence artificielle est momentanément indisponible. Veuillez réessayer dans un instant.",
        )

    stripped = message.strip()
    if not stripped or stripped.startswith("{") or stripped.startswith("["):
        return (
            502,
            "Une erreur technique est survenue lors de l'appel au service d'intelligence artificielle. Veuillez réessayer.",
        )

    return (code if code and 400 <= code < 600 else 500), message
