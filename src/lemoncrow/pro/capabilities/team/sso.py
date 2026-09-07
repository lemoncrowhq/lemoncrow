"""Minimal Google OIDC stub for local workspace flows."""

from __future__ import annotations

import hashlib
import secrets
from urllib.parse import urlencode


def begin_google_oidc(email: str, *, redirect_uri: str, hosted_domain: str | None = None) -> dict[str, str]:
    state = secrets.token_urlsafe(12)
    query = {
        "client_id": "lemoncrow-local",
        "response_type": "code",
        "scope": "openid email profile",
        "redirect_uri": redirect_uri,
        "state": state,
        "login_hint": email,
    }
    if hosted_domain:
        query["hd"] = hosted_domain
    return {
        "state": state,
        "authorization_url": f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(query)}",
    }


def finish_google_oidc(code: str, *, state: str, email: str, hosted_domain: str | None = None) -> dict[str, str]:
    if not code.strip():
        raise ValueError("google auth code is required")
    digest = hashlib.sha256(f"{email}:{state}:{code}".encode()).hexdigest()
    payload = {
        "user_id": email.strip().lower(),
        "email": email.strip().lower(),
        "provider": "google",
        "state": state,
        "access_token": digest[:32],
        "id_token": digest[32:],
    }
    if hosted_domain:
        payload["hosted_domain"] = hosted_domain
    return payload
