"""Compiled, fail-closed savings-cap enforcement.

Production builds pin the issuer's Ed25519 public key in this mypyc-compiled
module. Access requires a valid v2 cap verdict bound to the local account,
device, and canonical plan. Absent, expired, malformed, copied, or mismatched
tokens are dormant immediately; there is no editable local or fresh-install
grace when the signing authority is configured.

A local meter remains only as a development fallback for builds with no pinned
key. Tests exercise that fallback by patching the key accessor rather than by
using CI/environment bypasses.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, NamedTuple

# Pinned Ed25519 server public key (hex, 32-byte raw). BAKED INTO THE .so AT
# RELEASE from the license-issuer's keypair. There is deliberately NO env
# override: one would let anyone point verification at their OWN key and
# self-sign an "under cap" verdict to bypass the cap. Rotation requires a client
# rebuild (recompile the .so with a new pinned key) — not a runtime setting.
_DEFAULT_PUBLIC_KEY_HEX = "30873bf4c34131f57b4b17d35da62079650ec7f2f95f01f0ecfc70c4a5df063a"


def _public_key_hex() -> str:
    return _DEFAULT_PUBLIC_KEY_HEX.strip()


def is_configured() -> bool:
    """Open-source runtime: there is no signing authority and no savings cap.

    Always ``False`` — nothing is ever gated or made dormant. The Ed25519 verify
    helpers below are retained only for backward-compatible imports/tests; the
    runtime never enforces a cap. See docs/maintenance-mode-transition.md."""
    return False


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def verify_cap_token(token: str, *, public_key_hex: str | None = None) -> dict[str, Any] | None:
    """Verify the Ed25519 signature and return the payload, or ``None``.

    Signature-only: does NOT check expiry (see :func:`cap_over_from_token`).
    Fail-safe: any malformed input / bad signature / missing key -> ``None``.
    """
    key_hex = (public_key_hex if public_key_hex is not None else _public_key_hex()).strip()
    if not key_hex or not token or "." not in token:
        return None
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        payload_b64, sig_b64 = token.split(".", 1)
        payload_bytes = _b64url_decode(payload_b64)
        signature = _b64url_decode(sig_b64)
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(key_hex))
        try:
            pub.verify(signature, payload_bytes)
        except InvalidSignature:
            return None
        payload = json.loads(payload_bytes)
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001 — any parse/crypto error = untrusted
        return None


_TOKEN_VERSION = 2
_MAX_TOKEN_TTL_SECONDS = 9 * 60 * 60
_CLOCK_SKEW_SECONDS = 5 * 60
_CANONICAL_PLANS = frozenset({"anonymous", "free", "lite", "pro", "enterprise"})


def _typed_token_payload(
    token: str | None,
    *,
    token_type: str,
    now: int,
    account_id: str,
    device_id: str,
    public_key_hex: str | None,
) -> dict[str, Any] | None:
    if not token or not account_id or not device_id:
        return None
    payload = verify_cap_token(token, public_key_hex=public_key_hex)
    if payload is None:
        return None
    if payload.get("v") != _TOKEN_VERSION or payload.get("typ") != token_type:
        return None
    if payload.get("account_id") != account_id or payload.get("device_id") != device_id:
        return None

    issued_at = payload.get("issued_at")
    expires_at = payload.get("expires_at")
    if type(issued_at) is not int or type(expires_at) is not int:
        return None
    if issued_at > now + _CLOCK_SKEW_SECONDS or now >= expires_at:
        return None
    if expires_at <= issued_at or expires_at - issued_at > _MAX_TOKEN_TTL_SECONDS:
        return None
    return payload


def _cap_payload(
    token: str | None,
    *,
    now: int,
    account_id: str,
    device_id: str,
    plan: str,
    public_key_hex: str | None = None,
) -> dict[str, Any] | None:
    """The verified ``cap``-type payload bound to *plan*, or None when untrusted.

    Shared by :func:`cap_over_from_token` (the boolean) and
    :func:`resolve_cap_verdict` (which also wants the server's signed dollar
    figures, ``monthly_savings_usd`` / ``cap_usd`` -- both embedded in this
    SAME payload by the issuer, see services/license-issuer/src/usage.ts::
    signVerdict). One verify, two readers -- never re-parse the token twice.
    """
    normalized_plan = plan.strip().lower()
    if normalized_plan not in _CANONICAL_PLANS:
        return None
    payload = _typed_token_payload(
        token,
        token_type="cap",
        now=now,
        account_id=account_id,
        device_id=device_id,
        public_key_hex=public_key_hex,
    )
    if payload is None or payload.get("plan") != normalized_plan:
        return None
    return payload


def cap_over_from_token(
    token: str | None,
    *,
    now: int,
    account_id: str,
    device_id: str,
    plan: str,
    public_key_hex: str | None = None,
) -> bool | None:
    """Return a bound signed cap decision, or None when untrusted."""

    payload = _cap_payload(
        token, now=now, account_id=account_id, device_id=device_id, plan=plan, public_key_hex=public_key_hex
    )
    if payload is None:
        return None
    over = payload.get("savings_over_cap")
    return over if type(over) is bool else None


def _cap_amount(payload: dict[str, Any], key: str) -> float | None:
    raw = payload.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    return float(raw)


def plan_from_token(
    token: str | None,
    *,
    now: int,
    account_id: str,
    device_id: str,
    public_key_hex: str | None = None,
) -> str | None:
    """Return a canonical account-and-device-bound signed plan."""

    payload = _typed_token_payload(
        token,
        token_type="plan",
        now=now,
        account_id=account_id,
        device_id=device_id,
        public_key_hex=public_key_hex,
    )
    if payload is None:
        return None
    plan = payload.get("plan")
    return plan if isinstance(plan, str) and plan in _CANONICAL_PLANS else None


def _cap_verdict_token(root: str | Path) -> str | None:
    """The server's signed cap verdict token, if the account has one."""
    from lemoncrow.core.capabilities.plugin_runtime import (
        _read_json,
        auth_state_path,
        subscription_state_path,
    )

    auth = _read_json(auth_state_path(root), {})
    sub = auth.get("subscriptionStatus") if isinstance(auth, dict) else None
    if not isinstance(sub, dict):
        sub = _read_json(subscription_state_path(root), {})
    tok = sub.get("capVerdictToken") if isinstance(sub, dict) else None
    return tok if isinstance(tok, str) and tok else None


def _local_device_id() -> str:
    from lemoncrow.core.capabilities.licensing import store

    return store.load_or_create_device_id()


def _local_device_hash() -> str:
    import hashlib

    return hashlib.sha256(_local_device_id().encode("utf-8")).hexdigest()


def _anonymous_device_hash() -> str:
    """Device hash the ANON verdict binds to -- the non-overridable machine id
    (``store.stable_machine_device_id``), so a signed anon verdict can't be
    replayed on another machine via the ``LEMONCROW_DEVICE_ID`` override.

    Distinct from :func:`_local_device_hash` (which honors the override for the
    AUTHENTICATED path, where CI/containers forward an already-bound device).
    """
    import hashlib

    from lemoncrow.core.capabilities.licensing import store

    return hashlib.sha256(store.stable_machine_device_id().encode("utf-8")).hexdigest()


def _anonymous_cap_identity(token: str | None) -> tuple[str, str, str] | None:
    if not token:
        return None
    payload = verify_cap_token(token)
    if payload is None or payload.get("v") != _TOKEN_VERSION or payload.get("typ") != "cap":
        return None
    account_id = payload.get("account_id")
    device_id = payload.get("device_id")
    plan = payload.get("plan")
    if (
        not isinstance(account_id, str)
        or not account_id.startswith("anon:")
        or not isinstance(device_id, str)
        or device_id != _anonymous_device_hash()
        or plan not in {"anonymous", "free"}
    ):
        return None
    # "free" is accepted for the short lifetime of verdicts minted by older
    # servers during rollout; new anonymous verdicts use "anonymous".
    return account_id, device_id, plan


class CapVerdict(NamedTuple):
    """The one true cap/dormancy decision.

    Every surface that reports whether LemonCrow is "active" or "dormant" —
    MCP tool-gating, ``lc account cap``, ``lc savings``, the statusline — must
    resolve through :func:`resolve_cap_verdict` (or the ``cap_exhausted``
    convenience wrapper below), never recompute its own opinion from the
    local usage meter. Two independent opinions is how a CLI can say "active"
    while the MCP server has already hidden every tool for the same account.
    """

    dormant: bool
    verified: bool
    """True when backed by an actual signature-checked cap-verdict token.
    False means either no trustworthy token was available (fail-closed
    default) or this is the unsigned dev-build local-meter fallback."""
    plan: str | None
    """Canonical plan from a verified source; None when unverified."""
    reason: str
    """Machine-readable cause: "signed" | "signed_anonymous" | "no_token" |
    "local_fallback" | "gate_error"."""
    server_saved_usd: float | None = None
    """The account's signed, server-accumulated savings total ($), lifted
    straight out of the verified token payload -- None when unverified.
    Self-heals a wiped local ledger: the server owns this number (see
    services/license-issuer/src/usage.ts::accumulateUsage, keyed by
    account_id, never a client-writable file), so deleting every local
    ``sessions/**/savings.jsonl`` cannot zero it -- the NEXT verified verdict
    (session start, MCP self-heal, periodic reconciler) restores the true
    figure here regardless of what the local ledger says."""
    server_cap_usd: float | None = None
    """The plan's cap ($) from that same verified payload; None when
    unverified or the plan is uncapped."""


def resolve_cap_verdict(root: str | Path) -> CapVerdict:
    """Resolve the single authoritative cap/dormancy decision for *root*.

    Trust order: a real account's signed verdict token, then an anonymous
    signed verdict token, then (only when no signing key is pinned — never
    true in a real build) the local unsigned usage meter as a development
    fallback. Absent, expired, malformed, copied, or mismatched tokens are
    dormant immediately when a key is pinned: there is no editable local or
    fresh-install grace once the signing authority is configured.
    """
    # Open-source runtime: no savings cap, no dormancy, ever. Every surface that
    # asks "active or dormant?" resolves to active/uncapped, with no token, no
    # account, and no network. Retained as the single chokepoint so callers keep
    # a stable API. See docs/maintenance-mode-transition.md.
    return CapVerdict(
        dormant=False,
        verified=True,
        plan="free",
        reason="oss",
        server_saved_usd=None,
        server_cap_usd=None,
    )


def cap_exhausted(root: str | Path) -> bool:
    """Whether the bound server verdict says the savings cap is exhausted.

    Thin convenience wrapper over :func:`resolve_cap_verdict` for callers
    that only need the boolean (the MCP server's tool-gating, mostly).
    """
    return resolve_cap_verdict(root).dormant
