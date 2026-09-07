"""Configuration helpers for cross-vendor routing."""

from __future__ import annotations

import functools
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import ValidationError

from lemoncrow.core.capabilities.cross_vendor_routing_contract import (
    ROUTE_CONFIG_VERSION as ROUTE_CONFIG_VERSION,
)
from lemoncrow.core.capabilities.cross_vendor_routing_contract import (
    SUPPORTED_ROUTE_VENDORS as SUPPORTED_ROUTE_VENDORS,
)
from lemoncrow.core.capabilities.cross_vendor_routing_contract import (
    AgentMode as AgentMode,
)
from lemoncrow.core.capabilities.cross_vendor_routing_contract import (
    EditMode as EditMode,
)
from lemoncrow.core.capabilities.cross_vendor_routing_contract import (
    ReadMode as ReadMode,
)
from lemoncrow.core.capabilities.cross_vendor_routing_contract import (
    RouteConfig as RouteConfig,
)
from lemoncrow.core.capabilities.cross_vendor_routing_contract import (
    RouteConfigError as RouteConfigError,
)
from lemoncrow.core.foundation.paths import default_store_root

_VENDOR_ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "bedrock": ("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_BEARER_TOKEN_BEDROCK"),
    "vertex": ("VERTEXAI_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"),
    "azure": ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "ollama": ("OLLAMA_HOST",),  # Ollama uses base URL, not API key
    "together": ("TOGETHER_API_KEY",),
    "fireworks": ("FIREWORKS_API_KEY",),
    "zen": ("OPENCODE_API_KEY",),
}
_VENDOR_HOST_COMMANDS: dict[str, tuple[str, ...]] = {
    "anthropic": ("claude",),
    "openai": ("codex",),
    # No host-CLI runner is implemented for google (_PROVIDER_RUNNERS["google"] is
    # empty) -- an owned turn routed to "google" always executes over litellm's
    # direct Vertex/Gemini HTTP transport, which needs a real GOOGLE_API_KEY /
    # GEMINI_API_KEY / Vertex ADC. Treating agy/antigravity's mere presence as
    # "configured" let routes.yaml enable google with zero credentials, so an
    # auto-routed turn would pick it and fail with "Missing Gemini API key" even
    # though nothing ever offered the user a key. Unlike claude/codex (which do
    # have real runners for anthropic/openai), host-CLI presence is not a valid
    # signal for google -- only the env-var check below is.
    "google": (),
    "bedrock": (),
    "vertex": (),
    "azure": (),
    "openrouter": (),
    "groq": (),
    "mistral": (),
    "ollama": ("ollama",),  # detect local ollama
    "together": (),
    "fireworks": (),
    "zen": (),
}

# RouteConfig / RouteConfigError / mode aliases are re-exported from the open
# contract above (imported at module top); logic below stays compiled.


def route_config_path(root: Path | str | None = None) -> Path:
    base = Path(root).expanduser().resolve() if root is not None else default_store_root()
    return base / "route.yaml"


@functools.cache
def _ollama_reachable(base_url: str) -> bool:
    """Whether an Ollama daemon actually answers at *base_url*.

    An installed ``ollama`` binary is not a usable vendor on its own: routing a
    turn to a daemon that is not running fails the whole turn with a connection
    error. Probed once per process with a short timeout.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=0.3) as resp:
            return bool(200 <= resp.status < 300)
    except Exception:
        return False


def _ollama_available(source: Mapping[str, str]) -> bool:
    host = str(source.get("OLLAMA_HOST", "")).strip()
    if not host and shutil.which("ollama") is None:
        return False
    if host and not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return _ollama_reachable(host or "http://localhost:11434")


@functools.cache
def _detect_configured_vendors_cached() -> tuple[str, ...]:
    """Process-level cached vendor detection using os.environ (invariant for process lifetime)."""
    enabled: list[str] = []
    for vendor in SUPPORTED_ROUTE_VENDORS:
        if vendor == "zen":
            continue  # appended below once we know whether anything else is usable
        if vendor == "ollama":
            if _ollama_available(os.environ):
                enabled.append(vendor)
            continue
        has_env = any(str(os.environ.get(key, "")).strip() for key in _VENDOR_ENV_VARS[vendor])
        has_host_surface = any(shutil.which(command) is not None for command in _VENDOR_HOST_COMMANDS[vendor])
        if has_env or has_host_surface:
            enabled.append(vendor)
    if _zen_available(os.environ, any_other_vendor=bool(detect_api_key_vendors_without_zen())):
        enabled.append("zen")
    return tuple(enabled)


def detect_configured_vendors(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    if env is not None:
        # Custom env provided — bypass cache and compute directly.
        enabled: list[str] = []
        for vendor in SUPPORTED_ROUTE_VENDORS:
            if vendor == "zen":
                continue
            if vendor == "ollama":
                if _ollama_available(env):
                    enabled.append(vendor)
                continue
            has_env = any(str(env.get(key, "")).strip() for key in _VENDOR_ENV_VARS[vendor])
            has_host_surface = any(shutil.which(command) is not None for command in _VENDOR_HOST_COMMANDS[vendor])
            if has_env or has_host_surface:
                enabled.append(vendor)
        if _zen_available(env, any_other_vendor=bool(detect_api_key_vendors_without_zen(env))):
            enabled.append("zen")
        return tuple(enabled)
    return _detect_configured_vendors_cached()


def detect_api_key_vendors(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Return vendors reachable via an API key in the environment.

    Unlike :func:`detect_configured_vendors`, this ignores installed host CLIs:
    owned execution runs through the litellm/openai HTTP transports, which need
    a real API key, so a host-CLI subscription alone cannot execute an owned
    turn. Used to seed a default route config that only enables vendors that can
    actually run.
    """
    enabled = list(detect_api_key_vendors_without_zen(env))
    source = env if env is not None else os.environ
    if _zen_available(source, any_other_vendor=bool(enabled)):
        enabled.append("zen")
    return tuple(enabled)


def detect_api_key_vendors_without_zen(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """API-key vendors excluding Zen, whose availability depends on this result."""
    source = env if env is not None else os.environ
    enabled: list[str] = []
    for vendor in SUPPORTED_ROUTE_VENDORS:
        if vendor == "zen":
            continue
        if vendor == "ollama":
            if _ollama_available(source):
                enabled.append(vendor)
            continue
        if any(str(source.get(key, "")).strip() for key in _VENDOR_ENV_VARS[vendor]):
            enabled.append(vendor)
    return tuple(enabled)


def _zen_available(source: Mapping[str, str], *, any_other_vendor: bool) -> bool:
    """Whether OpenCode Zen should be offered as a route vendor.

    An explicit Zen credential always enables it. The keyless ``public`` tier is
    only enabled when nothing else can run: its models are free because the
    upstream vendors may train on the traffic, so it must never silently
    out-compete a vendor the user actually configured.
    """
    from lemoncrow.core.capabilities.providers.zen import has_account_key, public_tier_enabled

    if str(source.get("OPENCODE_API_KEY", "")).strip() or has_account_key():
        return True
    return not any_other_vendor and public_tier_enabled()


def load_route_config(root: Path | str | None = None, *, path: Path | str | None = None) -> RouteConfig:
    config_path = Path(path).expanduser().resolve() if path is not None else route_config_path(root)
    if not config_path.exists():
        raise RouteConfigError(f"route config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RouteConfigError(f"route config is not valid YAML: {config_path}") from exc
    if not isinstance(raw, dict):
        raise RouteConfigError(f"route config at {config_path} must be a mapping")
    try:
        config = RouteConfig.model_validate(raw)
    except ValidationError as exc:
        raise RouteConfigError(f"route config is invalid: {exc}") from exc
    if config.version != ROUTE_CONFIG_VERSION:
        raise RouteConfigError(f"unsupported route config version {config.version}; expected {ROUTE_CONFIG_VERSION}")
    return config


# Module-level cache for load_route_config_or_default.
# Key: (resolved_path_str, mtime_ns, file_size) — or (resolved_path_str, None, None) when absent.
# Value: RouteConfig | RouteConfigError  (we cache errors too so absent-file calls don't stat repeatedly)
_route_config_cache: dict[tuple[str, int | None, int | None], RouteConfig | RouteConfigError] = {}


def load_route_config_or_default(
    root: Path | str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    path: Path | str | None = None,
) -> RouteConfig:
    """Load ``route.yaml`` or synthesise a default from detected vendors.

    Owned routing should work out of the box: when no ``route.yaml`` has been
    written yet, build a low-risk config enabling every vendor reachable via an
    API key. Host-CLI-only vendors are intentionally excluded — owned execution
    runs over the litellm/openai HTTP transports and needs a real key, so a
    bare CLI subscription cannot execute a turn. Re-raises ``RouteConfigError``
    only when the file is genuinely missing *and* no API-key vendor is present,
    or when the file exists but is invalid (so real config mistakes are never
    silently masked).

    Results are cached keyed on (resolved_path, mtime_ns, file_size) so repeated
    calls within a session pay no I/O cost.  A custom ``env`` mapping bypasses
    the cache because the synthesised default depends on the caller-supplied env.
    """
    config_path = Path(path).expanduser().resolve() if path is not None else route_config_path(root)
    resolved = str(config_path)

    # Build cache key from file metadata (or sentinel when absent).
    if config_path.exists():
        stat = config_path.stat()
        cache_key: tuple[str, int | None, int | None] = (resolved, stat.st_mtime_ns, stat.st_size)
    else:
        cache_key = (resolved, None, None)

    # Only use cache when env is None (default os.environ path).
    if env is None and cache_key in _route_config_cache:
        cached = _route_config_cache[cache_key]
        if isinstance(cached, RouteConfigError):
            raise cached
        return cached

    # Cache miss (or custom env) — compute the result.
    try:
        result = load_route_config(root, path=path)
    except RouteConfigError as exc:
        if config_path.exists():
            # File present but invalid — cache and re-raise.
            if env is None:
                _route_config_cache[cache_key] = exc
            raise
        vendors = list(detect_api_key_vendors(env))
        if not vendors:
            if env is None:
                _route_config_cache[cache_key] = exc
            raise
        result = RouteConfig(enabled_vendors=vendors)

    if env is None:
        _route_config_cache[cache_key] = result
    return result


def save_route_config(
    root: Path | str | None = None,
    config: RouteConfig | None = None,
    *,
    path: Path | str | None = None,
) -> Path:
    if config is None:
        raise RouteConfigError("route config is required")
    config_path = Path(path).expanduser().resolve() if path is not None else route_config_path(root)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


__all__ = [
    "ROUTE_CONFIG_VERSION",
    "RouteConfig",
    "RouteConfigError",
    "detect_configured_vendors",
    "load_route_config",
    "route_config_path",
    "save_route_config",
]
