"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from lemoncrow.core.runtime import LemonCrowRuntimeCore
    from lemoncrow.infra.storage.bundle import StoreBundle


@pytest.fixture(autouse=True)
def _isolate_workspace_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Isolate tests from host workspace env vars and default runtime roots."""
    for env_var in (
        "LEMONCROW_WORKSPACE_ROOT",
        "CLAUDE_WORKSPACE_ROOT",
        "CURSOR_WORKSPACE_ROOT",
        "VSCODE_CWD",
        "LEMONCROW_LESSONS_ROOT",
        "LEMONCROW_STORE_ROOT",
        "LEMONCROW_MEM_ROOT",
        # Live credentials/device identity must never leak into a test: the
        # suite has to behave identically on a logged-in pro machine, in CI,
        # and inside a benchmark container. Signed-path tests build their own
        # keypair-signed fixtures; live checks are explicit opt-in tests.
        "LEMONCROW_AUTH_TOKEN",
        "LEMONCROW_DEVICE_ID",
    ):
        monkeypatch.delenv(env_var, raising=False)
    isolated_root = tmp_path / ".lemoncrow"
    monkeypatch.setenv("LEMONCROW_ROOT", str(isolated_root))
    monkeypatch.setenv("LEMONCROW_STORE_ROOT", str(isolated_root))
    # Complete the isolation: point the workspace root at tmp_path too. Without
    # this, _workspace_root() falls through to os.getcwd() (the real repo), so
    # the new read/projection workspace-confinement rejects files tests create
    # under tmp_path. Tests that need a specific workspace set it themselves.
    monkeypatch.setenv("LEMONCROW_WORKSPACE_ROOT", str(tmp_path))
    # The agent running pytest may itself be Codex/Cursor/etc. Pin ordinary
    # tests to the product's documented fallback host; host-detection tests
    # explicitly override or remove this value.
    monkeypatch.setenv("LEMONCROW_AGENT", "claude")
    # Point host-transcript discovery at an isolated, empty dir. Savings/recall/
    # statusline code falls back to the developer's real ~/.claude/projects when
    # CLAUDE_CONFIG_DIR is unset, so an in-process test would replay every real
    # host session -- tens of seconds and non-hermetic. The dir is left absent so
    # scans short-circuit on `projects.is_dir()`; tests needing transcripts set
    # CLAUDE_CONFIG_DIR to their own fixture dir, which overrides this.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "_isolated_claude_home"))
    yield


@pytest.fixture(autouse=True)
def _development_cap_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the compiled gate's documented no-key development mode in unit tests.

    Production wheels keep their pinned key. Tests that exercise signatures
    explicitly replace this mock with their own generated keypair; ordinary
    tool-handler tests remain hermetic and never need a network-issued token.
    """

    from lemoncrow.pro.capabilities import licensing_gate

    monkeypatch.setattr(licensing_gate, "_public_key_hex", lambda: "")


@pytest.fixture(autouse=True)
def _no_network_sync() -> Iterator[None]:
    """Block all outbound sync_usage calls so no test ever hits lemoncrow.beseam.com."""
    with patch("lemoncrow.core.service.sync.sync_usage", return_value=True):
        yield


@pytest.fixture(autouse=True)
def _no_usage_report_network() -> Iterator[None]:
    """Block the usage-report POST fallback so no test hits the live cap
    endpoints (``/api/usage/report[-anon]``). CLI flows like ``init
    --no-login`` and login/logout force a verdict mint through this path;
    unpatched, every such test would attempt a real 5s-timeout POST.
    Tests that exercise reporting pass their own ``http_post``."""
    with patch(
        "lemoncrow.core.capabilities.licensing.usage_report._default_post",
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def _no_external_search_in_tests(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate tests from zoekt and semantic search backends.

    * ``LEMONCROW_ZOEKT_LOC_THRESHOLD=100000000`` — tiny test repos (< 10 k LOC)
      never route through zoekt (which requires a built index the test fixtures
      don't provide).  The real default is 1 (all repos).
    * ``LEMONCROW_CODE_AUTOSYNC=0`` — one-shot test engines should not start
      background autosync workers. Tests that exercise autosync opt in explicitly.

    Tests that explicitly exercise zoekt, autosync, or semantic behaviour should
    override these env vars via their own ``monkeypatch.setenv`` call or engine
    constructor arguments.
    """
    from lemoncrow.infra.embeddings.factory import make_code_embedder

    make_code_embedder.cache_clear()
    monkeypatch.setenv("LEMONCROW_ZOEKT_LOC_THRESHOLD", "100000000")
    monkeypatch.setenv("LEMONCROW_CODE_AUTOSYNC", "0")
    monkeypatch.setenv("LEMONCROW_CODE_EMBEDDER", "null")
    yield
    make_code_embedder.cache_clear()


@pytest.fixture(autouse=True)
def _no_ollama() -> Iterator[None]:
    """Block real Ollama calls so no test waits on a local LLM.

    Patches _ollama_module() — the single gateway all ollama_client functions
    (summarize, chat) use — so the mock works even for callers that did
    ``from lemoncrow.infra.internal_llm.ollama_client import summarize``.

    Tests that explicitly need LLM behaviour should override via monkeypatch.
    """
    from lemoncrow.infra.internal_llm import OllamaUnavailable

    with patch(
        "lemoncrow.infra.internal_llm.ollama_client._ollama_module",
        side_effect=OllamaUnavailable("ollama blocked in tests"),
    ):
        yield


@pytest.fixture(scope="session")
def retrieval_eval_runtime(tmp_path_factory: pytest.TempPathFactory) -> LemonCrowRuntimeCore:
    """Initialize lemoncrow runtime and seed blocks once per session for retrieval evaluation."""
    from tests.core.test_retriever_eval import _ensure_eval_blocks_exist, _init_runtime

    # Note: Using tmp_path_factory to get a persistent session directory
    root = tmp_path_factory.mktemp("retrieval_eval_session")
    runtime = _init_runtime(root)
    _ensure_eval_blocks_exist(runtime)
    return runtime


@pytest.fixture()
def store(tmp_path: Path) -> StoreBundle:
    """A StoreBundle over six fresh, physically-split SQLite files.

    Access the store you need explicitly: ``store.history``, ``store.knowledge``,
    ``store.lessons``, ``store.jobs``, ``store.memory``, ``store.telemetry``.
    """
    from lemoncrow.infra.storage.bundle import build_sqlite_store_bundle

    root = tmp_path / "lemoncrow"
    store = build_sqlite_store_bundle(root)
    store.init()
    return store


# --- Engine availability ----------------------------------------------------
# The engine is the entire ``src/lemoncrow/pro/`` package. Its source is
# published (Apache-2.0, like the rest of the tree) and present in every
# checkout, so every test runs against the real modules and nothing below fires.
#
# The guard below is a defensive net for an install that lacks the engine
# entirely (e.g. a wheel built with ``pro`` stripped): pro-importing test
# modules are deselected at collection time so pytest reports skips instead of
# erroring on import.


def _pro_importable() -> bool:
    """True when the compiled engine (``lemoncrow.pro``) can be imported."""
    import importlib.util

    try:
        return importlib.util.find_spec("lemoncrow.pro") is not None
    except (ImportError, ValueError):
        return False


_PRO_AVAILABLE = _pro_importable()


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    """Skip collecting test modules that import the engine when it is absent.

    No-op when ``lemoncrow.pro`` is importable (always, in this private repo).
    On a source-only public checkout it prevents a wall of collection-time
    ImportErrors by declining to collect any ``test_*.py`` that references
    ``lemoncrow.pro``. Self-maintaining: no static file list to keep in sync.
    """
    if _PRO_AVAILABLE:
        return None
    if collection_path.suffix != ".py" or not collection_path.name.startswith("test_"):
        return None
    try:
        source = collection_path.read_text(encoding="utf-8")
    except OSError:
        return None
    return "lemoncrow.pro" in source or None
