"""Tests for the consolidated MCP contract."""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from lemoncrow.core.capabilities.licensing import entitlements
from lemoncrow.core.environment import HIDDEN_LLM_TOOLS
from lemoncrow.core.service.bootstrap_context import build_bootstrap_plan, persist_bootstrap_plan
from lemoncrow.core.service.jobs import JOB_BOOTSTRAP_CONTEXT
from lemoncrow.gateway.adapters import mcp_server
from lemoncrow.gateway.adapters.mcp_server import TOOLS, _handle, tool_smart_edit
from lemoncrow.gateway.cli import cli
from lemoncrow.infra.code_intel.astgrep import (
    AstGrepToolUnavailable,
    PatternMatch,
    PatternRewriteResult,
    PatternSearchResult,
)
from lemoncrow.infra.storage.factory import create_store, make_memory_store
from lemoncrow.pro.capabilities.code_context import CodeContextEngine
from tests.helpers import grant_oauth_pro, init_store_at

# Single-primary retrieval surface: `explore` (ranked source + call-graph
# relations + blast-radius in one call) + `read`, plus edit/bash/web_fetch.
# `grep`, `relations`, `search`, `memory`, `sql`, `codemod` are registered but
# hidden from agents (grep/relations stay callable as escape hatch / drill-in).
EXPECTED_TOOLS = {
    "read",
    "edit",
    "code_search",
    "bash",
    "web_fetch",
}


def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    req: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    }
    resp = _handle(req)
    assert isinstance(resp, dict)
    return resp


def _result(resp: dict[str, Any]) -> Any:
    assert "result" in resp, resp
    text = resp["result"]["content"][0]["text"]
    # Clean edit renders "ok" (no ranges) or "applied path:line[, ...]" (the minimal
    # orientation echo); normalize both to a dict so callers can assert structurally.
    if text == "ok":
        return {}
    if text.startswith("applied "):
        return {"applied": text[len("applied ") :].split(", ")}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _op_result(render_name: str, op_fn: Any, **kwargs: Any) -> Any:
    """Mirror _handle's render path for a direct _op_* call: returns rendered
    markdown when a code renderer applies, else the raw payload dict."""
    mcp_server._tool_call_rendered_text.value = None
    payload = op_fn(**kwargs)
    rendered = mcp_server.render_tool_result_text(render_name, payload)
    return rendered if rendered is not None else payload


def _preindex(repo_root: str | Path) -> None:
    """Explicitly index a repo for deterministic code-context tests.

    Also indexes workspace siblings if ``.lemoncrow/workspace.toml`` exists.
    The gateway conftest disables the background autosync worker so tests that
    need a populated index build it explicitly via ``_op_index``.
    """
    import tomllib

    mcp_server._op_index(repo_root=str(repo_root), force=True)
    workspace_config = Path(repo_root) / ".lemoncrow" / "workspace.toml"
    if workspace_config.exists():
        config = tomllib.loads(workspace_config.read_text())
        for entry in config.get("workspace", {}).get("repos", []):
            entry_path = (Path(repo_root) / entry["path"]).resolve()
            if entry_path.resolve() != Path(repo_root).resolve():
                mcp_server._op_index(repo_root=str(entry_path), force=True)


def _mock_client(return_values: dict[str, dict[str, Any]]) -> MagicMock:
    client = MagicMock()
    for method_name, retval in return_values.items():
        getattr(client, method_name).return_value = retval
    return client


def _write_bootstrap_fixture_repo(root: Path) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "src" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "app.py").write_text(
        "from src.worker import run_worker\n\ndef main() -> str:\n    return run_worker()\n",
        encoding="utf-8",
    )
    (root / "src" / "worker.py").write_text(
        "def run_worker() -> str:\n    return 'ready'\n",
        encoding="utf-8",
    )
    (root / "scripts" / "cli.py").write_text(
        "from src.app import main\n\ndef cli() -> str:\n    return main()\n",
        encoding="utf-8",
    )


def _write_workspace_fixture_repo(root: Path, *, module_name: str, class_name: str = "SharedConfig") -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "config.py").write_text(
        f"class {class_name}:\n    SOURCE = '{module_name}'\n",
        encoding="utf-8",
    )


def _write_workspace_fixture_config(workspace_root: Path, sibling_root: Path) -> None:
    (workspace_root / ".lemoncrow").mkdir(parents=True, exist_ok=True)
    (workspace_root / ".lemoncrow" / "workspace.toml").write_text(
        "\n".join(
            [
                "[workspace]",
                'id = "fixture-workspace"',
                "",
                "[[workspace.repos]]",
                'name = "lemoncrow"',
                'path = "."',
                "",
                "[[workspace.repos]]",
                'name = "billing"',
                f'path = "{os.path.relpath(sibling_root, workspace_root)}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def store_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / ".lemoncrow"
    init_store_at(str(root))
    monkeypatch.setenv("LEMONCROW_ROOT", str(root))
    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("LEMONCROW_MEMORY_BACKEND", "sqlite")
    # Trace persistence tests exercise the local ledger path; an ambient
    # LEMONCROW_SERVICE_URL would force remote dispatch and skip _current_ledger.
    monkeypatch.delenv("LEMONCROW_SERVICE_URL", raising=False)
    mcp_server._ledger._current_ledger = None
    mcp_server._ledger._realtime_ctx = None
    mcp_server._remote_client = _mock_client(
        {
            "get_context": {"context": "Here are the relevant procedures.", "run_ledger": []},
            "rescue_failure": {
                "rescue": "Try a narrower reproduction.",
                "analysis": "repeat failure",
            },
            "record_trace": {"id": "trace-123", "event_recorded": True},
            "run_rubric_gate": {"status": "pass"},
        }
    )
    return root


def test_initialize_returns_server_info() -> None:
    resp = _handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
        }
    )
    assert resp is not None
    assert resp["result"]["serverInfo"]["name"] == "lemoncrow"
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    # Server-level steering: injected into the host system prompt by every MCP
    # client automatically — the surface that reaches hosts and subagents that
    # never see LemonCrow's persona files.
    instructions = resp["result"]["instructions"]
    assert "code_search" in instructions
    assert "grep" in instructions and "read" in instructions


def test_notifications_initialized_returns_none() -> None:
    resp = _handle({"jsonrpc": "2.0", "id": None, "method": "notifications/initialized", "params": {}})
    assert resp is None


def test_tools_list_returns_exact_public_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "full")
    resp = _handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert resp is not None
    names = {tool["name"] for tool in resp["result"]["tools"]}
    assert names == EXPECTED_TOOLS
    assert EXPECTED_TOOLS | HIDDEN_LLM_TOOLS == set(TOOLS)


def test_tools_list_hides_internal_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEMONCROW_MCP_TOOL_PROFILE", "full")
    resp = _handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert resp is not None
    tools = resp["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert names == EXPECTED_TOOLS
    assert not (names & HIDDEN_LLM_TOOLS)
    assert "read" in names
    assert all("passive" not in tool["description"] for tool in tools if tool["name"] in EXPECTED_TOOLS)


def test_memory_tool_call_works_without_dev_mode(store_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = store_root
    monkeypatch.delenv("LEMONCROW_SERVICE_URL", raising=False)
    mcp_server._remote_client = None
    resp = _call(
        "memory",
        {
            "op": "store_fact",
            "agent_id": "lemoncrow:non-dev",
            "subject": "test",
            "fact": "Memory should be active in non-dev mode.",
            "citations": 'Test: "direct"',
            "reason": "Verifying non-dev memory works.",
            "scope": "user",
        },
    )
    payload = _result(resp)
    assert payload["fact"] == "Memory should be active in non-dev mode."

    recalled = _result(
        _call(
            "memory",
            {
                "op": "recall",
                "query": "Memory should be active in non-dev mode.",
            },
        )
    )
    assert "passages" in recalled


def test_cli_tools_list_hides_internal_tools_even_with_legacy_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()

    stable = runner.invoke(cli, ["--root", str(tmp_path / ".lemoncrow"), "tools", "list"])
    assert stable.exit_code == 0, stable.output
    assert set(stable.output.splitlines()) == EXPECTED_TOOLS

    dev = runner.invoke(cli, ["--root", str(tmp_path / ".lemoncrow"), "tools", "list", "--dev"])
    assert dev.exit_code == 0, dev.output
    assert set(dev.output.splitlines()) == EXPECTED_TOOLS


def test_cli_tools_call_invokes_stable_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "--root",
            str(tmp_path / ".lemoncrow"),
            "tools",
            "call",
            "compact",
            "--args",
            "{}",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "tokens_freed" in payload


def test_tools_list_each_entry_has_schema() -> None:
    resp = _handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert resp is not None
    for tool in resp["result"]["tools"]:
        assert tool["name"]
        assert isinstance(tool.get("inputSchema"), dict)


def test_tools_list_grep_is_lean_and_relations_is_the_drill_in() -> None:
    # grep is a lean regex tool that rides call-graph COUNTS inline on definition
    # matches; the dedicated `relations` tool expands a count into the list.
    # `search` stays registered but hidden (semantic-only).
    assert "search" in TOOLS
    assert "search" in HIDDEN_LLM_TOOLS
    assert "relations" in TOOLS
    grep_tool = TOOLS["grep"]
    grep_props = grep_tool["inputSchema"]["properties"]
    # grep advertises the inline counts but carries no relation/symbol/map params.
    assert "counts" in grep_tool["description"].lower()
    assert "relation" not in grep_props
    assert "symbol" not in grep_props
    assert "seed_files" not in grep_props
    mode_desc = grep_props["mode"]["description"]
    for mode_name in ("with_content", "ranked_map", "paths_only", "count_only"):
        assert mode_name in mode_desc, mode_desc
    assert "file_path" not in grep_props
    assert ":Lx-Ly" in grep_props["path"]["description"]
    # relations is single-purpose: symbol + kind.
    rel_props = TOOLS["relations"]["inputSchema"]["properties"]
    assert "symbol" in rel_props
    assert "kind" in rel_props


def test_tools_list_grep_schema_covers_native_mode() -> None:
    grep_tool = TOOLS["grep"]
    properties = grep_tool["inputSchema"]["properties"]

    assert "regex" in grep_tool["description"].lower()
    assert "path" in properties
    assert "file_path" not in properties
    assert "regex" in properties
    assert "summary" in properties


def test_grep_accepts_single_glob_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare string for file_glob_patterns is coerced to a one-element list.

    The model frequently reaches for a single glob string; accepting it avoids a
    schema-validation rejection against the array type. A list passes through.
    """
    captured: dict[str, Any] = {}

    def _fake_run_native_grep(**kwargs: Any) -> dict[str, Any]:
        captured.clear()
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(mcp_server, "_run_native_grep", _fake_run_native_grep)
    handler = TOOLS["grep"]["handler"]

    handler({"content_regex": "x", "file_glob_patterns": "src/**/*.py"})
    assert captured["file_glob_patterns"] == ["src/**/*.py"]

    handler({"content_regex": "x", "file_glob_patterns": ["a", "b"]})
    assert captured["file_glob_patterns"] == ["a", "b"]


def test_grep_param_aliases_reach_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Old (content_regex) and new (regex) arg names both reach the handler.

    The published schema only shows the short names, but the alias layer remaps
    deprecated names before validation; when both are passed, the new name wins.
    """
    captured: dict[str, Any] = {}

    def _fake_run_native_grep(**kwargs: Any) -> dict[str, Any]:
        captured.clear()
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(mcp_server, "_run_native_grep", _fake_run_native_grep)
    handler = TOOLS["grep"]["handler"]

    # New name only.
    handler({"regex": "needle", "before": 2, "after": 3, "i": True})
    assert captured["content_regex"] == "needle"
    assert captured["lines_before"] == 2
    assert captured["lines_after"] == 3
    assert captured["ignore_case"] is True

    # Old alias only — remapped to the new param before the handler runs.
    handler({"content_regex": "legacy", "lines_before": 1, "ignore_case": True})
    assert captured["content_regex"] == "legacy"
    assert captured["lines_before"] == 1
    assert captured["ignore_case"] is True

    # Both present — the new name wins.
    handler({"regex": "winner", "content_regex": "loser"})
    assert captured["content_regex"] == "winner"


def test_tools_list_edit_schema_documents_flat_shape() -> None:
    edit_tool = TOOLS["edit"]
    schema = edit_tool["inputSchema"]
    edits_schema = schema["properties"]["edits"]
    item_props = edits_schema["items"]["properties"]

    assert schema["required"] == ["edits"]
    assert "anyOf" not in edits_schema["items"]
    assert set(item_props) == {"path", "old", "new", "replace"}
    assert edits_schema["items"].get("additionalProperties") is False
    path_desc = item_props["path"]["description"]
    assert ":Lx" in path_desc and ":Lx-Ly" in path_desc


def test_tools_list_memory_schema_describes_ops_and_required_fields() -> None:
    memory_tool = TOOLS["memory"]
    properties = memory_tool["inputSchema"]["properties"]

    assert "fact storage/voting and recall" in memory_tool["description"]
    assert "store_fact" in properties["op"]["description"]
    assert "vote_fact" in properties["op"]["description"]
    assert "need query" in properties["op"]["description"]
    assert "query used by recall" in properties["query"]["description"].lower()
    assert "subject" in properties
    assert "fact" in properties
    assert "citations" in properties
    assert "direction" in properties
    assert "label" not in properties
    assert "session_id" not in properties
    assert "expected_version" not in properties


def test_unknown_method_returns_error() -> None:
    resp = _handle({"jsonrpc": "2.0", "id": 3, "method": "unknown/method", "params": {}})
    assert resp is not None
    assert resp["error"]["code"] == -32601


def test_unknown_tool_returns_error() -> None:
    resp = _call("does_not_exist", {})
    assert "error" in resp
    assert "unknown tool" in resp["error"]["message"]


def test_get_context_can_include_folded_state(store_root: Path) -> None:
    resp = _call(
        "context",
        {"task": "Fix publish regression", "include_run_ledger": True},
    )
    payload = _result(resp)
    assert isinstance(payload.get("context"), str)
    assert "run_ledger" in payload


@pytest.mark.slow
def test_context_enqueues_single_bootstrap_job_for_cold_repo(
    store_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = Path(os.environ["CLAUDE_WORKSPACE_ROOT"])
    _write_bootstrap_fixture_repo(workspace_root)
    mcp_server._reset_runtime_cache_for_testing()
    monkeypatch.setattr(mcp_server, "_run_worker_tick_safe", lambda root: None)

    first = mcp_server.tool_get_context({"task": "Map the repo entry points"})
    second = mcp_server.tool_get_context({"task": "Map the repo entry points"})

    store = create_store(store_root)
    store.init()
    jobs = [
        job
        for job in store.jobs.list_jobs(job_type=JOB_BOOTSTRAP_CONTEXT, limit=20)
        if job["status"] in {"pending", "running"}
    ]

    assert len(jobs) == 1
    assert first["bootstrap"]["queued"] is True
    assert second["bootstrap"]["queued"] is False


def test_context_worker_tick_persists_bootstrap_blocks_without_blocking_initial_response(
    store_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEMONCROW_DEV_MODE", "1")
    workspace_root = Path(os.environ["CLAUDE_WORKSPACE_ROOT"])
    monkeypatch.delenv("LEMONCROW_SERVICE_URL", raising=False)
    mcp_server._remote_client = None
    _write_bootstrap_fixture_repo(workspace_root)
    mcp_server._reset_runtime_cache_for_testing()

    payload = mcp_server.tool_get_context({"task": "Warm the repository context"})

    assert "Repository bootstrap" not in payload["context"]
    mcp_server._run_worker_tick_safe(store_root)

    plan = build_bootstrap_plan(workspace_root)
    bootstrap_count = 0
    for _ in range(6):
        import time

        time.sleep(0.1)
        blocks = make_memory_store(store_root).list_pinned_blocks(plan.agent_id)
        bootstrap_count = len([block for block in blocks if block.label.startswith(f"bootstrap/{plan.repo_id}/")])
        if bootstrap_count == 4:
            break
        mcp_server._run_worker_tick_safe(store_root)

    assert bootstrap_count == 4
    assert bootstrap_count == 4


def test_context_reuses_bootstrap_blocks_instead_of_enqueuing_duplicate_work(
    store_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_root = Path(os.environ["CLAUDE_WORKSPACE_ROOT"])
    monkeypatch.delenv("LEMONCROW_SERVICE_URL", raising=False)
    mcp_server._remote_client = None
    _write_bootstrap_fixture_repo(workspace_root)
    mcp_server._reset_runtime_cache_for_testing()

    mcp_server.tool_get_context({"task": "Warm the repository context"})
    mcp_server._run_worker_tick_safe(store_root)
    mcp_server._reset_runtime_cache_for_testing()
    payload = mcp_server.tool_get_context({"task": "Warm the repository context"})

    store = create_store(store_root)
    store.init()
    jobs = store.jobs.list_jobs(job_type=JOB_BOOTSTRAP_CONTEXT, limit=20)

    assert len(jobs) == 1
    assert payload["bootstrap"]["status"] in {"warm", "warming"}
    if payload["bootstrap"]["status"] == "warm":
        assert "Repository bootstrap" in payload["context"]


def test_context_injects_preseeded_bootstrap_blocks_without_recomputing(
    store_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_root = Path(os.environ["CLAUDE_WORKSPACE_ROOT"])
    monkeypatch.delenv("LEMONCROW_SERVICE_URL", raising=False)
    mcp_server._remote_client = None
    _write_bootstrap_fixture_repo(workspace_root)
    memory_store = make_memory_store(store_root)
    persist_bootstrap_plan(workspace_root, memory_store)
    mcp_server._reset_runtime_cache_for_testing()

    payload = mcp_server.tool_get_context({"task": "Use the warmed bootstrap state"})

    store = create_store(store_root)
    store.init()
    assert store.jobs.list_jobs(job_type=JOB_BOOTSTRAP_CONTEXT, limit=20) == []
    assert payload["bootstrap"]["status"] == "warm"
    assert "architecture-sketch" in payload["context"]


def test_context_pull_threads_keywords_and_excluded_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    grant_oauth_pro(monkeypatch)
    seen: dict[str, Any] = {}

    def fake_pull(self: Any, subtask: Any) -> Any:
        seen["subtask"] = subtask
        from lemoncrow.pro.capabilities.scoped_context import ScopedContext

        return ScopedContext(
            chunks=[],
            rationale="ok",
            excluded=[],
            trace_id="trace",
            total_tokens=0,
            dropped_for_budget=0,
        )

    monkeypatch.setattr(mcp_server, "_code_context_engine", lambda root: object())
    monkeypatch.setattr(
        "lemoncrow.pro.capabilities.scoped_context.ScopedContextCapability.pull",
        fake_pull,
    )

    payload = mcp_server.tool_get_context(
        {
            "task": "fix auth flow",
            "mode": "pull",
            "files": ["src/auth.py"],
            "keywords": ["auth", "login"],
            "excluded_paths": ["src/legacy"],
            "token_budget": 321,
        }
    )

    subtask = seen["subtask"]
    assert subtask.description == "fix auth flow"
    assert subtask.affected_paths == ["src/auth.py"]
    assert subtask.keywords == ["auth", "login"]
    assert subtask.excluded_paths == ["src/legacy"]
    assert subtask.budget_tokens == 321
    assert payload["rationale"] == "ok"
    entitlements.reload()


def test_context_pull_reuses_cached_scoped_context(monkeypatch: pytest.MonkeyPatch) -> None:
    grant_oauth_pro(monkeypatch)

    class _FakePullRecord:
        def __init__(self, file_path: str, symbol_name: str) -> None:
            self.file_path = file_path
            self.symbol_name = symbol_name
            self.kind = "function"
            self.language = "python"
            self.qualified_name = symbol_name
            self.signature = f"def {symbol_name}(): ..."
            self.snippet = "return 1"
            self.score = 0.9

    class _FakePullEngine:
        def __init__(self) -> None:
            self.index_version = 0

        def _current_index_version(self) -> int:
            return self.index_version

        def search_symbols(
            self,
            query: str,
            *,
            limit: int = 20,
            mode: str = "auto",
            snippet: str = "head",
            file_glob: str | None = None,
            **_: object,
        ) -> list[_FakePullRecord]:
            ignored = (query, limit, mode, snippet)
            assert ignored
            records = [_FakePullRecord("src/auth.py", "auth_flow")]
            if file_glob is None:
                return records
            return [record for record in records if record.file_path == file_glob]

    mcp_server._reset_runtime_cache_for_testing()
    monkeypatch.setattr(mcp_server, "_code_context_engine", lambda repo_root=".": _FakePullEngine())

    first = mcp_server.tool_get_context(
        {"task": "fix auth flow", "mode": "pull", "files": ["src/auth.py"], "token_budget": 400}
    )
    second = mcp_server.tool_get_context(
        {"task": "fix auth flow", "mode": "pull", "files": ["src/auth.py"], "token_budget": 400}
    )

    assert first["provenance"] == "fresh"
    assert second["provenance"] == "cached"
    entitlements.reload()


def test_rescue_failure_returns_procedure(store_root: Path) -> None:
    _ = store_root
    payload = _result(
        _call(
            "rescue",
            {
                "task": "Run tests",
                "error": "pytest AssertionError",
                "recent_actions": ["run pytest", "run pytest"],
            },
        )
    )
    assert "rescue" in payload
    assert "analysis" in payload


def test_record_trace_accepts_monitor_event_payload(store_root: Path) -> None:
    _ = store_root
    payload = _result(
        _call(
            "trace",
            {
                "agent": "codex",
                "domain": "coding",
                "task": "Fix failing tests",
                "status": "partial",
                "event_type": "monitor.warning",
                "event_payload": {"message": "saw repeated command"},
            },
        )
    )
    assert "trace_id" in payload
    assert payload["event_recorded"] is True


def test_record_trace_persists_structured_workflow_progress(store_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = store_root
    monkeypatch.setattr(mcp_server, "_remote_client", None)
    mcp_server._ledger._current_ledger = None
    payload = _result(
        _call(
            "trace",
            {
                "agent": "copilot",
                "domain": "coding",
                "task": "Track plan progress",
                "status": "partial",
                "event_type": "plan_review",
                "event_payload": {
                    "workflow_step": "review",
                    "review_decision": "revise",
                    "plan_id": "02-01",
                },
            },
        )
    )
    assert "trace_id" in payload
    assert payload["event_recorded"] is True
    assert mcp_server._ledger._current_ledger is not None
    snapshot = mcp_server._ledger._current_ledger.snapshot()
    assert snapshot["plan_review"] == {
        "workflow_step": "review",
        "review_decision": "revise",
        "plan_id": "02-01",
    }


@pytest.mark.parametrize("review_decision", ["approve", "rerun"])
def test_record_trace_preserves_plan_review_receipt_for_other_decisions(
    store_root: Path, monkeypatch: pytest.MonkeyPatch, review_decision: str
) -> None:
    _ = store_root
    monkeypatch.setattr(mcp_server, "_remote_client", None)
    mcp_server._ledger._current_ledger = None
    payload = _result(
        _call(
            "trace",
            {
                "agent": "copilot",
                "domain": "coding",
                "task": "Track plan review",
                "status": "partial",
                "event_type": "plan_review",
                "event_payload": {
                    "workflow_step": "review",
                    "review_decision": review_decision,
                    "plan_id": "02-01",
                },
            },
        )
    )

    assert "trace_id" in payload
    assert payload["event_recorded"] is True
    assert mcp_server._ledger._current_ledger is not None
    assert mcp_server._ledger._current_ledger.snapshot()["plan_review"]["review_decision"] == review_decision


def test_record_trace_persists_task_progress_workflow_event(store_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = store_root
    monkeypatch.setattr(mcp_server, "_remote_client", None)
    mcp_server._ledger._current_ledger = None
    payload = _result(
        _call(
            "trace",
            {
                "agent": "copilot",
                "domain": "coding",
                "task": "Track task progress",
                "status": "partial",
                "event_type": "task_progress",
                "event_payload": {
                    "workflow_step": "execute",
                    "task_id": "02-02/task-1",
                    "completed_tasks": 2,
                    "remaining_tasks": 1,
                },
            },
        )
    )

    assert "trace_id" in payload
    assert payload["event_recorded"] is True
    assert mcp_server._ledger._current_ledger is not None
    assert mcp_server._ledger._current_ledger.snapshot()["task_progress"] == {
        "workflow_step": "execute",
        "task_id": "02-02/task-1",
        "completed_tasks": 2,
        "remaining_tasks": 1,
    }


def test_run_rubric_gate_pass(store_root: Path) -> None:
    _ = store_root
    payload = _result(
        _call(
            "verify",
            {
                "rubric_id": "rubric_state_change_safety",
                "checks": {
                    "canonical_identifier_used": True,
                    "pre_change_state_captured": True,
                    "read_after_write_completed": True,
                    "observed_state_matches_intent": True,
                    "rollback_plan_available": True,
                    "user_visible_surface_checked": True,
                },
            },
        )
    )
    assert payload["status"] == "pass"


def test_compact_session_call_returns_summary(store_root: Path) -> None:
    _ = store_root
    payload = _result(_call("compact", {}))
    assert "tokens_freed" in payload
    assert "prompt_block" in payload


def test_compact_auto_gate_requires_boundary_and_turns(store_root: Path) -> None:
    _ = store_root
    led = mcp_server._get_ledger()
    led.token_count = 160_000
    for idx in range(16):
        led.record("agent_message", f"working turn {idx}", {"idx": idx})

    waiting = mcp_server._compact_advise()
    assert waiting["should_advise"] is True
    assert waiting["should_compact"] is False
    assert waiting["task_boundary_detected"] is False

    led.record_test("pytest", passed=True, detail="tests passed")
    ready = mcp_server._compact_advise()
    assert ready["should_auto_compact"] is True
    assert ready["should_compact"] is True
    assert ready["task_boundary_detected"] is True


def test_compact_high_utilisation_bypasses_turns_gate(store_root: Path) -> None:
    # Five huge turns push utilisation to >=90% before the 15-turn gate is met.
    # The high-utilisation override should fire auto-compact at a task boundary
    # even though turn_count < AUTO_COMPACT_MIN_TURNS.
    _ = store_root
    led = mcp_server._get_ledger()
    led.token_count = 181_000  # 90.5% of 200k
    for idx in range(5):
        led.record("agent_message", f"dense turn {idx}", {"idx": idx})
    led.record_test("pytest", passed=True, detail="all green")

    result = mcp_server._compact_advise()
    assert result["turn_count"] < mcp_server.AUTO_COMPACT_MIN_TURNS
    assert result["should_auto_compact"] is True
    assert "override" in result["reason"] or "auto-compact threshold" in result["reason"]


def test_compact_handover_writes_markdown(store_root: Path) -> None:
    root = store_root
    led = mcp_server._get_ledger()
    led.session_id = "handover-session"
    led.task = "Finish a large refactor"
    led.token_count = 190_000
    led.record_file_event("src/app.py", "edit", diff="--- a/src/app.py\n+++ b/src/app.py\n")

    payload = mcp_server._compact_advise()

    assert payload["should_handover"] is True
    assert payload["handover_file"]
    handover_path = Path(payload["handover_file"])
    from lemoncrow.core.foundation.paths import session_dir

    assert handover_path == session_dir(root, "claude", "handover-session") / "HANDOVER.md"
    assert "Session Handover" in handover_path.read_text(encoding="utf-8")


def test_model_recommendation_emitted_before_tool_dispatch(store_root: Path) -> None:
    _ = store_root
    _result(_call("compact", {}))

    led = mcp_server._get_ledger()
    recommendations = [event for event in led.events if event.kind == "model_recommendation"]
    assert recommendations
    assert recommendations[-1].payload["tool_name"] == "compact"
    assert recommendations[-1].payload["tier"] in {"cheap", "medium", "expensive"}
    assert recommendations[-1].payload["lever"] == "model_routing"
    assert recommendations[-1].payload["tokens_saved"] == 0
    assert recommendations[-1].payload["cost_saved_usd"] >= 0


def test_model_recommendation_fallback_records_route_decision(
    monkeypatch: pytest.MonkeyPatch, store_root: Path
) -> None:
    from lemoncrow.infra.runtime.run_ledger import RunLedger
    from lemoncrow.pro.capabilities.cross_vendor_routing.configuration import RouteConfigError

    def fail_recommend(*_: object, **__: object) -> dict[str, object]:
        raise RouteConfigError("disabled")

    monkeypatch.setattr(
        "lemoncrow.pro.capabilities.cross_vendor_routing.router.CrossVendorRouter.recommend",
        fail_recommend,
    )
    ledger = RunLedger(session_id="route-fallback", root=store_root)

    payload = mcp_server._emit_model_recommendation("read", {}, ledger)
    route_decisions = [event for event in ledger.events if event.kind == "route_decision"]

    assert payload["kind"] == "model_recommendation"
    assert route_decisions
    assert route_decisions[-1].payload["kind"] == "route_decision"


def test_compact_session_op_emits_session_compaction_savings(monkeypatch: pytest.MonkeyPatch, store_root: Path) -> None:
    _ = store_root
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(mcp_server, "_append_live_savings_event", lambda event: events.append(event))
    led = mcp_server._get_ledger()
    led.token_count = 48_000
    for idx in range(4):
        led.record("agent_message", f"working turn {idx}", {"idx": idx})

    payload = mcp_server._compress_context()

    session_events = [event for event in events if event.get("kind") == "session_compaction"]
    assert session_events
    assert session_events[-1]["lever"] == "session_compaction"
    assert session_events[-1]["tokens_saved"] > 0
    assert session_events[-1]["cost_saved_usd"] >= 0
    assert payload["tokens_freed"] == session_events[-1]["tokens_saved"]
    assert payload["cost_saved_usd"] == session_events[-1]["cost_saved_usd"]


def test_compact_advise_emits_session_compaction_savings_when_auto_compacting(
    monkeypatch: pytest.MonkeyPatch, store_root: Path
) -> None:
    _ = store_root
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(mcp_server, "_append_live_savings_event", lambda event: events.append(event))
    led = mcp_server._get_ledger()
    led.token_count = 160_000
    for idx in range(16):
        led.record("agent_message", f"working turn {idx}", {"idx": idx})
    led.record_test("pytest", passed=True, detail="tests passed")

    payload = mcp_server._compact_advise()

    session_events = [event for event in events if event.get("kind") == "session_compaction"]
    assert payload["should_compact"] is True
    assert session_events
    assert session_events[-1]["trigger"] == "compact_advise"
    assert session_events[-1]["tokens_saved"] == payload["tokens_freed"]
    assert session_events[-1]["cost_saved_usd"] == payload["cost_saved_usd"]


def test_detect_agent_supports_all_cli_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    for host in ("claude", "codex", "copilot", "opencode", "lemoncode", "antigravity"):
        monkeypatch.setenv("LEMONCROW_AGENT", host)
        assert mcp_server._detect_agent() == host
        monkeypatch.delenv("LEMONCROW_AGENT", raising=False)


def test_detect_host_prefers_lemoncode_over_opencode(monkeypatch: pytest.MonkeyPatch) -> None:
    """LemonCode forks OpenCode, so a LemonCode session must never be labelled
    ``opencode`` -- its env vars are checked first."""
    from lemoncrow.core.foundation.paths import detect_host

    for var in ("LEMONCROW_AGENT", "CLAUDE_CODE", "CODEX_SESSION_ID", "CODEX_CLI"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LEMONCODE_SESSION_ID", "ses_1")
    assert detect_host() == "lemoncode"

    monkeypatch.setenv("OPENCODE_SESSION_ID", "ses_2")
    assert detect_host() == "lemoncode"

    monkeypatch.delenv("LEMONCODE_SESSION_ID")
    assert detect_host() == "opencode"

    monkeypatch.delenv("OPENCODE_SESSION_ID")
    monkeypatch.setenv("LEMONCODE_CLI", "1")
    assert detect_host() == "lemoncode"


def test_smart_read_and_search_surfaces(store_root: Path, tmp_path: Path) -> None:
    _ = store_root
    target = tmp_path / "sample.py"
    target.write_text("def alpha():\n    return 'needle'\n", encoding="utf-8")

    read_payload = _result(_call("read", {"path": str(target)}))
    assert "def alpha()" in read_payload
    assert "needle" in read_payload

    # `search` stays callable by name (hidden semantic tool) for the embedding path.
    search_payload = _result(_call("search", {"query": "needle", "path": str(tmp_path)}))
    assert "sample.py" in json.dumps(search_payload)

    # The `relations` drill-in tool routes a symbol's call-graph relation. (This
    # tmp repo isn't indexed, so the symbol may be absent -- we only assert the
    # tool is registered and dispatches cleanly, not that it finds `alpha`.)
    assert "relations" in TOOLS
    relations_resp = _call("relations", {"symbol": "alpha", "kind": "self"})
    assert "result" in relations_resp or "error" in relations_resp

    grep_payload = _result(_call("grep", {"path": str(target), "content_regex": "needle"}))
    assert grep_payload
    assert "_meta" not in grep_payload

    legacy_payload = _result(_call("grep", {"path": str(target), "content_regex": "needle", "include_meta": True}))
    assert "sample.py" in legacy_payload


def test_smart_read_batch_accepts_string_paths(store_root: Path, tmp_path: Path) -> None:
    """Batch read must accept plain string paths, dict specs, and a mix of both.

    Regression: `files` previously required `list[dict]`, so the natural
    `read(files=["a.py", "b.py"])` call failed Pydantic validation with a
    list/dict type error before reaching the handler.
    """
    _ = store_root
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("alpha_val = 1\n", encoding="utf-8")
    b.write_text("beta_val = 2\n", encoding="utf-8")

    # Plain strings. Reaching a non-error result at all proves the list[str]
    # input passed Pydantic validation; both files must be present.
    payload = _result(_call("read", {"files": [str(a), str(b)]}))
    assert "alpha_val" in payload
    assert "beta_val" in payload

    # Mixed strings and dict specs in one batch.
    mixed = _result(_call("read", {"files": [str(a), {"path": str(b), "range": "1-1"}]}))
    assert "alpha_val" in mixed
    assert "beta_val" in mixed


def test_smart_read_batch_honors_top_level_full(store_root: Path, tmp_path: Path) -> None:
    """A top-level ``full=True`` must apply to every batched file.

    Regression: the batch loop read ``expand`` only from each per-file spec
    (``spec.get("full", False)``), silently dropping a top-level
    ``full=True``. Plain-string entries therefore fell back to the >200-LOC
    outline projection (bodies omitted) even though the caller asked for full
    bodies. All prior ``expand`` coverage used single-path reads, which take a
    different code path, so the batch gap went untested.
    """
    _ = store_root
    big = tmp_path / "big_module.py"
    # >500 LOC so the default projection is outline (bodies omitted). The marker
    # lives inside a function body, which outline drops and expand keeps.
    # Use long function bodies (not module-level constants) so the outline omits
    # 75%+ of the source and passes the _outline_saves_enough guard.
    body = ["def head():"]
    body += [f"    x_{i} = {i}" for i in range(510)]
    body += ["    return x_0", ""]
    body += ["def carries_marker():", "    leaf = 'UNIQUE_BODY_TOKEN'", "    return leaf", ""]
    big.write_text("\n".join(body), encoding="utf-8")

    # Without expand: outline projection, the in-body marker is omitted.
    outline = _result(_call("read", {"files": [str(big)]}))
    assert "UNIQUE_BODY_TOKEN" not in outline

    # Top-level full=True must reach every plain-string batch entry.
    expanded = _result(_call("read", {"files": [str(big)], "full": True}))
    assert "UNIQUE_BODY_TOKEN" in expanded

    # A per-file expand still works and overrides the top-level default.
    # Use a second file (same content, different path) so the response text
    # differs from the prior `expanded` call and dedup does not fire.
    big2 = tmp_path / "big_module2.py"
    big2.write_text(big.read_text(encoding="utf-8"), encoding="utf-8")
    per_file = _result(_call("read", {"files": [{"path": str(big2), "full": True}]}))
    assert "UNIQUE_BODY_TOKEN" in per_file


def test_smart_read_batch_honors_top_level_max_lines(store_root: Path, tmp_path: Path) -> None:
    """A top-level ``max_lines`` must apply to every batched file.

    Same bug class as the ``expand`` drop: the batch loop read ``max_lines``
    only from each per-file spec (``spec.get("max_lines")``), discarding a
    top-level ``max_lines``. A caller capping every file in a batch silently
    got the default projection instead of the head-summary cap.
    """
    _ = store_root
    big = tmp_path / "big_module.py"
    big.write_text("\n".join(f"line_{i} = {i}" for i in range(300)), encoding="utf-8")

    # Top-level max_lines must reach each batched file -> summary (head-cap) mode.
    capped = mcp_server.tool_smart_read({"files": [str(big)], "max_lines": 3})
    assert capped["files"][0].get("mode") == "summary"

    # Without it, the same large file is not summary-capped (proves the cap came
    # from the top-level arg, not the file size).
    plain = mcp_server.tool_smart_read({"files": [str(big)]})
    assert plain["files"][0].get("mode") != "summary"


def test_scope_search_matches_to_range_filters_snippets() -> None:
    payload: dict[str, Any] = {
        "matches": [
            {"path": "a.py", "snippets": [{"line_start": 1, "line_end": 5}, {"line_start": 80, "line_end": 90}]},
            {"path": "b.py", "snippets": [{"line_start": 200, "line_end": 210}]},
            {"path": "c.py"},  # no snippet line data -> cannot filter, kept
        ],
        "match_paths": ["a.py", "b.py", "c.py"],
    }
    mcp_server._scope_search_matches_to_range(payload, (1, 50))
    assert [m["path"] for m in payload["matches"]] == ["a.py", "c.py"]
    a = next(m for m in payload["matches"] if m["path"] == "a.py")
    assert a["snippets"] == [{"line_start": 1, "line_end": 5}]
    assert payload["match_paths"] == ["a.py", "c.py"]


def test_smart_edit_surface_applies_patch(store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = store_root
    monkeypatch.chdir(tmp_path)
    target = Path("edit.txt")
    target.write_text("hello world", encoding="utf-8")

    # Through the dispatcher a clean exact edit echoes the minimal applied range,
    # change confirmed on disk.
    payload = _result(
        _call(
            "edit",
            {
                "edits": [
                    {
                        "path": str(target),
                        "old_string": "world",
                        "new_string": "lemoncrow",
                    }
                ]
            },
        )
    )
    assert payload == {"applied": ["edit.txt:1"]}
    assert target.read_text(encoding="utf-8") == "hello lemoncrow"


def test_smart_edit_compacts_hunks_by_path(store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = store_root
    monkeypatch.chdir(tmp_path)
    first = Path("first.txt")
    second = Path("second.txt")
    first.write_text("one\ntwo\nthree\n", encoding="utf-8")
    second.write_text("alpha\nbeta\n", encoding="utf-8")

    # A clean multi-file edit is success-silent on its body, but the cross-file
    # savings credit survives on the structured handler return (the dispatcher
    # reads calls_saved for content[].saved). The compact `applied` formatting is
    # exercised directly in test_compact_applied_entries_groups_by_path.
    payload = tool_smart_edit(
        {
            "post_edit_hooks": False,
            "edits": [
                {"path": str(first), "old_string": "one", "new_string": "ONE"},
                {"path": str(first), "old_string": "three", "new_string": "THREE"},
                {"path": str(second), "old_string": "alpha\nbeta", "new_string": "ALPHA\nBETA"},
            ],
        }
    )

    assert payload["applied"] == ["first.txt:1,3", "second.txt:1-2"]
    # 3 hunks but only 2 distinct files: built-in MultiEdit already batches the
    # two same-file hunks, so the honest cross-file saving is distinct_files - 1.
    assert payload["calls_saved"] == 1
    assert first.read_text(encoding="utf-8") == "ONE\ntwo\nTHREE\n"
    assert second.read_text(encoding="utf-8") == "ALPHA\nBETA\n"


def test_smart_edit_same_file_hunks_credit_no_calls(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple hunks in ONE file are not a saving — MultiEdit already batches them."""
    _ = store_root
    monkeypatch.chdir(tmp_path)
    target = Path("only.txt")
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    payload = tool_smart_edit(
        {
            "post_edit_hooks": False,
            "edits": [
                {"path": str(target), "old_string": "one", "new_string": "ONE"},
                {"path": str(target), "old_string": "two", "new_string": "TWO"},
                {"path": str(target), "old_string": "three", "new_string": "THREE"},
            ],
        }
    )

    # Single file => no cross-file saving; clean edit echoes the minimal range.
    assert payload["applied"] == ["only.txt:1,2,3"]
    assert payload.get("calls_saved", 0) == 0
    assert target.read_text(encoding="utf-8") == "ONE\nTWO\nTHREE\n"


def test_smart_edit_cross_file_credit_matches_distinct_files(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One hunk each across N files => calls_saved == N - 1."""
    _ = store_root
    monkeypatch.chdir(tmp_path)
    files = [Path(f"f{i}.txt") for i in range(3)]
    for f in files:
        f.write_text("target\n", encoding="utf-8")

    payload = tool_smart_edit(
        {
            "post_edit_hooks": False,
            "edits": [{"path": str(f), "old_string": "target", "new_string": "DONE"} for f in files],
        }
    )

    assert payload["calls_saved"] == len(files) - 1


def test_bash_omitted_tokens_saved_caps_at_vanilla_baseline() -> None:
    """Bash trim credit is measured against vanilla CC's own ~30k-char Bash cap,
    not the raw firehose: a multi-MB build log is not millions of tokens saved."""
    # Under the vanilla cap: the full omission is credited.
    assert mcp_server._bash_omitted_tokens_saved({"stdout": "x" * 1_000, "stderr": ""}, 4_000) == 1_000
    # Firehose: the naive side is capped at what vanilla would have shown.
    assert (
        mcp_server._bash_omitted_tokens_saved({"stdout": "x" * 8_000, "stderr": ""}, 10_000_000)
        == (30_000 - 8_000) // 4
    )
    # Already showing more than vanilla ever would: no credit.
    assert mcp_server._bash_omitted_tokens_saved({"stdout": "x" * 40_000, "stderr": ""}, 5_000) == 0
    # Nothing omitted: no credit.
    assert mcp_server._bash_omitted_tokens_saved({"stdout": "x"}, 0) == 0


def test_trimmed_tokens_saved_caps_at_host_inline_guard() -> None:
    """Dispatcher trim credit (spill/compact/truncate) caps the naive side at the
    host's inline MCP-output guard: anything larger would have been dumped to a
    file the model never pays for, so it was never a real context cost."""
    cap = mcp_server._HOST_INLINE_RESULT_CHARS
    assert mcp_server._trimmed_tokens_saved(10_000, 2_000) == 8_000 // 4
    assert mcp_server._trimmed_tokens_saved(cap * 10, 2_000) == (cap - 2_000) // 4
    # Nothing trimmed, or final text already over the cap -> no credit.
    assert mcp_server._trimmed_tokens_saved(2_000, 2_000) == 0
    assert mcp_server._trimmed_tokens_saved(cap * 10, cap + 1) == 0


def test_finish_code_result_credits_distinct_files_not_items() -> None:
    """20 symbols across 3 files ~= one grep + 3 reads avoided, not 19 calls."""
    items = [{"name": f"sym{i}", "path": f"src/f{i % 3}.py"} for i in range(20)]
    assert mcp_server._finish_code_result({"items": items})["calls_saved"] == 3
    # Items without file info credit only the single locate scan they replaced.
    assert mcp_server._finish_code_result({"routes": [{"method": "GET"}, {"method": "POST"}]})["calls_saved"] == 1
    # Single-file results credit the grep+read they replaced (2 calls - 1 made).
    assert mcp_server._finish_code_result({"items": [{"path": "a.py"}]})["calls_saved"] == 1
    # Empty lists still credit nothing.
    assert "calls_saved" not in mcp_server._finish_code_result({"items": []})
    # An explicit handler-set credit is never overwritten.
    assert mcp_server._finish_code_result({"items": items, "calls_saved": 7})["calls_saved"] == 7


def test_finish_code_result_counterfactual_token_credit(tmp_path: Path) -> None:
    """Surfaced files credit capped vanilla reads minus returned bytes — max
    with any engine packing credit, never the sum."""
    big = tmp_path / "big.py"
    big.write_text("x" * 40_000, encoding="utf-8")
    small = tmp_path / "small.py"
    small.write_text("y" * 4_000, encoding="utf-8")
    items = [{"name": "a", "path": str(big)}, {"name": "b", "path": str(small)}]

    result = mcp_server._finish_code_result({"items": items})
    returned = len(json.dumps({"items": items, "calls_saved": 2}, default=str))
    assert result["tokens_saved"] == (44_000 - returned) // 4

    # A larger engine-stamped packing credit is preserved (max, not sum).
    stamped = mcp_server._finish_code_result({"items": list(items), "tokens_saved": 999_999})
    assert stamped["tokens_saved"] == 999_999

    # Per-file cap: a giant file cannot inflate the credit past the vanilla dump.
    giant = tmp_path / "giant.py"
    giant.write_text("z" * 500_000, encoding="utf-8")
    capped_items = [{"name": "g", "path": str(giant)}, {"name": "a", "path": str(big)}]
    capped = mcp_server._finish_code_result({"items": capped_items})
    assert capped["tokens_saved"] <= (mcp_server._VANILLA_READ_FILE_CAP_CHARS + 40_000) // 4

    # Unreadable paths contribute nothing (no counterfactual read to avoid).
    ghost = mcp_server._finish_code_result({"items": [{"path": "nope/a.py"}, {"path": "nope/b.py"}]})
    assert "tokens_saved" not in ghost


def test_smart_read_batch_credits_calls_saved(store_root: Path, tmp_path: Path) -> None:
    """N files in one read call replace N single-file read calls => N - 1 saved;
    errored entries earned nothing and are excluded from the credit."""
    _ = store_root
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("alpha_val = 1\n", encoding="utf-8")
    b.write_text("beta_val = 2\n", encoding="utf-8")

    payload = mcp_server.tool_smart_read({"files": [str(a), str(b), str(tmp_path / "missing.py")]})
    assert payload["calls_saved"] == 1  # 2 ok entries -> 1 avoided call

    single = mcp_server.tool_smart_read({"files": [str(a)]})
    assert "calls_saved" not in single


def test_compact_applied_entries_groups_by_path() -> None:
    """Compaction groups same-path hunks and buckets "kind" entries by kind.

    This formatting only reaches the model on a LOUD result (clean exact edits are
    silenced), so it is verified directly on the helper rather than via a clean
    edit's dispatched body.
    """
    from lemoncrow.gateway.adapters.mcp_server import _compact_applied_entries

    entries = [
        {"path": "first.txt", "hunks": [{"line_start": 1, "line_end": 1}]},
        {"path": "first.txt", "hunks": [{"line_start": 3, "line_end": 3}]},
        {"path": "second.txt", "hunks": [{"line_start": 1, "line_end": 2}]},
        {"path": "sym.py", "kind": "symbol"},
        {"path": "nb.ipynb", "kind": "notebook"},
        {"path": "whole.txt", "kind": "replace"},
        {
            "path": "p.py",
            "kind": "projection",
            "projection_kind": "minified",
            "hunks": [{"line_start": 3, "line_end": 4}],
        },
    ]
    compact = _compact_applied_entries(entries)
    assert "first.txt:1,3" in compact
    assert "second.txt:1-2" in compact
    # "kind" entries with nothing beyond path+kind bucket as a bare path list,
    # grouped one entry per kind rather than one raw dict per file.
    assert {"symbol": ["sym.py"]} in compact
    assert {"notebook": ["nb.ipynb"]} in compact
    # "replace" reads as its past-tense outcome, not the raw kind value.
    assert {"replaced": ["whole.txt"]} in compact
    # A richer "kind" entry (extra fields beyond path) keeps those fields,
    # grouped as a dict under the kind label instead of a bare path.
    assert {
        "projection": [{"path": "p.py", "projection_kind": "minified", "hunks": [{"line_start": 3, "line_end": 4}]}]
    } in compact


def test_smart_edit_blocks_test_assertion_removal(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deleting an assertion from an existing test is the reward-hack signal:
    # roll back and surface a counterexample (the detector trips only on weakening).
    _ = store_root
    monkeypatch.chdir(tmp_path)
    target = Path("tests/test_parser.py")
    target.parent.mkdir()
    target.write_text("def test_x():\n    assert compute() == 5\n    assert other() == 9\n", encoding="utf-8")

    payload = _result(
        _call(
            "edit",
            {
                "edits": [
                    {
                        "file_path": str(target),
                        "old_string": "    assert compute() == 5\n    assert other() == 9\n",
                        "new_string": "    assert other() == 9\n",
                    }
                ],
                "post_edit_hooks": False,
            },
        )
    )

    assert payload["rolled_back"] is True
    # `writes` is no longer emitted (atomic edits are all-or-nothing; the count is
    # pure noise). The rollback is signalled by rolled_back/test_weakening.
    assert "writes" not in payload
    assert payload["test_weakening"][0]["path"] == "tests/test_parser.py"
    assert "assertion" in payload["test_weakening"][0]["reason"]
    assert "assert compute() == 5" in target.read_text(encoding="utf-8")


def test_smart_edit_blocks_test_skip_addition(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = store_root
    monkeypatch.chdir(tmp_path)
    target = Path("tests/test_parser.py")
    target.parent.mkdir()
    target.write_text("def test_x():\n    assert compute() == 5\n", encoding="utf-8")

    payload = _result(
        _call(
            "edit",
            {
                "edits": [
                    {
                        "file_path": str(target),
                        "old_string": "def test_x():\n",
                        "new_string": "@pytest.mark.skip\ndef test_x():\n",
                    }
                ],
                "post_edit_hooks": False,
            },
        )
    )

    assert payload["rolled_back"] is True
    assert "skip/xfail" in payload["test_weakening"][0]["reason"]
    assert "@pytest.mark.skip" not in target.read_text(encoding="utf-8")


def test_smart_edit_allows_additive_test_edit(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Adding a new assertion to an existing test passes freely -- no friction.
    _ = store_root
    monkeypatch.chdir(tmp_path)
    target = Path("tests/test_parser.py")
    target.parent.mkdir()
    target.write_text("def test_x():\n    assert compute() == 5\n", encoding="utf-8")

    payload = _result(
        _call(
            "edit",
            {
                "edits": [
                    {
                        "file_path": str(target),
                        "old_string": "    assert compute() == 5\n",
                        "new_string": "    assert compute() == 5\n    assert compute() != 0\n",
                    }
                ],
                "post_edit_hooks": False,
            },
        )
    )

    assert payload.get("rolled_back") is not True
    assert "test_weakening" not in payload
    assert "assert compute() != 0" in target.read_text(encoding="utf-8")


def test_smart_edit_allows_assertion_value_change(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In-place assertion modification (changing an expected value) nets to zero
    # assertions and passes -- the detector guards removal/skip, not value edits.
    _ = store_root
    monkeypatch.chdir(tmp_path)
    target = Path("tests/test_parser.py")
    target.parent.mkdir()
    target.write_text("def test_x():\n    assert compute() == 5\n", encoding="utf-8")

    payload = _result(
        _call(
            "edit",
            {
                "edits": [
                    {
                        "file_path": str(target),
                        "old_string": "assert compute() == 5",
                        "new_string": "assert compute() == 6",
                    }
                ],
                "post_edit_hooks": False,
            },
        )
    )

    assert payload.get("rolled_back") is not True
    assert "test_weakening" not in payload
    assert "assert compute() == 6" in target.read_text(encoding="utf-8")


def test_smart_edit_allows_test_weakening_paired_with_production_change(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pair signal: a test weakening that rides with a production-code change reads
    # as a genuine refactor (contract moved with the code), not a reward-hack.
    _ = store_root
    monkeypatch.chdir(tmp_path)
    test_file = Path("tests/test_parser.py")
    test_file.parent.mkdir()
    test_file.write_text("def test_x():\n    assert compute() == 5\n    assert other() == 9\n", encoding="utf-8")
    src_file = Path("src/parser.py")
    src_file.parent.mkdir()
    src_file.write_text("def compute():\n    return 5\n", encoding="utf-8")

    payload = _result(
        _call(
            "edit",
            {
                "edits": [
                    {
                        "file_path": str(test_file),
                        "old_string": "    assert compute() == 5\n    assert other() == 9\n",
                        "new_string": "    assert other() == 9\n",
                    },
                    {
                        "file_path": str(src_file),
                        "old_string": "    return 5\n",
                        "new_string": "    return 6\n",
                    },
                ],
                "post_edit_hooks": False,
            },
        )
    )

    assert payload.get("rolled_back") is not True
    assert "test_weakening" not in payload
    assert "assert compute() == 5" not in test_file.read_text(encoding="utf-8")
    assert "return 6" in src_file.read_text(encoding="utf-8")


def test_smart_edit_does_not_flag_new_regression_test(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = store_root
    monkeypatch.chdir(tmp_path)
    target = Path("tests/regression/issue.rs")

    payload = _result(
        _call(
            "edit",
            {
                "edits": [
                    {
                        "file_path": str(target),
                        "new_string": "#[test]\nfn regression() {}\n",
                        "overwrite": True,
                    }
                ],
                "post_edit_hooks": False,
            },
        )
    )

    assert "FIXME" not in payload


def test_smart_edit_rejects_protected_paths(store_root: Path, tmp_path: Path) -> None:
    _ = store_root
    protected = tmp_path / ".lemoncrow" / "state.txt"
    protected.write_text("hello world", encoding="utf-8")

    payload = _result(
        _call(
            "edit",
            {
                "edits": [
                    {
                        "path": str(protected),
                        "old_string": "world",
                        "new_string": "lemoncrow",
                    }
                ]
            },
        )
    )

    assert payload["rolled_back"] is True
    assert "protected path denied" in payload["failed"][0]["error"]
    assert protected.read_text(encoding="utf-8") == "hello world"


def test_smart_edit_records_workspace_relative_diff_after_hooks(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = store_root
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other_cwd = tmp_path / "cwd"
    other_cwd.mkdir()
    target = workspace / "edit.txt"
    target.write_text("hello world", encoding="utf-8")
    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(workspace))
    monkeypatch.chdir(other_cwd)

    def fake_hooks(files: list[str], *, repo_root: Path, config: object) -> object:
        target.write_text("hello hooks", encoding="utf-8")

        class HookResult:
            diagnostics: tuple[object, ...] = ()
            steps_ran: tuple[str, ...] = ("fake-format",)
            steps_skipped: tuple[str, ...] = ()
            steps_failed: tuple[str, ...] = ()
            total_ms: int = 1

        return HookResult()

    monkeypatch.setattr(
        "lemoncrow.pro.capabilities.tool_supervision.post_edit_hooks.run_post_edit_hooks",
        fake_hooks,
    )

    payload = _result(
        _call(
            "edit",
            {
                "post_edit_hooks": True,
                "edits": [
                    {
                        "file_path": "edit.txt",
                        "old_string": "world",
                        "new_string": "lemoncrow",
                    }
                ],
            },
        )
    )

    # Clean edit (the fake hook reports no diagnostics) is success-silent.
    assert payload == {"applied": ["edit.txt:1"]}
    assert target.read_text(encoding="utf-8") == "hello hooks"
    file_events = [event for event in mcp_server._get_ledger().events if event.kind == "file_edit"]
    assert file_events[-1].payload["path"] == "edit.txt"
    assert "hello hooks" in file_events[-1].payload["diff"]
    assert "hello lemoncrow" not in file_events[-1].payload["diff"]


def test_code_context_workspace_search_returns_repo_tagged_hits_and_repo_filter(
    store_root: Path,
    tmp_path: Path,
) -> None:
    _ = store_root
    billing_root = tmp_path.parent / "billing"
    _write_workspace_fixture_repo(tmp_path, module_name="lemoncrow")
    _write_workspace_fixture_repo(billing_root, module_name="billing")
    _write_workspace_fixture_config(tmp_path, billing_root)
    _preindex(tmp_path)

    payload = mcp_server._op_search(repo_root=str(tmp_path), query="SharedConfig", budget_tokens=4000)
    billing_only = mcp_server._op_search(
        repo_root=str(tmp_path),
        query="SharedConfig",
        repo="billing",
        budget_tokens=4000,
    )

    assert [(item["repo_name"], item["path"]) for item in payload["items"]] == [
        ("lemoncrow", "src/config.py"),
        ("billing", "src/config.py"),
    ]
    assert [item["repo_name"] for item in billing_only["items"]] == ["billing"]


def test_repo_map_and_seed_files_dropped_from_grep(store_root: Path, tmp_path: Path) -> None:
    _ = store_root
    target = tmp_path / "sample.py"
    target.write_text("def alpha():\n    return 1\n", encoding="utf-8")

    # The repo-map capability (and its `seed_files` param) is gone from grep --
    # grep's `mode='map'` now just means the ranked FILE map, an output shape, not
    # a seed-expanded repo map. `seed_files` is no longer a grep param.
    grep_props = mcp_server.TOOLS["grep"]["inputSchema"]["properties"]
    assert "seed_files" not in grep_props
    # Modes were renamed to self-documenting names (fe520724); the schema keeps
    # mode as a described string (aliases accepted) rather than a hard enum.
    mode_desc = grep_props["mode"]["description"]
    for mode_name in ("with_content", "ranked_map", "paths_only", "count_only"):
        assert mode_name in mode_desc, mode_desc
    # The ranked-file-map output shape is reached normally.
    resp = _call("grep", {"regex": "alpha", "path": str(tmp_path), "mode": "ranked_map"})
    assert "result" in resp


def test_code_context_mcp_surfaces(store_root: Path, tmp_path: Path) -> None:
    _ = store_root
    (tmp_path / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("from a import alpha\n\ndef beta():\n    return alpha()\n", encoding="utf-8")

    indexed = _result(_call("index", {"repo_root": str(tmp_path)}))
    _m = re.search(r"symbols=(\d+)", indexed)
    assert _m is not None
    assert int(_m.group(1)) >= 2

    searched = _op_result("symbols", mcp_server._op_search, repo_root=str(tmp_path), query="alpha")
    assert searched and "no matches" not in searched
    assert "snippet:" not in searched
    cached_search = _op_result("symbols", mcp_server._op_search, repo_root=str(tmp_path), query="alpha")
    assert cached_search == searched

    symbol = _op_result(
        "node",
        mcp_server._op_node,
        repo_root=str(tmp_path),
        qualified_name="alpha",
        path="a.py",
    )
    assert "def alpha" in symbol

    context = _result(
        _call(
            "context",
            {
                "task": "change alpha",
                "files": ["a.py"],
                "token_budget": 4000,
                "mode": "symbols",
            },
        )
    )
    assert isinstance(context, dict)
    assert context.get("task") == "change alpha"


def test_code_context_search_surface_supports_snippet_scope_and_glob(store_root: Path, tmp_path: Path) -> None:
    _ = store_root
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "orders.py").write_text(
        "class OrderService:\n    def calculate_total(self, items: list[int]) -> int:\n        return sum(items)\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_orders.py").write_text(
        "from src.orders import OrderService\n",
        encoding="utf-8",
    )
    _preindex(tmp_path)

    payload = mcp_server._op_search(
        repo_root=str(tmp_path),
        query="OrderService",
        snippet="head",
        snippet_lines=2,
        file_glob="src/*.py",
        scope="repo",
        budget_tokens=4000,
    )

    assert "provenance" not in payload
    assert "provenance_breakdown" not in payload
    assert payload["items"][0]["path"] == "src/orders.py"
    assert (
        payload["items"][0]["snippet"] == "class OrderService:\n    def calculate_total(self, items: list[int]) -> int:"
    )


def test_lean_code_search_view_returns_related_symbol_ranges() -> None:
    payload = {
        "exact_match": True,
        "entry_points": [
            {
                "qualified_name": "OrderService.calculate_total",
                "path": "src/orders.py",
                "line": 12,
                "end_line": 20,
                "kind": "method",
                "score": 10.0,
            },
            {
                "qualified_name": "PaymentService.capture",
                "path": "src/payments.py",
                "line": 30,
                "end_line": 41,
                "kind": "method",
                "score": 8.0,
            },
        ],
        "files": [
            {
                "path": "src/orders.py",
                "source_sections": [
                    {
                        "qualified_name": "OrderService.calculate_total",
                        "path": "src/orders.py",
                        "line": 12,
                        "end_line": 20,
                        "content": "12\tdef calculate_total(self):\n",
                    }
                ],
            }
        ],
    }

    lean = mcp_server._lean_code_search_view(payload, max_files=1)

    # A span already returned as source is NOT repeated as a pointer; only
    # symbols beyond the returned sections make the map.
    assert lean["related_symbols"] == [
        {
            "qualified_name": "PaymentService.capture",
            "path": "src/payments.py",
            "line": 30,
            "end_line": 41,
            "kind": "method",
        }
    ]


def test_lean_code_search_view_preserves_learned_file_order() -> None:
    payload = {
        "exact_match": False,
        "experiment": {"name": "explore_reranker_v2_lambdamart"},
        "entry_points": [
            {"qualified_name": "legacy", "path": "src/legacy.py", "line": 1, "end_line": 2, "score": 10.0},
            {"qualified_name": "learned", "path": "src/learned.py", "line": 1, "end_line": 2, "score": 1.0},
        ],
        "files": [
            {
                "path": "src/learned.py",
                "source_sections": [
                    {
                        "qualified_name": "learned",
                        "path": "src/learned.py",
                        "line": 1,
                        "end_line": 2,
                        "content": "1\tx\n",
                    }
                ],
            },
            {
                "path": "src/legacy.py",
                "source_sections": [
                    {"qualified_name": "legacy", "path": "src/legacy.py", "line": 1, "end_line": 2, "content": "1\ty\n"}
                ],
            },
        ],
    }

    lean = mcp_server._lean_code_search_view(payload, max_files=1)

    assert [entry["path"] for entry in lean["files"]] == ["src/learned.py"]
    assert lean["candidate_files"][0] == "src/legacy.py"


def test_outline_lean_view_low_confidence_outlines_small_non_top_section() -> None:
    # A "ranked candidates" (no exact match, no include_source -> keep_top2=False)
    # call is a gamble: only the single top-ranked section stays inline. Rank 1+
    # outlines to a pointer even when tiny -- size alone must not be a free pass
    # past low confidence.
    lean = {
        "exact_match": False,
        "files": [
            {
                "path": "a.py",
                "sections": [
                    {"path": "a.py", "qualified_name": "top", "line": 1, "end_line": 2, "content": "rank 0\n"},
                    {"path": "a.py", "qualified_name": "tiny", "line": 3, "end_line": 4, "content": "x = 1\n"},
                ],
            }
        ],
    }
    shaped = mcp_server._outline_lean_view(lean, keep_top2=False)
    secs = shaped["files"][0]["sections"]
    assert secs[0]["content"] == "rank 0\n"
    assert "outline" not in secs[0]
    assert "content" not in secs[1]
    assert secs[1]["outline"] == "tiny — read a.py:L3-L4"


def test_outline_lean_view_confident_call_keeps_small_non_top2_inline() -> None:
    # Unchanged prior behavior: a confident call (exact_match/include_source ->
    # keep_top2=True) still ships a small non-top2 section inline.
    lean = {
        "exact_match": True,
        "files": [
            {
                "path": "a.py",
                "sections": [
                    {"path": "a.py", "qualified_name": "top", "line": 1, "end_line": 2, "content": "rank 0\n"},
                    {"path": "a.py", "qualified_name": "second", "line": 3, "end_line": 4, "content": "rank 1\n"},
                    {"path": "a.py", "qualified_name": "tiny", "line": 5, "end_line": 6, "content": "x = 1\n"},
                ],
            }
        ],
    }
    shaped = mcp_server._outline_lean_view(lean, keep_top2=True)
    secs = shaped["files"][0]["sections"]
    assert secs[2]["content"] == "x = 1\n"
    assert "outline" not in secs[2]


def test_lean_related_symbol_flattens_multiline_names() -> None:
    # Markdown headings can index a whole frontmatter block as their name; the
    # symbol map must stay a one-line pointer, never a prose dump.
    out = mcp_server._lean_related_symbol(
        {
            "qualified_name": "description: agent one-liner\nmodel: gpt\ntools:\n  [a, b]",
            "path": "integrations/agents/general.md",
            "line": 2,
            "end_line": 30,
            "kind": "heading",
        }
    )
    assert out["qualified_name"] == "description: agent one-liner"


def test_render_read_outline_md_caps_symbol_lines() -> None:
    outline = {
        "symbols": [
            {"name": f"f{i}", "kind": "function", "start_line": i * 10, "end_line": i * 10 + 5} for i in range(1, 121)
        ],
    }
    rendered = mcp_server._render_read_outline_md("big.py", outline, "python")
    lines = rendered.splitlines()
    assert lines[0] == "symbols:"
    assert len(lines) == 1 + mcp_server._READ_OUTLINE_MAX_LINES + 1
    assert lines[-1] == "... +40 more symbols (use :Lx-Ly or :full)"


def test_lean_code_search_view_reports_section_retrieval_savings(tmp_path: Path) -> None:
    first = tmp_path / "src" / "orders.py"
    second = tmp_path / "src" / "payments.py"
    first.parent.mkdir()
    first.write_text("def calculate_total():\n    return 1\n" * 100, encoding="utf-8")
    second.write_text("def capture_payment():\n    return 1\n" * 100, encoding="utf-8")
    payload = {
        "exact_match": False,
        "entry_points": [
            {"qualified_name": "calculate_total", "path": "src/orders.py", "line": 1, "end_line": 2, "score": 10.0},
            {"qualified_name": "capture_payment", "path": "src/payments.py", "line": 1, "end_line": 2, "score": 9.0},
        ],
        "files": [
            {
                "path": "src/orders.py",
                "source_sections": [{"line": 1, "end_line": 2, "content": "1\tdef calculate_total():\n"}],
            },
            {
                "path": "src/payments.py",
                "source_sections": [{"line": 1, "end_line": 2, "content": "1\tdef capture_payment():\n"}],
            },
        ],
    }

    lean = mcp_server._attach_code_search_savings(mcp_server._lean_code_search_view(payload, max_files=2), tmp_path)

    assert lean["tokens_saved"] > 0
    # Two whole-source files returned = 2 avoided Reads; the symbol map only
    # points back at those same two files, so no extra locate is credited.
    assert lean["calls_saved"] == 2


def test_lean_code_search_view_excludes_truncated_and_credits_cross_file_nav(tmp_path: Path) -> None:
    """A Read is credited only for source returned whole — a truncated section is
    a preview. A symbol map pointing to a file we did NOT return source for adds
    one avoided locate (the grep-for-usages you'd otherwise run)."""
    orders = tmp_path / "src" / "orders.py"
    payments = tmp_path / "src" / "payments.py"
    orders.parent.mkdir()
    orders.write_text("def calculate_total():\n    return 1\n" * 100, encoding="utf-8")
    payments.write_text("def capture_payment():\n    return 1\n" * 100, encoding="utf-8")
    payload = {
        "exact_match": False,
        "entry_points": [
            {"qualified_name": "calculate_total", "path": "src/orders.py", "line": 1, "end_line": 2, "score": 10.0},
            {"qualified_name": "capture_payment", "path": "src/payments.py", "line": 1, "end_line": 2, "score": 9.0},
            # A usage in a file we do NOT return source for — an extra grep avoided.
            {"qualified_name": "audit_log", "path": "src/audit.py", "line": 5, "end_line": 6, "score": 8.0},
        ],
        "files": [
            {
                "path": "src/orders.py",
                "source_sections": [{"line": 1, "end_line": 2, "content": "1\tdef calculate_total():\n"}],
            },
            {
                "path": "src/payments.py",
                "source_sections": [
                    {"line": 1, "end_line": 2, "content": "1\tdef capture_payment(): [truncated after L2]\n"}
                ],
            },
        ],
    }
    lean = mcp_server._attach_code_search_savings(mcp_server._lean_code_search_view(payload, max_files=2), tmp_path)
    # orders.py = 1 whole-source Read; payments.py truncated (not counted);
    # audit.py usage in the symbol map = 1 avoided locate. => 2.
    assert lean["tokens_saved"] > 0
    assert lean["calls_saved"] == 2, lean


def test_lean_code_search_view_credits_repeat_file_only_once(tmp_path: Path) -> None:
    """A file returned by a later search re-reads nothing — credit it once per
    session, not on every repeat search of the same hot file."""
    src = tmp_path / "src" / "orders.py"
    src.parent.mkdir()
    src.write_text("def calculate_total():\n    return 1\n" * 100, encoding="utf-8")
    payload = {
        "exact_match": True,
        "entry_points": [
            {"qualified_name": "calculate_total", "path": "src/orders.py", "line": 1, "end_line": 2, "score": 10.0}
        ],
        "files": [
            {
                "path": "src/orders.py",
                "source_sections": [{"line": 1, "end_line": 2, "content": "1\tdef calculate_total():\n"}],
            }
        ],
    }
    first = mcp_server._attach_code_search_savings(mcp_server._lean_code_search_view(payload, max_files=1), tmp_path)
    second = mcp_server._attach_code_search_savings(mcp_server._lean_code_search_view(payload, max_files=1), tmp_path)
    assert first["calls_saved"] >= 1 and first["tokens_saved"] > 0
    # Same file, same session -> no new read or token credit.
    assert second.get("calls_saved", 0) == 0
    assert second.get("tokens_saved", 0) == 0


def test_tool_code_search_response_includes_saved_section_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "src" / "orders.py"
    source.parent.mkdir()
    source.write_text("def calculate_total():\n    return 1\n" * 100, encoding="utf-8")
    fake_engine = MagicMock()
    fake_engine.tool_explore.return_value = {
        "exact_match": True,
        "entry_points": [
            {"qualified_name": "calculate_total", "path": "src/orders.py", "line": 1, "end_line": 2, "score": 10.0}
        ],
        "files": [
            {
                "path": "src/orders.py",
                "source_sections": [{"line": 1, "end_line": 2, "content": "1\tdef calculate_total():\n"}],
            }
        ],
    }
    appended: list[tuple[str, int, int]] = []
    monkeypatch.setattr("lemoncrow.gateway.adapters.mcp_server._workspace_root", lambda: tmp_path)
    monkeypatch.setattr("lemoncrow.gateway.adapters.mcp_server._code_context_engine", lambda repo_root=".": fake_engine)
    monkeypatch.setattr(
        "lemoncrow.gateway.adapters.mcp_server._append_workspace_savings",
        lambda tool, tokens, calls, rid="": appended.append((tool, tokens, calls)),
    )

    response = _call("code_search", {"query": "calculate_total", "max_files": 1})

    saved = response["result"]["content"][0].get("saved")
    assert saved is not None
    assert saved["tokens"] > 0
    assert saved["calls"] == 1  # single whole-source Read avoided; symbol map stays in-file
    assert appended == [("code_search", saved["tokens"], 1)]


def test_resolve_query_as_existing_file_pins_verbatim_and_prefixed_paths(tmp_path: Path) -> None:
    (tmp_path / "benchmarks" / "harbor").mkdir(parents=True)
    target = tmp_path / "benchmarks" / "harbor" / "rebuild_bundle.sh"
    target.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")

    # Verbatim relpath.
    assert (
        mcp_server._resolve_query_as_existing_file(tmp_path, "benchmarks/harbor/rebuild_bundle.sh")
        == "benchmarks/harbor/rebuild_bundle.sh"
    )
    # A bogus leading segment (e.g. a phantom repo-name prefix) is stripped once and retried.
    assert (
        mcp_server._resolve_query_as_existing_file(tmp_path, "/lemoncrow/benchmarks/harbor/rebuild_bundle.sh")
        == "benchmarks/harbor/rebuild_bundle.sh"
    )
    # Path traversal out of the workspace never resolves.
    assert mcp_server._resolve_query_as_existing_file(tmp_path, "../../etc/passwd") is None
    # A prose query (whitespace) is never treated as a path.
    assert mcp_server._resolve_query_as_existing_file(tmp_path, "how does rebuild_bundle work") is None
    # A bare symbol name with no path shape and no matching file resolves to nothing.
    assert mcp_server._resolve_query_as_existing_file(tmp_path, "OrderService") is None


def test_tool_code_search_fast_paths_a_literal_file_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "benchmarks" / "harbor").mkdir(parents=True)
    (tmp_path / "benchmarks" / "harbor" / "rebuild_bundle.sh").write_text(
        "#!/bin/sh\necho rebuilding\n", encoding="utf-8"
    )
    monkeypatch.setattr("lemoncrow.gateway.adapters.mcp_server._workspace_root", lambda: tmp_path)

    response = _call("code_search", {"query": "/lemoncrow/benchmarks/harbor/rebuild_bundle.sh"})

    # code_search renders to compact markdown (no code renderer applies to a
    # shell script), so assert on that text rather than parsed JSON.
    payload = _result(response)
    assert isinstance(payload, str)
    assert "no exact match" not in payload
    assert "benchmarks/harbor/rebuild_bundle.sh" in payload
    assert "echo rebuilding" in payload


def test_tool_code_search_dispatches_mode_without_gateway_ranking_logic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_engine = MagicMock()
    fake_engine.tool_search.return_value = {
        "items": [{"symbol_name": "issue_access_token", "provenance": "local"}],
        "cache_hit": False,
        "provenance": "local",
        "tokens_saved": 10,
        "total_tokens": 80,
        "mode": "semantic",
    }
    monkeypatch.setattr(
        "lemoncrow.gateway.adapters.mcp_server._code_context_engine",
        lambda repo_root=".": fake_engine,
    )

    payload = mcp_server._op_search(
        repo_root=str(tmp_path),
        query="create login token for authenticated user",
        mode="semantic",
        budget_tokens=220,
    )

    assert "mode" not in payload
    fake_engine.tool_search.assert_called_once_with(
        "create login token for authenticated user",
        limit=20,
        mode="semantic",
        intent="auto",
        kind=None,
        language=None,
        seed_files=None,
        snippet="none",
        snippet_lines=8,
        file_glob=None,
        scope="repo",
        budget_tokens=220,
    )


def test_tool_code_search_dispatches_grounded_seed_files_without_gateway_ranking_logic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_engine = MagicMock()
    fake_engine.tool_search.return_value = {
        "items": [{"symbol_name": "OrderService", "provenance": "local"}],
        "cache_hit": False,
        "provenance": "local",
        "tokens_saved": 6,
        "total_tokens": 60,
        "mode": "lexical",
    }
    monkeypatch.setattr(
        "lemoncrow.gateway.adapters.mcp_server._code_context_engine",
        lambda repo_root=".": fake_engine,
    )

    payload = mcp_server._op_search(
        repo_root=str(tmp_path),
        query="OrderService",
        seed_files=["src/orders.py"],
        budget_tokens=220,
    )

    assert "mode" not in payload
    fake_engine.tool_search.assert_called_once_with(
        "OrderService",
        limit=20,
        mode="auto",
        intent="auto",
        kind=None,
        language=None,
        seed_files=["src/orders.py"],
        snippet="none",
        snippet_lines=8,
        file_glob=None,
        scope="repo",
        budget_tokens=220,
    )


def test_tool_code_search_dispatches_deleted_scope_filters_without_gateway_history_logic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_engine = MagicMock()
    fake_engine.tool_search.return_value = {
        "items": [
            {
                "symbol_name": "LegacyCheckout",
                "provenance": "graveyard",
                "deleted_at_sha": "abc123",
                "rename_target": "modern.py",
            }
        ],
        "cache_hit": False,
        "provenance": "graveyard",
        "tokens_saved": 11,
        "total_tokens": 120,
        "mode": "auto",
    }
    monkeypatch.setattr(
        "lemoncrow.gateway.adapters.mcp_server._code_context_engine",
        lambda repo_root=".": fake_engine,
    )

    payload = mcp_server._op_search(
        repo_root=str(tmp_path),
        query="ModernCheckout",
        scope="deleted",
        since="2025-01-01",
        touched_by="history@example.com",
        budget_tokens=220,
    )

    assert "provenance" not in payload
    assert payload["items"][0]["rename_target"] == "modern.py"
    fake_engine.tool_search.assert_called_once_with(
        "ModernCheckout",
        limit=20,
        mode="auto",
        kind=None,
        language=None,
        snippet="none",
        snippet_lines=8,
        file_glob=None,
        scope="deleted",
        since="2025-01-01",
        touched_by="history@example.com",
        budget_tokens=220,
    )


def test_tool_code_blame_dispatches_additively_without_gateway_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_engine = MagicMock()
    fake_engine.tool_blame.return_value = {
        "symbol_name": "risk_score",
        "qualified_name": "risk_score",
        "file_path": "service.py",
        "freshness": "fresh",
        "last_author": "carol@example.com",
        "last_commit_sha": "abc123",
        "local_edits": False,
        "distinct_authors": 2,
        "cache_hit": False,
        "provenance": "blame",
        "tokens_saved": 12,
        "total_tokens": 150,
    }
    monkeypatch.setattr(
        "lemoncrow.gateway.adapters.mcp_server._code_context_engine",
        lambda repo_root=".": fake_engine,
    )

    payload = mcp_server._op_blame(
        repo_root=str(tmp_path),
        query="risk_score",
        include_churn=False,
        budget_tokens=220,
    )

    assert "provenance" not in payload
    assert payload["symbol_name"] == "risk_score"
    fake_engine.tool_blame.assert_called_once_with(
        query="risk_score",
        symbol_id=None,
        qualified_name=None,
        symbol_name=None,
        file_path=None,
        include_churn=False,
        budget_tokens=220,
    )


def test_tool_code_include_churn_remains_additive_for_non_blame_ops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_engine = MagicMock()
    fake_engine.tool_search.return_value = {
        "items": [{"symbol_name": "OrderService", "file_path": "src/orders.py", "provenance": "local"}],
        "cache_hit": False,
        "provenance": "local",
        "tokens_saved": 10,
        "total_tokens": 100,
        "mode": "auto",
    }
    monkeypatch.setattr(
        "lemoncrow.gateway.adapters.mcp_server._code_context_engine",
        lambda repo_root=".": fake_engine,
    )

    payload = mcp_server._op_search(
        repo_root=str(tmp_path),
        query="OrderService",
        budget_tokens=220,
    )

    assert "provenance" not in payload
    fake_engine.tool_search.assert_called_once_with(
        "OrderService",
        limit=20,
        mode="auto",
        kind=None,
        language=None,
        snippet="none",
        snippet_lines=8,
        file_glob=None,
        scope="repo",
        budget_tokens=220,
        intent="auto",
        seed_files=None,
    )


def test_code_context_usages_surface_groups_references(store_root: Path, tmp_path: Path) -> None:
    _ = store_root
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "orders.py").write_text(
        "class OrderService:\n    def calculate_total(self, items: list[int]) -> int:\n        return sum(items)\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "checkout.py").write_text(
        "from src.orders import OrderService\n\n"
        "def checkout(items: list[int]) -> int:\n"
        "    return OrderService().calculate_total(items)\n",
        encoding="utf-8",
    )
    _preindex(tmp_path)

    payload = _op_result(
        "usages",
        mcp_server._op_usages,
        repo_root=str(tmp_path),
        query="OrderService",
    )

    assert "src/checkout.py" in payload
    assert "checkout" in payload


def test_code_context_pattern_search_surface_is_cached(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = store_root
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("requests.get(url)\n", encoding="utf-8")

    monkeypatch.setattr(
        "lemoncrow.pro.capabilities.code_context.engine.AstGrepAdapter.search",
        lambda self, *, pattern, language=None, file_glob=None, limit=20: PatternSearchResult(
            matches=[
                PatternMatch(
                    file_path="src/app.py",
                    line=1,
                    column=0,
                    end_line=1,
                    end_column=17,
                    snippet="requests.get(url)",
                    captures={"URL": "url"},
                )
            ],
            truncated=False,
            total_matches=1,
        ),
    )

    first = _op_result(
        "codemod",
        mcp_server._op_pattern,
        repo_root=str(tmp_path),
        pattern="requests.get($URL)",
        budget_tokens=220,
    )
    cached = _op_result(
        "codemod",
        mcp_server._op_pattern,
        repo_root=str(tmp_path),
        pattern="requests.get($URL)",
        budget_tokens=220,
    )

    # Pattern search now surfaces the compact markdown the agent receives, not
    # the raw JSON payload (path emitted once, snippet preserved).
    assert isinstance(first, str)
    assert first.startswith("- src/app.py")
    assert "src/app.py" in first
    assert "requests.get(url)" in first
    assert "provenance" not in first
    # Cache state is internal bookkeeping; the cached response must be identical.
    assert cached == first


def test_code_context_cache_diagnostics_surface_is_additive(store_root: Path, tmp_path: Path) -> None:
    _ = store_root
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "orders.py").write_text(
        "class OrderService:\n    def calculate_total(self, items: list[int]) -> int:\n        return sum(items)\n",
        encoding="utf-8",
    )
    _preindex(tmp_path)

    _op_result(
        "search",
        mcp_server._op_search,
        repo_root=str(tmp_path),
        query="OrderService",
        budget_tokens=4000,
    )
    _op_result(
        "node",
        mcp_server._op_node,
        repo_root=str(tmp_path),
        qualified_name="OrderService",
        path="src/orders.py",
        budget_tokens=4000,
    )

    status = _result(_call("cache", {"op": "status", "repo_root": str(tmp_path), "budget_tokens": 200}))
    invalidated = _result(
        _call(
            "cache",
            {
                "op": "invalidate",
                "repo_root": str(tmp_path),
                "cache_tool": "search",
                "budget_tokens": 200,
            },
        )
    )

    assert "code.search=1" in status and "code.symbol=1" in status
    assert "items" not in status
    assert "matches" not in status
    assert invalidated["scope"]["cache_tool"] == "search"
    assert invalidated["invalidated_entries"] == 1


def test_code_context_pattern_rewrite_reindexes_changed_files(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = store_root
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("requests.get(url)\n", encoding="utf-8")

    def fake_rewrite(self, *, pattern, rewrite, language=None, file_glob=None, dry_run=True):  # type: ignore[no-untyped-def]
        before = target.read_text(encoding="utf-8")
        after = before.replace("requests.get(url)", "requests.get(url, timeout=30)")
        diff = "--- a/src/app.py\n+++ b/src/app.py\n@@\n-requests.get(url)\n+requests.get(url, timeout=30)\n"
        if not dry_run:
            target.write_text(after, encoding="utf-8")
        return PatternRewriteResult(diff=diff, files_changed=["src/app.py"])

    reindexed: list[list[str]] = []

    monkeypatch.setattr("lemoncrow.pro.capabilities.code_context.engine.AstGrepAdapter.rewrite", fake_rewrite)
    monkeypatch.setattr(
        CodeContextEngine,
        "_reindex_files",
        lambda self, file_paths: reindexed.append(list(file_paths)),
        raising=False,
    )

    preview = _op_result(
        "codemod",
        mcp_server._op_pattern,
        repo_root=str(tmp_path),
        pattern="requests.get($URL)",
        rewrite="requests.get($URL, timeout=30)",
        dry_run=True,
    )
    applied = _op_result(
        "codemod",
        mcp_server._op_pattern,
        repo_root=str(tmp_path),
        pattern="requests.get($URL)",
        rewrite="requests.get($URL, timeout=30)",
        dry_run=False,
    )

    assert "--- a/src/app.py" in preview["diff"]
    assert applied["files_changed"] == ["src/app.py"]
    assert reindexed == [["src/app.py"]]
    assert target.read_text(encoding="utf-8") == "requests.get(url, timeout=30)\n"


def test_code_context_pattern_returns_structured_tool_unavailable(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = store_root
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("requests.get(url)\n", encoding="utf-8")

    payload = {
        "error": "tool_unavailable",
        "tool": "ast-grep",
        "expected_binary": "ast-grep",
        "message": "ast-grep is unavailable",
        "checked": [],
        "hint": "install ast-grep",
    }
    monkeypatch.setattr(
        "lemoncrow.pro.capabilities.code_context.engine.AstGrepAdapter.search",
        lambda self, *, pattern, language=None, file_glob=None, limit=20: (_ for _ in ()).throw(
            AstGrepToolUnavailable(payload)
        ),
    )

    result = mcp_server._op_pattern(
        repo_root=str(tmp_path),
        pattern="requests.get($URL)",
    )

    assert result["error"] == "tool_unavailable"
    assert result["expected_binary"] == "ast-grep"


# ---------------------------------------------------------------------------
# Remaining-gap regression tests (Issues 4, 13, 14 and shell failure fix)
# ---------------------------------------------------------------------------


def test_path_safety_module_is_importable_and_has_protected_parts() -> None:
    """Centralised PROTECTED_PARTS frozenset must exist and cover the canonical dirs."""
    from lemoncrow.pro.capabilities.tool_supervision.path_safety import PROTECTED_PARTS

    required = {".git", ".lemoncrow", "node_modules", ".venv"}
    assert required <= set(PROTECTED_PARTS), f"Missing entries: {required - set(PROTECTED_PARTS)}"


def test_trace_compact_receipt_always_present(store_root: Path) -> None:
    """tool_record_trace must always return trace_id and event_recorded — the compact receipt."""
    _ = store_root
    payload = _result(
        _call(
            "trace",
            {
                "agent": "lemoncrow:code",
                "domain": "mcp-server",
                "task": "Verify compact receipt",
                "status": "success",
            },
        )
    )
    assert payload.get("event_recorded") is True, f"'event_recorded' missing or False in trace receipt: {payload}"
    assert (
        isinstance(payload.get("trace_id"), str) and payload["trace_id"]
    ), f"'trace_id' missing or empty in trace receipt: {payload}"


def test_shell_failure_preserves_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """For failing commands, the tail of stdout must be preserved even when output is long."""
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    # Generate 300 numbered lines then exit 1 — only tail should survive truncation
    result = _run_bash_tool(
        "python3 -c \"import sys; [print(f'line-{i}') for i in range(300)]; sys.exit(1)\"",
        max_lines=60,
    )

    assert result["exit_code"] == 1
    stdout = result["stdout"]
    # The last line must be visible (line-299)
    assert "line-299" in stdout, f"tail not preserved for failing command; stdout tail:\n{stdout[-500:]}"


def test_shell_falls_back_when_workspace_root_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-existent CLAUDE_WORKSPACE_ROOT must not hard-fail the shell tool.

    A host path leaking into a container (e.g. via the environment) used to make
    every cwd-less command raise FileNotFoundError from Popen -> MCP -32000.
    The handler now falls back to the process cwd so the command still runs.
    """
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    missing = tmp_path / "does" / "not" / "exist"
    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(missing))

    result = _run_bash_tool("pwd")

    assert result["exit_code"] == 0, result
    assert result["stdout"].strip()


def test_shell_short_timeout_does_not_kill_the_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`timeout` is a soft response budget, not a kill deadline (see
    bash_exec._MANAGED_COMMAND_HARD_CAP_S) -- a command that legitimately
    takes longer than the requested `timeout` must be allowed to keep running
    and complete normally instead of being killed out from under the caller.
    This is the fix for tasks that background a long-running service with a
    modest timeout and expect it to survive."""
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    result = _run_bash_tool(
        "python3 -c \"import time; time.sleep(1); print('done')\"",
        timeout=0.5,
    )

    assert result["exit_code"] == 0
    assert result["stdout"] == "done"


def test_shell_run_blocks_until_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    # Foreground run blocks until the command finishes -- no artificial
    # window, no detach, no session, no poll -- even for a slow-ish command.
    result = _run_bash_tool(
        "python3 -c \"import time; time.sleep(0.3); print('done')\"",
        timeout=30,
    )
    assert result.get("status") != "running"
    assert "session_id" not in result
    assert result["exit_code"] == 0
    assert result["stdout"] == "done"


def test_shell_large_timeout_does_not_detach_fast_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    # A 30-minute timeout budget must not make a fast command detach: it
    # blocks only as long as the command actually runs, then returns.
    result = _run_bash_tool("echo hi", timeout=1800)
    assert result.get("status") != "running"
    assert "session_id" not in result
    assert result["exit_code"] == 0
    assert result["stdout"] == "hi"


def test_shell_poll_blocks_until_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    # Detach immediately, then a SINGLE blocking poll returns the finished
    # result -- no manual retry loop, no busy-polling.
    started = _run_bash_tool(
        "python3 -c \"import time; time.sleep(0.5); print('done')\"",
        timeout=10,
        background=True,
    )
    assert started["status"] == "running"

    completed = _run_bash_tool(session_id=started["session_id"], action="poll")
    assert completed["status"] == "completed"
    assert completed["exit_code"] == 0
    assert completed["stdout"] == "done"


def test_shell_poll_timeout_returns_running_handle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    started = _run_bash_tool(
        "python3 -c \"import time; time.sleep(1); print('done')\"",
        timeout=10,
        background=True,
    )
    try:
        poll_started = time.monotonic()
        result = _run_bash_tool(
            session_id=started["session_id"],
            action="poll",
            timeout=0.2,
        )
        elapsed = time.monotonic() - poll_started

        assert elapsed < 1.0
        assert result["status"] == "running"
        assert result["session_id"] == started["session_id"]
    finally:
        _run_bash_tool(session_id=started["session_id"], action="kill")


def test_shell_background_return_reports_timeout_remaining(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    # Detaching surfaces an honest upper bound (the timeout), not a fake ETA.
    started = _run_bash_tool(
        'python3 -c "import time; time.sleep(5)"',
        timeout=10,
        background=True,
    )
    try:
        assert started["status"] == "running"
        assert started["session_id"]
        assert started["duration_ms"] >= 0
        assert 0 < started["timeout_remaining_ms"] <= 10_000
    finally:
        _run_bash_tool(session_id=started["session_id"], action="kill")


def test_render_shell_text_running_ships_handle_and_logs() -> None:
    # pid/elapsed/timeout values are noise, but a running command's logs are
    # actionable while the process remains live.
    from lemoncrow.gateway.adapters.mcp_server import _render_bash_text

    text = _render_bash_text(
        {
            "status": "running",
            "session_id": "abc123",
            "pid": 42,
            "duration_ms": 95_000,
            "timeout_remaining_ms": 1_765_000,
            "log_file": "/tmp/x.stdout.txt",
        }
    )
    assert text == "running id=abc123\n[logs: /tmp/x.stdout.txt]"
    assert "pid=" not in text
    assert "elapsed=" not in text
    assert "timeout_in=" not in text
    assert "log_file=" not in text


def test_render_shell_text_warns_once_past_its_soft_budget() -> None:
    from lemoncrow.gateway.adapters.mcp_server import _render_bash_text

    text = _render_bash_text(
        {
            "status": "running",
            "session_id": "abc123",
            "timeout_remaining_ms": 0,
            "over_budget": True,
        }
    )
    # The handle line must carry the wait recipe: benchmark transcripts showed
    # agents sleep-polling for up to ~30 turns because nothing told them
    # bash(id=...) blocks until the command finishes.
    assert text == "still running id=abc123; bash(id=abc123) waits for it — don't sleep-poll"


def test_render_shell_text_identifies_explicit_background_job() -> None:
    from lemoncrow.gateway.adapters.mcp_server import _render_bash_text

    text = _render_bash_text(
        {
            "status": "running",
            "session_id": "abc123",
            "explicit_background": True,
            "log_file": "/tmp/x.stdout.txt",
        }
    )
    assert text == "background running id=abc123; bash(id=abc123) waits for it\n[logs: /tmp/x.stdout.txt]"


def test_render_shell_text_running_before_budget_has_no_nudge() -> None:
    # A plain mid-flight peek that's still well inside its window gets no
    # act-now nudge -- that's reserved for the actually-overrunning case.
    from lemoncrow.gateway.adapters.mcp_server import _render_bash_text

    text = _render_bash_text(
        {
            "status": "running",
            "session_id": "abc123",
            "timeout_remaining_ms": 500_000,
            "over_budget": False,
        }
    )
    assert text == "running id=abc123"
    assert "act now" not in text


def test_render_bash_text_includes_spill_hint_in_truncation_notice() -> None:
    from lemoncrow.gateway.adapters.mcp_server import _render_bash_text

    text = _render_bash_text(
        {
            "stdout": "line1\n... (50 lines omitted) ...\nline300",
            "stderr": "",
            "exit_code": 0,
            "truncated": True,
            "lines_omitted": 50,
            "spill_hint": "[lc: shrunk 5000→123; full: /tmp/x.txt]",
        }
    )
    # The spill notice subsumes the bare truncation marker -- exactly one
    # truncation footer ships, never both.
    assert "[lc: shrunk 5000→123; full: /tmp/x.txt]" in text
    assert "[output truncated: 50 lines omitted]" not in text


def test_unknown_session_error_names_the_live_handles(monkeypatch: pytest.MonkeyPatch) -> None:
    from lemoncrow.gateway.adapters.mcp import bash as bash_mod

    monkeypatch.setattr(bash_mod, "live_managed_bash_ids", lambda: ["v334oo", "8muo9e"])

    text = bash_mod.tool_bash({"id": "zzzzzz", "action": "status"})

    assert "unknown shell session: zzzzzz" in text
    assert "live sessions: v334oo, 8muo9e" in text


def test_unknown_session_error_flags_a_host_task_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # A host that caps MCP calls hands back its own task id; feeding that back
    # as a shell handle can only fail, so the error has to say which id-space
    # it landed in rather than repeating "unknown".
    from lemoncrow.gateway.adapters.mcp import bash as bash_mod

    monkeypatch.setattr(bash_mod, "live_managed_bash_ids", lambda: [])

    text = bash_mod.tool_bash({"id": "k31b54d8q", "action": "status"})

    assert "host background-task id" in text
    assert "no live sessions in this MCP process" in text


def test_render_bash_text_omits_spill_hint_when_absent() -> None:
    from lemoncrow.gateway.adapters.mcp_server import _render_bash_text

    text = _render_bash_text(
        {
            "stdout": "line1\n... (50 lines omitted) ...\nline300",
            "stderr": "",
            "exit_code": 0,
            "truncated": True,
            "lines_omitted": 50,
        }
    )
    assert "[output truncated: 50 lines omitted]" in text


def test_shell_background_session_can_be_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    started = _run_bash_tool(
        'python3 -c "import time; time.sleep(10)"',
        timeout=10,
        background=True,
    )
    cancelled = _run_bash_tool(session_id=started["session_id"], action="kill")
    time.sleep(0.1)

    assert cancelled["status"] == "cancelled"
    assert cancelled["exit_code"] == -1
    assert "cancelled" in cancelled["stderr"].lower()


def test_shell_background_session_survives_past_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A backgrounded session's `timeout` no longer triggers a kill -- the
    session survives past it and a blocking poll returns its real, completed
    result. Explicit cancellation is still available via action="kill" (see
    test_shell_background_session_can_be_cancelled) for callers that do want
    to give up on it."""
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    started = _run_bash_tool(
        "python3 -c \"import time; time.sleep(1); print('done')\"",
        timeout=0.5,
        background=True,
    )
    time.sleep(0.65)  # past the requested timeout; the command is still running
    completed = _run_bash_tool(session_id=started["session_id"], action="poll")

    assert completed["status"] == "completed"
    assert completed["exit_code"] == 0
    assert completed["stdout"] == "done"


def test_shell_action_update_installs_an_enforced_kill_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before any action="update" call, `timeout` is only ever a soft
    response budget (see test_shell_background_session_survives_past_timeout)
    -- update is the only thing that installs a real, enforced kill."""
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    started = _run_bash_tool(
        'python3 -c "import time; time.sleep(30)"',
        timeout=3600,
        background=True,
    )
    assert started["status"] == "running"
    updated = _run_bash_tool(session_id=started["session_id"], action="update", timeout=0.3)
    assert updated["status"] == "running"
    assert updated["updated"] is True
    assert updated["timeout"] == 0.3

    result = _run_bash_tool(session_id=started["session_id"], action="poll")
    assert result["status"] == "timed_out"


def test_shell_action_update_extends_its_own_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    started = _run_bash_tool(
        "python3 -c \"import time; time.sleep(1); print('done')\"",
        timeout=10,
        background=True,
    )
    _run_bash_tool(session_id=started["session_id"], action="update", timeout=0.2)
    extended = _run_bash_tool(session_id=started["session_id"], action="update", timeout=5)
    assert extended["updated"] is True
    assert extended["timeout"] == 5.0

    result = _run_bash_tool(session_id=started["session_id"], action="poll")
    assert result["status"] == "completed"
    assert result["stdout"] == "done"


def test_shell_action_update_requires_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    started = _run_bash_tool('python3 -c "import time; time.sleep(1)"', timeout=10, background=True)
    try:
        with pytest.raises(ValueError, match="timeout"):
            _run_bash_tool(session_id=started["session_id"], action="update")
    finally:
        _run_bash_tool(session_id=started["session_id"], action="kill")


def test_render_shell_text_reports_update_confirmation() -> None:
    from lemoncrow.gateway.adapters.mcp_server import _render_bash_text

    text = _render_bash_text(
        {
            "status": "running",
            "session_id": "abc123",
            "updated": True,
            "timeout": 5.0,
            "timeout_remaining_ms": 4800,
        }
    )
    assert "id=abc123" in text
    assert "4s" in text


def test_render_shell_text_reports_update_failure_when_already_finished() -> None:
    from lemoncrow.gateway.adapters.mcp_server import _render_bash_text

    text = _render_bash_text({"status": "completed", "session_id": "abc123", "updated": False})
    assert "abc123" in text
    assert "completed" in text


def test_shell_status_action_is_nonblocking_and_reports_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    started = _run_bash_tool(
        'python3 -c "import sys, time; [print(i, flush=True) for i in range(20)]; time.sleep(5)"',
        timeout=10,
        background=True,
    )
    try:
        time.sleep(0.5)  # let the printed lines land before peeking

        before = time.monotonic()
        status = _run_bash_tool(session_id=started["session_id"], action="status")
        elapsed = time.monotonic() - before

        assert elapsed < 2.0, "status must not block until the command finishes"
        assert status["status"] == "running"
        assert status["session_id"] == started["session_id"]
        assert status["tail_lines"] == 10
        assert status["stdout"].splitlines() == [str(i) for i in range(10, 20)]

        # A status peek must not reap the session -- poll/cancel still work after.
        cancelled = _run_bash_tool(session_id=started["session_id"], action="kill")
        assert cancelled["status"] == "cancelled"
    finally:
        with contextlib.suppress(Exception):
            _run_bash_tool(session_id=started["session_id"], action="kill")


def test_shell_status_action_reports_finished_session_without_reaping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    started = _run_bash_tool(
        "python3 -c \"print('done')\"",
        timeout=10,
        background=True,
    )
    time.sleep(0.3)  # let the process finish

    status = _run_bash_tool(session_id=started["session_id"], action="status")
    assert status["status"] == "completed"
    assert status["exit_code"] == 0
    assert "done" in status["stdout"]

    # The finished session is still reap-able by an explicit poll afterward.
    polled = _run_bash_tool(session_id=started["session_id"], action="poll")
    assert polled["status"] == "completed"
    assert polled["stdout"] == "done"


def test_render_shell_text_status_action_notes_tail_window() -> None:
    from lemoncrow.gateway.adapters.mcp_server import _render_bash_text

    text = _render_bash_text(
        {
            "status": "running",
            "session_id": "abc123",
            "pid": 42,
            "duration_ms": 1_000,
            "timeout_remaining_ms": 9_000,
            "stdout": "line1\nline2",
            "stderr": "",
            "tail_lines": 10,
        }
    )
    assert "[tail: last 10 lines]" in text
    assert "line1\nline2" in text


def test_render_bash_text_tail_braces_shared_log_prefix() -> None:
    from lemoncrow.gateway.adapters.mcp_server import _render_bash_text

    text = _render_bash_text(
        {
            "stdout": "x",
            "stderr": "",
            "exit_code": 0,
            "tail_lines": 5,
            "log_file": "/tmp/lemoncrow-bash/abc.stdout.txt",
            "log_file_stderr": "/tmp/lemoncrow-bash/abc.stderr.txt",
        }
    )
    assert "[tail: last 5 lines; full: /tmp/lemoncrow-bash/abc.{stdout.txt, stderr.txt}]" in text


def test_shell_mcp_call_returns_managed_session_for_background_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    response = _call(
        "bash",
        {
            "command": 'python3 -c "import time; time.sleep(10)"',
            "timeout": 30,
            "background": True,
        },
    )
    text = response["result"]["content"][0]["text"]
    match = re.search(r"\bid=([^;\s]+)", text)

    assert "running" in text
    assert match is not None
    cancelled = mcp_server._run_bash_tool(session_id=match.group(1), action="kill")
    assert cancelled["status"] == "cancelled"


def test_truncate_result_text_passes_small_through() -> None:
    small = "hello world"
    assert mcp_server._truncate_result_text(small, 1024) == small


def test_truncate_result_text_caps_oversized_with_notice() -> None:
    out = mcp_server._truncate_result_text("x" * 5000, 1024)
    assert len(out.encode("utf-8")) <= 1024
    assert "truncated" in out
    assert "5000" in out  # canonical footer: "[lc: truncated 5000→1024; narrow the query for full]"


def test_truncate_result_text_keeps_valid_utf8_on_multibyte_boundary() -> None:
    # 'é' encodes to 2 bytes; an odd byte limit must not yield a partial char.
    out = mcp_server._truncate_result_text("é" * 1000, 101)
    out.encode("utf-8")  # raises if the head was split mid-codepoint


def test_truncate_result_text_spills_full_text_when_tool_name_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This is the last-resort wire-byte backstop; a bare 'narrow the query'
    used to discard everything past the cut. With T7 spill enabled (default)
    and a tool_name given, the full text is persisted first.
    """
    monkeypatch.setenv("LEMONCROW_MCP_SPILL_DIR", str(tmp_path / "spill"))
    monkeypatch.delenv("LEMONCROW_TOOL_OUTPUT_SPILL", raising=False)  # default on
    middle_marker = "UNIQUE-MIDDLE-MARKER"
    text = "HEAD" + ("x" * 5000) + middle_marker + ("x" * 5000) + "TAIL"
    out = mcp_server._truncate_result_text(text, 1024, "bash")

    assert len(out.encode("utf-8")) <= 1024
    assert "[lc: truncated" in out
    match = re.search(r"full: (\S+\.txt)\]", out)
    assert match is not None
    recovered = Path(match.group(1)).read_text(encoding="utf-8")
    assert recovered == text
    assert middle_marker in recovered


def test_truncate_result_text_without_tool_name_keeps_bare_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LEMONCROW_TOOL_OUTPUT_SPILL", raising=False)
    out = mcp_server._truncate_result_text("x" * 5000, 1024)
    assert "spilled to" not in out
    assert "narrow the query" in out


def test_write_jsonrpc_backstop_replaces_oversized_frame(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_server, "_MAX_WIRE_BYTES", 256)
    huge = {"jsonrpc": "2.0", "id": 7, "result": {"content": [{"type": "text", "text": "z" * 4000}]}}
    mcp_server._write_jsonrpc(huge)
    line = capsys.readouterr().out.strip()
    frame = json.loads(line)  # a single valid JSON-RPC line, not a 4 KB blob
    assert frame["id"] == 7
    assert "error" in frame
    assert len(line.encode("utf-8")) < 4000


def test_write_jsonrpc_passes_normal_frame_through(capsys: pytest.CaptureFixture[str]) -> None:
    msg = {"jsonrpc": "2.0", "id": 9, "result": {"content": [{"type": "text", "text": "ok"}]}}
    mcp_server._write_jsonrpc(msg)
    assert json.loads(capsys.readouterr().out.strip()) == msg


def test_read_oversized_result_is_capped_not_dropped(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: an exact (expand) read of a large file produced one >16 MiB
    # JSON-RPC frame, tripping the host stdout guard and disconnecting the
    # server. The result must be truncated in place, never dropped.
    _ = store_root
    monkeypatch.setenv("LEMONCROW_MCP_MAX_RESULT_BYTES", "70000")
    big = tmp_path / "big.txt"
    big.write_text("A" * 200_000, encoding="utf-8")
    text = _result(_call("read", {"path": str(big), "full": True}))
    assert isinstance(text, str)
    assert len(text.encode("utf-8")) <= 70000
    assert "truncated" in text


def test_smart_read_single_caps_oversized_expand_at_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The source-side guard bounds an exact (expand) read of a huge file before
    # it is ever fully loaded, returning a truncated payload with the byte count.
    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("LEMONCROW_MCP_MAX_RESULT_BYTES", "70000")
    big = tmp_path / "big.log"
    big.write_text("L" * 500_000, encoding="utf-8")
    payload = mcp_server._smart_read_single(str(big), expand=True)
    assert payload["truncated"] is True
    assert payload["bytes_total"] == 500_000
    assert len(payload["content"].encode("utf-8")) <= 70000


def test_render_memory_md_compact_recall() -> None:
    out = mcp_server._render_memory_md(
        {
            "passages": [
                {"id": "pas-1", "text": "Prefer LemonCrow memory.", "source_ref": "sess#1", "tags": ["pref"]},
                {"id": "pas-2", "text": "Use uv run.", "source_ref": "sess#2", "tags": []},
            ]
        }
    )
    assert out is not None
    assert out.startswith("### memory")
    assert "- sess#1 [pref]" in out
    assert "Prefer LemonCrow memory." in out
    # repeated JSON field keys are dropped
    assert "source_ref" not in out


def test_render_memory_md_non_recall_falls_back_to_json() -> None:
    # store_fact/vote_fact responses have no `passages` list -> keep JSON.
    assert mcp_server._render_memory_md({"id": "mem-1", "fact": "x"}) is None


def test_render_search_map_md_is_compact() -> None:
    out = mcp_server._render_search_md(
        {"mode": "map", "outline": "pkg/\n  mod.py", "ranked_files": ["pkg/mod.py"], "token_count": 10}
    )
    assert out is not None
    assert out.startswith("### repo_map")
    assert "pkg/mod.py" in out
    # the JSON wrapper / bookkeeping keys are gone
    assert "token_count" not in out
    assert "ranked_files" not in out


def test_check_auto_update_respects_explicit_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEMONCROW_AUTO_UPDATE", "0")

    def _boom(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - must not run
        raise AssertionError("auto-update ran a subprocess while LEMONCROW_AUTO_UPDATE=0")

    import subprocess as _subprocess

    monkeypatch.setattr(_subprocess, "run", _boom)
    # Returns without touching subprocess.run.
    mcp_server._check_auto_update()


def _outline_payload(*section_sizes: int) -> dict[str, Any]:
    sections = [
        {
            "path": "pkg/mod.py",
            "qualified_name": f"sym{i}",
            "line": 10,
            "end_line": 99,
            "content": "x" * size,
        }
        for i, size in enumerate(section_sizes)
    ]
    return {"files": [{"path": "pkg/mod.py", "sections": sections}]}


def test_outline_lean_view_outlines_large_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_server, "_CODESEARCH_OUTLINE", True)
    payload = mcp_server._outline_lean_view(_outline_payload(100, 2000), keep_top2=False)
    small, large = payload["files"][0]["sections"]
    assert small["content"] == "x" * 100  # under _CODESEARCH_OUTLINE_MAX_CHARS: inline
    assert "content" not in large
    assert large["outline"] == "sym1 — read pkg/mod.py:L10-L99"


def test_outline_lean_view_caps_kept_top2_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    # include_source (hidden) keeps the top-2 inline, but bounded: past
    # _CODESEARCH_TOP2_MAX_CHARS the section keeps its head plus a precise read
    # pointer instead of the full dump.
    monkeypatch.setattr(mcp_server, "_CODESEARCH_OUTLINE", True)
    monkeypatch.setattr(mcp_server, "_CODESEARCH_TOP2_MAX_CHARS", 500)
    payload = mcp_server._outline_lean_view(_outline_payload(2000, 400, 2000), keep_top2=True)
    first, second, third = payload["files"][0]["sections"]
    assert first["content"].startswith("x" * 500)
    assert first["content"].endswith("[truncated — read pkg/mod.py:L10-L99]")
    assert len(first["content"]) < 600  # head + pointer, not the 2000-char dump
    assert second["content"] == "x" * 400  # top-2 within budget: untouched
    assert "content" not in third  # rank 3: outlined as usual
    assert third["outline"].startswith("sym2 — read")


def test_outline_lean_view_disabled_is_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_server, "_CODESEARCH_OUTLINE", False)
    raw = _outline_payload(5000)
    assert mcp_server._outline_lean_view(raw, keep_top2=False) == _outline_payload(5000)


def test_outline_lean_view_disabled_still_caps_pathological_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    # The hard head cap is UNCONDITIONAL: even with outline mode off, a single
    # pathological section is bounded to head + precise read pointer.
    monkeypatch.setattr(mcp_server, "_CODESEARCH_OUTLINE", False)
    monkeypatch.setattr(mcp_server, "_CODESEARCH_TOP2_MAX_CHARS", 500)
    payload = mcp_server._outline_lean_view(_outline_payload(2000, 100), keep_top2=False)
    big, small = payload["files"][0]["sections"]
    assert big["content"].startswith("x" * 500)
    assert big["content"].endswith("[truncated — read pkg/mod.py:L10-L99]")
    assert small["content"] == "x" * 100


def test_render_code_search_md_is_compact() -> None:
    payload = {
        "exact_match": True,
        "files": [
            {
                "path": "src/orders.py",
                "sections": [
                    {
                        "qualified_name": "OrderService.calculate_total",
                        "line": 12,
                        "end_line": 20,
                        "content": "12\tdef calculate_total(self):\n13\t    return 1\n",
                    },
                    {
                        "qualified_name": "big",
                        "line": 30,
                        "end_line": 99,
                        "outline": "big — read src/orders.py:L30-L99.",
                    },
                ],
            }
        ],
        "related_symbols": [
            {
                "qualified_name": "tokens_saved",
                "path": "frontend/src/api.ts",
                "line": 537,
                "end_line": 537,
                "kind": "variable",
            },
            {
                "qualified_name": "tokens_saved",
                "path": "frontend/src/api.ts",
                "line": 611,
                "end_line": 611,
                "kind": "variable",
            },
            {"qualified_name": "helper", "path": "src/util.py", "line": 5, "end_line": 40, "kind": "function"},
        ],
        "candidate_files": ["src/a.py", "src/b.py"],
        "tokens_saved": 999,
        "calls_saved": 3,
    }
    text = mcp_server._render_code_search_md(payload)
    assert text is not None
    # sections: heading + sparse-gutter source (first line of a contiguous run
    # keeps its number, followers drop theirs); outline pointer verbatim
    assert "## src/orders.py" in text
    assert "### OrderService.calculate_total L12-L20" in text
    assert "12\tdef calculate_total(self):" in text
    assert "13\t" not in text
    assert "\n    return 1" in text
    assert "big — read src/orders.py:L30-L99." in text
    # related symbols: one line per symbol, same-symbol ranges merged
    assert "frontend/src/api.ts:L537,L611 variable tokens_saved" in text
    # majority kind "function" is untagged; non-function kinds keep theirs
    assert "src/util.py:L5-L40 helper" in text
    assert "function" not in text.split("related_symbols:")[1]
    # candidates: single comma-joined line, adjacent same-dir entries grouped;
    # bookkeeping never rendered
    assert "candidate_files: src/{a.py,b.py}" in text
    assert "999" not in text
    assert "exact match" not in text  # exact hits carry no banner
    # the render is line-shaped, not JSON
    assert '"related_symbols"' not in text


def test_compress_candidate_files_groups_adjacent_same_directory() -> None:
    paths = [
        "src/a/one.py",
        "src/a/two.py",
        "src/b/three.py",
        "src/a/four.py",
        "README.md",
    ]
    text = mcp_server._compress_candidate_files(paths)
    # a/one.py + a/two.py are ADJACENT -> grouped. src/b/three.py breaks the
    # run, so the later src/a/four.py is a separate single-file segment (never
    # reordered into the earlier group -- that would shift MRR-scored rank).
    assert text == "src/a/{one.py,two.py}, src/b/three.py, src/a/four.py, README.md"


@pytest.mark.slow  # importing eval_external_provider_mrr loads the full gold-pairs
# corpus at module level (~7k pairs) -- correct, just too slow for the default run.
def test_compress_candidate_files_round_trips_through_eval_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cross-module contract: LemonCrow's compressor and the retrieval eval's
    # parser must agree, or every LemonCrow-channel MRR run silently corrupts
    # candidate_files ranks (the eval has no structuredContent to fall back to
    # -- this compact-markdown path is the ONLY one it ever exercises).
    import importlib
    import sys
    import types
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2]
    import benchmarks

    benchmark_paths = list(getattr(benchmarks, "__path__", []))
    for path in (str(root / "benchmarks"), str(root / "src" / "benchmarks")):
        if path not in benchmark_paths:
            benchmark_paths.append(path)
    benchmarks.__path__ = benchmark_paths
    codebench_pkg = sys.modules.get("benchmarks.codebench")
    if codebench_pkg is None:
        codebench_pkg = types.ModuleType("benchmarks.codebench")
        sys.modules["benchmarks.codebench"] = codebench_pkg
    codebench_paths = list(getattr(codebench_pkg, "__path__", []))
    root_codebench_path = str(root / "benchmarks" / "codebench")
    if root_codebench_path not in codebench_paths:
        codebench_paths.append(root_codebench_path)
    codebench_pkg.__path__ = codebench_paths
    sys.modules.pop("benchmarks.codebench.eval_external_provider_mrr", None)
    # The script does module-level `_parser.parse_known_args()` (it's a CLI
    # entry point, not a library) with a required --provider -- give it a
    # valid argv for the duration of the import.
    monkeypatch.setattr(sys, "argv", ["eval_external_provider_mrr.py", "--provider", "lemoncrow"])
    _eval_mrr = importlib.import_module("benchmarks.codebench.eval_external_provider_mrr")

    paths = [
        "src/a/one.py",
        "src/a/two.py",
        "src/a/three.py",
        "src/b/four.py",
        "src/c/five.py",
        "src/c/six.py",
        "README.md",
        "lib/seven.py",
    ]
    line = mcp_server._compress_candidate_files(paths)
    recovered: list[str] = []
    for seg in _eval_mrr._split_candidate_files_line(line):
        recovered.extend(_eval_mrr._expand_candidate_segment(seg))
    assert recovered == paths


def test_render_code_search_md_flags_inexact_and_truncated() -> None:
    payload = {"exact_match": False, "files": [], "candidate_files": ["src/a.py"], "truncated": True}
    text = mcp_server._render_code_search_md(payload)
    assert text is not None
    assert text.startswith("no exact match -- ranked candidates")
    assert "(truncated; narrow with paths=)" in text


def test_sparse_gutter_reanchors_after_elisions_and_markers() -> None:
    content = "\n".join(
        [
            "10\tdef f():",  # run start: keeps number
            "11\t    x = 1",  # consecutive: dropped
            "12\t    return x",  # consecutive: dropped
            "\t…",  # skeleton elision: resets the run
            "20\tdef g():",  # new run after elision: keeps number
            "21\t    pass",  # consecutive: dropped
            "30\tdef h():",  # non-consecutive jump: keeps number
            "... [truncated after L30 — L45]",  # marker: resets, passes through
        ]
    )
    out = mcp_server._sparse_gutter(content)
    assert out.split("\n") == [
        "10\tdef f():",
        "    x = 1",
        "    return x",
        "\t…",
        "20\tdef g():",
        "    pass",
        "30\tdef h():",
        "... [truncated after L30 — L45]",
    ]


def test_sparse_gutter_reanchors_every_k_lines_within_a_run() -> None:
    # 12 consecutive lines: anchors at run positions 0, 5, 10 bound the offset
    # counting for any in-between line to <=4 lines from the anchor above.
    content = "\n".join(f"{100 + i}\tline{i}" for i in range(12))
    out = mcp_server._sparse_gutter(content).split("\n")
    numbered = [i for i, line in enumerate(out) if mcp_server._SECTION_GUTTER_RE.match(line)]
    assert numbered == [0, 5, 10]
    assert out[0] == "100\tline0"
    assert out[5] == "105\tline5"
    assert out[10] == "110\tline10"
    assert out[1] == "line1"


def test_sparse_gutter_leaves_unnumbered_content_untouched() -> None:
    content = "plain text\nno numbers here"
    assert mcp_server._sparse_gutter(content) == content


def test_lean_code_search_view_candidates_keep_symbol_map_paths() -> None:
    # candidate_files is the RANKED recall surface (rank-consumers read files +
    # candidate_files, never the symbol map): a path carried only by a
    # related_symbols pointer MUST still appear here. Excluding map-covered
    # paths deleted the top entry-point files from the ranking and was part of
    # a measured -0.10 overall retrieval MRR regression (2026-07-06). Only
    # returned-source paths are excluded.
    payload = {
        "exact_match": True,
        "entry_points": [
            {
                "qualified_name": "f",
                "path": "src/covered.py",
                "line": 1,
                "end_line": 9,
                "kind": "function",
                "score": 9.0,
            }
        ],
        "files": [
            {
                "path": "src/main.py",
                "source_sections": [
                    {"qualified_name": "main", "path": "src/main.py", "line": 1, "end_line": 3, "content": "1\tx\n"}
                ],
            },
            {"path": "src/covered.py", "source_sections": []},
            {"path": "src/fresh.py", "source_sections": []},
        ],
    }
    lean = mcp_server._lean_code_search_view(payload, max_files=1)
    # main.py: source returned -> excluded. covered.py: symbol-map pointer AND
    # ranked candidate -> kept. fresh.py: plain ranked candidate -> kept.
    assert lean["candidate_files"] == ["src/covered.py", "src/fresh.py"]


def test_lean_code_search_view_candidate_files_capped_tighter_when_dominant() -> None:
    # The exact-single-winner case (`dominant`: exact match, no seed scope,
    # one file scoring far above any rival) is the SAME signal that already
    # collapses `files` to n_src=1 -- it also shrinks the candidate_files
    # tail to _LEAN_CANDIDATE_FILES_DOMINANT (5) instead of the general 8.
    # This is deliberately NOT keyed on any per-candidate score: that was
    # tried and falsified by
    # test_lean_code_search_view_candidates_keep_symbol_map_paths's fresh.py
    # case. Reusing the whole-response `dominant` signal instead avoids that
    # trap.
    payload = {
        "exact_match": True,
        "entry_points": [
            {
                "qualified_name": "f",
                "path": "src/winner.py",
                "line": 1,
                "end_line": 9,
                "kind": "function",
                "score": 9.0,
            }
        ],
        "files": [{"path": "src/winner.py", "source_sections": []}]
        + [{"path": f"src/other{i}.py", "source_sections": []} for i in range(10)],
    }
    lean = mcp_server._lean_code_search_view(payload, max_files=1)
    assert len(lean["candidate_files"]) == 5
    # 11 total candidates, 5 shown -> 6 hidden, named explicitly.
    assert lean["candidate_files_more"] == 6


def test_lean_code_search_view_candidate_files_more_absent_when_nothing_hidden() -> None:
    # No cap was actually hit -> no candidate_files_more key at all (its mere
    # presence, even at 0, would read as "something is being withheld").
    payload = {
        "exact_match": False,
        "entry_points": [],
        "files": [{"path": f"src/other{i}.py", "source_sections": []} for i in range(3)],
    }
    lean = mcp_server._lean_code_search_view(payload, max_files=1)
    assert len(lean["candidate_files"]) == 3
    assert "candidate_files_more" not in lean


def test_render_code_search_md_hints_hidden_candidate_files_via_limit() -> None:
    # `limit` is code_search's one real, visible count param (candidate_files
    # cap, collapsed from the old separate max_candidates/max_files split) --
    # the hint must point at it directly, not the retired name.
    payload = {
        "exact_match": False,
        "files": [],
        "candidate_files": ["src/a.py", "src/b.py", "src/c.py"],
        "candidate_files_more": 4,
    }
    text = mcp_server._render_code_search_md(payload)
    assert text is not None
    assert "(+4 more; pass limit=7 to see them)" in text
    assert "max_candidates=" not in text


def test_query_words_normalizes_and_drops_short_tokens() -> None:
    assert mcp_server._query_words("fail-on-stale stale-days debt_cmd lc-debt") == {
        "fail",
        "stale",
        "days",
        "debt",
        "cmd",
    }
    assert "lc" not in mcp_server._query_words("lc debt stale")  # 2 chars, dropped
    # "on" (2 chars) dropped; case-insensitive.
    assert "on" not in mcp_server._query_words("turn ON the fail on stale flag")


def test_check_repeat_query_flags_the_debt_benchmark_regression() -> None:
    # The exact near-duplicate query sequence observed in a real debt-
    # benchmark rep (2026-07-25, 11+ near-identical code_search calls in one
    # run) -- at least one consecutive pair must be flagged, proving this
    # would have caught that regression.
    mcp_server._RECENT_CODE_SEARCH_QUERIES.clear()
    queries = [
        "fail-on-stale stale-days debt_cmd lc-debt",
        "stale_days fail_on_stale debt stale age git blame",
        "test_debt fail_on_stale stale_days debt_cmd",
        "_line_age_days git blame annotate stale",
        "line age days git blame last modified",
    ]
    flags = [mcp_server._check_repeat_query(q) for q in queries]
    assert all(isinstance(f, bool) for f in flags)
    assert flags[0] is False  # nothing recorded yet
    assert any(flags[1:]), "expected at least one repeat-query flag"


def test_check_repeat_query_false_for_distinct_topics() -> None:
    mcp_server._RECENT_CODE_SEARCH_QUERIES.clear()
    assert mcp_server._check_repeat_query("parse yaml config loader") is False
    assert mcp_server._check_repeat_query("websocket reconnect backoff timer") is False
    assert mcp_server._check_repeat_query("database connection pool sizing") is False


def test_check_repeat_query_records_into_session_bucket() -> None:
    mcp_server._RECENT_CODE_SEARCH_QUERIES.clear()
    mcp_server._check_repeat_query("parse yaml config loader")
    bucket = mcp_server._recent_code_search_queries()
    assert len(bucket) == 1
    assert bucket[0][1] == "parse yaml config loader"
    assert bucket[0][2] == mcp_server._query_words("parse yaml config loader")


def test_check_repeat_query_ignores_entries_outside_time_window() -> None:
    # A daemon can outlive many separate conversations (confirmed this
    # session) -- a query from a previous, unrelated chat must not blank a
    # brand-new conversation's first call just because the words overlap.
    mcp_server._RECENT_CODE_SEARCH_QUERIES.clear()
    mcp_server._check_repeat_query("fail-on-stale stale-days debt_cmd")
    bucket = mcp_server._recent_code_search_queries()
    ts, text, words = bucket[0]
    bucket[0] = (ts - mcp_server._RECENT_QUERY_WINDOW_SECONDS - 1, text, words)
    assert mcp_server._check_repeat_query("stale_days fail_on_stale debt marker age") is False


def test_render_code_search_md_renders_blank_result_with_no_hint_text() -> None:
    # A flagged repeat query returns blank results with NO explanatory text --
    # renders as just the bare "no exact match" line, nothing more (measured:
    # the caller skims past prose hints and searches again anyway).
    payload = {"exact_match": False, "files": []}
    text = mcp_server._render_code_search_md(payload)
    assert text == "no exact match -- ranked candidates"
    assert "[lc:" not in text


def test_lean_code_search_view_candidate_files_keeps_full_window_when_not_dominant() -> None:
    # Same 20-file shape, but exact_match=False -> `dominant` never fires
    # (it requires exact_match). The recall tail stays the full 8-wide net.
    payload = {
        "exact_match": False,
        "entry_points": [],
        "files": [{"path": f"src/other{i}.py", "source_sections": []} for i in range(20)],
    }
    lean = mcp_server._lean_code_search_view(payload, max_files=1)
    assert len(lean["candidate_files"]) == 8


def test_lean_code_search_view_max_candidates_overrides_both_defaults() -> None:
    # An explicit max_candidates wins over BOTH the dominant (5) and general
    # (8) defaults -- the caller's own confidence, not a heuristic, decides.
    payload = {
        "exact_match": False,
        "entry_points": [],
        "files": [{"path": f"src/other{i}.py", "source_sections": []} for i in range(20)],
    }
    lean = mcp_server._lean_code_search_view(payload, max_files=1, max_candidates=3)
    assert len(lean["candidate_files"]) == 3
    # A dominant response asking for MORE than its own 5-default also gets it.
    dominant_payload = {
        "exact_match": True,
        "entry_points": [
            {"qualified_name": "f", "path": "src/winner.py", "line": 1, "end_line": 9, "kind": "function", "score": 9.0}
        ],
        "files": [{"path": "src/winner.py", "source_sections": []}]
        + [{"path": f"src/other{i}.py", "source_sections": []} for i in range(20)],
    }
    lean_dominant = mcp_server._lean_code_search_view(dominant_payload, max_files=1, max_candidates=12)
    assert len(lean_dominant["candidate_files"]) == 12


def _doc_heavy_payload() -> dict[str, object]:
    # 6 markdown headings outscore 2 code symbols; without the filetype
    # diversity pass the whole related_symbols window is documentation.
    eps = [
        {
            "qualified_name": f"title: doc {i}",
            "path": f"docs/page_{i}.md",
            "line": 1,
            "end_line": 4,
            "kind": "heading",
            "score": 20.0 - i,
        }
        for i in range(6)
    ]
    eps += [
        {
            "qualified_name": "handler",
            "path": "src/server.py",
            "line": 5,
            "end_line": 9,
            "kind": "function",
            "score": 2.0,
        },
        {"qualified_name": "parse", "path": "src/parse.py", "line": 1, "end_line": 3, "kind": "function", "score": 1.0},
    ]
    files = [{"path": e["path"], "source_sections": []} for e in eps]
    return {"exact_match": False, "entry_points": eps, "files": files}


def test_lean_code_search_view_caps_doc_filetype_in_ranked_windows() -> None:
    lean = mcp_server._lean_code_search_view(_doc_heavy_payload(), max_files=1, query="read tool png support")
    # related_symbols window (8): docs capped at a quarter (2); the code
    # symbols are promoted ahead of the demoted markdown overflow.
    rel_paths = [s["path"] for s in lean["related_symbols"]]
    assert rel_paths[:4] == ["docs/page_0.md", "docs/page_1.md", "src/server.py", "src/parse.py"]
    # candidate_files keeps every path (permutation, not a filter) with the
    # same promotion applied.
    cands = lean["candidate_files"]
    assert cands[:4] == ["docs/page_0.md", "docs/page_1.md", "src/server.py", "src/parse.py"]
    assert sorted(cands) == sorted({str(e["path"]) for e in _doc_heavy_payload()["entry_points"]})  # type: ignore[union-attr]


def test_lean_code_search_view_doc_query_skips_doc_capping() -> None:
    lean = mcp_server._lean_code_search_view(_doc_heavy_payload(), max_files=1, query="where are the docs for setup")
    rel_paths = [s["path"] for s in lean["related_symbols"]]
    # Doc-seeking query: score order untouched, markdown keeps its slots.
    assert rel_paths[:6] == [f"docs/page_{i}.md" for i in range(6)]


def test_shell_high_timeout_waits_out_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`timeout` is a pure wait budget: set high it blocks to the finished
    result in one call; past a low one a still-running handle comes back
    (bench transcripts: agents sleep-polled handles for up to ~30 turns
    because they read `timeout` as a kill deadline and never raised it)."""
    from lemoncrow.gateway.adapters.mcp_server import _run_bash_tool

    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))

    result = _run_bash_tool("sleep 1.3; echo waited-out", timeout=30)
    assert result.get("status") != "running"
    assert result["exit_code"] == 0
    assert "waited-out" in result["stdout"]

    # Control: a low timeout hands back a running handle (and never kills).
    handle = _run_bash_tool("sleep 30", timeout=1)
    assert handle.get("status") == "running"
    _run_bash_tool(session_id=str(handle["session_id"]), action="kill")


def test_smart_edit_new_file_sources_replacement_content(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = store_root
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))
    payload_file = tmp_path / "payload.txt"
    payload_file.write_text("BIG GENERATED BODY\n" * 5, encoding="utf-8")
    target = Path("out.txt")

    result = tool_smart_edit(
        {"post_edit_hooks": False, "edits": [{"path": str(target), "new_file": str(payload_file), "replace": True}]}
    )

    assert not result.get("failed")
    assert target.read_text(encoding="utf-8") == payload_file.read_text(encoding="utf-8")


def test_smart_edit_new_file_missing_is_a_clear_error(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = store_root
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(Exception, match="new_file"):
        tool_smart_edit(
            {"post_edit_hooks": False, "edits": [{"path": "out.txt", "new_file": "nope-missing.txt", "replace": True}]}
        )


def test_smart_edit_old_without_new_rejected(store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Schema now requires only `path` (so a hidden new_file retry validates
    # client-side); a bare {path, old} must be rejected loudly -- treating a
    # missing new as "" would silently DELETE the matched text.
    _ = store_root
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(Exception, match="new"):
        tool_smart_edit({"post_edit_hooks": False, "edits": [{"path": "out.txt", "old": "something"}]})


def test_smart_edit_failed_edit_preserves_large_new_payload(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed edit must never force the model to REGENERATE a big `new`
    payload: the content is preserved to the spill store and the retry
    references it via new_file."""
    _ = store_root
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("LEMONCROW_MCP_SPILL_DIR", str(tmp_path / "spill"))
    target = Path("mod.py")
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    big = "# regenerated-content-line\n" * 200  # comfortably past the 2000-char preserve floor

    result = tool_smart_edit(
        {
            "post_edit_hooks": False,
            "edits": [{"path": str(target), "old_string": "ANCHOR THAT DOES NOT EXIST ANYWHERE", "new_string": big}],
        }
    )

    failed = result.get("failed") or []
    assert failed, result
    preserved = failed[0].get("new_preserved_at")
    assert preserved, failed[0]
    assert Path(preserved).read_text(encoding="utf-8") == big

    # The no-regeneration retry: same content, referenced by path.
    retry = tool_smart_edit(
        {"post_edit_hooks": False, "edits": [{"path": str(target), "new_file": preserved, "replace": True}]}
    )
    assert not retry.get("failed")
    assert target.read_text(encoding="utf-8") == big
