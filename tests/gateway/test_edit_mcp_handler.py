"""Comprehensive MCP-level tests for the `edit` tool handler (tool_smart_edit).

These tests exercise tool_smart_edit end-to-end through the _handle dispatcher,
covering all six descriptor families, atomicity, hooks, diff recording, and edge
cases that were previously untested.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from lemoncrow.gateway.adapters import mcp_server
from lemoncrow.gateway.adapters.mcp_server import tool_smart_edit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
from tests.helpers import init_store_at


def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    req: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    }
    resp = mcp_server._handle(req)
    assert isinstance(resp, dict)
    return resp


def _result(resp: dict[str, Any]) -> dict[str, Any]:
    assert "result" in resp, resp
    text = resp["result"]["content"][0]["text"]
    # Clean success renders "ok" (no ranges) or "applied path:line[, ...]" (the
    # minimal orientation echo); normalize both so callers can assert structurally.
    if text == "ok":
        return {}
    if text.startswith("applied "):
        return {"applied": text[len("applied ") :].split(", ")}
    payload = json.loads(text)
    assert isinstance(payload, dict)
    return payload


def _edit(args: dict[str, Any]) -> dict[str, Any]:
    """Shortcut: call edit and return the parsed result."""
    return _result(_call("edit", args))


def _edit_text(args: dict[str, Any]) -> str:
    """Call edit and return the raw model-facing text (clean success == "applied path:line")."""
    resp = _call("edit", args)
    assert "result" in resp, resp
    return str(resp["result"]["content"][0]["text"])


def _assert_silent_success(payload: dict[str, Any]) -> None:
    """A clean exact-match edit carries only the minimal `applied` range echo.

    `applied` is a list of compact "path:line" strings (orientation, not a diff)
    so the model need not re-read the file it just edited; nothing else actionable
    (`failed`/`rolled_back`/`writes`/`hooks`/`diagnostics`). (`calls_saved` is
    internal accounting the dispatcher strips before rendering.)
    """
    assert payload.get("applied"), f"clean success must echo applied ranges: {payload}"
    assert all(isinstance(a, str) for a in payload["applied"]), payload
    for key in ("failed", "rolled_back", "writes", "hooks", "diagnostics"):
        assert key not in payload, f"clean success must not carry {key!r}: {payload}"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / ".lemoncrow"
    init_store_at(str(root))
    monkeypatch.setenv("LEMONCROW_ROOT", str(root))
    monkeypatch.setenv("CLAUDE_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "test-gnd-session")
    monkeypatch.chdir(tmp_path)
    mcp_server._ledger._current_ledger = None
    mcp_server._ledger._realtime_ctx = None
    mcp_server._remote_client = MagicMock()
    mcp_server._remote_client.get_context.return_value = {"context": "", "run_ledger": []}
    return tmp_path


# ---------------------------------------------------------------------------
# #1  Rich descriptor family
# ---------------------------------------------------------------------------


def test_rich_replace_basic(workspace: Path) -> None:
    """Rich replace updates text and returns an applied hunk."""
    f = workspace / "hello.py"
    f.write_text("def greet():\n    return HELLO\n", encoding="utf-8")

    payload = _edit(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "hello.py", "old_string": "HELLO", "new_string": "WORLD"}],
        }
    )

    # Clean exact-match edit echoes the minimal applied range (orientation only).
    assert payload.get("applied") == ["hello.py:2"]
    assert "failed" not in payload
    assert "rolled_back" not in payload
    assert "writes" not in payload
    assert "WORLD" in f.read_text(encoding="utf-8")
    assert "HELLO" not in f.read_text(encoding="utf-8")


def test_rich_create_new_file_via_overwrite(workspace: Path) -> None:
    """Rich overwrite with no existing file creates it from scratch."""
    f = workspace / "new_module.py"
    assert not f.exists()

    payload = _edit(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "new_module.py", "new_string": "# new\n", "overwrite": True}],
        }
    )

    # An overwrite/create is a clean success too: minimal body, file created.
    assert "failed" not in payload
    assert f.exists()
    assert f.read_text(encoding="utf-8") == "# new\n"


def test_rich_overwrite_replaces_existing_file(workspace: Path) -> None:
    """Rich overwrite with an existing file replaces its full content."""
    f = workspace / "config.txt"
    f.write_text("old content\n", encoding="utf-8")

    payload = _edit(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "config.txt", "new_string": "new content\n", "overwrite": True}],
        }
    )

    assert "failed" not in payload
    assert f.read_text(encoding="utf-8") == "new content\n"


def test_rich_overwrite_with_line_range_rejected(workspace: Path) -> None:
    """overwrite=true + a #line range is a contradiction: reject, don't truncate."""
    f = workspace / "big.py"
    original = "".join(f"line_{i} = {i}\n" for i in range(1, 21))
    f.write_text(original, encoding="utf-8")

    payload = _edit({"edits": [{"file_path": "big.py:L5-L10", "new_string": "replacement\n", "overwrite": True}]})

    assert payload["rolled_back"] is True
    assert payload["failed"]
    assert "ignores the :L5-L10 line range" in payload["failed"][0]["error"]
    # The file must be untouched — not emptied, not partially overwritten.
    assert f.read_text(encoding="utf-8") == original


def test_rich_overwrite_empty_string_truncation_rejected(workspace: Path) -> None:
    """overwrite=true with an empty new_string must not silently zero a non-empty file."""
    f = workspace / "keep.py"
    f.write_text("def keep():\n    return 1\n", encoding="utf-8")

    payload = _edit({"edits": [{"file_path": "keep.py", "overwrite": True}]})

    assert payload["rolled_back"] is True
    assert payload["failed"]
    assert "truncate non-empty file" in payload["failed"][0]["error"]
    assert f.read_text(encoding="utf-8") == "def keep():\n    return 1\n"


def test_rich_overwrite_empty_string_allowed_for_new_file(workspace: Path) -> None:
    """Creating an empty file via overwrite stays allowed — nothing is destroyed."""
    f = workspace / "fresh.py"
    assert not f.exists()

    payload = _edit(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "fresh.py", "new_string": "", "overwrite": True}],
        }
    )

    assert "failed" not in payload
    assert f.exists()
    assert f.read_text(encoding="utf-8") == ""


def test_rich_line_anchor_restricts_scope(workspace: Path) -> None:
    """file_path#line scopes the replacement to that line only."""
    f = workspace / "scope.py"
    f.write_text("x = 1\nx = 2\nx = 3\n", encoding="utf-8")

    # Only replace the x = 2 on line 2 — the other 'x = 1' must stay
    payload = _edit(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "scope.py:L2", "old_string": "x = 2", "new_string": "x = 99"}],
        }
    )

    assert "failed" not in payload
    text = f.read_text(encoding="utf-8")
    assert "x = 99" in text
    assert "x = 1" in text
    assert "x = 3" in text


def test_rich_multi_file_batch_all_applied(workspace: Path) -> None:
    """Multiple rich edits across different files succeed atomically."""
    f1 = workspace / "a.py"
    f2 = workspace / "b.py"
    f1.write_text("alpha\n", encoding="utf-8")
    f2.write_text("beta\n", encoding="utf-8")

    edits = [
        {"file_path": "a.py", "old_string": "alpha", "new_string": "ALPHA"},
        {"file_path": "b.py", "old_string": "beta", "new_string": "BETA"},
    ]
    payload = _edit({"post_edit_hooks": False, "edits": edits})

    # Clean multi-file success echoes the minimal applied ranges, one per file.
    assert "failed" not in payload
    assert sorted(payload.get("applied", [])) == ["a.py:1", "b.py:1"]
    assert f1.read_text(encoding="utf-8") == "ALPHA\n"
    assert f2.read_text(encoding="utf-8") == "BETA\n"
    # ...but the internal cross-file savings credit is preserved on the
    # structured result (the dispatcher reads calls_saved before rendering).
    f1.write_text("alpha\n", encoding="utf-8")
    f2.write_text("beta\n", encoding="utf-8")
    structured = tool_smart_edit({"post_edit_hooks": False, "edits": edits})
    assert structured == {"applied": ["a.py:1", "b.py:1"], "calls_saved": 1}


def test_recovered_fuzzy_edit_credits_retry_loop(workspace: Path) -> None:
    """A hunk that landed via non-exact recovery credits the avoided
    failed-edit → re-read → re-edit chain: 2 roundtrips per recovered file
    (a byte-exact vanilla Edit would have rejected the stale old_string)."""
    f = workspace / "fz.py"
    f.write_text("def alpha():\n    return   1\n", encoding="utf-8")
    structured = tool_smart_edit(
        {
            "post_edit_hooks": False,
            "edits": [
                {
                    "file_path": "fz.py",
                    # Interior whitespace differs from disk → exact match fails,
                    # the normalized/fuzzy recovery lands it.
                    "old_string": "def alpha():\n    return 1",
                    "new_string": "def alpha():\n    return 2",
                }
            ],
        }
    )
    assert f.read_text(encoding="utf-8") == "def alpha():\n    return 2\n"
    assert structured.get("calls_saved") == 2
    # The recovered entry stays a dict carrying match_mode so the agent verifies.
    modes = [e.get("match_mode") for e in structured.get("applied", []) if isinstance(e, dict)]
    assert modes and modes[0] in ("normalized", "placeholder", "fuzzy", "minified")


def test_rich_multi_file_atomic_rollback(workspace: Path) -> None:
    """Second edit fails → atomic=True rolls back the first edit too."""
    f1 = workspace / "r1.py"
    f1.write_text("original\n", encoding="utf-8")

    payload = _edit(
        {
            "atomic": True,
            "edits": [
                {"file_path": "r1.py", "old_string": "original", "new_string": "changed"},
                {"file_path": "r1.py", "old_string": "does-not-exist", "new_string": "x"},
            ],
        }
    )

    assert payload["rolled_back"] is True
    assert f1.read_text(encoding="utf-8") == "original\n"


def test_rich_non_atomic_partial_success(workspace: Path) -> None:
    """With atomic=False a failing second edit doesn't undo the first."""
    f = workspace / "partial.py"
    f.write_text("keep this\nbad target here\n", encoding="utf-8")

    payload = _edit(
        {
            "atomic": False,
            "edits": [
                {"file_path": "partial.py", "old_string": "keep this", "new_string": "KEPT"},
                {"file_path": "partial.py", "old_string": "no such string", "new_string": "x"},
            ],
        }
    )

    # Partial success is loud (non-empty failed) but sheds noise: a false
    # rolled_back is stripped, while applied + the real failures stay.
    assert "rolled_back" not in payload
    assert len(payload["applied"]) >= 1
    assert len(payload["failed"]) >= 1
    assert "KEPT" in f.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# #2  Legacy descriptor family
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# #3  Notebook cell edits
# ---------------------------------------------------------------------------


def test_notebook_cell_insert_through_handler(workspace: Path) -> None:
    """Notebook cell_action=insert_after adds a new cell via MCP."""
    nb_path = workspace / "nb.ipynb"
    nb_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "source": "x = 1",
                        "outputs": [],
                        "execution_count": None,
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )

    payload = _edit(
        {
            "edits": [
                {
                    "file_path": "nb.ipynb#cell=0",
                    "cell_action": "insert_after",
                    "cell_type": "markdown",
                    "new_string": "# heading",
                }
            ]
        }
    )

    assert "failed" not in payload
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 2
    assert notebook["cells"][1]["cell_type"] == "markdown"
    assert notebook["cells"][1]["source"] == "# heading"


def test_notebook_cell_overwrite_clears_outputs(workspace: Path) -> None:
    """Overwriting a notebook cell via #cell=N resets its outputs."""
    nb_path = workspace / "out.ipynb"
    nb_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "source": "print(1)",
                        "outputs": [{"output_type": "stream", "text": "1\n"}],
                        "execution_count": 3,
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )

    payload = _edit({"edits": [{"file_path": "out.ipynb#cell=0", "overwrite": True, "new_string": "print(2)"}]})

    assert "failed" not in payload
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    assert notebook["cells"][0]["source"] == "print(2)"
    assert notebook["cells"][0]["outputs"] == []
    assert notebook["cells"][0]["execution_count"] is None


# ---------------------------------------------------------------------------
# #4  post_edit_hooks behaviour
# ---------------------------------------------------------------------------


def test_post_edit_hooks_false_no_diagnostics_key(workspace: Path) -> None:
    """With post_edit_hooks=False the result has no diagnostics/hooks keys."""
    f = workspace / "nohook.py"
    f.write_text("x = 1\n", encoding="utf-8")

    payload = _edit(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "nohook.py", "old_string": "x = 1", "new_string": "x = 2"}],
        }
    )

    assert "failed" not in payload
    assert "diagnostics" not in payload
    assert "hooks" not in payload


def test_post_edit_hooks_false_diff_still_recorded(workspace: Path) -> None:
    """Diffs are recorded in the ledger even when hooks are disabled."""
    f = workspace / "difftest.py"
    f.write_text("before\n", encoding="utf-8")

    _edit(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "difftest.py", "old_string": "before", "new_string": "after"}],
        }
    )

    led = mcp_server._get_ledger()
    file_events = [e for e in led.events if e.kind == "file_edit"]
    assert any(e.payload.get("path") == "difftest.py" for e in file_events)
    matching = [e for e in file_events if e.payload.get("path") == "difftest.py"]
    assert "+after" in matching[-1].payload["diff"]


def test_edit_result_omits_inline_diff(workspace: Path) -> None:
    """The edit result never carries an inline diff; the agent re-reads to verify.

    A unified diff echoes old+new content back into context (cache-write now,
    cache-read on every later turn), so it is not returned. The diff is still
    recorded to the ledger for audit/undo.
    """
    f = workspace / "nodiff.py"
    f.write_text("x = 1\n", encoding="utf-8")

    payload = _edit({"edits": [{"file_path": "nodiff.py", "old_string": "x = 1", "new_string": "x = 2"}]})

    assert "failed" not in payload
    assert "diff" not in payload
    assert f.read_text(encoding="utf-8") == "x = 2\n"
    led = mcp_server._get_ledger()
    file_events = [e for e in led.events if e.kind == "file_edit" and e.payload.get("path") == "nodiff.py"]
    assert file_events and "+x = 2" in file_events[-1].payload["diff"]


def test_hook_exception_does_not_fail_successful_edit(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hook crash must not surface as an edit failure — the file was already written."""
    f = workspace / "hookcrash.py"
    f.write_text("old\n", encoding="utf-8")

    def exploding_hooks(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("hook toolchain not found")

    monkeypatch.setattr(
        "lemoncrow.pro.capabilities.tool_supervision.post_edit_hooks.run_post_edit_hooks",
        exploding_hooks,
    )

    payload = _edit(
        {
            "post_edit_hooks": True,
            "edits": [{"file_path": "hookcrash.py", "old_string": "old", "new_string": "new"}],
        }
    )

    # Edit must succeed even when post-edit hooks crash. Hook diagnostics are
    # intentionally stripped, so a crashed-hook success is still silent.
    assert "failed" not in payload
    assert "rolled_back" not in payload
    assert f.read_text(encoding="utf-8") == "new\n"
    assert "hooks" not in payload


def test_hook_exception_diff_still_recorded(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Diff is recorded in the ledger even when the hooks crash."""
    f = workspace / "hookdiff.py"
    f.write_text("original\n", encoding="utf-8")

    monkeypatch.setattr(
        "lemoncrow.pro.capabilities.tool_supervision.post_edit_hooks.run_post_edit_hooks",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    _edit(
        {
            "post_edit_hooks": True,
            "edits": [{"file_path": "hookdiff.py", "old_string": "original", "new_string": "modified"}],
        }
    )

    led = mcp_server._get_ledger()
    file_events = [e for e in led.events if e.kind == "file_edit" and e.payload.get("path") == "hookdiff.py"]
    assert file_events, "diff event must be recorded even after hook crash"
    assert "+modified" in file_events[-1].payload["diff"]


# ---------------------------------------------------------------------------
# #5  Edge cases and validation
# ---------------------------------------------------------------------------


def test_empty_edits_returns_error(workspace: Path) -> None:
    """An empty edits array is a protocol error."""
    resp = _call("edit", {"edits": []})
    # An execution failure surfaces as an isError tool result (MCP spec), a
    # JSON-RPC error, or a result with failed containing the message.
    result = resp.get("result") or {}
    has_error = "error" in resp or result.get("isError") is True or result.get("failed")
    assert has_error


def test_rich_path_escape_rejected(workspace: Path) -> None:
    """A path that escapes the repo root is blocked by path safety."""
    payload = _edit({"edits": [{"file_path": "../../../etc/passwd", "old_string": "root", "new_string": "hacked"}]})
    assert payload["rolled_back"] is True
    assert payload["failed"]


# ---------------------------------------------------------------------------
# #6  Ledger diff recording (multi-file)
# ---------------------------------------------------------------------------


def test_diff_recorded_per_file_multi_edit(workspace: Path) -> None:
    """Each touched file gets its own file_edit event in the ledger."""
    f1 = workspace / "m1.py"
    f2 = workspace / "m2.py"
    f1.write_text("alpha\n", encoding="utf-8")
    f2.write_text("beta\n", encoding="utf-8")

    _edit(
        {
            "post_edit_hooks": False,
            "edits": [
                {"file_path": "m1.py", "old_string": "alpha", "new_string": "ALPHA"},
                {"file_path": "m2.py", "old_string": "beta", "new_string": "BETA"},
            ],
        }
    )

    led = mcp_server._get_ledger()
    file_events = [e for e in led.events if e.kind == "file_edit"]
    paths_recorded = {e.payload["path"] for e in file_events}
    assert "m1.py" in paths_recorded
    assert "m2.py" in paths_recorded


# ---------------------------------------------------------------------------
# #7  Schema contract
# ---------------------------------------------------------------------------


def test_schema_top_level_params_have_descriptions() -> None:
    """atomic, hooks must each have a description."""
    from lemoncrow.gateway.adapters.mcp_server import EDIT_TOOL_INPUT_SCHEMA, TOOLS

    props = EDIT_TOOL_INPUT_SCHEMA["properties"]
    # atomic/hooks are hidden policy knobs: absent from the advertised schema,
    # still accepted by the handler by name.
    assert set(props) == {"edits"}
    assert props["edits"]["description"].strip()
    for hidden in ("atomic", "hooks"):
        assert hidden in TOOLS["edit"]["handler"].__wrapped__.__code__.co_varnames


def test_schema_registered_as_edit_tool() -> None:
    """The edit tool must be registered in TOOLS with a non-trivial description."""
    from lemoncrow.gateway.adapters.mcp_server import TOOLS

    assert "edit" in TOOLS
    desc = TOOLS["edit"]["description"]
    assert "batch" in desc.lower(), "description should mention batching"
    assert "re-read to verify" not in desc and "re-read after" in desc.lower()


def test_schema_edits_array_requires_min_one_item() -> None:
    """The JSON schema specifies minItems: 1 on the edits array."""
    from lemoncrow.gateway.adapters.mcp_server import EDIT_TOOL_INPUT_SCHEMA

    assert EDIT_TOOL_INPUT_SCHEMA["properties"]["edits"].get("minItems") == 1


def test_schema_documents_flat_item_shape() -> None:
    """The edits array items schema is a flat object with path/old/new/overwrite.

    The schema is intentionally lean — notebook/symbol/projection edits are
    accepted by the handler but not enumerated in the schema.
    """
    from lemoncrow.gateway.adapters.mcp_server import EDIT_TOOL_INPUT_SCHEMA

    items = EDIT_TOOL_INPUT_SCHEMA["properties"]["edits"]["items"]
    assert "anyOf" not in items
    assert set(items["properties"]) == {"path", "old", "new", "replace"}
    assert items.get("additionalProperties") is False
    assert ":Lx" in items["properties"]["path"]["description"]


def test_description_advertises_edits_wrapper() -> None:
    """The short description must show the edits=[...] wrapper, not just the
    item shape -- a cold model that reads only the description otherwise calls
    edit(path=..., new=...) flat (observed in Harbor runs)."""
    from lemoncrow.gateway.adapters.mcp_server import TOOLS

    assert "edits=[" in TOOLS["edit"]["description"]


def test_flattened_create_call_auto_recovered(workspace: Path) -> None:
    """edit(path=..., new=..., replace=True) at top level (no edits[]) is
    auto-wrapped into edits=[{...}] instead of erroring."""
    payload = _edit({"path": "gen.py", "new": "print('hi')\n", "replace": True, "hooks": False})
    # Clean create renders as silent success ({} after normalization); the
    # file on disk is the real check.
    assert "failed" not in payload, payload
    assert (workspace / "gen.py").read_text(encoding="utf-8") == "print('hi')\n"


def test_flattened_replace_call_auto_recovered(workspace: Path) -> None:
    """A flattened old/new replace at top level is lifted into edits=[...]."""
    f = workspace / "a.py"
    f.write_text("x = 1\n", encoding="utf-8")
    payload = _edit({"path": "a.py", "old": "x = 1", "new": "x = 2", "hooks": False})
    assert payload.get("applied"), payload
    assert f.read_text(encoding="utf-8") == "x = 2\n"


def test_flattened_content_alias_create_call_auto_recovered(workspace: Path) -> None:
    """edit(path=..., content=...) -- the native-host file-write convention a
    model reaches for when creating a brand-new file -- is recovered the same
    way, not rejected as an unknown argument."""
    payload = _edit({"path": "gen2.py", "content": "print('hi')\n", "hooks": False})
    assert "failed" not in payload, payload
    assert (workspace / "gen2.py").read_text(encoding="utf-8") == "print('hi')\n"


def test_flattened_call_without_content_still_rejected(workspace: Path) -> None:
    """A stray path with no new-content key is NOT an unambiguous edit -- the
    unknown-argument error must still surface."""
    resp = _call("edit", {"path": "a.py"})
    assert "unknown argument" in json.dumps(resp), resp


# ---------------------------------------------------------------------------
# #8  Success-silent / minimal model-facing contract
# ---------------------------------------------------------------------------


def test_clean_exact_match_edit_is_silent(workspace: Path) -> None:
    """A clean exact-match edit returns the minimal model-facing body.

    Success is implied by the call returning, so the rendered result is a single
    minimal token ("ok") with NO applied/failed/rolled_back/writes/hooks body.
    """
    f = workspace / "silent.py"
    f.write_text("value = 1\n", encoding="utf-8")

    text = _edit_text(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "silent.py", "old_string": "value = 1", "new_string": "value = 2"}],
        }
    )
    assert text == "applied silent.py:1"
    # The normalized dispatched payload carries the minimal applied echo.
    dispatched = _edit(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "silent.py", "old_string": "value = 2", "new_string": "value = 3"}],
        }
    )
    _assert_silent_success(dispatched)
    assert f.read_text(encoding="utf-8") == "value = 3\n"

    # The structured handler return is empty (single-file => no cross-file credit).
    f.write_text("value = 1\n", encoding="utf-8")
    structured = tool_smart_edit(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "silent.py", "old_string": "value = 1", "new_string": "value = 2"}],
        }
    )
    assert structured == {"applied": ["silent.py:1"]}


def test_failing_edit_stays_loud(workspace: Path) -> None:
    """A failing edit is actionable, so its structured failure body is preserved."""
    f = workspace / "loud.py"
    f.write_text("keep me\n", encoding="utf-8")

    payload = _edit({"edits": [{"file_path": "loud.py", "old_string": "no such text", "new_string": "x"}]})

    assert payload.get("rolled_back") is True
    assert payload["failed"]
    assert f.read_text(encoding="utf-8") == "keep me\n"


def test_rich_fuzzy_match_stays_loud(workspace: Path) -> None:
    """A rich-edit non-exact match keeps its applied entry + match_mode re-read warning.

    A `...` placeholder forces a non-exact (placeholder) match; that is
    actionable -- the agent should re-read to confirm the match did not diverge --
    so the result is NOT silenced.
    """
    f = workspace / "fuzz.py"
    f.write_text("start = 1\nmiddle = 2\nend = 3\n", encoding="utf-8")

    # The `...` placeholder spans the middle line, so the match is not literal:
    # rich edit resolves it as a placeholder match and stamps match_mode on the
    # applied entry. Hooks off so the surfaced body is driven purely by match_mode.
    payload = _edit(
        {
            "post_edit_hooks": False,
            "edits": [
                {
                    "file_path": "fuzz.py",
                    "old_string": "start = 1\n...\nend = 3",
                    "new_string": "start = 10\nend = 30",
                }
            ],
        }
    )

    assert "applied" in payload, payload
    modes = [e.get("match_mode") for e in payload["applied"] if isinstance(e, dict)]
    assert any(m and m != "exact" for m in modes), payload
    assert "start = 10" in f.read_text(encoding="utf-8")


def test_dispatcher_preserves_cross_file_calls_saved(workspace: Path) -> None:
    """Silencing the body must not break the savings accounting path.

    The model-facing text is silent, but the dispatcher still reads calls_saved
    off the structured result and writes it into content[].saved before stripping.
    """
    f1 = workspace / "acc1.py"
    f2 = workspace / "acc2.py"
    f1.write_text("one\n", encoding="utf-8")
    f2.write_text("two\n", encoding="utf-8")

    resp = _call(
        "edit",
        {
            "post_edit_hooks": False,
            "edits": [
                {"file_path": "acc1.py", "old_string": "one", "new_string": "ONE"},
                {"file_path": "acc2.py", "old_string": "two", "new_string": "TWO"},
            ],
        },
    )
    content_item = resp["result"]["content"][0]
    # Body is the minimal applied echo...
    assert content_item["text"] == "applied acc1.py:1, acc2.py:1"
    # ...but the cross-file calls_saved credit rode into content[].saved.
    assert content_item["saved"]["calls"] == 1


# ---------------------------------------------------------------------------
# Blind range edits: pre-batch coordinates + freshness guard
# ---------------------------------------------------------------------------


def test_edit_accepts_read_style_path_suffixes(workspace: Path) -> None:
    f = workspace / "selectors.txt"
    f.write_text("head\nbody\ntail\n", encoding="utf-8")

    _edit({"post_edit_hooks": False, "edits": [{"file_path": "selectors.txt:head=1", "old": "head", "new": "HEAD"}]})
    _edit({"post_edit_hooks": False, "edits": [{"file_path": "selectors.txt:tail=1", "old": "tail", "new": "TAIL"}]})
    _edit({"post_edit_hooks": False, "edits": [{"file_path": "selectors.txt:summary", "old": "body", "new": "BODY"}]})
    _edit({"post_edit_hooks": False, "edits": [{"file_path": "selectors.txt:outline", "old": "BODY", "new": "middle"}]})
    _edit({"post_edit_hooks": False, "edits": [{"file_path": "selectors.txt:full", "new": "replacement\n"}]})

    assert f.read_text(encoding="utf-8") == "replacement\n"

    open_range = workspace / "open_range.txt"
    open_range.write_text("one\ntwo\nthree\n", encoding="utf-8")
    _edit(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "open_range.txt:L2-", "old": "two\nthree\n", "new": "rest\n"}],
        }
    )
    assert open_range.read_text(encoding="utf-8") == "one\nrest\n"


def _read(path: str) -> None:
    """Serve a file through the read tool so its stat signature is recorded."""
    resp = _call("read", {"path": path})
    assert "result" in resp, resp


def test_batch_range_edits_use_pre_batch_line_numbers(workspace: Path) -> None:
    """Two range edits to one file: the later range must hit the lines the
    caller READ, not coordinates shifted by the earlier splice (regression:
    L2 growing by one line made a later L5-L5 replace pre-batch L4)."""
    f = workspace / "seq.txt"
    f.write_text("a1\na2\na3\na4\na5\na6\n", encoding="utf-8")
    _read("seq.txt")

    _edit(
        {
            "post_edit_hooks": False,
            "edits": [
                {"file_path": "seq.txt:L2-L2", "new_string": "B2\nB2b\n"},
                {"file_path": "seq.txt:L5-L5", "new_string": "B5\n"},
            ],
        }
    )

    # a5 (pre-batch L5) replaced; a4 intact despite the +1 shift above it.
    assert f.read_text(encoding="utf-8") == "a1\nB2\nB2b\na3\na4\nB5\na6\n"


def test_batch_range_edits_overlap_fails_loudly(workspace: Path) -> None:
    f = workspace / "ovl.txt"
    f.write_text("a1\na2\na3\n", encoding="utf-8")
    _read("ovl.txt")

    payload = _edit(
        {
            "post_edit_hooks": False,
            "edits": [
                {"file_path": "ovl.txt:L1-L2", "new_string": "X\n"},
                {"file_path": "ovl.txt:L2-L3", "new_string": "Y\n"},
            ],
        }
    )

    assert payload["rolled_back"] is True
    assert "overlaps" in payload["failed"][0]["error"]
    assert f.read_text(encoding="utf-8") == "a1\na2\na3\n"  # untouched


def test_range_edit_after_content_edit_same_file_coapplies(workspace: Path) -> None:
    """A content-located edit that grows the file is recorded in the ledger, so a
    later range edit's pre-batch line number is translated and both co-apply."""
    f = workspace / "mix.txt"
    f.write_text("a1\na2\na3\n", encoding="utf-8")
    _read("mix.txt")

    payload = _edit(
        {
            "post_edit_hooks": False,
            "edits": [
                {"file_path": "mix.txt", "old_string": "a1\n", "new_string": "Z1\nZ1b\n"},
                {"file_path": "mix.txt:L3-L3", "new_string": "Z3\n"},  # pre-batch L3 == 'a3'
            ],
        }
    )

    assert payload.get("applied"), payload
    assert "mix.txt" in str(payload["applied"])
    assert f.read_text(encoding="utf-8") == "Z1\nZ1b\na2\nZ3\n"


def test_range_old_string_edit_after_content_edit_same_file_coapplies(workspace: Path) -> None:
    """Same as test_range_edit_after_content_edit_same_file_coapplies, but the
    second edit pairs the :Lx-Ly scope with old_string (falls into
    _replace_in_scope). The scope is translated through the ledger, so the
    scoped search hits the right window instead of 'old_string not found'."""
    f = workspace / "mix2.txt"
    f.write_text("a1\na2\na3\n", encoding="utf-8")
    _read("mix2.txt")

    payload = _edit(
        {
            "post_edit_hooks": False,
            "edits": [
                {"file_path": "mix2.txt", "old_string": "a1\n", "new_string": "Z1\nZ1b\n"},
                {"file_path": "mix2.txt:L3-L3", "old_string": "a3\n", "new_string": "Z3\n"},
            ],
        }
    )

    assert payload.get("applied"), payload
    assert "mix2.txt" in str(payload["applied"])
    assert f.read_text(encoding="utf-8") == "Z1\nZ1b\na2\nZ3\n"


def test_blind_range_edit_rejected_without_prior_read(workspace: Path) -> None:
    """No old anchor + file never served by read/code_search → reject, do not splice."""
    f = workspace / "unread.txt"
    f.write_text("a1\na2\na3\n", encoding="utf-8")

    payload = _edit({"post_edit_hooks": False, "edits": [{"file_path": "unread.txt:L2-L2", "new_string": "X\n"}]})

    assert payload["rolled_back"] is True
    assert "was not served by read/code_search" in payload["failed"][0]["error"]
    assert f.read_text(encoding="utf-8") == "a1\na2\na3\n"


def test_blind_range_edit_follows_lines_shifted_by_unrelated_drift(workspace: Path) -> None:
    """Drift is not staleness: an external change ABOVE the target shifts the
    line numbers but not the content the edit was written against, so the guard
    relocates the range instead of rejecting a perfectly well-defined edit.
    """
    f = workspace / "drift.txt"
    f.write_text("a1\na2\na3\n", encoding="utf-8")
    _read("drift.txt")
    f.write_text("NEW\na1\na2\na3\n", encoding="utf-8")  # external change shifts every line

    payload = _edit({"post_edit_hooks": False, "edits": [{"file_path": "drift.txt:L2-L2", "new_string": "X\n"}]})

    # a2 was served as L2 and now sits at L3 -- the splice follows it, and the
    # echoed line is where it LANDED so the model stays oriented.
    assert payload.get("applied") == ["drift.txt:3"], payload
    assert f.read_text(encoding="utf-8") == "NEW\na1\nX\na3\n"


def test_blind_range_edit_rejected_when_the_target_lines_themselves_changed(workspace: Path) -> None:
    """The one case that must still fail: the served lines are gone, so there is
    no honest place to put the replacement.
    """
    f = workspace / "drift.txt"
    f.write_text("a1\na2\na3\n", encoding="utf-8")
    _read("drift.txt")
    f.write_text("a1\nsomething else entirely\na3\n", encoding="utf-8")

    payload = _edit({"post_edit_hooks": False, "edits": [{"file_path": "drift.txt:L2-L2", "new_string": "X\n"}]})

    assert payload["rolled_back"] is True
    assert "no longer hold what was served" in payload["failed"][0]["error"]
    assert f.read_text(encoding="utf-8") == "a1\nsomething else entirely\na3\n"


def test_blind_range_edit_allowed_after_fresh_read_and_old_anchor_exempt(workspace: Path) -> None:
    f = workspace / "fresh.txt"
    f.write_text("a1\na2\na3\n", encoding="utf-8")
    _read("fresh.txt")

    _edit({"post_edit_hooks": False, "edits": [{"file_path": "fresh.txt:L2-L2", "new_string": "X\n"}]})
    assert f.read_text(encoding="utf-8") == "a1\nX\na3\n"

    # Our own write staled the stat signature, but it changed L2 only -- L3 is
    # still the a3 that was served, so a second blind range edit from the SAME
    # read applies without a re-read turn.
    _edit({"post_edit_hooks": False, "edits": [{"file_path": "fresh.txt:L3-L3", "new_string": "Y\n"}]})
    assert f.read_text(encoding="utf-8") == "a1\nX\nY\n"

    # …while an old-anchored edit needs no freshness at all: it self-verifies.
    _edit({"post_edit_hooks": False, "edits": [{"file_path": "fresh.txt", "old_string": "Y\n", "new_string": "Z\n"}]})
    assert f.read_text(encoding="utf-8") == "a1\nX\nZ\n"


def test_blind_range_edits_survive_our_own_line_shifting_edits(workspace: Path) -> None:
    """The common real sequence: one read, then several edits that each shift the
    lines below them. Every later range still refers to the ORIGINAL read, and
    each one is relocated to where its content actually lives now.
    """
    f = workspace / "shift.txt"
    f.write_text("a1\na2\na3\na4\n", encoding="utf-8")
    _read("shift.txt")

    # Grow L1 into three lines -- everything below moves down by two.
    _edit({"post_edit_hooks": False, "edits": [{"file_path": "shift.txt:L1-L1", "new_string": "b1\nb2\nb3\n"}]})
    payload = _edit({"post_edit_hooks": False, "edits": [{"file_path": "shift.txt:L3-L3", "new_string": "C\n"}]})

    assert payload.get("applied") == ["shift.txt:5"], payload
    assert f.read_text(encoding="utf-8") == "b1\nb2\nb3\na2\nC\na4\n"


def test_blind_range_edit_rejected_when_relocation_is_ambiguous(workspace: Path) -> None:
    """A served block that now occurs in several places (even with its context)
    is not relocatable -- guessing one would splice the wrong copy.
    """
    f = workspace / "dup.txt"
    f.write_text("a\nDUP\nb\n", encoding="utf-8")
    _read("dup.txt")
    # L2's content is gone from where it was and now appears twice elsewhere,
    # and the served neighbours no longer surround either copy.
    f.write_text("a\nZ\nb\nDUP\nc\nDUP\n", encoding="utf-8")

    payload = _edit({"post_edit_hooks": False, "edits": [{"file_path": "dup.txt:L2-L2", "new_string": "Y\n"}]})

    assert payload["rolled_back"] is True
    assert f.read_text(encoding="utf-8") == "a\nZ\nb\nDUP\nc\nDUP\n"


def test_blind_range_edit_never_relocates_on_a_bare_content_match(workspace: Path) -> None:
    """A moved block must bring its neighbours with it.

    The served line is gone from where it was and happens to exist exactly once
    elsewhere -- "unique in the file" alone would relocate the splice into
    unrelated code, silently, which is the failure the guard exists to prevent.
    """
    f = workspace / "coincidence.txt"
    f.write_text("alpha\nTARGET\nbeta\ngamma\n", encoding="utf-8")
    _read("coincidence.txt")
    f.write_text("alpha\nreplaced\nbeta\ngamma\nunrelated\nTARGET\n", encoding="utf-8")

    payload = _edit({"post_edit_hooks": False, "edits": [{"file_path": "coincidence.txt:L2-L2", "new_string": "X\n"}]})

    assert payload["rolled_back"] is True
    assert f.read_text(encoding="utf-8") == "alpha\nreplaced\nbeta\ngamma\nunrelated\nTARGET\n"


def test_blind_range_edit_uses_gutter_disk_line_directly(workspace: Path) -> None:
    """A bare read of a real .py file defaults to the minified view, but each
    line is prefixed with its REAL disk line number (a gutter) -- so a blind
    range edit using that shown number is already disk-accurate and needs no
    translation at all, the same trust level as a normal exact read.
    """
    f = workspace / "calc.py"
    src = "import os\n\n\ndef add(a, b):\n    return a + b\n\n\ndef other():\n    # a trailing comment\n    return 0\n"
    f.write_text(src, encoding="utf-8")
    rendered = _call("read", {"path": "calc.py"})["result"]["content"][0]["text"]
    assert "5\t    return a + b" in rendered  # gutter shows the real disk line

    payload = _edit(
        {"post_edit_hooks": False, "edits": [{"file_path": "calc.py:L5", "new_string": "    return a + b + 1\n"}]}
    )
    assert payload.get("applied"), payload
    assert f.read_text(encoding="utf-8") == src.replace("return a + b\n", "return a + b + 1\n")
    assert "# a trailing comment" in f.read_text(encoding="utf-8")  # untouched, not swallowed


def test_blind_range_edit_minified_flag_translates_explicitly(workspace: Path) -> None:
    """The explicit `:minified` suffix declares a range's numbers as MINIFIED-
    view positions rather than disk-direct ones -- it changes how a FRESH
    ledger entry is interpreted, for a model referencing an earlier minified
    view without re-reading. It does NOT bypass the freshness requirement
    itself (see test_blind_range_edit_minified_flag_rejected_when_stale for
    that). Disk L3 is blank here, so the flag genuinely changes the target vs.
    a plain (unflagged) `:L3`.
    """
    f = workspace / "calc.py"
    src = "import os\n\n\ndef add(a, b):\n    return a + b\n\n\ndef other():\n    # a trailing comment\n    return 0\n"
    f.write_text(src, encoding="utf-8")
    _read("calc.py")  # ledger now trusts calc.py's gutter-numbered (disk-direct) view

    # Minified view: 1 import os / 2 def add(a, b): / 3     return a + b / ...
    # Minified L3 is disk L5 -- NOT disk L3 (which is blank).
    payload = _edit(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "calc.py:L3:minified", "new_string": "    return a + b + 1\n"}],
        }
    )
    assert payload.get("applied"), payload
    assert f.read_text(encoding="utf-8") == src.replace("return a + b\n", "return a + b + 1\n")


def test_blind_range_edit_minified_flag_rejected_when_ambiguous(workspace: Path) -> None:
    """`:minified` still refuses to guess through a dropped comment -- which
    side of the removed comment did the edit mean? -- same as `apply_minified_edit`
    already refuses via old_string. No retry_with is attached when even a
    single-line probe can't find an honest disk anchor to show (never mislead
    with content from the wrong lines).
    """
    f = workspace / "mod.py"
    src = "import os\n\n\ndef helper():\n    # a comment worth stripping\n    x = 1\n    return x\n"
    f.write_text(src, encoding="utf-8")
    _read("mod.py")

    # Minified view: 1 import os / 2 def helper(): / 3     x = 1 / 4     return x.
    # Minified L3 sits immediately after the dropped comment -> ambiguous.
    payload = _edit(
        {"post_edit_hooks": False, "edits": [{"file_path": "mod.py:L3:minified", "new_string": "    x = 2\n"}]}
    )
    assert payload["rolled_back"] is True
    assert "explicitly flagged :minified" in payload["failed"][0]["error"]
    assert "retry_with" not in payload["failed"][0]
    assert f.read_text(encoding="utf-8") == src

    # The real disk line (shown by the gutter on a bare read) is unambiguous.
    assert "result" in _call("read", {"path": "mod.py"})
    _edit({"post_edit_hooks": False, "edits": [{"file_path": "mod.py:L6", "new_string": "    x = 2\n"}]})
    assert "x = 2" in f.read_text(encoding="utf-8")
    assert "# a comment worth stripping" in f.read_text(encoding="utf-8")


def test_blind_range_edit_minified_flag_rejected_when_stale(workspace: Path) -> None:
    """`:minified` must NOT be honored against a file whose ledger entry is
    stale (changed on disk, or never read, since) -- it only picks the
    INTERPRETATION of an already-fresh serve, not a bypass of freshness.

    Regression: an earlier version translated a `:minified`-flagged range
    against whatever is on disk NOW regardless of the ledger, so a since-
    edited file resolved the SAME minified line number against different
    (but structurally valid) content and silently spliced the wrong function.
    """
    f = workspace / "h2.py"
    f.write_text("def old_func():\n    return 1\n", encoding="utf-8")
    _read("h2.py")  # minified L2 == 'return 1' (inside old_func) as of THIS read

    # External rewrite: a new function prepended shifts what minified L2 means
    # -- it now falls inside brand_new(), not old_func(). Ledger's recorded
    # size no longer matches, so the read above is stale.
    f.write_text("def brand_new():\n    return 2\n\n\ndef old_func():\n    return 1\n", encoding="utf-8")

    payload = _edit(
        {"post_edit_hooks": False, "edits": [{"file_path": "h2.py:L2:minified", "new_string": "    return 99\n"}]}
    )
    assert payload["rolled_back"] is True
    assert "wasn't read fresh" in payload["failed"][0]["error"]
    # Neither function was touched -- no silent wrong-line splice.
    current = f.read_text(encoding="utf-8")
    assert "return 2" in current
    assert "return 1" in current
    assert "return 99" not in current


def test_blind_range_edit_rejection_includes_retry_content(workspace: Path) -> None:
    """A rejected blind range edit (the served lines are gone from disk) ships
    the exact current content around the requested range as retry_with, so the
    retry doesn't cost a separate read turn.
    """
    f = workspace / "drift.txt"
    f.write_text("a1\na2\na3\na4\na5\n", encoding="utf-8")
    _read("drift.txt")
    f.write_text("a1\nrewritten\nalso rewritten\na4\na5\n", encoding="utf-8")

    payload = _edit({"post_edit_hooks": False, "edits": [{"file_path": "drift.txt:L2-L2", "new_string": "X\n"}]})
    assert "changed on disk since" in payload["failed"][0]["error"]
    retry_with = payload["failed"][0]["retry_with"]
    assert retry_with["old_string"] == "a1\nrewritten\nalso rewritten\na4\n"
    assert retry_with["path"] == "drift.txt:L1-L4"


def test_blind_range_edit_retry_anchor_is_bounded(workspace: Path) -> None:
    """The retry anchor is an exact-substring hint, not a content channel.

    A batch of rejected range edits used to echo each target region back
    verbatim -- an 11,433-char rejection payload was measured against Cursor,
    versus 204 chars for the native edit result, and every later turn re-pays
    for it. The anchor is now capped to whole leading lines, which keeps it a
    valid substring of the region so it still anchors a retry passed as ``old=``.
    """
    f = workspace / "big.txt"
    body = "".join(f"line {i:03d} " + "x" * 40 + "\n" for i in range(1, 61))
    f.write_text(body, encoding="utf-8")

    payload = _edit({"post_edit_hooks": False, "edits": [{"file_path": "big.txt:L10-L40", "new_string": "X\n"}]})
    retry_with = payload["failed"][0]["retry_with"]
    anchor = retry_with["old_string"]

    assert 0 < len(anchor) <= 400, "anchor must stay bounded"
    assert anchor in body, "anchor must remain an exact substring so it works as old="
    assert anchor.endswith("\n"), "anchor must cut on a line boundary"


def test_relocation_never_lands_on_different_content_under_random_drift() -> None:
    """The invariant the guard is judged on, fuzzed.

    Rejecting a resolvable range costs a turn; relocating onto the WRONG lines
    corrupts a file silently, so the only property that must hold across every
    drift shape is: whenever a range IS relocated, the lines it now points at
    are byte-identical to the lines that were served. The vocabulary is
    deliberately tiny and repetitive -- far more self-similar than real source
    -- so near-miss anchors come up constantly.
    """
    vocab = ["", "    pass", "    return None", "}", "def f():", "x = 1", "# note", "import os", "    return x"]
    rnd = random.Random(20260729)
    relocated = 0
    for _ in range(3000):
        served_lines = [rnd.choice(vocab) for _ in range(rnd.randint(3, 40))]
        current_lines = list(served_lines)
        for _ in range(rnd.randint(1, 3)):
            at = rnd.randrange(0, max(1, len(current_lines)))
            op = rnd.choice(("insert", "delete", "replace"))
            if op == "insert":
                current_lines[at:at] = [rnd.choice(vocab) for _ in range(rnd.randint(1, 3))]
            elif current_lines and op == "delete":
                del current_lines[at : at + rnd.randint(1, 3)]
            elif current_lines:
                current_lines[at] = rnd.choice(vocab) + "!"
        start = rnd.randint(1, len(served_lines))
        end = min(len(served_lines), start + rnd.randint(0, 4))
        served_text = "\n".join(served_lines) + "\n"
        current_text = "\n".join(current_lines) + "\n"
        got = mcp_server._relocate_served_range(
            mcp_server._line_digests(served_text), mcp_server._line_digests(current_text), start, end
        )
        if got is None:  # fails closed -- always allowed
            continue
        relocated += 1
        assert (
            current_text.splitlines()[got[0] - 1 : got[1]] == served_text.splitlines()[start - 1 : end]
        ), f"L{start}-L{end} -> L{got[0]}-L{got[1]} changed the content under the edit"
    assert relocated > 1000, f"fuzz degenerated into all-rejections ({relocated} resolved)"


def test_blind_range_guard_disabled_by_env(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEMONCROW_RANGE_EDIT_GUARD", "0")
    f = workspace / "off.txt"
    f.write_text("a1\na2\n", encoding="utf-8")

    _edit({"post_edit_hooks": False, "edits": [{"file_path": "off.txt:L1-L1", "new_string": "X\n"}]})
    assert f.read_text(encoding="utf-8") == "X\na2\n"


# ---------------------------------------------------------------------------
# Native full-file rewrite: edit(path, new) on an EXISTING file with no
# old/replace -- the shape a host-native model sends for "write this file".
# A plausible complete body applies as replace; fragments still error.
# ---------------------------------------------------------------------------


def test_full_rewrite_of_existing_file_without_replace_applies(workspace: Path) -> None:
    f = workspace / "rw.py"
    f.write_text("A = 1\nB = 2\nC = 3\n", encoding="utf-8")
    body = "A = 10\nB = 20\nC = 30\nD = 40\n"
    payload = _edit({"path": "rw.py", "new": body, "hooks": False})
    assert "failed" not in payload, payload
    assert f.read_text(encoding="utf-8") == body


def test_full_rewrite_with_elision_marker_still_rejected(workspace: Path) -> None:
    f = workspace / "el.py"
    original = "A = 1\nB = 2\nC = 3\n"
    f.write_text(original, encoding="utf-8")
    body = "A = 10\n# ... existing code ...\nZ = 99\n"
    payload = _edit({"path": "el.py", "new": body, "hooks": False})
    assert "failed" in payload, payload
    assert f.read_text(encoding="utf-8") == original  # untouched


def test_new_file_missing_names_actual_path_and_inline_fallback(workspace: Path) -> None:
    """A hallucinated new_file path fails with the real path + the inline-new fallback.

    Regression: the daemon boundary redacted absolute paths, so the model saw
    "new_file <path> could not be read: ... '<path>'" and could not tell which
    path was wrong or how to recover.
    """
    resp = _call(
        "edit",
        {
            "path": "tests/new_test.py",
            "new_file": "/tmp/lemoncrow-spill/does-not-exist-xyz.py",
            "hooks": False,
        },
    )
    message = str(resp["error"]["message"])
    assert "/tmp/lemoncrow-spill/does-not-exist-xyz.py" in message, resp
    assert "inline via new" in message, resp


def test_new_file_creates_target_with_full_content(workspace: Path, tmp_path: Path) -> None:
    """{path: x, new_file: y} = x gets y's FULL content -- create when x is new."""
    src = tmp_path / "payload.py"
    body = "def created():\n    return 42\n"
    src.write_text(body, encoding="utf-8")
    payload = _edit({"path": "made_from_file.py", "new_file": str(src), "hooks": False})
    assert "failed" not in payload, payload
    assert (workspace / "made_from_file.py").read_text(encoding="utf-8") == body


def test_new_file_overwrites_existing_whole_file(workspace: Path, tmp_path: Path) -> None:
    f = workspace / "ow.py"
    f.write_text("OLD = 1\n" * 30, encoding="utf-8")
    src = tmp_path / "payload2.py"
    body = "NEW = 2\n"
    src.write_text(body, encoding="utf-8")
    payload = _edit({"path": "ow.py", "new_file": str(src), "hooks": False})
    assert "failed" not in payload, payload
    assert f.read_text(encoding="utf-8") == body


def test_new_file_with_old_anchor_still_targets_splice(workspace: Path, tmp_path: Path) -> None:
    """An old anchor alongside new_file keeps targeted-replace semantics."""
    f = workspace / "anchor.py"
    f.write_text("KEEP = 0\nTARGET = 1\nKEEP2 = 2\n", encoding="utf-8")
    src = tmp_path / "payload3.py"
    src.write_text("TARGET = 99\n", encoding="utf-8")
    payload = _edit({"path": "anchor.py", "old": "TARGET = 1\n", "new_file": str(src), "hooks": False})
    assert "failed" not in payload, payload
    assert f.read_text(encoding="utf-8") == "KEEP = 0\nTARGET = 99\nKEEP2 = 2\n"


def test_tiny_fragment_without_old_still_rejected(workspace: Path) -> None:
    f = workspace / "frag.py"
    original = "X = 1\n" * 40
    f.write_text(original, encoding="utf-8")
    payload = _edit({"path": "frag.py", "new": "Y = 2\n", "hooks": False})
    assert "failed" in payload, payload
    assert f.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# Worktree-aware relative path resolution
#
# The MCP server is long-lived and its own cwd is fixed at launch, so a session
# that enters a git worktree used to have its RELATIVE edit paths silently
# resolved against the main checkout -- writing the right change to the wrong
# copy of the file, and passing write-confinement because in-repo worktrees sit
# under the workspace root.
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> None:
    import subprocess

    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _repo_with_worktree(root: Path) -> Path:
    """Init a git repo at `root` and add a linked worktree; return the worktree."""
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "t@example.com", cwd=root)
    _git("config", "user.name", "t", cwd=root)
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git("add", "-A", cwd=root)
    _git("commit", "-qm", "seed", cwd=root)
    wt = root / ".claude" / "worktrees" / "wt"
    wt.parent.mkdir(parents=True, exist_ok=True)
    _git("worktree", "add", "-q", str(wt), "-b", "wt-branch", cwd=root)
    return wt


def test_relative_edit_follows_the_session_into_a_worktree(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative path writes to the worktree the session is in, not the main checkout."""
    wt = _repo_with_worktree(workspace)
    (workspace / "target.txt").write_text("MAIN\n", encoding="utf-8")
    (wt / "target.txt").write_text("WORKTREE\n", encoding="utf-8")
    monkeypatch.setattr(mcp_server, "_last_session_cwd", str(wt))

    payload = _edit(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "target.txt", "old_string": "WORKTREE", "new_string": "EDITED"}],
        }
    )

    assert "failed" not in payload, payload
    assert (wt / "target.txt").read_text(encoding="utf-8") == "EDITED\n"
    # The whole point: the main checkout is untouched.
    assert (workspace / "target.txt").read_text(encoding="utf-8") == "MAIN\n"


def test_worktree_redirect_is_disclosed_to_the_model(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The redirect is inferred from the last bash cwd, so it is never silent.

    A clean success is normally squelched to `applied path:line`; a wrong
    inference would then be invisible exactly when it misdirected a write.
    """
    wt = _repo_with_worktree(workspace)
    (wt / "target.txt").write_text("WORKTREE\n", encoding="utf-8")
    monkeypatch.setattr(mcp_server, "_last_session_cwd", str(wt))

    text = _edit_text(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "target.txt", "old_string": "WORKTREE", "new_string": "EDITED"}],
        }
    )

    assert "resolved against worktree" in text, text
    assert str(wt) in text, text


def test_no_worktree_evidence_keeps_the_main_checkout(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a worktree cwd, resolution is unchanged -- and says nothing."""
    _repo_with_worktree(workspace)
    (workspace / "target.txt").write_text("MAIN\n", encoding="utf-8")
    monkeypatch.setattr(mcp_server, "_last_session_cwd", str(workspace))

    text = _edit_text(
        {
            "post_edit_hooks": False,
            "edits": [{"file_path": "target.txt", "old_string": "MAIN", "new_string": "EDITED"}],
        }
    )

    assert (workspace / "target.txt").read_text(encoding="utf-8") == "EDITED\n"
    assert "resolved against" not in text, text


def test_a_worktree_of_another_repo_does_not_redirect(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a worktree of THIS repo counts; a foreign one falls back to the root."""
    _repo_with_worktree(workspace)
    other = tmp_path.parent / "other-repo"
    other.mkdir(parents=True, exist_ok=True)
    foreign_wt = _repo_with_worktree(other)
    monkeypatch.setattr(mcp_server, "_last_session_cwd", str(foreign_wt))

    assert mcp_server._session_worktree_root(workspace) is None


def test_bash_cwd_is_what_teaches_edit_where_the_session_is(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a bash call's cwd is recorded -- it is the sole session-cwd signal."""
    monkeypatch.setattr(mcp_server, "_last_session_cwd", None)

    mcp_server._record_session_cwd("read", {"cwd": "/not/recorded"})
    assert mcp_server._last_session_cwd is None

    mcp_server._record_session_cwd("bash", {"cwd": "/recorded"})
    assert mcp_server._last_session_cwd == "/recorded"

    # A bash call without an explicit cwd must not clear what we know.
    mcp_server._record_session_cwd("bash", {})
    assert mcp_server._last_session_cwd == "/recorded"
