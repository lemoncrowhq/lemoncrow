"""scripts/mirror.py: `!`-prefixed private denies beat broad allows."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

_MIRROR = Path(__file__).resolve().parents[1] / "scripts" / "mirror.py"
_spec = importlib.util.spec_from_file_location("_mirror_under_test", _MIRROR)
assert _spec and _spec.loader
mirror = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mirror)


def test_deny_beats_allow_regardless_of_order() -> None:
    prefixes = ["src", "!src/lemoncrow/core/capabilities/code_context/renderer.py"]
    assert mirror.is_public("src/lemoncrow/gateway/adapters/mcp_server.py", prefixes) is True
    assert mirror.is_public("src/lemoncrow/core/capabilities/code_context/renderer.py", prefixes) is False
    # deny listed BEFORE the allow still wins
    rev = list(reversed(prefixes))
    assert mirror.is_public("src/lemoncrow/core/capabilities/code_context/renderer.py", rev) is False


def test_subtree_deny() -> None:
    prefixes = ["src", "!src/lemoncrow/core/capabilities/source_projection"]
    assert mirror.is_public("src/lemoncrow/core/capabilities/source_projection/minify.py", prefixes) is False
    assert mirror.is_public("src/lemoncrow/core/capabilities/licensing/models.py", prefixes) is True


def test_no_allow_no_public() -> None:
    assert mirror.is_public("internal/secret.py", ["src", "tests"]) is False


def test_plain_allowlist_unchanged() -> None:
    prefixes = ["docs", "src"]
    assert mirror.is_public("docs/x.md", prefixes) is True
    assert mirror.is_public("src/a.py", prefixes) is True
    assert mirror.is_public("deploy/x", prefixes) is False


# --- What the real allowlist would publish -----------------------------------
# `src/` is allowed wholesale, so nothing under it is protected by a deny any
# more (the `!src/lemoncrow/pro` deny is gone: the engine is published under
# PolyForm Noncommercial). These two tests are the replacement guard -- they run
# the REAL release/public-paths.txt over the REAL tracked tree, so a new private
# directory or a committed credential fails here instead of on GitHub.

_PRIVATE_TREES = (
    "services",  # license-issuer + hosted control plane
    "tools",
    "deploy",
    "experiments",
    "docs-internal",
    "signatures",
    "release",
    "reports",
    "lemoncode",
)

# High-signal committed-credential shapes. Deliberately narrow: provider key
# formats and PEM private-key headers, not "the word secret".
_SECRET_RE = (
    r"BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY"
    r"|ghp_[A-Za-z0-9]{30,}"
    r"|github_pat_[A-Za-z0-9_]{40,}"
    r"|sk-ant-[A-Za-z0-9-]{40,}"
    r"|sk-[A-Za-z0-9]{32,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9-]{20,}"
    r"|AIza[0-9A-Za-z_-]{35}"
)

# Files whose credential-shaped strings are VERIFIED synthetic fixtures for the
# redaction feature itself (AKIA-shaped literals, a bare "sk-", the PEM header
# text). Adding an entry here means someone read the file and confirmed the
# match is fake -- it is not a way to silence the check.
_VERIFIED_SYNTHETIC = frozenset(
    {
        "tests/core/test_output_redaction_wiring.py",
        "tests/core/test_redaction.py",
        "tests/gateway/test_cli_memory_commands.py",
    }
)

_REPO = Path(__file__).resolve().parents[1]


def _tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=_REPO, capture_output=True, text=True, check=True).stdout
    return [p for p in out.split("\0") if p]


def test_private_trees_never_reach_the_public_mirror() -> None:
    prefixes = mirror.load_public_prefixes()
    leaked = [
        path
        for path in _tracked_files()
        if path.split("/", 1)[0] in _PRIVATE_TREES and mirror.is_public(path, prefixes)
    ]
    assert not leaked, f"release/public-paths.txt would publish private files: {leaked[:10]}"


def test_no_credentials_in_the_files_the_mirror_would_publish() -> None:
    """Committed key material must never be inside the public allowlist.

    Replaces the wheel-side guard that used to keep `lemoncrow/pro` source out of
    releases: the engine is public on purpose now, so the thing worth guarding is
    credentials, not source.
    """
    hits = subprocess.run(
        ["git", "grep", "-lIE", _SECRET_RE, "HEAD", "--", "."],
        cwd=_REPO,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    prefixes = mirror.load_public_prefixes()
    # `git grep <rev>` prints "HEAD:<path>".
    paths = [line.split(":", 1)[1] for line in hits if ":" in line]
    leaked = sorted(p for p in paths if p not in _VERIFIED_SYNTHETIC and mirror.is_public(p, prefixes))
    assert not leaked, f"credential-shaped content inside public paths: {leaked}"


def test_public_workflows_are_rewritten_to_github_workflows() -> None:
    assert mirror.public_output_path(".github/public-workflows/tests.yml") == ".github/workflows/tests.yml"
    assert mirror.public_output_path(".github/public-workflows") == ".github/workflows"
    assert mirror.public_output_path("src/lemoncrow/__init__.py") == "src/lemoncrow/__init__.py"
