# Licensing report

Prepared for the maintenance-mode / open-source transition. This records the
licensing position after relicensing the whole project to Apache-2.0.

## 1. Project license

- **LemonCrow: Apache-2.0** in its entirety, including the `lemoncrow.pro`
  engine. Full text: `LICENSE-APACHE`; summary: `LICENSE`; attribution notice:
  `NOTICE`.
- Previously the project was "open-core": Apache-2.0 source plus a proprietary,
  compiled-only `lemoncrow.pro` engine. That split has been removed — the engine
  source is now published and Apache-2.0 licensed.

## 2. Contributor terms

- `CONTRIBUTING.md` states contributions are under Apache-2.0.
- `CLA.md` (Contributor License Agreement) additionally grants the maintainer
  broad relicensing rights. **Owner review recommended:** with the project now
  fully Apache-2.0, decide whether the CLA's relicensing clause should be
  narrowed to match the OSS positioning (it is not required for Apache-2.0, and
  a DCO sign-off is a lighter alternative).

## 3. Bundled / vendored code

- `vendor/babel-stub/` + `vendor/babel-99.0.0-py3-none-any.whl` — a minimal
  functional replacement for a single `babel` API (BCP-47 prefix check), written
  in-project to save install size. It is our own code; ship under Apache-2.0.
  **Verified:** the stub is a 32-line first-party reimplementation
  (`Locale.parse().language` only) with no upstream Babel copyright, source, or
  data — it copies nothing from upstream (item 7.5 resolved).
- No other vendored third-party source is bundled. No private PyPI index or
  `git+ssh`/`git+https` dependencies are used (`uv.lock` resolves only
  `pypi.org` plus the local `vendor/babel` path).

## 4. Dependency licenses (runtime)

All runtime dependencies in `pyproject.toml` are permissively licensed and
compatible with Apache-2.0 redistribution. Spot check of the principal ones:

| Dependency | License (typical) |
|---|---|
| pydantic, tiktoken, rapidfuzz, blake3, markdownify, diff-match-patch | MIT |
| click, prompt-toolkit, GitPython, beautifulsoup4, uvicorn | BSD-3-Clause |
| cryptography | Apache-2.0 / BSD |
| rich, fastapi, tenacity, opentelemetry-*, prometheus-client | Apache-2.0 |
| tree-sitter, tree-sitter-language-pack | MIT / Apache-2.0 |
| pyyaml, urllib3, aiohttp, yarl | MIT / Apache-2.0 |
| trafilatura, pdfplumber, pypdf | Apache-2.0 / BSD / MIT |
| pygit2 (libgit2 bindings) | GPL-2.0-with-linking-exception |

**Owner review recommended:** `pygit2` carries GPLv2-with-a-linking-exception.
The linking exception is designed to permit use from otherwise-licensed
programs, so this is generally compatible, but confirm the exception's terms are
acceptable for redistribution, or gate `pygit2` behind an optional extra if you
prefer to avoid it. Optional extras (`letta`, `torch`/`sentence-transformers`,
`ollama`, `openai`, `litellm`, `langfuse`, `psycopg`, `ortools`, `rope`, etc.)
are not installed by default; their licenses apply only when a user opts in.

A machine-generated inventory of the 35 runtime dependencies is in
`docs/dependency-licenses.md`. A full SPDX report (including transitive deps, via
`pip-licenses`) should still be produced in CI before a formal release.

**Copyleft scan of the full installed environment (runtime + optional + dev):**

| Package | License | Assessment |
|---|---|---|
| `pygit2` | GPLv2 + linking exception | Runtime dep. The linking exception permits use from Apache-2.0 code; accept or move to an optional extra (owner review). |
| `pathspec` | MPL-2.0 | Transitive. File-level weak copyleft; compatible with Apache-2.0 distribution. |
| `psycopg2-binary` | LGPL | Only via the `postgres` optional extra. Dynamic-link/LGPL is compatible; not installed by default. |
| `pyinstaller` | GPLv2 | **Dev/build tool only** (bundling binaries); its packaging exception means output is not GPL-encumbered, and it is not redistributed with the runtime. |

No copyleft license blocks an Apache-2.0 release of LemonCrow's own source.

## 5. Source files with unclear ownership / generated code

- Generated agent-context and host-integration files under `integrations/` are
  produced from `integrations/agents/shared/` by `scripts/sync_agent_context.py`
  — first-party, Apache-2.0.
- `benchmarks/**` raw results and fixtures are first-party measurement data.
- No third-party source of unclear provenance was identified in the runtime
  tree. Confirm none of the benchmark fixtures embed third-party code under an
  incompatible license before publishing them.

## 6. Recommendation

- **Adopt Apache-2.0** for the whole project (done). It is permissive, patent-
  grant-bearing, and compatible with every runtime dependency identified above.
- Keep `LICENSE`, `LICENSE-APACHE`, and `NOTICE` in sync.

## 7. Steps requiring owner confirmation

1. Confirm the intent to publish the `lemoncrow.pro` engine source as
   Apache-2.0 (irreversible disclosure).
2. Decide the CLA's future (keep, narrow the relicensing clause, or switch to a
   DCO).
3. Confirm `pygit2`'s GPLv2-linking-exception is acceptable, or move it to an
   optional extra.
4. Generate and attach a full SPDX dependency-license inventory before release.
   (A runtime-dependency inventory is already generated in
   `docs/dependency-licenses.md`; a full transitive SPDX run in CI is still
   recommended.)
5. ~~Confirm the vendored `babel` stub contains no copied upstream Babel
   source.~~ **Done** — verified first-party (see §3).
