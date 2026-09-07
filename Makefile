.DEFAULT_GOAL := help

PY_PATHS := src benchmarks tests scripts integrations
MYPY_PATHS := src/lemoncrow
LEMONCROW_STORE ?= $(HOME)/.lemoncrow
LEMONCROW_CMD ?= uv run lemoncrow
TEST_PRINT_TIME ?= 0
# Coverage floor for the full slow-inclusive suite (make test-full / nightly-coverage.yml).
# Conservative provisional floor pending first-CI calibration (see 22-01-SUMMARY.md):
# local measurement could not complete the full suite (slow-service + xdist tree-sitter
# limitations); a partial subset run measured 68% (a strict lower bound). Calibrate to
# ~2 points below the first nightly run's reported total.
COV_FAIL_UNDER ?= 66
FORCE_ARG := $(if $(f),--force,)
.PHONY: help uninstall dev build release/build prod build-lemoncode-host status start restart build-host-skills sync-agent-context mirror release \
	docs-check worktree-env runtime-evidence \
	test test-fast test-cov test-full lint format-check format typecheck verify pre-commit \
	proof-cost-quality import clean \
	_ensure_hooks

# --------------------------------------------------------------------------- #
# Lifecycle                                                                   #
# --------------------------------------------------------------------------- #

#    * To do a clean development install (editable mode):
#         make dev
#    * To build and install a local production binary:
#         make prod

dev: ## Install LemonCrow in dev mode (stable source COPY, no auto-update); re-run to pick up edits, then /mcp reconnect
	bash scripts/local.sh
	@$(MAKE) build-lemoncode-host

build: ## Build and package for production distribution
	bash scripts/build.sh

release/build: build ## Alias for build release jobs

mirror: ## Incremental mirror bench → public repo (history-preserving): make mirror [f=1] [ARGS="--resync"]
	LEMONCROW_MIRROR_RUNNING=1 uv run python -m scripts.mirror $(FORCE_ARG) $(ARGS)

release: ## Bump version, commit, push, tag: make release tag=v0.4.X [f=1] [SKIP_MIRROR=1]
	@set -e; \
 TAG=$${tag:-}; \
 [ -n "$$TAG" ] || { echo "Usage: make release tag=vX.Y.Z"; exit 1; }; \
 FORCE=$${f:-0}; \
 SKIP_MIRROR=$${SKIP_MIRROR:-0}; \
 VER=$${TAG#v}; \
 sed -i.bak "s/^version = .*/version = \"$$VER\"/" pyproject.toml && rm -f pyproject.toml.bak; \
 echo "Bumped pyproject.toml to $$VER"; \
 uv lock; \
 git add pyproject.toml uv.lock; \
 git commit --no-verify -m "chore: bump to $$TAG" || echo "  → nothing new to commit, continuing..."; \
 git push --no-verify -u origin HEAD || echo "  → nothing to push, continuing..."; \
 if [ "$$FORCE" = "1" ]; then \
   git tag -d $$TAG 2>/dev/null || true; \
   git push --no-verify --delete origin $$TAG 2>/dev/null || true; \
 fi; \
 git tag $$TAG; \
 PUSH_FLAG=; \
 [ "$$FORCE" = "1" ] && PUSH_FLAG=--force; \
 git push --no-verify $$PUSH_FLAG origin $$TAG; \
 if [ "$$SKIP_MIRROR" = "1" ]; then \
   echo "✓ Released $$TAG (private only, mirror skipped)"; \
   exit 0; \
 fi; \
 if [ "$$(uname -s)" = "Darwin" ] && command -v gh >/dev/null 2>&1; then \
   gh auth setup-git >/dev/null 2>&1 || true; \
 fi; \
 echo "Mirroring to public repo..."; \
 MIRROR_FORCE_ARG=; \
 [ "$$FORCE" = "1" ] && MIRROR_FORCE_ARG=--force; \
 LEMONCROW_MIRROR_RUNNING=1 uv run python -m scripts.mirror $$MIRROR_FORCE_ARG; \
 PUB_SHA=$$(git rev-parse refs/mirror/last-pub); \
 git push --no-verify $$PUSH_FLAG https://github.com/lemoncrow-lab/lemoncrow.git "$$PUB_SHA:refs/tags/$$TAG"; \
 echo "✓ Released $$TAG (dev + public)"

prod: ## Build and install from local production build (includes mypyc compilation; expects ~2-3 min build time)
	LEMONCROW_ENABLE_MYPYC=1 bash scripts/build.sh
	# Run the local installer: copies bundle/ → ~/.local/ and sets up host integrations,
	# exactly mirroring the remote path (download → extract → bundle.sh).
	bash scripts/install.sh --local
	@$(MAKE) build-lemoncode-host

build-lemoncode-host: ## Provision the LemonCode host binary: builds from the vendored lemoncode/ submodule if bun is available, else downloads a release. Skips if already installed; force rebuild with 'make build-lemoncode-host f=1'
	@if [ -z "$(f)" ] && [ -x "$(LEMONCROW_STORE)/bin/lemoncode-host" ]; then \
		:; \
	elif command -v bun >/dev/null 2>&1 && git submodule update --init lemoncode >/dev/null 2>&1 && [ -f lemoncode/package.json ]; then \
		echo "Building LemonCode host from source (lemoncode/ submodule)..."; \
		$(LEMONCROW_CMD) code host build --source lemoncode; \
	else \
		echo "bun/lemoncode submodule unavailable -- downloading prebuilt LemonCode host release..."; \
		$(LEMONCROW_CMD) code host install; \
	fi

uninstall: ## Remove all LemonCrow agent-host integrations, hooks, and bin wrappers
	@bash scripts/uninstall.sh $${ARGS:-}

status: ## Show LemonCrow installation status
	@bash scripts/status.sh

start: ## Start the service and frontend natively
	@if [ -f .env.worktree ]; then set -a; . ./.env.worktree; set +a; fi; \
	$(LEMONCROW_CMD) --root "$${LEMONCROW_STACK_ROOT:-$(LEMONCROW_STORE)}" stack start
	@if [ -f .env.worktree ]; then set -a; . ./.env.worktree; set +a; fi; \
	$(LEMONCROW_CMD) --root "$${LEMONCROW_STACK_ROOT:-$(LEMONCROW_STORE)}" stack logs -f
restart: ## Restart the service and frontend natively
	@if [ -f .env.worktree ]; then set -a; . ./.env.worktree; set +a; fi; \
	$(LEMONCROW_CMD) --root "$${LEMONCROW_STACK_ROOT:-$(LEMONCROW_STORE)}" stack stop --force || true
	@if [ -f .env.worktree ]; then set -a; . ./.env.worktree; set +a; fi; \
	$(LEMONCROW_CMD) --root "$${LEMONCROW_STACK_ROOT:-$(LEMONCROW_STORE)}" stack start
	@if [ -f .env.worktree ]; then set -a; . ./.env.worktree; set +a; fi; \
	$(LEMONCROW_CMD) --root "$${LEMONCROW_STACK_ROOT:-$(LEMONCROW_STORE)}" stack logs -f

# --------------------------------------------------------------------------- #
# Development                                                                 #
# --------------------------------------------------------------------------- #

build-host-skills: ## Generate Codex/Gemini skill bundles from integrations/skills (set LEMONCROW_DEV_MODE=1 to include dev-only skills)
	@bash scripts/build_host_skills.sh --host all $$( [ "$${LEMONCROW_DEV_MODE:-0}" = "1" ] && echo --include-dev )

sync-agent-context: ## Regenerate host instruction surfaces from integrations/agents/shared/
	uv run python scripts/sync_agent_context.py

docs-check: ## Run docs and repo-governance checks
	uv run pytest tests/gateway/test_docs.py tests/gateway/test_generated_agent_contexts.py -q

worktree-env: ## Write a per-worktree .env file for local stack bootstraps
	uv run python scripts/worktree_env.py --env-file .env.worktree --json

runtime-evidence: ## Capture runtime evidence from a local LemonCrow stack
	uv run python scripts/runtime_evidence.py

# Auto-configure git hooks path so .githooks/pre-commit runs on every commit.
# Developers never need to run `git config core.hooksPath .githooks` by hand.
_ensure_hooks:
	@current=$$(git config core.hooksPath 2>/dev/null || echo ""); \
	if [ "$$current" != ".githooks" ]; then \
		git config core.hooksPath .githooks; \
		echo "  → Configured git hooks path → .githooks"; \
	fi

test: | _ensure_hooks ## Run all tests
ifeq ($(TEST_PRINT_TIME),1)
	@time bash -lc 'if uv run python -c "import xdist" >/dev/null 2>&1; then uv run pytest -q -ra --durations=0 -n auto --dist=worksteal; else uv run pytest -q -ra --durations=0; fi'
else
	@bash -lc 'if uv run python -c "import xdist" >/dev/null 2>&1; then uv run pytest -q -ra --durations=0 -n auto --dist=worksteal; else uv run pytest -q -ra --durations=0; fi'
endif

test-fast: | _ensure_hooks ## Run fast tests: stop on first failure, skip slow/Postgres-gated tests
	@bash -lc 'if uv run python -c "import xdist" >/dev/null 2>&1; then uv run pytest -q -x -n auto --dist=worksteal --ignore=tests/test_postgres_store.py --ignore=tests/test_worker_jobs.py -m "not slow"; else uv run pytest -q -x --ignore=tests/test_postgres_store.py --ignore=tests/test_worker_jobs.py -m "not slow"; fi'

test-cov: ## Run tests with terminal and HTML coverage reports
	uv run pytest --cov=lemoncrow --cov-report=term-missing --cov-report=html

test-full: ## Run the FULL suite (incl. slow) with measured coverage floor
	uv run pytest -m "" --timeout=300 --cov=lemoncrow --cov-report=term-missing --cov-fail-under=$(COV_FAIL_UNDER)

lint: | _ensure_hooks ## Run ruff lint checks
	uv run ruff check $(PY_PATHS)

format-check: ## Check Python formatting without rewriting files
	uv run black --check src tests

format: | _ensure_hooks ## Format all code: Python (ruff+black) and frontend (prettier if available)
	uv run ruff check --fix $(PY_PATHS)
	uv run black src tests
	@if [ -d "frontend" ]; then \
		if [ -f "frontend/package.json" ] && grep -q "prettier" frontend/package.json 2>/dev/null; then \
			cd frontend && npx prettier --write "src/**/*.{ts,tsx,js,jsx,json,css,md}" 2>/dev/null || true; \
		fi; \
	fi

typecheck: | _ensure_hooks ## Run mypy strict type-checking
	uv run mypy --explicit-package-bases $(MYPY_PATHS)

pre-commit: | _ensure_hooks format lint typecheck docs-check test ## Full pre-commit gate: format + lint + typecheck + docs + test

verify: | _ensure_hooks lint format-check typecheck docs-check test ## Verify code, docs, runtime smoke tests, and agent integrations
	bash scripts/verify_lemoncrow_service.sh
	bash scripts/verify_lemoncrow_postgres.sh
	bash scripts/verify_agent_clis.sh

proof-cost-quality: ## Run cost-quality proof gate tests and write proof-report.json
	LOCAL=1 uv run pytest tests/core/test_cost_quality_proof_gate.py tests/gateway/test_cli_proof_gate.py -v
	LOCAL=1 uv run lemoncrow proof run --session-id wp32-proof --context-reduction-pct 60 --json
	@test -s $(LEMONCROW_STORE)/proof/proof-report.json

# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #

import: ## Import sessions: make import [f=1]
	LOCAL=1 $(LEMONCROW_CMD) --root "$(LEMONCROW_STORE)" import $(FORCE_ARG)

flow-dump: ## Extract chat from a .flow file or directory: make flow-dump path=/path/to/file_or_dir
	@if [ -z "$(path)" ]; then \
		echo "Error: 'path' argument is required. Usage: make flow-dump path=/path/to/file_or_dir"; \
		exit 1; \
	fi
	uv run --project benchmarks python -m benchmarks.flowlib.dump $(path)

clean: ## Remove build artifacts, caches, and coverage data
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

help: ## Show this help message
	@echo "LemonCrow - AI reasoning/procedure/runtime layer"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@printf "%-20s %s\n" "Target" "Description"
	@printf "%-20s %s\n" "------" "-----------"
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | \
		sed 's/:.*## /\t/' | \
		awk -F'\t' '{ printf "  %-18s %s\n", $$1, $$2 }'
