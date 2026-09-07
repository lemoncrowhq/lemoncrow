# LemonCrow Maintenance-Mode Transition — Audit & Plan

Status: **implemented.** This document began as the Phase 1 audit and Phase 2
implementation plan; the plan has since been executed (owner decisions in §5.1).
It converts LemonCrow from a partially-commercial, account-and-server-dependent
open-core product into a complete, account-free, locally-usable open-source
runtime that can be left in low-maintenance mode.

**Verification:** full fast test suite green (4435 passed, 1 skipped) excluding 5
pre-existing failures unrelated to this work (they depend on
`integrations/agents/shared/agent-rule.md`, modified before this transition).
`ruff` clean. `lc init` + `lc status` verified end-to-end in a fresh repo with no
account and no network. See the final report for details.

All file:line references are against the private dev repo
(`lemoncrow-dev`). The public GitHub repo is a filtered mirror produced by
`scripts/mirror.py` from `release/public-paths.txt`.

---

## 0. Executive summary of findings

LemonCrow's local runtime is **coupled to a hosted control plane** in four ways
that break the account-free / server-free goal:

1. **A fail-closed savings-cap gate.** The compiled
   `lemoncrow.pro.capabilities.licensing_gate.resolve_cap_verdict` requires a
   server-signed Ed25519 cap-verdict token bound to `(account_id, device_id,
   plan)`. Absent/expired/mismatched token ⇒ **dormant** ⇒ the MCP server
   **advertises zero tools and rejects every tool call**. A fresh install, or
   any machine offline for >8h, collapses to dormant. This is the single
   hardest blocker on offline use.
2. **Mandatory account.** `lc init` defaults to `--login`, popping a browser
   OAuth flow against `https://lemoncrow.com`; the MCP server also runs a
   *seamless background browser login* at startup. Anonymous use is
   server-capped at $100 measured savings.
3. **A hardware-derived device id.** `store.stable_machine_device_id()` reads
   `/etc/machine-id` (Linux) / `IOPlatformUUID` (macOS) / registry `MachineGuid`
   (Windows) and hashes it — a hardware fingerprint used for cap binding.
4. **Telemetry on by default**, plus two always-on LemonCrow egress endpoints
   (`/api/telemetry/rollup`, `lemoncrow.beseam.com/api/sync`) not fully covered
   by the opt-out.

Additionally, the **entire runtime engine** (`lemoncrow.pro`, 265 source files:
retrieval, ranking, prompt compilation, memory, routing, verification) is
**excluded from the public mirror** and shipped `.so`-only, so the public repo
**cannot reproduce the core runtime from source** today.

The good news: the account/cap system is a self-contained subsystem
(`core/capabilities/licensing/**` + one compiled gate + a handful of call
sites), telemetry already has CLI controls and a privacy-conscious event schema,
the local-security API key is cleanly separated from account identity, and the
`pro` engine source is physically present in this repo — so opening it is a
mirror/build-config change, not a rewrite.

---

## 1. Architecture before

```
         ┌─────────────────────── local machine ───────────────────────┐
         │  lc CLI / lcd daemon / MCP server                            │
         │    ├─ core.foundation  (paths, identity, stores — LOCAL)     │
         │    ├─ core.service     (api, telemetry, sync, auth)          │
         │    ├─ core.capabilities.licensing  ── OAuth + cap gate ──────┼──► lemoncrow.com
         │    ├─ lemoncrow.pro    (compiled .so engine, cap gate) ──────┤     /account, /api/auth/me
         │    └─ gateway (cli, mcp_server, hosts, sdk)                  │     /api/usage/report[-anon]
         │  data: ~/.lemoncrow/**   config: ~/.config/lemoncrow/**      │     /api/telemetry/rollup
         └──────────────────────────────────────────────────────────────┘  ► beseam.com/api/sync
                                                                            ► us.i.posthog.com (dormant)
```

Control-plane server code lives outside this runtime in `services/`
(license-issuer, private) and `landing/` (Cloudflare Worker submodule): OAuth,
Stripe, Ed25519 cap signing, device/seat enforcement (SQLite migrations
0019–0022).

## 1b. Architecture after (target)

```
         ┌─────────────────────── local machine ───────────────────────┐
         │  lc CLI / lcd daemon / MCP server — all features unlocked    │
         │    ├─ core.foundation  (paths, identity=random installation_id)
         │    ├─ core.service     (api, telemetry OFF by default)       │
         │    ├─ (licensing → inert: everything free, no network)       │
         │    ├─ lemoncrow.pro    (OPEN source, optionally mypyc-compiled)
         │    └─ gateway                                                │
         │  data: ~/.lemoncrow/**   config: ~/.config/lemoncrow/**      │
         └──────────────────────────────────────────────────────────────┘
   Outbound only when the user asks: `lc update` (GitHub), user-configured model
   providers, opt-in telemetry, opt-in dependency downloads. No LemonCrow
   control plane. No account. No cap. No hardware id.
```

---

## 2. Phase 1 audit — inventory & classification

Classification key: **(1)** core local runtime · **(2)** local security · **(3)**
commercial licensing · **(4)** analytics/attribution · **(5)** hosted-service ·
**(6)** historical/unused.

### 2.1 Authentication & commercial state

| Mechanism | Location | Class | Blocks offline? | Disposition |
|---|---|---|---|---|
| `lc account` group (login/logout/status/subscription/cap) | `gateway/cli/commands/admin.py:1474-1790` | 3/5 | login yes | remove (keep inert `status`) |
| Browser OAuth flow | `core/capabilities/licensing/oauth_flow.py` | 5 | login yes | delete |
| `/api/auth/me` entitlement fetch | `core/capabilities/licensing/entitlements.py:59-83` | 5 | Pro only | delete network; local unlock |
| Signed cap-verdict gate (compiled) | `pro/capabilities/licensing_gate.py` | 3 | **YES** | make always-active, drop key |
| MCP dormancy tool-gating | `gateway/adapters/mcp_server.py:10664-10700,11723-11861` | 3 | **YES** | remove |
| MCP seamless background login | `mcp_server.py:12201-12363` | 5 | no | remove |
| Usage report client (`/api/usage/report[-anon]`) | `core/capabilities/licensing/usage_report.py` | 3/5 | cap | delete |
| Feature registry + gates (`require`, `require_pro`) | `licensing/features.py`, `entitlements.py`, `_shared.py`, call sites | 3 | no | unlock all |
| Anonymous trial / referral seed | `core/capabilities/plugin_runtime.py:437-540,853` | 4/6 | no | remove referral, keep local auth-state |
| `lc init --login` (default on) | `admin.py:715-800` | 3/5 | no | remove login path |
| Local service API key (`verify_api_key`) | `core/service/auth.py` | **2** | no | **RETAIN** (local security) |
| Local OpenAI-gateway bearer (`LEMONCROW_GATEWAY_TOKEN`) | `gateway/openai_gateway/app.py:40-66` | **2** | no | **RETAIN** |
| Team workspace HMAC signing secret | `pro/capabilities/team/workspace.py:289` | **2** | no | **RETAIN** (random-local already) |

### 2.2 Outbound network activity

Automatic LemonCrow/GitHub calls that fire **without explicit user action**:

| # | Trigger | Destination | Data | Required? | Disable today | Disposition |
|---|---|---|---|---|---|---|
| 1 | MCP startup seamless login | `lemoncrow.com/account`, `/api/auth/me` | device_id (hw hash), hostname | no | `lc init --no-login` | remove |
| 2 | MCP startup git auto-update (default ON) | `github.com/lemoncrow-lab/lemoncrow` | git fetch | no | `LEMONCROW_AUTO_UPDATE=0` | default OFF |
| 3 | MCP request boundary usage tick (120s) | `lemoncrow.com/api/usage/report[-anon]` | report_id, savings, `machine_id`(hw hash) | no | — | remove |
| 4 | Savings-reconciler daemon (30m) | same | same | no | — | remove |
| 5 | Session-end hook public rollup | `lemoncrow.com/api/telemetry/rollup` | hashed install/session key, aggregate $ | no | **not gated by opt-out** | gate on opt-in |
| 6 | Daily daemon public rollup | same | same | no | same gap | gate on opt-in |
| 7 | Daemon periodic auto-update (`--auto-update`) | GitHub | git fetch / installer | no | run w/o flag | keep opt-in, unit default off |
| 8 | Daemon session import → sync | `lemoncrow.beseam.com/api/sync` | **raw** anon UUID, per-session stats | no | `remote_enabled=false` | off by default + hash id |
| 9 | Entitlement check `/api/auth/me` | `lemoncrow.com` | Bearer token | no | not signed in | delete |
| 10 | CLI/MCP product telemetry | local `:4318` (PostHog only if key set) | scrubbed events + anon_id | no | `remote_enabled=false` | off by default |

User-initiated / legitimate (retain): `lc update` (GitHub releases API);
user-configured model-provider + embedding calls (Anthropic/OpenAI/Google/…
only when the user sets a key); optional dependency downloads (ast-grep — SHA
pinned; HF embedder models; mem0/openmemory clone); Langfuse (opt-in, keys
required). No crash reporting, no remote feature flags, no runtime price refresh
exist.

Hardcoded LemonCrow-controlled hosts to remove/neutralize: `lemoncrow.com`
(`store.py:153`, `oauth_flow.py:122`, `public_rollup.py:38`,
`settings_registry.py:1362`, `licensing/__init__.py:43`, `mcp_server.py:11269`),
`lemoncrow.beseam.com` (`sync.py:43`), `lemoncrow.dev/telemetry`
(`banner.py:15`).

### 2.3 Device / installation identity

Three identity subsystems:

- **System A — commercial device id (DELETE).** `store._read_os_machine_id()`
  + `stable_machine_device_id()` + `load_or_create_device_id()` +
  `LEMONCROW_DEVICE_ID` env (`core/capabilities/licensing/store.py:180-292`).
  Hardware-derived; sole purpose is cap/entitlement binding. Consumers:
  `licensing_gate` device-hash checks, `usage_report._machine_hash`,
  `entitlements._verified_plan`, `oauth_flow` (also sends hostname). Delete with
  the licensing subsystem.
- **System B — anonymous telemetry id (RETAIN, already compliant).**
  `core/foundation/identity.py:get_anon_id()` — a random `uuid4` at
  `~/.config/lemoncrow/telemetry_id` (0600), resettable via `reset_anon_id()`,
  never hardware-derived. This is the canonical **local installation id**. Only
  transmitted under opt-in telemetry. No change needed except documentation and
  making sure it is never transmitted when telemetry is off.
- **System C — team machine binding (REGENERATE random-local).**
  `pro/capabilities/cross_vendor_memory/audit_log.local_machine_id()` reads
  `/etc/machine-id`/hostname independently. Genuine local need (distinguish
  machines in a local team workspace) but hardware-derived → replace with a
  random-local id (reuse `get_anon_id`/a dedicated `instance_id` file).

Local cryptographic material to RETAIN, separated from account identity: team
workspace HMAC secret (`secrets.token_hex`), `LEMONCROW_GATEWAY_TOKEN`,
`LEMONCROW_API_KEY`. The server-issued OAuth `auth_token` (System A adjunct) is
deleted with the account system.

### 2.4 Proprietary / closed components — reproducibility

**Verdict: the public repo cannot currently build/run the core runtime from
source.** The whole engine is `lemoncrow.pro` (265 `.py` here), excluded from
the mirror (`release/public-paths.txt` `!src/lemoncrow/pro/` plus pre-split
history denies) and shipped `.so`-only; `hatch_build.py` compiles it and deletes
the `.py`, with an "IP-leak guard" that fails the build if any `pro` module
can't be compiled. 34 module-top-level `from lemoncrow.pro …` imports across
`core/runtime/engine.py`, `core/service/{api,bootstrap_context}.py`,
`gateway/adapters/{mcp_server,runtime}.py`, and most CLI commands mean importing
the public source without the compiled wheel raises `ModuleNotFoundError`. No
open fallback exists.

Other closed/downloaded pieces: license-issuer service (private `services/`);
ast-grep (SHA-pinned optional download, MIT upstream); Zoekt (opt-in, off by
default, lexical fallback); telemetry rollup + sync endpoints (hosted, opt-in
after fix); vendored babel stub (clean open replacement — keep). No private PyPI
index or `git+ssh` deps.

**Resolution:** the `pro` source exists in this repo, so **open-source it**
(include in the mirror, stop deleting the `.py`; keep mypyc as an optional perf
build). This is the only path that makes the repo genuinely complete — but it is
a relicensing decision requiring owner sign-off (§3, §11).

### 2.5 Telemetry & privacy

- Defaults **ON**: `TelemetryConfig.remote_enabled = True`,
  `lexical_frustration_enabled = True` (`telemetry/config.py:19-21,42`).
- Event schema is privacy-conscious: allowlisted props, bucketed durations,
  SHA-256-hashed repo/block/cluster ids, PII/secret/path scrubbing; no raw
  source, symbol names, file paths, or command args. No crash reporting.
- PostHog cloud export is **effectively dead code** (`init_product_telemetry`
  never called; `LEMONCROW_POSTHOG_KEY` empty by default; live export path
  targets local `:4318`).
- Two real always-on LemonCrow egress paths: **public rollup**
  (`lemoncrow.com/api/telemetry/rollup`, *not* gated by `remote_enabled`, no CLI
  off-switch) and **session sync** (`beseam.com/api/sync`, gated by
  `remote_enabled` but default-on, ships the **raw** anon UUID, blocking 10s).
- CLI controls exist (`lc telemetry status/show/remote/lexical/reset-id`) but
  don't cover the public rollup. No env kill-switch (no `DO_NOT_TRACK`).

### 2.6 Install / uninstall lifecycle

Install (`scripts/install.sh` → `bundle.sh`/`local.sh` → `lib/common.sh`):
top-level tarball is checksum-verified (fails **open** if the sidecar is
missing); ast-grep/ripgrep SHA-pinned; uv/node/go/rustup piped unverified. Good
dry-run coverage. Writes `~/.lemoncrow/**`, `~/.config/lemoncrow/**`,
user-level systemd/launchd units, `~/.local/bin` symlinks, two different PATH
sentinel blocks, per-host configs (surgical JSON/TOML/YAML merges).
**`run_setup` runs `lc init` (browser login) and prompts `lc login`.** Telemetry
defaults on. git-install auto-update is automatic; release auto-update is opt-in.

Uninstall (`scripts/uninstall.sh`): per-host removal is surgical (low clobber
risk); data preserved by default; `--purge` removes `~/.lemoncrow`. **Gaps:**
does NOT remove systemd/launchd units (orphaned, `Restart=always`); `--purge`
never touches `~/.config/lemoncrow` so the telemetry id survives; `~/.local/bin`
symlinks and the second PATH block leak; no account/identity purge on default
uninstall; no removed-vs-preserved summary.

### 2.7 Documentation & marketing

Commercial claims to remove/rewrite: `README.md` (Pro $20/mo, account required,
"uncapped after sign-in", proprietary-engine badge, `lemoncrow.com`
savings/pricing/vs links, `install.lemoncrow.com`); `docs/pricing.md` (whole
file); `docs/licensing.md` (de-commercialize); `docs/roadmap.md`,
`docs/production-readiness.md` (enterprise/governance/SSO/SLA);
`docs/marketing/**`; `docs-site` footer/config; `landing/**` (submodule — full
pricing/billing UI, out of runtime scope, note only). Cap-value contradiction
($100 vs $50) and telemetry-default contradiction (README "on" vs terms "off")
become moot once caps are removed and telemetry defaults off. **Benchmarks are
well-supported (raw data + methodology + reproduction in-repo) — keep them**
(Phase 10 forbids deleting benchmark history); only drop the unbacked "live
savings" counter claim.

Community-health files: present — `LICENSE`, `LICENSE-APACHE`, `CONTRIBUTING.md`,
`CLA.md`. **Missing — `NOTICE`, `CODE_OF_CONDUCT.md`, `SECURITY.md`,
`CHANGELOG.md`, `.github/ISSUE_TEMPLATE/`, PR template.**

---

## 3. Device-minting decision

**Delete** the commercial device identity (System A) entirely: its only purpose
is cap/entitlement binding, and it is hardware-derived. No local runtime
function depends on it (DB namespacing uses path-based `workspace_key`;
migrations are version-based; workers key on session_id/host; local service auth
uses a user-provided key; trace namespacing uses the random anon_id).

**Retain** System B (`get_anon_id`, random uuid4, local, resettable) as the
canonical local installation id — already meets every requirement of the policy
(random UUID, not machine-derived, in the data dir, resettable, not access-
control, not transmitted unless opted in). Document it; it is renamed in
concept to *installation id* though the on-disk file stays `telemetry_id` for
compatibility.

**Regenerate** System C (team `local_machine_id`) as a random-local id.

Migration for existing installs: on first run after upgrade, delete the legacy
account/device files (`~/.lemoncrow/{auth_token,auth_user.json,auth_base,
device_id,login_declined,cap_anon_token}`) — never transmit them — behind a
versioned one-time migration. `telemetry_id` is preserved.

---

## 4. Phase 2 implementation plan

Execution order (each step verified before the next):

**Step A — Telemetry off by default (Phase 5).** Smallest, highest-trust.
**Step B — Remove account/cap/entitlement system (Phases 3, 4, 10 core).**
**Step C — Remove device minting + hardware readers (Phase 4).**
**Step D — Neutralize remaining hosted network calls (Phase 5 tail).**
**Step E — Open the `pro` engine in the mirror/build (Phase 6).** DONE, under a
split license — see §5.2.
**Step F — Install/uninstall hygiene (Phases 7, 8).**
**Step G — Docs rewrite + community-health files (Phases 9, 10, 11).**
**Step H — Offline/no-network test + full suite (acceptance).**

### 4.1 Files to modify (primary)

- `core/service/telemetry/config.py` — `remote_enabled` default → False;
  `lexical_frustration_enabled` default → False; honor `DO_NOT_TRACK` /
  `LEMONCROW_TELEMETRY=off` env.
- `core/service/telemetry/public_rollup.py` + callers
  (`plugin_runtime.py:2835`, `servicectl_lifecycle.py:869`) — gate on
  `remote_enabled()`; make empty endpoint truly disable.
- `core/service/telemetry/banner.py` — reword to opt-in disclosure (telemetry
  is OFF; how to enable); drop non-tty auto-ack of an active collector.
- `core/service/sync.py` — hash `machine_id`; already `remote_enabled`-gated.
- `gateway/cli/commands/telemetry.py` — cover the rollup path in on/off/status.
- `core/capabilities/licensing/entitlements.py` — rewrite: local, always
  unlocked, no network (`is_pro→True`, `has_feature→True`, `require→pass`,
  `status→local`).
- `core/capabilities/licensing/__init__.py` — drop `pro_url`, keep unlock API.
- `core/capabilities/licensing/store.py` — delete auth-token/user/base +
  device-id machinery; keep nothing account-related.
- `pro/capabilities/licensing_gate.py` — `is_configured→False`;
  `resolve_cap_verdict→CapVerdict(dormant=False, verified=True, plan="free",
  reason="oss")`; `cap_exhausted→False`; drop pinned key.
- `gateway/adapters/mcp_server.py` — remove dormancy gating (10664-10700,
  11723-11861), seamless login (12201-12363), usage tick, feature-locked upsell;
  default git auto-update OFF.
- `gateway/cli/commands/admin.py` — remove `account` login/logout/cap subcommands
  and `_bootstrap_cap_verdict`; keep inert `lc account status`; strip `--login`
  from `lc init` (no browser prompt).
- `gateway/cli/commands/_shared.py` — `require_pro` → no-op; and call sites in
  `code.py`, `knowledge.py`, `savings.py`, `memory.py`, `playbooks.py`.
- `core/capabilities/plugin_runtime.py` — remove cap constants/meter
  (`ANONYMOUS_SAVINGS_CAP_USD`, `SAVINGS_CAP_BY_PLAN`, `savingsOverCap`,
  `persist_cap_verdict_token`, `cap_exhausted`, `refresh_subscription_meter`,
  referral code); keep local savings metering for display only.
- `pro/capabilities/cross_vendor_memory/audit_log.py` — `local_machine_id` →
  random-local id.
- `core/service/code_warm.py:187`, `pro/capabilities/code_context/engine.py:179,
  3994`, `pro/capabilities/knowledge_extract.py:36` — drop free-tier caps.
- `release/public-paths.txt` — remove the `pro`/pre-split denies.
- `hatch_build.py` — never delete `.py`; ship source; mypyc optional.
- `scripts/lib/common.sh`, `scripts/install.sh`, `scripts/uninstall.sh` — remove
  login/init prompts; don't enable telemetry; fix uninstall gaps; harden
  checksum; print summary.
- `README.md`, `docs/licensing.md`, `docs/architecture.md`, `docs/cli.md`,
  `docs/installation.md`, `docs-site` config, `pyproject.toml` license string.
- `mcp_server.py` git auto-update default; `core/settings_registry.py` telemetry
  defaults + drop rollup endpoint default.

### 4.2 Files to delete

- `core/capabilities/licensing/oauth_flow.py`
- `core/capabilities/licensing/usage_report.py`
- `core/capabilities/licensing/cap_verdict.py` (client sign mirror)
- `docs/pricing.md`
- Consider deleting `core/service/savings_reconcile.py`'s report path (keep
  local reconcile only).

(`licensing/features.py`, `models.py`, `entitlements.py` are kept but neutered so
the ~dozen import sites stay valid — deletion of the package would touch every
caller; a thin always-unlocked module is the smaller, clearer change.)

### 4.3 New files

- `NOTICE`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`,
  `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.md`,
  `.github/pull_request_template.md`.
- `docs/licensing-report.md` (dependency-license + relicensing report).
- Tests: `tests/privacy/test_offline_core.py` (network-blocked smoke),
  `tests/core/test_installation_id.py`, migration test.

### 4.4 Database / config migrations

- No LemonCrow *runtime* SQLite schema is keyed on device id (server-side
  migrations 0019–0022 live in the private `services/` and are out of scope).
- One-time on-disk migration: delete legacy `~/.lemoncrow/{auth_token,
  auth_user.json,auth_base,device_id,login_declined,cap_anon_token}` and
  `~/.lemoncrow/subscription.json`/`auth.json` cap fields; write a
  `migration_version` marker in the store root. Preserve `telemetry_id`,
  sessions, lessons, code index, memory. Add a versioned + tested migration.
- `telemetry.toml`: existing `remote_enabled=true` files are honored (explicit
  user choice); only the **default** for new installs flips to false.

### 4.5 Compatibility risks

- Existing users who *were* signed in lose Pro-gated remote features (cross-
  vendor memory sync etc.) — but every local feature is now unlocked, so net
  local capability increases. Document.
- Tests that assert cap/dormancy/entitlement behavior
  (`tests/core/test_cap_verdict.py`, `test_usage_report.py`,
  `test_entitlement_signed_plan.py`, `test_device_id.py`, cap-period tests) must
  be rewritten or deleted to assert the new always-unlocked behavior.
- `pro` opening changes the wheel contents/size and the mirror diff; CI mirror
  checks and `hatch_build` IP-leak guard must be updated.

### 4.6 Security risks

- Do **not** weaken the local service API key (`service/auth.py`) or the OpenAI-
  gateway bearer — they protect network-accessible local services. Keep them,
  separated from (now-deleted) account identity.
- Removing the fail-closed gate is intentional; no local security depends on it.
- Publishing the Ed25519 *public* key is harmless, but we drop it entirely.
- Opening `pro` exposes previously-secret source — a business decision (§11),
  not a security regression.

### 4.7 Network behavior — before vs after

| Command | Before | After |
|---|---|---|
| `lc init` | browser OAuth to lemoncrow.com | fully local, no network |
| MCP startup | seamless login + git fetch + cap mint | no network (auto-update opt-in) |
| MCP tool list/call | usage report every 120s; dormant if no token | no network; always active |
| daemon tick | usage report, rollup, sync, auto-update | none by default (all opt-in) |
| any `is_pro()` check | `/api/auth/me` | local, always unlocked |
| CLI start | product telemetry (local; PostHog if key) | nothing unless telemetry opted in |
| `lc update` | GitHub releases API | unchanged (user-initiated) |
| model calls | user-configured provider | unchanged |

### 4.8 Features that will intentionally stop working

- Account login/logout, subscription/plan display, savings cap & dormancy,
  device/seat management, referral codes, the hosted "live savings" counter,
  the anonymous-trial mint. All Pro/Enterprise *gating* is gone (the features
  themselves become free). Optional hosted telemetry/sync remain available but
  off by default.

### 4.9 Tests to add / modify

- Add: offline-core smoke (init, service start, index, retrieval, MCP list,
  status — all with LemonCrow domains blocked, no credentials); installation-id
  randomness/reset; legacy-identity migration; "telemetry off ⇒ zero outbound";
  `lc account status` reports healthy-local, no login-as-error.
- Modify/remove: all cap/entitlement/device-id/usage-report tests.

### 4.10 Documentation to update

README (status, account-free quick start, what it does/doesn't, privacy/network,
supported environments, removal, maintenance expectations); delete pricing;
de-commercialize licensing/architecture/cli/installation; docs-site chrome;
remove enterprise/SSO/SLA promises; keep benchmarks with methodology.

---

## 5. Owner decisions required before implementation

1. **Open-source the `lemoncrow.pro` engine?** Required for the repo to build
   from source (Phase 6). The source is present here; opening it means removing
   the mirror denies, shipping `.py`, and relicensing `pro` under Apache-2.0.
   This is an irreversible disclosure of the "moat" and needs explicit consent.
   Alternative: keep it closed and honestly label the OSS edition as a
   source-readable shell that needs the proprietary wheel (fails the
   "complete/reproducible" acceptance criterion).
2. **Remove the account/commercial system entirely, or keep it as an optional
   package?** The plan assumes full removal (no active hosted service worth
   preserving for a maintenance-mode local tool). Keeping an optional login
   would mean isolating it behind an explicit `lemoncrow[account]` extra.
3. **Relicense to pure Apache-2.0 and update `LICENSE`?** Depends on (1) and on
   the CLA (which already grants relicensing rights). Legal files are not
   overwritten without confirmation (Phase 11).

Everything else (telemetry default, device-minting removal, install/uninstall
hygiene, docs) proceeds regardless of these answers.

### 5.1 Owner decisions — RESOLVED

1. **Open-source the `lemoncrow.pro` engine: YES.** Include `pro/` source in the
   mirror, stop stripping `.py`, keep mypyc as an optional perf build.
2. **Account system: KEEP AS AN OPTIONAL, FULLY-DECOUPLED EXTRA.** The local
   runtime must never require, prompt for, warn about, or depend on login, and
   login must never gate a feature or block operation. `lc account
   login/logout/status` remain available as an explicitly optional convenience
   for anyone who wants to link a hosted account, but grant nothing required
   (all features are free/local). The savings-cap, dormancy, seamless
   background login, `lc init` login prompt, usage-report/cap-mint, and
   entitlement gating are removed regardless. Practically: entitlements resolve
   locally to always-unlocked; `oauth_flow.py` is retained but only reachable
   via the explicit optional command.
3. **Relicense to Apache-2.0: YES.** Rewrite `LICENSE` to pure Apache-2.0, add
   `NOTICE`, update the `pyproject.toml` license string, and produce
   `docs/licensing-report.md` for the record.

### 5.2 Step E executed (2026-09-07)

Decisions 1 and 3 both stand, as written: the `pro` source is published in the
mirror under Apache-2.0, the `!src/lemoncrow/pro` deny is gone, and the
`hatch_build` IP-leak guard is removed (the remaining wheel check only rejects a
`.so`/`.py` twin). The whole repository is Apache-2.0 and builds from public
source.

A split license (Apache-2.0 + PolyForm Noncommercial on `pro/`) was briefly
shipped and reverted the same day — protecting against a commercial fork was
judged not worth excluding every commercial adopter at this stage. Tighten
future releases if that ever changes; see `docs/legal/licensing.md`.

One mechanical lesson, now fixed in `scripts/mirror.py`: widening the allowlist
publishes nothing on its own. Incremental mirroring applies only the paths each
commit touched, so removing a deny needs `make mirror ARGS="--resync"` (full
filtered-tree rebuild as one fast-forward commit) or `--force` (full replay,
rewrites public history).
