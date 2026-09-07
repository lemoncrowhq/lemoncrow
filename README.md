<!-- cspell:ignore Alamofire Excalidraw ast-grep codegraph ctags django jcodemunch nohit okhttp scip serena tokio vscode zoekt beasm Trendshift telegraphese -->

<div align="center">

<img src="docs-site/favicon.png" width="36" height="36" alt="" style="vertical-align: middle;">

# LemonCrow Runtime

### Keep your coding agent sharp on real codebases

**Context engineering, done right.**

LemonCrow runs underneath Claude Code, Codex, and other supported hosts with a local code graph, exact-range reads, bounded output, durable memory, and verified runtime controls — fully local, no account required.

**State-of-the-art context engineering.** Read less, Output less, without compromising correctness. LemonCrow is tuned end to end across input context and output — ranked retrieval, exact-range reads, and bounded, compacted output — and out-measures grep-class code-index and output-compression tooling on the [numbers below](#results) (~1.9x retrieval MRR vs ripgrep, 27.9% fewer output tokens on SWE-bench Verified).

[![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square)](LICENSE)
[![Latest release](https://img.shields.io/github/v/release/lemoncrow-lab/lemoncrow?style=flat-square)](https://github.com/lemoncrow-lab/lemoncrow/releases)
[![Stars](https://img.shields.io/github/stars/lemoncrow-lab/lemoncrow?style=flat-square)](https://github.com/lemoncrow-lab/lemoncrow)

[![Claude Code](https://img.shields.io/badge/Claude_Code-supported-blue?style=flat-square)](integrations/claude)
[![Codex](https://img.shields.io/badge/Codex-supported-blue?style=flat-square)](integrations/codex)
[![opencode](https://img.shields.io/badge/opencode-supported-blue?style=flat-square)](integrations/opencode)
[![LemonCode](https://img.shields.io/badge/LemonCode-supported-blue?style=flat-square)](integrations/lemoncode)
[![Copilot](https://img.shields.io/badge/Copilot-supported-blue?style=flat-square)](integrations/copilot)
[![Copilot CLI](https://img.shields.io/badge/Copilot_CLI-supported-blue?style=flat-square)](integrations/copilot-cli)

[Results](#results) · [What it does](#what-lemoncrow-does) · [Quick start](#quick-start) · [Limitations](#what-lemoncrow-does-not-do) · [Privacy](#privacy-and-network-behavior) · [Removal](#removal)

</div>

```bash
curl -fsSL https://github.com/lemoncrow-lab/lemoncrow/releases/latest/download/install.sh | bash
cd your-project && lc init
```

<div align="center"><sub>Checksummed GitHub release · no login, no network — details in <a href="#quick-start">Quick start</a>.</sub></div>

---

## Results

These are fixed results from pinned benchmark runs — not a live counter. Every
headline number links back to committed raw runs and methodology in
[BENCHMARKS.md](BENCHMARKS.md). The model, tasks, containers, turn limits, and
verification harness were held constant. Results are mixed by design and include
a regression (SWE-bench Lite below).

| Benchmark                                         | Baseline correct | LemonCrow correct | Correct delta |        Baseline cost |    LemonCrow cost | Cost delta |
| --------------------------------------------------- | -----------------: | ------------------: | --------------: | ---------------------: | ------------------: | -----------: |
| SWE-bench Verified, 50 tasks x 5 reps             |            80.8% |         **92.8%** |  **+12.0 pp** | $234.84 |**$165.45** | **29.5% cheaper** |            |
| SWE-bench Lite, 10 tasks x 5 reps                 |            98.0% |             96.0% |       -2.0 pp |   $19.83 |**$17.51** | **11.7% cheaper** |            |
| SWE-bench Pro, 10 tasks x 5 reps                  |            88.0% |         **90.0%** |   **+2.0 pp** |   $39.01 |**$30.61** | **21.5% cheaper** |            |
| Exploration tasks across 7 large repos x 5 reps   |                - |                 - |             - |    $19.11 |**$6.29** |   **67% cheaper** |            |
| Telegraphic Q&A, 20 prompts x 5 reps              |                - |                 - |             - |     $8.40 |**$4.48** | **46.7% cheaper** |            |
| Terminal-Bench 2.1, 89 tasks x 5 reps, Opus 4.8 (matched)\* |            78.9% |             78.9% |  0.0 pp (tied) |               $73.75 |          **$61.98** | **16.0% cheaper** |
| Terminal-Bench 2.1, 89 tasks x 5 reps, Opus 5 (standalone)† |                - |             80.7% |             - |                    - |             $38.68 |                 **47% cheaper** than opus 4.8 |

<sub> Both arms 89 tasks x 5 reps = 445 trials on the same dataset — LemonCrow's Harbor run, public at [Harbor Hub job `47e1713b`](https://hub.harborframework.com/jobs/47e1713b-cad9-4715-a9e7-ca71ff202ba7), vs the Claude Code 2.1.205 leaderboard run — so correctness is directly comparable; this run ties baseline exactly (351/445 both sides). LemonCrow sends 98.6% fewer fresh input tokens (182K vs 12.87M). Cost is normalized to the 1-hour cache-write rate on both sides (LemonCrow's harness bills prompt-cache writes at that tier; baseline's real run used the cheaper 5-minute tier, so it's re-priced at 1-hour for a same-rate comparison) on the 86 of 89 tasks with a priceable trajectory both sides. † The Opus 5 row is LemonCrow-only ([Harbor Hub job `18239ddc`](https://hub.harborframework.com/jobs/18239ddc-556a-4631-a20d-bcf5da8d16a2), 359/445 resolved, `reasoning_effort=high`): no official Claude Code + Opus 5 leaderboard run exists yet, so there is nothing to compare against — the empty cells are missing baselines, not zeros. Its $38.68 is real own-tier billing (nothing to normalize) over its own 86-of-89 priceable tasks, a different exclusion set from the Opus 4.8 cut, so do **not** subtract the two rows from each other: different model, different task set, no controlled comparison. See [BENCHMARKS.md](BENCHMARKS.md#terminal-bench).</sub>

<p align="center">
  <img src="benchmarks/cost_vs_savings_scatter.svg" alt="LemonCrow vs baseline: dollars saved per run against baseline task cost" width="720">
</p>

SWE-bench Verified detail (250 runs a side) — one-shot search collapses the
grep-and-read loop, so turns, wall-clock, and tool calls drop together:

| Metric           | Baseline | LemonCrow |            Delta |
| ------------------ | ---------: | ----------: | -----------------: |
| Turns            |    6,962 |     4,336 |  **37.7% fewer** |
| Wall-clock       |    14.3h |     10.9h | **23.7% faster** |
| Total tool calls |    6,700 |     4,167 |       **-37.8%** |
| Output tokens    |    3.04M |     2.19M |  **27.9% fewer** |

### Scale

Indexing throughput and search quality hold up at repository sizes agents
actually hit. A cold full rebuild of the Linux kernel core (1.24M symbols,
4.5M lines) and retrieval quality vs grep-class tools on ~7,200 query/answer
pairs across 14 repos:

| Metric                                    |                          LemonCrow | Grep-class baseline |
| ------------------------------------------- | -----------------------------------: | --------------------: |
| Linux cold index, lexical (1.24M symbols) |                  **179s** (~3 min) |                  — |
| Linux cold index, zoekt trigram           |                          **13.7s** |                  — |
| Retrieval MRR (higher = better)           | **0.727** semantic / 0.676 lexical |     0.376 (ripgrep) |
| Query latency, p95                        |     134ms lexical / 390ms semantic |  **66ms** (ripgrep) |

Ranked search is ~1.9x more accurate than ripgrep at a still-interactive p95;
ripgrep wins raw latency but not what it finds. Per-repo indexing table and the
full 13-tool retrieval comparison: [BENCHMARKS.md](BENCHMARKS.md#indexing-time).

Reproduce any of this from committed raw data: see [BENCHMARKS.md](BENCHMARKS.md)
and [docs/benchmarks/results.md](docs/benchmarks/results.md).

## What LemonCrow does

LemonCrow keeps your existing coding agent and changes the working set around it:

<p align="center">
  <img src="docs/assets/screenshots/source-map.jpg" alt="LemonGraph, LemonCrow's local code graph, showing a full repository code universe: 28,462 indexed symbols, 10,349 tracked files, 38,811 map nodes, and 23,894 resolved calls, with one function focused to show its callers and callees." width="720">
</p>
<p align="center"><sub>LemonGraph — your codebase's code universe — 28,462 symbols · 38,811 nodes · 23,894 calls. Live, local, on this repo.</sub></p>

| Stage      | Runtime behavior                                                                                              |
| ------------ | --------------------------------------------------------------------------------------------------------------- |
| **Find**   | Rank symbols, definitions, callers, callees, usages, and exact source ranges before broad file exploration.   |
| **Read**   | Return an outline or only the requested lines; cap noisy command and web output with recoverable spill files. |
| **Carry**  | Preserve useful task state through memory, deduplication, compaction manifests, and handover packets.         |
| **Verify** | Notice code changes without tests or checks, then nudge the agent before it declares completion.              |

**One shot to answer, one shot to action.** Every stage above is built around a
single round trip: one search returns the symbol, its callers, and the exact
source ranges — not a grep loop; one edit applies every hunk across every file —
not a patch-per-file series. Re-asking the same ground is the cost, not the
model.

### What actually gets replaced

On Claude Code, `lc init` gives the agent five grounded tools and hides the
equivalent built-ins — one way to do each job, not two. Other hosts use the
strongest equivalent controls they expose.

| LemonCrow tool | Replaces (hidden from the model) | Why                                                                                                                                                                       |
| ---------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `code_search`  | Grep, Glob                       | One call returns the symbol, its callers/callees, and ranked source — no grep-loop-then-read-whole-file. Ranked by LemonGraph centrality over a tree-sitter symbol table |
| `read`         | Read                             | Returns an outline or the exact`:L10-L40` range, budgeted, instead of the full file                                                                                       |
| `edit`         | Edit, Write                      | Verified, cross-file edits in one call instead of per-file patch-or-create guessing                                                                                       |
| `bash`         | Bash                             | Output is capped and structured so a noisy build log can't blow the context window                                                                                        |
| `web_fetch`    | WebFetch                         | Strips a page to clean Markdown instead of a raw HTML dump                                                                                                                |

What's unchanged: the host, the model, your workflow. Full internals:
[Architecture](docs/reference/architecture.md).

**Caveat — Cursor (CLI vs IDE).** Built-ins can't be hidden there, so
LemonCrow is additive — Claude Code and Codex can displace most of their
built-in toolset, and that does not apply on Cursor. Measured on SWE-bench
Lite (10 tasks, `cursor-grok-4.5-high`, matched prompts): **Cursor CLI +
LemonCrow was ~40% cheaper** than Cursor CLI baseline (tokens −39.8%, cost
−41.2%). The same tasks in **Cursor IDE did not show that saving** — CLI is
the cheaper Cursor path today. Reproduce from
`reports/benchmark/swe/20260802T121526Z/`.

NOTE: One inference, flagged: that Cursor selectively chooses server-side what to cache-write
is derived from implied hit rates, not confirmed in their docs.
Cursor stores no local cache counters, so every hit
rate is computed as 1 − billed/integral. Treat the caching mechanism as
unproven; the CLI cost delta above is from Usage/token totals on the pinned run.
## Quick start

The [two lines at the top](#lemoncrow-runtime) are the whole setup: the installer
pulls a checksummed GitHub release, and `lc init` indexes the repo it is run in —
locally, no login, no network, nothing sent anywhere.

Install once, then `lc init` in every project where you use your coding agent:

```bash
cd another-project
lc init
```

## More ways to run

### Code from your chat app — free

`lc mcp serve` publishes this workspace as a **remote MCP server**: a public
`https://…/mcp` URL behind OAuth 2.1. Nothing about it is vendor-specific — any
client that accepts a remote MCP server URL (ChatGPT connectors, Claude
connectors, Cursor, VS Code, …) gets the same LemonCrow tools (search, read,
edit, bash) your local agent uses, so you can code from a chat window or a
phone. Chat usage is typically billed differently from coding-agent usage, so
this is often the cheaper seat.

```bash
lc mcp serve
```

Prints a pairing code and, by default, an auto-launched cloudflared tunnel URL
(installs cloudflared on first use if missing). Paste the printed MCP server URL
into your client, set Authentication to **OAuth**, and approve the browser
prompt with the pairing code:

| Client                          | Where to paste it                                                  |
| ------------------------------- | ------------------------------------------------------------------ |
| ChatGPT                         | Settings → Plugins → Browse Plugins → (next to search) + → Create |
| Claude (web, desktop, mobile)   | Settings → Connectors → Add custom connector                       |
| Cursor / VS Code / Zed / others | Add a remote (streamable-HTTP) MCP server                          |

The pairing code is stored per server, so restarting `lc mcp serve` keeps the
same code — nothing to re-type. Use `--new-pairing-code` to rotate it, or
`--reset` to wipe the pairing and every issued token.

Use `--persistent` so the URL survives restarts too — a rotating quick-tunnel
URL has to be re-pasted every run, and some clients drop the connector when it
changes. `--persistent` also **registers the server as a background service**
(systemd on Linux, launchd on macOS) bound to the directory you started it in:
it keeps running after you close the terminal and comes back on reboot, with
the same URL and the same pairing code. Add `--foreground` to run it in the
terminal instead.

```bash
lc mcp serve --persistent --hostname mcp.example.com   # install + start, prints the code
lc mcp service list                                     # every installed server, URL, workspace, state
lc mcp service restart mcp.example.com --new-pairing-code
lc mcp service logs mcp.example.com
lc mcp service stop|start|remove mcp.example.com
```

Most clients register themselves with the server automatically. For the ones
that ask you to supply an OAuth client ID instead, `lc mcp client` mints a
stable one; pass `--redirect-uri` if your client shows a per-app callback URL
(ChatGPT does, for newly created apps).

`lc chatgpt serve` still works as a hidden alias of `lc mcp serve`.

| Flag                        | Effect                                                                                                                   |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `--no-tunnel`               | Bring your own tunnel (named cloudflared tunnel, ngrok).                                                                 |
| `--persistent --hostname X` | Stable URL via a Cloudflare named tunnel (needs a domain in your Cloudflare DNS); survives restarts instead of rotating, and installs it as an always-on user service for that directory. Each hostname gets its own tunnel, state, OAuth store and service, so several projects can serve at once. |
| `--foreground`              | With `--persistent`: serve in this terminal instead of registering the background service.                                |
| `--no-auth`                 | Serve`/mcp` with no authentication — the tunnel URL alone grants access. Prefer OAuth (default).                        |
| `--new-pairing-code`        | Rotate the stored pairing code. Already-authorized clients keep working; only re-pairing needs the new one.               |

Full request/response traffic is logged locally per run (path printed at
startup; credentials and tokens are redacted) so you can audit exactly what the
client sent and got back.

**Known ChatGPT-side quirk:** Persistent connection is much more reliable. Sometimes chatgpt looses the tool aceess on a new chat message conversation and without reattaching it can't access the tool. Workaround is branchoff the chat and then reattach the tool and continue with your message.
**Permissions**: If it complains about permissions or asks to reconnect, check in the Setting -> Plugins, it has `Allow All` permission

> ⚠ The pairing code is a password — don't share the tunnel URL. This
> exposes shell-grade tool access (`bash`, `edit`) to this machine while the
> server runs. Stop it (Ctrl-C) when you're done.

### Inspect your past sessions (Offline Replay, dry mode)

LemonCrow records all sessions locally so you can inspect, audit, and debug exactly what your agent did.

```bash
lc session stats     # read-only report of wasted tool calls and round-trips
lc session replay    # replay a recorded session through the real LemonCrow tools
```

Both are local and read-only — no model re-run, nothing transmitted.

<p align="center">
  <img src="docs/assets/screenshots/session-replay.gif" alt="LemonCrow session replay demonstration" width="720">
</p>
<p align="center"><sub>Replay recorded agent sessions locally with full tool visibility and resource usage breakdown.</sub></p>

## Agents and skills

### Agents

Packaged in [integrations/agents/](integrations/agents/) — each a distinct
capability grant (subagent name `lemoncrow:<mode>`):

| Agent      | Writes? | Use                                               |
| ------------ | :-------: | --------------------------------------------------- |
| `code`     |   Yes   | default interactive — edits, refactors, features |
| `auto`     |   Yes   | fully autonomous — CI/headless runs              |
| `solve`    |   Yes   | end-to-end solving of a well-defined task         |
| `execute`  |   Yes   | one verified pass of an accepted plan             |
| `general`  |   Yes   | catch-all for mixed work                          |
| `bare`     |   Yes   | minimal toolset, same discipline                  |
| `explore`  |   No   | read-only exploration — locate and cite          |
| `plan`     |   No   | read-only planning, stops for human checkpoint    |
| `review`   |   No   | adversarial read-only review                      |
| `research` |   No   | external web research — cited memo               |

### Skills

Optional Packaged in [integrations/skills/](integrations/skills/): `/lemoncrow`, `/benchmark`,
`/orchestrate`, `/swarm`, `/perf-review`, `/ux-review`, `/recall`.

## Code hygiene

The best code is the code you never wrote. Every LemonCrow persona climbs a fixed
ladder before writing anything, stopping at the first rung that holds:

| # | Rung | Do |
| --- | --- | --- |
| 1 | **Need it at all?** | Skip what the task doesn't require (YAGNI). |
| 2 | **Already here?** | Reuse the helper, util, or pattern already in the repo. |
| 3 | **Stdlib?** | Use the standard library before rolling your own. |
| 4 | **Native feature?** | Reach for the platform capability that already exists. |
| 5 | **Installed dep?** | Solve it with a dependency already in the tree. |
| 6 | **One line?** | If it collapses to one line, make it one line. |
| 7 | **Otherwise** | Write the minimum new code that works. |

The ladder runs _after_ the agent understands the problem, not instead of it, and
it never trades away validation, error handling, security, or accessibility. Lazy
about the solution, never about reading the code first.

When a change deliberately cuts a corner with a known ceiling (a global lock, an
O(n²) scan, a naive heuristic), the agent leaves an `lc-debt: <ceiling>; <upgrade
path>` marker. Harvest them any time into a ledger — `lc debt` flags any marker
that names no upgrade path (`no-trigger`), the ones that silently rot:

```bash
lc debt          # ceiling + upgrade per deferred simplification, no-trigger flagged
lc debt --json   # same, machine-readable
```

The packaged `code-audit` workflow (Claude Code) adds an over-engineering lens
that returns a concrete delete-list — code to remove, not rewrite.

## What LemonCrow does not do

- It is **not** a hosted service. There is no cloud backend, dashboard account,
  or team collaboration server.
- It does **not** run your model for you — you bring and configure your own
  provider/API key (Anthropic, OpenAI, Ollama, …).
- It does **not** guarantee the benchmark deltas above on your repository;
  results vary by task, codebase, and model.
- Some integrations are early or in progress; behavior varies by host (e.g.
  session-close verification is enforced on Claude Code, advisory elsewhere).

## Privacy and network behavior

- **Runs locally.** Indexing, search, edits, and memory all stay on your
  machine; core functionality works offline.
- **Anonymous telemetry is ON by default.** Turn it off with `lc telemetry remote off`,
  or set `DO_NOT_TRACK=1` / `LEMONCROW_TELEMETRY=off`.
- **No** source code, prompt, repository path, or symbol name is ever sent —
  only aggregate counts, durations, and dollar estimates plus a hashed install
  key. There is no crash reporting.
- Apart from that rollup, the only network calls the runtime makes are ones you
  initiate (`lc update`, which checks GitHub Releases).
- Full detail: [docs/setup/privacy.md](docs/setup/privacy.md).

## Supported environments

- **Operating systems:** Linux and macOS (primary); Windows is partially, never tested.
- **Runtime:** Python 3.12–3.13, managed with [`uv`](https://docs.astral.sh/uv/).
- **Agent hosts:** Claude Code, Codex, opencode and LemonCode today — LemonCode is
  LemonCrow's own fork of opencode, so the frontend itself can be optimized;
  Copilot, Cursor, Hermes, and Antigravity are in progress. Any MCP-compatible agent can
  connect to the same tools.
- **Build requirements:** `uv`, a C toolchain (only if you opt into the `mypyc`
  performance build; a pure-Python build works without it), and `git`.
- **Known limitations:** see [What LemonCrow does not do](#what-lemoncrow-does-not-do)
  and [Troubleshooting](docs/setup/troubleshooting.md).

## Roadmap — Savings Optimization

The LemonCode host/control plane and all six savings-runtime implementations
are shipped locally: closed-loop routing (**LemonRoute**), Output Governor V2,
provider-aware cache economics, the bounded local retrieval firewall
(**LemonScout**), hybrid MCP exposure, and verified cross-session reuse. The
five learned/policy levers remain measurement-pending; MCP exposure is
adapted-complete without a mandatory search-first call.

See the
[detailed savings optimization status and roadmap](docs/planning/savings-optimization-roadmap.md)
for shipped behavior, missing work, original estimates, acceptance gates, and
the proposed implementation order. Planning estimates there are non-additive
and are not presented as measured savings.

## Learn more

- [Installation](docs/setup/installation.md)
- [Troubleshooting](docs/setup/troubleshooting.md)
- [Benchmarks](BENCHMARKS.md) · [full results with methodology](docs/benchmarks/results.md)
- [CLI reference](docs/reference/cli.md)
- [Architecture](docs/reference/architecture.md)
- [Privacy & network behavior](docs/setup/privacy.md)
- [Maintenance-mode transition (audit & rationale)](docs/operations/maintenance-mode-transition.md)

## Removal

Uninstall LemonCrow and its host integrations, preserving your data by default:

```bash
bash scripts/uninstall.sh
```

To also remove all LemonCrow-managed local state (databases, caches, logs, the
local installation identifier, and configuration):

```bash
bash scripts/uninstall.sh --purge
```

The uninstaller stops background services, removes user-level systemd/launchd
units, removes LemonCrow-owned host-integration entries (without touching
unrelated agent-host configuration), reverts LemonCrow's PATH changes, and prints
exactly what was removed and preserved. Preview with `--dry-run`.

## Why I built this

I am a solo builder, previously at Google doing performance optimizations and cost savings. I kept burning my weekly credits before the week was out. Every
"token-saving" claimed tools only every shows me a curated list of tasks where they save. Only showing partial wins. Claiming 50-60-70% wins infact they never shows on all varaties of tasks. In reality they either same so little to justify complexity or they don't save at all because they add fat system prompts on their own that the savings are offset.

So I built LemonCrow. Every number below is an absolute-dollar measurement
([BENCHMARKS.md](BENCHMARKS.md)) — on swe, terminalbench and infact some of the claimed tools task themselves. Result? **lemoncrow beat them all**.

## Development & Building from Source

If you want to build LemonCrow from source or run a local development setup, clone the repository and run the local installation script (see [Installation](docs/setup/installation.md) for full details):

```bash
git clone https://github.com/lemoncrow-lab/lemoncrow
cd lemoncrow
bash scripts/local.sh
```

---

## License

LemonCrow is free and open-source software under the
[Apache License, Version 2.0](LICENSE) — in its entirety, engine included.
There is no open-core split, no paid tier, and no proprietary component: the
whole repository builds and runs from source. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
