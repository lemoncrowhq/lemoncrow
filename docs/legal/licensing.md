# Licensing

LemonCrow publishes **all** of its source. It is licensed by directory:

| Path | License | Commercial use |
| --- | --- | --- |
| everything except `src/lemoncrow/pro/` | Apache-2.0 | allowed |
| `src/lemoncrow/pro/` (the engine) | PolyForm Noncommercial 1.0.0 | separate license required |

The runtime, the CLI, the MCP server, the SDK, and host integrations are open
source under the Apache License, Version 2.0. The engine — code intelligence,
retrieval, prompt compilation, memory, routing, and verification — is
**source-available**: readable, buildable, and free for any noncommercial
purpose (personal use, study, research, hobby projects, education, charities,
public research, government), but commercial use needs a separate license.

There is no proprietary binary component: the whole repository builds and runs
from source.

- Full license terms: [`/LICENSE`](../../LICENSE), [`/LICENSE-APACHE`](../../LICENSE-APACHE),
  [`/src/lemoncrow/pro/LICENSE`](../../src/lemoncrow/pro/LICENSE)
- Attribution and third-party notices: [`/NOTICE`](../../NOTICE)
- SPDX for the distribution as a whole: `Apache-2.0 AND PolyForm-Noncommercial-1.0.0`

## No account, no entitlement gate

Every feature runs locally and is available to everyone at no cost. There is no
license check, no entitlement server, no usage or savings cap, and no plan
tiers (no Free/Pro/Enterprise split).

`lc account login` and `lc account logout` still exist as an **optional**
convenience for linking a hosted account, but they gate nothing: they are never
required, never prompted, and can be omitted entirely. If you never run them,
LemonCrow behaves identically.

## Optional performance build

The engine ships as readable Python source. Compiling it with mypyc is an
**optional performance build** — never required to run LemonCrow, and it changes
no behavior and unlocks no features.

## Network behavior

Indexing, search, edits, and memory all stay on your machine. Anonymous remote
telemetry is on by default — turn it off with `lc telemetry remote off`,
`DO_NOT_TRACK=1`, or `LEMONCROW_TELEMETRY=off`. For the full details
of what does and does not leave your machine, see
[Privacy & network behavior](../setup/privacy.md).
