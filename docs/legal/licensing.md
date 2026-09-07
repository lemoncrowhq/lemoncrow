# Licensing

LemonCrow is open source under the **Apache License, Version 2.0**, in its
entirety. The runtime, the CLI, the MCP server, the SDK, host integrations, and
the engine (the `lemoncrow.pro` package) are all published as readable source
under the same license. There is no open-core split and no proprietary
component.

- Full license text: [`/LICENSE`](../../LICENSE) and [`/LICENSE-APACHE`](../../LICENSE-APACHE)
- Attribution and third-party notices: [`/NOTICE`](../../NOTICE)

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
