# Contributing

Thanks for helping with LemonCrow.

LemonCrow is early, experimental, and especially open to contributors who want to help shape the core abstractions before they harden.

Start with the setup and architecture docs linked from [README.md](README.md). Good areas to help:

- MCP integrations
- coding-agent workflows
- Python backend/runtime work
- memory and retrieval
- evals and benchmarks
- docs, tutorials, and examples
- security review

## License and contribution terms

LemonCrow is licensed by directory: [Apache-2.0](LICENSE) for the runtime, CLI,
MCP server, SDK, and integrations, and the [PolyForm Noncommercial License
1.0.0](src/lemoncrow/pro/LICENSE) for the engine (`src/lemoncrow/pro/`).
Contributions are accepted under the license that already covers the files you
touch.

Every contributor must agree to our [Contributor License Agreement (CLA)](CLA.md)
before a pull request can be merged. The CLA lets the Maintainer relicense and
sublicense the project while you keep copyright in your contribution.

You grant these rights automatically when you open a pull request, so there is
nothing to do in advance. The first time you contribute, a bot also asks you to
confirm the CLA with a one-line comment — a one-time step that records your
agreement (you are never asked again on later PRs). Bots and the Maintainer are
allowlisted.

## Scope boundaries

LemonCrow is governance for AI-assisted coding. It does not own the agent loop, replace CI, replace linting, become a Slack/GitHub bot, auto-apply lesson candidates, or add a web editor for Playbooks. Before proposing adoption tooling, keep this scope in mind (the "What NOT to build" list above).

Before opening a pull request, please run:

```bash
make pre-commit
```

To enable the repository-managed pre-push hook:

```bash
git config core.hooksPath .githooks
```

That hook runs `make install` before push-time validation.

Small issues, design critiques, bug reports, docs fixes, and "this abstraction feels wrong" feedback are all welcome.
