# Security Policy

## Reporting a vulnerability

Please report security issues **privately** — do not open a public
issue for anything exploitable.

- Preferred: GitHub's [private vulnerability reporting](https://github.com/trevhud/rote/security/advisories/new)
  (Security tab → "Report a vulnerability").
- Alternatively, email **trevhud@gmail.com** with `rote security` in
  the subject.

Please include a description, affected version, and a minimal
reproduction if you have one. You'll get an acknowledgement within a
few days; fixes for confirmed issues are prioritized over other work.

## Supported versions

`rote` is pre-1.0. Security fixes are released against the latest
published version on PyPI ([`rote-cli`](https://pypi.org/project/rote-cli/))
only. Pin a version you've reviewed if that matters to you.

## Threat model — read this before graduating third-party skills

`rote` turns a `SKILL.md` into a `pipeline.yaml` (the IR) and then
**emits runtime source code** from that IR. Two properties follow:

1. **A `pipeline.yaml` is untrusted input.** It is LLM-generated from a
   possibly third-party skill, or shared directly. IR string fields
   that reach emitted code or filenames (`Node.id`, `signal`, and the
   `impl` / `signature` references) are charset-constrained by
   validators at the IR boundary in [`src/rote/ir.py`](src/rote/ir.py);
   prose fields are escaped at emission. If you find a way to make a
   crafted `pipeline.yaml` inject executable code, break out of a
   docstring/comment, or write outside the output directory, that's a
   vulnerability — report it.

2. **The graduator runs an LLM agent with tools.** `rote graduate`
   spawns an agent (Claude Code, Codex, or the Anthropic SDK) that
   reads the source skill and writes files. Treat a skill you did not
   author as untrusted content in an agent's context. The emitted
   `extracted/*` modules are **stubs that raise `NotImplementedError`**
   — you fill them in — so review emitted code before running it, as
   you would any generated scaffold.

Emitted code is guaranteed **never to import or reference MCP**
(enforced by test); the graduator's job is to replace MCP tool calls
with direct vendor API calls.

## Handling of credentials

`rote` never bakes secrets into emitted code or the IR. Runtime
inference is configured via environment variables (`ANTHROPIC_API_KEY`,
`ROTE_MODEL_<ID>`, `ROTE_BASE_URL_<ID>`, etc.). `scripts/sanity-check.sh`
scans the repo for leaked keys/tokens and internal identifiers and is
part of the release gate; run it before any push.
