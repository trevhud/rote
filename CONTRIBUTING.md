# Contributing to rote

Thanks for your interest in contributing. `rote` is still pre-1.0, so
the most valuable contributions right now are the kinds that
stress-test the design against real skills, not polish on the existing
code.

## Most useful contributions, in order

1. **Run `rote compile` on a real skill of your own and report what
   happens.** The rubric was designed against one skill (the bundled
   BDR example). Every additional skill that passes through it is a
   stress test for both the classification rubric and the IR schema.
   If the compiler makes a weird call, open an issue with (a) the
   source skill's `SKILL.md`, (b) the produced `pipeline.yaml`, and
   (c) what you expected instead. Snapshots are cheap and the
   fastest way to improve the rubric.
2. **Add a runtime adapter.** Six adapters ship today — `dbos`
   (default), `temporal`, and `python` emit Python;
   [`cloudflare`](src/rote/adapters/cloudflare.py), `dbos-ts`, and
   `inngest` emit TypeScript. Restate and Hatchet are good next
   targets. Each new runtime that consumes the IR unchanged is more
   evidence the IR is genuinely runtime-agnostic; one that *can't* be
   expressed cleanly is a real signal to revisit the IR shape.
3. **Add a compiler driver.** The Protocol lives in
   [`src/rote/compiler/drivers/__init__.py`](src/rote/compiler/drivers/__init__.py).
   `ClaudeDriver` (subprocess) and `AnthropicApiDriver` (in-process
   SDK) are both implemented. `CodexDriver` is a stub that just
   needs the subprocess invocation filled in. Aider, Gemini CLI,
   and Cursor Agent are reasonable third-party additions.
4. **Improve the rubric.** Files under
   [`skills/rote-compile/references/`](skills/rote-compile/references/)
   are what the compiler agent actually reads. Small prose
   improvements often produce measurable differences in output
   quality. Every change is diffable and can be validated by
   re-running the compiler against the BDR example.

## Development setup

```sh
git clone https://github.com/trevhud/rote.git
cd rote
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs `rote` in editable mode plus the dev dependencies
(`pytest`, `pytest-asyncio`, `ruff`, `mypy`, `types-pyyaml`, and the
runtime extras: `temporalio`, `anthropic`, `dbos`, `fastmcp`,
`httpx`). You now have a `rote` command on `PATH` and can run tests.

## Running tests

```sh
pytest tests/                                         # full suite
pytest tests/test_ir.py                               # just the IR
pytest tests/test_temporal_adapter.py                 # just the adapter
pytest tests/test_compiler_bdr_regression.py         # BDR regression
```

The fast suite (what `pytest tests/` runs by default) finishes in
~15 seconds — no real API calls, subprocesses, or LLM invocations.
The toolchain-dependent integration tests are `slow`-marked and opt-in
via `pytest tests/ -m slow` (they need Node/npm/tsc, and DBOS-TS needs
Docker). The BDR regression tests skip gracefully if no snapshot
exists under `examples/bdr-outreach/runs/`.

## Running the real compiler

```sh
rote compile examples/bdr-outreach/skill \
  --runtime temporal \
  --out /tmp/bdr-compiled
```

This spawns an agent (auto-detected: `claude` CLI → `codex` CLI →
Anthropic SDK). Expect ~13 minutes wall clock on the BDR example
with the default `claude` driver and Sonnet 4.6.

To commit a fresh snapshot as the new regression baseline:

```sh
cp -r /tmp/bdr-compiled/compiled examples/bdr-outreach/runs/$(date -u +%Y-%m-%d)
pytest tests/test_compiler_bdr_regression.py
```

If the regression test passes against the new snapshot, commit it.

## Sanity-checking before a push

Before any `git push` to a public remote, run:

```sh
./scripts/sanity-check.sh
```

The script greps the repo for:

- internal identifiers from the original source of the bundled BDR
  example (company name, internal wiki URLs, internal script paths,
  customer friction claims)
- real-looking API keys / OAuth tokens / webhook secrets
- hard-coded absolute home paths

It exits non-zero if anything is found. The exact patterns are in
[`scripts/sanity-check.sh`](scripts/sanity-check.sh) — review them
when adding new example skills or test fixtures. If a contribution
legitimately needs to use one of the flagged strings (e.g. in a
regression test's documentation), adjust the exclusion list rather
than disabling the check.

## Adding a new runtime adapter

The Temporal adapter is the reference implementation. A new adapter
needs:

1. A module under `src/rote/adapters/<runtime>.py` that exposes an
   `<Runtime>Adapter` class with an `emit(pipeline, output_dir)`
   method matching the `Adapter` Protocol in
   [`src/rote/adapters/__init__.py`](src/rote/adapters/__init__.py).
   The method must write `workflow.py` (or the runtime's equivalent)
   + `activities.py` + `__init__.py` into `output_dir` and return a
   `dict[str, Path]` mapping labels to the written files.
2. A factory function in `src/rote/adapters/__init__.py` that
   lazy-imports the adapter module and appends a new entry to
   `ADAPTERS`. Lazy imports matter — users who don't use your
   runtime shouldn't pay for its import cost on every CLI invocation.
3. Tests under `tests/test_<runtime>_adapter.py` covering:
   - Emission produces files that parse as valid Python.
   - Every non-HITL node becomes an activity (or the runtime's
     equivalent).
   - Every `hitl_gate` becomes a signal handler.
   - The emitted code never references MCP (`test_emitted_activities_never_reference_mcp`
     is a template you can copy).
   - End-to-end: the emitted workflow runs against mocked activities
     in the runtime's testing environment (see
     [`tests/test_temporal_e2e.py`](tests/test_temporal_e2e.py) for
     the pattern).
4. A declaration in [`src/rote/cli.py`](src/rote/cli.py) adding the
   new runtime to the `--runtime` choices on the `compile` and
   `emit` subparsers (the list is derived from `ADAPTERS` so it
   updates automatically once the factory is registered).

The Temporal adapter is a few hundred lines, most of it string
templates; the TypeScript adapters share emit machinery in
`_ts_common.py` and the Python ones in `_py_common.py`, so a new
adapter in an existing language is smaller. Route every emitted-file
write through the `EmitWriter` in `_common.py` rather than bare
`write_text` — it powers hash-guarded re-emission.

## Adding a new compiler driver

Drivers are the thinnest layer. A new driver needs:

1. A module under `src/rote/compiler/drivers/<name>.py` exposing a
   class that implements the `CompilerDriver` Protocol:
   `name: str`, `is_available() -> (bool, str)`, `async run(
   skill_dir, compiler_skill_dir, work_dir) -> DriverResult`.
2. A factory + registry entry in
   [`src/rote/compiler/drivers/__init__.py`](src/rote/compiler/drivers/__init__.py).
3. A CLI `--agent` entry in
   [`src/rote/cli.py`](src/rote/cli.py).
4. Tests under `tests/test_<name>_driver.py`. For subprocess drivers,
   mock `asyncio.create_subprocess_exec` and have the mock simulate
   side effects (writing `pipeline.yaml` to `work_dir`) — see
   [`tests/test_claude_driver.py`](tests/test_claude_driver.py) for
   the pattern. For in-process drivers, mock the SDK client — see
   [`tests/test_anthropic_driver.py`](tests/test_anthropic_driver.py).

**Critical:** the driver's only contract is that `work_dir/pipeline.yaml`
exists after `run()` returns (or a `DriverError` is raised). Do not
invent new return-shape fields or new side-channel communication
mechanisms — the filesystem is the contract.

## Code style

- **Ruff** is configured in `pyproject.toml`. Run `ruff check .` and
  `ruff format .` before committing.
- **mypy strict** is enabled. New code should be fully typed.
- **Docstrings on public APIs** (anything importable from
  `rote.<module>`). Module-level docstrings explain the *why*; function
  docstrings explain the *what* and the *contract*.

## PRs

- Describe what the change does and why in 2-3 sentences.
- Include a note about how you verified it — usually "pytest tests/"
  plus something specific to the change (e.g. "ran `rote compile`
  end-to-end on BDR").
- New features land with matching tests.
- If the change affects the compiler rubric or the IR schema, run
  the BDR regression test and commit an updated snapshot if needed.
  Explain in the commit message whether the snapshot change is a
  *correction* (the previous snapshot was wrong) or a *drift* (the
  rubric changed; the new snapshot is the new ground truth).

## What the test suite looks like on main

The suite is organized by layer. A representative slice:

```
tests/
├── test_ir.py                              IR Pydantic models + validators
├── test_<runtime>_adapter.py               IR → emitted code (unit, per runtime)
├── test_<runtime>_e2e.py                   emitted code on a real runtime (slow)
├── test_compiler_drivers.py               Driver Protocol + registry
├── test_anthropic_driver.py                API driver tool dispatch
├── test_claude_driver.py                   Claude subprocess driver
├── test_compiler.py                       Orchestrator
├── test_compile_update.py                 Incremental re-compilation
├── test_cli.py / test_eval_*.py            CLI subcommands + eval harness
├── test_serve_*.py                         rote serve (MCP trigger)
└── test_compiler_bdr_regression.py        Real compiler output vs snapshot
```

Adapters (`temporal`, `cloudflare`, `dbos`, `dbos-ts`, `inngest`,
`python`) each have a `_adapter.py` unit test and an `_e2e.py`
integration test. A healthy PR leaves coverage broader, not narrower.
