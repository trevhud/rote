# Example: the compiler compiling itself

Source: [`skills/rote-compile`](../../skills/rote-compile), the actual skill
that drives rote's compiler. The September 4 run used the local Claude
subscription driver with `claude-sonnet-4-6` and requested DBOS output.

The agent produced a valid 12-node IR: four pure functions, two external
calls, three typed judges, and three agent loops. All six deterministic
modules contain implementations. Runtime emission failed, however, and
the offline audit found additional integration defects. This snapshot is
evidence of what the compiler produced, not a working recursive compiler.

## Preserved run

[`runs/2026-09-04-selfcompile/`](runs/2026-09-04-selfcompile/) contains the
agent-authored `compiled/` artifacts, the partial `runtime/dbos/` directory,
and the progress stream. Machine-specific home paths are normalized for
publication; generated behavior and recorded usage remain unchanged.
The source hashes are in
[`provenance.json`](runs/2026-09-04-selfcompile/compiled/provenance.json).
The generated report's claims are retained; the audit below is the verified
assessment. A byte-for-byte local copy is kept in the ignored
`.worktrees/selfcompile-evidence/original/` directory.

The run took about 22 minutes between its first and last progress events.
Claude reported a **$3.58 cost estimate**, not a billing record.

The run exposed a telemetry defect: rote counted repeated assistant blocks
multiple times and treated provisional output counts as final. The original
progress counters are therefore unreliable. Grouping finalized transcript
records by response ID gives 30 responses, 32 uncached input tokens,
226,483 cache-write tokens, 3,270,597 cache-read tokens, and 71,693 output
tokens. These cover the recorded main-agent responses; auxiliary-model
usage was not recoverable after the emission failure. See
[`measurements.json`](measurements.json) for scope and verification details.

## Offline audit

The following checks used local files and emitted code, with no inference:

| Boundary | Observed result |
| --- | --- |
| Skill reader | Read the real core skill and all six reference files; missing-file check passed. |
| Entry routing | Mode and phase-selection checks passed. |
| IR validation | All 12 nodes loaded successfully. |
| Signature emission | Failed: `StepDescription` has conflicting definitions in one judge's input and output schemas. |
| Report invocation | IR omits required `emitted_files`; `signature_designs` binds an object where the renderer expects a list. |
| Eval sidecar | Generated `notes` fields violate the strict step-estimate schema. |
| Adapter invocation | Omitting `out_dir` fails; successful emission of a known-good pipeline still returns an empty file list because the helper parses the wrong CLI output. |
| Agent-loop outputs | Runtime returns a text-bearing result envelope; the generated bindings assume structured fields directly. |

The three agent loops also require a `filesystem` MCP server with the
declared tools. That server was not provisioned, and the complete generated
workflow was not executed. Original `signatures/*.py` files are legacy
design sketches with unimplemented dispatch; runtime emission uses their
structured `signature_spec` definitions instead.

## Reproduce emission without inference

From the repository root:

```sh
rote emit examples/rote-compile/runs/2026-09-04-selfcompile/compiled/pipeline.yaml \
  --runtime dbos --out /tmp/rote-selfcompile-emit
```

This reproduces the schema collision. To repeat the paid compilation:

```sh
rote compile skills/rote-compile --local --no-deploy --runtime dbos \
  --agent claude --model claude-sonnet-4-6 --out /tmp/rote-selfcompile-rerun
```
