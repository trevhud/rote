# Example: BDR Outreach

This directory is the canonical real-world example for `rote` — a BDR
outreach skill compiled into a durable pipeline. It's the dogfooding
target and the regression suite for the compiler. The hand-drafted IR
is emitted to every runtime under `expected/runtimes/` (DBOS is the CLI
default; Temporal, Cloudflare, DBOS-TS, and Inngest are also emitted).

## Layout

```
examples/bdr-outreach/
├── README.md                         # this file
├── skill/                            # input: the source skill bundle
│   ├── SKILL.md
│   └── references/
│       ├── conference-enrichment.md
│       ├── email-templates.md
│       ├── hubspot-operations.md
│       ├── lead-generation.md
│       ├── quality-and-vetting.md
│       └── target-research.md
├── expected/                         # hand-drafted ground-truth artifacts
│   ├── pipeline.yaml                 # the IR
│   ├── types.py                      # shared Pydantic models
│   ├── extracted/                    # pure-function stubs
│   ├── signatures/                   # llm_judge stubs
│   └── runtimes/                     # emitted code, one dir per runtime
│       ├── dbos/                     #   Python (CLI default)
│       ├── temporal/                 #   Python
│       ├── cloudflare/               #   TypeScript
│       ├── dbos-ts/                  #   TypeScript
│       └── inngest/                  #   TypeScript
└── runs/                             # real compiler run snapshots
    └── <timestamp>/                  # one per committed run
```

The `skill/` directory is the **input** to `rote compile`. The
`expected/` directory is my hand-drafted IR and stubs — the ground
truth the IR schema was designed against, and the regression baseline
for every adapter. The `runs/` directory holds snapshots of
real compiler runs; `tests/test_compiler_bdr_regression.py` loads
the most recent one and asserts semantic invariants (node kinds,
mandatory flags, HITL gates, file references, codifiable percentage).

## About the company name

The skill uses **Acme** as a placeholder company name throughout. The
original source of this example was an internal skill at a real pharma
services company; all identifying references were replaced before
publishing. If you're adapting this example for your own use,
find-and-replace `Acme` with your organization's name and update the
"validated experience" sources referenced in
`references/email-templates.md` and `references/target-research.md`
to point at your actual internal databases.

## Why this example

The BDR skill is a good compilation target because it combines all five
node kinds in a single pipeline:

- **`pure_function`**: HubSpot batch upsert (fixed batch size of 100),
  pharma/non-pharma classifier (literal Python pasted in the prompt),
  openpyxl XLSX formatting, pre-enrollment report generation.
- **`llm_judge`**: contact vetting against the red-flags rubric (MSL,
  biomarker, translational, etc.), email personalization.
- **`agent_loop`**: target company research (Bright Data +
  ClinicalTrials + internal search), backfill query generation when
  initial searches fall short of quota.
- **`hitl_gate`**: Phase 3 contact review, Phase 7 manual HubSpot
  sequence enrollment, optionally Phase 2-alt excluded-list confirmation.
- **`external_call`**: ZoomInfo enrich batches (fixed size of 10),
  do-not-contact list lookup, recently-emailed check, active-sequence
  check.

Hitting every node kind means the BDR example also validates the IR:
if `rote` can compile this skill cleanly, the IR is probably
expressive enough for the long tail.

## Running the compiler on this example

From the repo root, with a valid agent driver available (see the top-
level README for installation):

```sh
rote compile examples/bdr-outreach/skill \
  --runtime temporal \
  --out /tmp/bdr-compiled
```

The run takes about 13 minutes wall-clock (mostly the agent loop; the
adapter step is instantaneous) and produces:

- `compiled/pipeline.yaml` — the IR
- `compiled/extracted/*.py` — deterministic function stubs
- `compiled/signatures/*.py` — typed LLM-judge signatures
- `compiled/evals/*.jsonl` — seed eval examples
- `compiled/compile-report.md` — human-readable summary
- `runtime/temporal/workflow.py` — the Temporal workflow class
- `runtime/temporal/activities.py` — one `@activity.defn` per node

To commit a run as a new regression snapshot:

```sh
cp -r /tmp/bdr-compiled/compiled examples/bdr-outreach/runs/$(date -u +%Y-%m-%d)
pytest tests/test_compiler_bdr_regression.py
```

If the regression test passes against the new snapshot, you're good to
commit.
