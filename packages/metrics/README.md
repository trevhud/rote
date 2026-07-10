# rote-metrics

The shared implementation of rote's empirical metrics — **determinism**,
**speed**, and **cost** — plus the run-record and scorecard type
contracts.

rote-cloud (the hosted platform) and rote's OSS self-hosted dashboards
both depend on this package, so a number shown in one is computed by the
exact same code as the number shown in the other. There is one algorithm,
in one place.

- **Zero runtime dependencies.** TypeScript strict. ESM + CJS + `.d.ts`.
- Node >= 20.

## Install

```bash
pnpm add rote-metrics
```

rote-cloud consumes it via a workspace `link:` during development; OSS
dashboards depend on the published package.

## What's in it

| Export | What it is |
| --- | --- |
| `canonicalize(value)` | Stable JSON string with lexicographically sorted keys. |
| `flattenLeaves(value)` | Flatten a nested object to `{ "a.b.c": "<json>" }` leaf entries. |
| `computeDeterminism(outputs)` | The determinism report over K repeated run outputs. |
| `percentile(sorted, p)` | Linear-interpolation percentile (PERCENTILE.INC). |
| `summarizeRuns(records)` | Speed + cost + token aggregation over run records. |
| `RunRecord`, `RunStatus` | The run-record contract (matches rote-cloud's `runs` table). |
| `Scorecard` (+ member types) | The static before/after scorecard JSON shape. |

## The determinism algorithm

Determinism is agreement across **K repeated runs** of the same compiled
pipeline on the same input. `computeDeterminism` reports it three ways.

1. **Canonicalize** every output to a key-sorted JSON string, so
   structurally-equal outputs compare equal regardless of key order.
2. **Exact agreement** — the fraction of runs whose whole canonical
   output equals the single most common (modal) output. The strictest
   view; one differing byte drops a run.
3. **Leaf flattening** — parse each canonical output and flatten it to
   dotted leaf paths (`flattenLeaves`), so a nested object is judged leaf
   by leaf rather than all-or-nothing. Arrays are treated as whole leaves.
   From the flattened leaves:
   - **`field_agreement`** — the fraction of leaf paths whose value is
     identical across **all** runs (a missing leaf counts as its own
     value). One flipped leaf out of six reads as `5/6`, not `0`.
   - **`per_field`** — for each leaf path, the fraction of runs that
     agree on that leaf's **modal** value. This localizes non-determinism
     to specific fields.

`field_agreement` is rote-cloud's original semantics; `per_field`
generalizes rote OSS's per-top-level-field `compute_agreement` down to
flattened leaves. The package's test suite pins `computeDeterminism`
against the original rote-cloud implementation on random structured
fixtures, so the two can never silently diverge.

### Example

```ts
import { computeDeterminism } from "rote-metrics";

computeDeterminism([
  { intent: "refund", priority: "high" },
  { intent: "refund", priority: "high" },
  { intent: "refund", priority: "low" },
]);
// {
//   n: 3,
//   distinct_outputs: 2,
//   exact_agreement: 0.666…,   // 2 of 3 whole outputs match
//   field_agreement: 0.5,      // `intent` agrees, `priority` doesn't
//   per_field: { intent: 1, priority: 0.666… },
// }
```

## Speed & cost

`summarizeRuns` mirrors rote-cloud's `pipelineRunSummary`: wall-clock,
cost, and token **averages** (and the `p50`/`p95` wall-clock percentiles)
are taken over `complete` runs only, while `total_cost_usd` sums **every**
run — spend is spend whether or not the run succeeded.

## Scorecard types

`Scorecard` mirrors `rote.eval.scorecard.Scorecard.to_dict()` in the OSS
Python exactly: the before/after estimate blocks, per-node detail,
per-model cost rows, and the estimator priors. rote-cloud stores that JSON
verbatim and serves it unchanged, so this type is the contract both sides
validate against. Types only — scorecards are produced by the Python
estimator, never in TypeScript.

## Develop

```bash
pnpm install
pnpm typecheck
pnpm build
pnpm test
```
