/**
 * Speed + cost aggregation over a set of run records.
 *
 * Wall-clock, cost, and token averages are taken over `complete` runs
 * only (an errored run's partial timings would poison the numbers),
 * mirroring rote-cloud's `pipelineRunSummary`. `total_cost_usd` is the
 * exception: it sums every run, because spend is spend whether or not
 * the run succeeded.
 */
import type { RunRecord } from "./run-record.js";

/**
 * Linear-interpolation percentile (Excel PERCENTILE.INC / NumPy default).
 *
 * @param sortedAscending values pre-sorted ascending; not re-sorted here.
 * @param p percentile in [0, 100].
 * @returns the interpolated value, or null for an empty input.
 */
export function percentile(sortedAscending: number[], p: number): number | null {
  const len = sortedAscending.length;
  if (len === 0) return null;
  if (len === 1) return sortedAscending[0]!;
  const rank = (p / 100) * (len - 1);
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  const loVal = sortedAscending[lo]!;
  if (lo === hi) return loVal;
  return loVal + (sortedAscending[hi]! - loVal) * (rank - lo);
}

export interface RunSummary {
  /** Total runs considered. */
  n: number;
  /** Count with status `complete`. */
  complete: number;
  /** Count with status `errored`. */
  errored: number;
  /** Wall-clock percentiles over complete runs (null when none complete). */
  p50_wall_ms: number | null;
  p95_wall_ms: number | null;
  /** Averages over complete runs (null when none complete). */
  avg_wall_ms: number | null;
  avg_cost_usd: number | null;
  /** Total spend across ALL runs, settled or not. */
  total_cost_usd: number;
  avg_input_tokens: number | null;
  avg_output_tokens: number | null;
}

/** Aggregate speed/cost/token stats from a set of run records. */
export function summarizeRuns(records: RunRecord[]): RunSummary {
  const n = records.length;
  const complete = records.filter((r) => r.status === "complete");
  const errored = records.reduce((a, r) => a + (r.status === "errored" ? 1 : 0), 0);
  const c = complete.length;

  const walls = complete.map((r) => r.wall_ms ?? 0).sort((a, b) => a - b);
  const avg = (nums: number[]): number | null =>
    nums.length ? nums.reduce((a, x) => a + x, 0) / nums.length : null;

  return {
    n,
    complete: c,
    errored,
    p50_wall_ms: percentile(walls, 50),
    p95_wall_ms: percentile(walls, 95),
    avg_wall_ms: c ? complete.reduce((a, r) => a + (r.wall_ms ?? 0), 0) / c : null,
    avg_cost_usd: avg(complete.map((r) => r.cost_usd)),
    total_cost_usd: records.reduce((a, r) => a + r.cost_usd, 0),
    avg_input_tokens: avg(complete.map((r) => r.input_tokens)),
    avg_output_tokens: avg(complete.map((r) => r.output_tokens)),
  };
}
