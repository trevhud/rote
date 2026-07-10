import { describe, it, expect } from "vitest";
import { percentile, summarizeRuns } from "../src/speed-cost.js";
import type { RunRecord } from "../src/run-record.js";

describe("percentile", () => {
  it("returns null for an empty array", () => {
    expect(percentile([], 50)).toBeNull();
  });

  it("returns the single value for n=1 at any p", () => {
    expect(percentile([42], 0)).toBe(42);
    expect(percentile([42], 50)).toBe(42);
    expect(percentile([42], 100)).toBe(42);
  });

  it("interpolates for n=2", () => {
    expect(percentile([10, 20], 50)).toBe(15);
    expect(percentile([10, 20], 0)).toBe(10);
    expect(percentile([10, 20], 100)).toBe(20);
  });

  it("interpolates linearly across ranks", () => {
    const xs = [10, 20, 30, 40];
    expect(percentile(xs, 50)).toBe(25);
    expect(percentile(xs, 95)).toBeCloseTo(38.5);
    expect(percentile(xs, 0)).toBe(10);
    expect(percentile(xs, 100)).toBe(40);
  });
});

function run(overrides: Partial<RunRecord>): RunRecord {
  return {
    status: "complete",
    output: null,
    wall_ms: null,
    input_tokens: 0,
    output_tokens: 0,
    cost_usd: 0,
    ...overrides,
  };
}

describe("summarizeRuns", () => {
  it("returns nulls (but zero total cost) for no runs", () => {
    expect(summarizeRuns([])).toEqual({
      n: 0,
      complete: 0,
      errored: 0,
      p50_wall_ms: null,
      p95_wall_ms: null,
      avg_wall_ms: null,
      avg_cost_usd: null,
      total_cost_usd: 0,
      avg_input_tokens: null,
      avg_output_tokens: null,
    });
  });

  it("aggregates over complete runs but totals cost across all runs", () => {
    const records: RunRecord[] = [
      run({ status: "complete", wall_ms: 100, cost_usd: 0.01, input_tokens: 10, output_tokens: 5 }),
      run({ status: "complete", wall_ms: 300, cost_usd: 0.03, input_tokens: 30, output_tokens: 15 }),
      run({ status: "errored", wall_ms: 50, cost_usd: 0.02, input_tokens: 5, output_tokens: 0 }),
      run({ status: "running", wall_ms: null, cost_usd: 0 }),
    ];
    const s = summarizeRuns(records);
    expect(s.n).toBe(4);
    expect(s.complete).toBe(2);
    expect(s.errored).toBe(1);
    expect(s.p50_wall_ms).toBe(200);
    expect(s.p95_wall_ms).toBeCloseTo(290);
    expect(s.avg_wall_ms).toBe(200);
    expect(s.avg_cost_usd).toBeCloseTo(0.02);
    // Errored + running spend is included in the total, excluded from averages.
    expect(s.total_cost_usd).toBeCloseTo(0.06);
    expect(s.avg_input_tokens).toBe(20);
    expect(s.avg_output_tokens).toBe(10);
  });

  it("treats a complete run with null wall_ms as zero (matching pipelineRunSummary)", () => {
    const records: RunRecord[] = [
      run({ status: "complete", wall_ms: null, cost_usd: 0.01 }),
      run({ status: "complete", wall_ms: 100, cost_usd: 0.01 }),
    ];
    const s = summarizeRuns(records);
    expect(s.avg_wall_ms).toBe(50);
    expect(s.p50_wall_ms).toBe(50);
  });
});
