import { describe, it, expect } from "vitest";
import { computeDeterminism } from "../src/determinism.js";
import { canonicalize } from "../src/canonical.js";

// ───────── Reference implementation ─────────
//
// The original rote-cloud `computeDeterminism` (src/runs.ts), verbatim,
// including its own `flattenLeaves`. It takes canonicalized strings. The
// property test below asserts the package's `computeDeterminism(unknown[])`
// agrees with it field-for-field (minus `per_field`, which is new), so
// the hosted platform and OSS dashboards can never silently diverge.

function refFlattenLeaves(
  value: unknown,
  prefix = "",
  out: Record<string, string> = {},
): Record<string, string> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      refFlattenLeaves(v, prefix ? `${prefix}.${k}` : k, out);
    }
  } else {
    out[prefix || "_"] = JSON.stringify(value);
  }
  return out;
}

interface RefReport {
  n: number;
  distinct_outputs: number;
  exact_agreement: number;
  field_agreement: number;
}

function refComputeDeterminism(outputs: string[]): RefReport {
  const n = outputs.length;
  if (n === 0) return { n: 0, distinct_outputs: 0, exact_agreement: 0, field_agreement: 0 };

  const counts = new Map<string, number>();
  for (const o of outputs) counts.set(o, (counts.get(o) ?? 0) + 1);
  const modal = Math.max(...counts.values());

  let fieldAgreement = 1;
  try {
    const parsed = outputs.map((o) => refFlattenLeaves(JSON.parse(o)));
    const keys = new Set<string>();
    for (const p of parsed) for (const k of Object.keys(p)) keys.add(k);
    if (keys.size > 0) {
      let agree = 0;
      for (const k of keys) {
        const vals = new Set(parsed.map((p) => p[k]));
        if (vals.size === 1) agree++;
      }
      fieldAgreement = agree / keys.size;
    }
  } catch {
    fieldAgreement = counts.size === 1 ? 1 : 0;
  }

  return { n, distinct_outputs: counts.size, exact_agreement: modal / n, field_agreement: fieldAgreement };
}

// ───────── Hand-computed fixtures ─────────

describe("computeDeterminism — hand-computed", () => {
  it("reports perfect agreement for identical outputs", () => {
    const r = computeDeterminism([
      { a: 1, b: { c: 2 } },
      { a: 1, b: { c: 2 } },
      { a: 1, b: { c: 2 } },
    ]);
    expect(r).toEqual({
      n: 3,
      distinct_outputs: 1,
      exact_agreement: 1,
      field_agreement: 1,
      per_field: { a: 1, "b.c": 1 },
    });
  });

  it("drops field_agreement by one leaf when a single nested leaf flips", () => {
    const r = computeDeterminism([
      { a: 1, b: { c: 2, d: 3 } },
      { a: 1, b: { c: 2, d: 3 } },
      { a: 1, b: { c: 2, d: 9 } },
    ]);
    expect(r.n).toBe(3);
    expect(r.distinct_outputs).toBe(2);
    expect(r.exact_agreement).toBeCloseTo(2 / 3);
    // 2 of 3 leaves (a, b.c) identical; b.d diverges.
    expect(r.field_agreement).toBeCloseTo(2 / 3);
    expect(r.per_field).toEqual({ a: 1, "b.c": 1, "b.d": 2 / 3 });
  });

  it("counts a missing leaf as its own value", () => {
    const r = computeDeterminism([{ a: 1 }, { a: 1, b: 2 }]);
    expect(r.distinct_outputs).toBe(2);
    expect(r.exact_agreement).toBeCloseTo(1 / 2);
    // a agrees (both present, equal); b does not (present in one only).
    expect(r.field_agreement).toBeCloseTo(1 / 2);
    expect(r.per_field).toEqual({ a: 1, b: 1 / 2 });
  });

  it("handles primitive outputs via the '_' leaf", () => {
    const r = computeDeterminism([5, 5, 6]);
    expect(r).toEqual({
      n: 3,
      distinct_outputs: 2,
      exact_agreement: 2 / 3,
      field_agreement: 0,
      per_field: { _: 2 / 3 },
    });
  });

  it("returns zeros and an empty per_field for no runs", () => {
    expect(computeDeterminism([])).toEqual({
      n: 0,
      distinct_outputs: 0,
      exact_agreement: 0,
      field_agreement: 0,
      per_field: {},
    });
  });

  it("is invariant to key order in the inputs (canonicalization)", () => {
    const r = computeDeterminism([
      { a: 1, b: 2 },
      { b: 2, a: 1 },
    ]);
    expect(r.distinct_outputs).toBe(1);
    expect(r.exact_agreement).toBe(1);
    expect(r.field_agreement).toBe(1);
  });

  it("falls back to whole-output identity for a non-JSON output", () => {
    // canonicalize(undefined) === undefined → JSON.parse throws → fallback.
    expect(computeDeterminism([undefined]).field_agreement).toBe(1);
    expect(computeDeterminism([undefined, undefined]).field_agreement).toBe(1);
  });
});

// ───────── Property test vs. the original cloud impl ─────────

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomOutput(rand: () => number): unknown {
  const leafDomain: unknown[] = [0, 1, 2, "x", "y", null, true, [1, 2], [1, 3]];
  const pick = <T,>(xs: T[]): T => xs[Math.floor(rand() * xs.length)]!;
  const obj: Record<string, unknown> = {};
  // Small, overlapping key space so agreements and divergences both occur.
  if (rand() < 0.85) obj["a"] = pick(leafDomain);
  if (rand() < 0.7) obj["b"] = pick(leafDomain);
  if (rand() < 0.6) obj["nested"] = { c: pick(leafDomain), d: pick(leafDomain) };
  // Occasionally emit a bare primitive instead of an object.
  return rand() < 0.15 ? pick(leafDomain) : obj;
}

describe("computeDeterminism — parity with the original rote-cloud impl", () => {
  it("matches n, distinct_outputs, exact_agreement, field_agreement on random fixtures", () => {
    const rand = mulberry32(0x9e3779b9);
    for (let fixture = 0; fixture < 20; fixture++) {
      const k = 1 + Math.floor(rand() * 8);
      const outputs = Array.from({ length: k }, () => randomOutput(rand));
      const mine = computeDeterminism(outputs);
      const ref = refComputeDeterminism(outputs.map(canonicalize));
      expect(mine.n).toBe(ref.n);
      expect(mine.distinct_outputs).toBe(ref.distinct_outputs);
      expect(mine.exact_agreement).toBeCloseTo(ref.exact_agreement, 12);
      expect(mine.field_agreement).toBeCloseTo(ref.field_agreement, 12);
    }
  });
});
