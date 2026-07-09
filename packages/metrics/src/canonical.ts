/**
 * Canonical serialization + leaf flattening.
 *
 * These two functions are the shared substrate under the determinism
 * metric: `canonicalize` gives a stable, key-sorted JSON string so two
 * structurally-equal outputs compare equal, and `flattenLeaves` explodes
 * a nested object into `{ "a.b.c": "<json>" }` leaf entries so field
 * agreement is measured over individual leaves rather than opaque
 * subtrees (one flipped leaf reads as e.g. 5/6, not 0).
 *
 * Ported verbatim from rote-cloud's `src/runs.ts` so the hosted platform
 * and OSS self-hosted dashboards compute byte-identical results.
 */

/** Deterministic JSON string with lexicographically sorted keys. */
export function canonicalize(value: unknown): string {
  const sort = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(sort);
    if (v && typeof v === "object") {
      return Object.fromEntries(
        Object.keys(v as Record<string, unknown>)
          .sort()
          .map((k) => [k, sort((v as Record<string, unknown>)[k])]),
      );
    }
    return v;
  };
  return JSON.stringify(sort(value));
}

/**
 * Flatten a nested object to `{ "a.b.c": "<json value>" }` leaf entries,
 * so field agreement is measured over individual leaves, not opaque
 * subtrees. Arrays are treated as leaves (stringified whole), matching
 * cloud semantics: element-wise array diffing is deliberately out of
 * scope for the determinism metric.
 */
export function flattenLeaves(
  value: unknown,
  prefix = "",
  out: Record<string, string> = {},
): Record<string, string> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      flattenLeaves(v, prefix ? `${prefix}.${k}` : k, out);
    }
  } else {
    out[prefix || "_"] = JSON.stringify(value);
  }
  return out;
}
