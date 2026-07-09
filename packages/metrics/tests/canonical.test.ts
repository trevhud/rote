import { describe, it, expect } from "vitest";
import { canonicalize, flattenLeaves } from "../src/canonical.js";

describe("canonicalize", () => {
  it("sorts object keys lexicographically", () => {
    expect(canonicalize({ b: 1, a: 2 })).toBe('{"a":2,"b":1}');
  });

  it("sorts keys recursively in nested objects", () => {
    expect(canonicalize({ b: { d: 1, c: 2 }, a: 3 })).toBe('{"a":3,"b":{"c":2,"d":1}}');
  });

  it("preserves array element order but sorts keys inside elements", () => {
    expect(canonicalize([3, 1, 2])).toBe("[3,1,2]");
    expect(canonicalize([{ b: 1, a: 2 }])).toBe('[{"a":2,"b":1}]');
  });

  it("passes primitives through as JSON", () => {
    expect(canonicalize(5)).toBe("5");
    expect(canonicalize("x")).toBe('"x"');
    expect(canonicalize(true)).toBe("true");
    expect(canonicalize(null)).toBe("null");
  });

  it("returns undefined for a bare undefined (matching JSON.stringify)", () => {
    expect(canonicalize(undefined)).toBeUndefined();
  });

  it("drops object keys whose value is undefined", () => {
    expect(canonicalize({ a: undefined, b: 1 })).toBe('{"b":1}');
  });

  it("is idempotent when re-canonicalizing the parsed result", () => {
    const once = canonicalize({ z: [{ b: 1, a: 2 }], m: 3 });
    expect(canonicalize(JSON.parse(once))).toBe(once);
  });
});

describe("flattenLeaves", () => {
  it("flattens nested objects to dotted leaf paths", () => {
    expect(flattenLeaves({ a: 1, b: { c: 2, d: { e: 3 } } })).toEqual({
      a: "1",
      "b.c": "2",
      "b.d.e": "3",
    });
  });

  it("treats arrays as whole leaves", () => {
    expect(flattenLeaves({ a: [1, 2], b: 3 })).toEqual({ a: "[1,2]", b: "3" });
  });

  it("maps a top-level primitive to the '_' key", () => {
    expect(flattenLeaves(5)).toEqual({ _: "5" });
    expect(flattenLeaves(null)).toEqual({ _: "null" });
    expect(flattenLeaves("x")).toEqual({ _: '"x"' });
  });

  it("returns no leaves for an empty object", () => {
    expect(flattenLeaves({})).toEqual({});
  });
});
