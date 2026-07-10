import { describe, it, expect } from "vitest";
import fixture from "./fixtures/scorecard.json";
import type { Scorecard } from "../src/index.js";

// Compile-time contract: the real scorecard fixture is assignable to the
// Scorecard type. (JSON imports widen string literals to `string`, so the
// tier/kind unions are checked structurally rather than by literal.)
const scorecard = fixture as Scorecard;

describe("Scorecard type ↔ demo fixture", () => {
  it("spot-checks the top-level shape at runtime", () => {
    expect(scorecard.pipeline).toBe("ticket-triage");
    expect(scorecard.generated_at).toBe("2026-07-04T03:53:01Z");
    // roteness: structural determinism share, 0.0–1.0.
    expect(scorecard.roteness).toBe(0.5);
    expect(scorecard.roteness).toBeGreaterThanOrEqual(0);
    expect(scorecard.roteness).toBeLessThanOrEqual(1);
  });

  it("carries a well-formed before/after estimate", () => {
    expect(scorecard.before).not.toBeNull();
    expect(scorecard.before!.turns).toEqual({ low: 6, high: 15 });
    expect(scorecard.after.critical_path_seconds).toEqual({ low: 6.41, high: 6.41 });
    expect(scorecard.after.hitl_gates).toEqual([]);
    expect(scorecard.after.nodes).toHaveLength(4);
    const kinds = scorecard.after.nodes.map((n) => n.kind);
    expect(kinds).toContain("llm_judge");
    expect(kinds).toContain("pure_function");
  });

  it("prices every cost row against both estimates", () => {
    expect(scorecard.costs.map((c) => c.model)).toEqual([
      "claude-fable-5",
      "claude-sonnet-5",
      "claude-haiku-4-5",
    ]);
    for (const row of scorecard.costs) {
      expect(row.after_usd).toHaveProperty("low");
      expect(row.after_usd).toHaveProperty("high");
      expect(row.before_usd).not.toBeNull();
    }
    expect(scorecard.costs[0]!.tier).toBe("flagship");
  });

  it("exposes the estimator priors (older fixtures omit the newer fields)", () => {
    expect(scorecard.priors.seconds_per_turn).toBe(13);
    expect(scorecard.priors.chars_per_token).toBe(3.8);
    // The demo fixture predates these two Priors fields; the type marks
    // them optional so both shapes validate.
    expect(scorecard.priors.tokens_per_external_call_result).toBeUndefined();
    expect(scorecard.priors.payload_tokens_per_tool).toBeUndefined();
  });
});

// Independent compile-time check of the interface itself (not routed
// through JSON widening): a hand-authored scorecard with the `before: null`
// branch, a noted node, and the newer priors omitted must satisfy the type.
describe("Scorecard type accepts the documented variants", () => {
  it("compiles a before-less scorecard literal", () => {
    const skillless = {
      pipeline: "no-baseline",
      generated_at: "2026-07-08T00:00:00Z",
      roteness: 1,
      before: null,
      after: {
        critical_path_seconds: { low: 1, high: 2 },
        hitl_gates: ["approve"],
        llm_input_tokens: { low: 0, high: 0 },
        llm_output_tokens: { low: 0, high: 0 },
        sampled_steps: 0,
        total_steps: 1,
        schema_constrained_steps: 0,
        sampled_output_tokens: { low: 0, high: 0 },
        nodes: [
          {
            id: "loop",
            kind: "agent_loop",
            calls: { low: 1, high: 10 },
            llm_input_tokens_per_call: 100,
            llm_output_tokens_per_call: 50,
            wall_seconds_per_call: 3,
            note: "unbounded loop estimated at the prior default",
          },
        ],
      },
      costs: [
        {
          model: "claude-fable-5",
          provider: "anthropic",
          tier: "flagship",
          price_source: "https://models.dev/api.json",
          fetched_at: "2026-07-08T00:00:00Z",
          input_per_mtok: 10,
          output_per_mtok: 50,
          before_usd: null,
          after_usd: { low: 0.01, high: 0.02 },
        },
      ],
      priors: {
        seconds_per_turn: 13,
        output_tokens_per_turn: 250,
        transcript_growth_per_turn: 900,
        system_overhead_tokens: 16000,
        turns_per_step_low: 1,
        turns_per_step_high: 2.5,
        llm_ttft_seconds: 1.2,
        llm_output_tokens_per_second: 60,
        tokens_per_input_field: 60,
        tokens_per_output_field: 40,
        external_call_seconds: 1.5,
        pure_function_seconds: 0.01,
        agent_loop_turns_per_iteration: 3,
        agent_loop_default_max_iterations: 10,
        chars_per_token: 3.8,
      },
    } satisfies Scorecard;
    expect(skillless.before).toBeNull();
  });
});
