/**
 * Stub for agent_loop node: target_research
 *
 * Run external research (Bright Data web search, ClinicalTrials.gov)
 *
 * Agent loops require an LLM agent runtime (e.g. the Anthropic Agent SDK
 * with bounded iterations). Implement this against the agent harness your
 * project already uses — the workflow only cares that the function
 * resolves with the loop's terminal output.
 *
 * Tools the agent should be allowed to call:
 *   - bright_data_search
 *   - bright_data_scrape
 *   - clinical_trials_search
 *   - airweave_search
 *   - salesforce_query
 */

import { type Env } from "../workflow";

export async function targetResearch(
    _input: unknown,
    _env: Env,
): Promise<never> {
    throw new Error("agent_loop target_research: requires an agent runtime — implement me");
}
