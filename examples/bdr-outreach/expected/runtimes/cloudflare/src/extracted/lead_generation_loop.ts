/**
 * Stub for agent_loop node: lead_generation_loop
 *
 * Iterative search-enrich-vet loop. Starts with three parallel ZoomInfo
 *
 * Agent loops require an LLM agent runtime (e.g. the Anthropic Agent SDK
 * with bounded iterations). Implement this against the agent harness your
 * project already uses — the workflow only cares that the function
 * resolves with the loop's terminal output.
 *
 * Tools the agent should be allowed to call:
 *   - zoominfo_search_contacts
 *   - zoominfo_search_companies
 *
 * Loop body sub-nodes (call once per iteration):
 *   - enrich_contact_batch
 *   - vet_contact
 */

import { type Env } from "../workflow";

export async function leadGenerationLoop(
    _input: unknown,
    _env: Env,
): Promise<never> {
    throw new Error("agent_loop lead_generation_loop: requires an agent runtime — implement me");
}
