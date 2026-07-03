/**
 * Stub for pure_function node: exclusion_check_recent
 *
 * For each contact, check if they were emailed (outbound) in the last
 *
 * Replace this stub with the deterministic API call. Direct vendor SDKs
 * are preferred over MCP wrappers — the rote graduator removes the MCP
 * layer at emit time, so production code calls the vendor APIs directly.
 *
 * MANDATORY: this node was marked mandatory in the source skill.
 * The workflow always calls it; do not make it conditional.
 *
 * Constants from the source skill (lifted into the IR):
 *   days_back = 30
 *   direction = "OUTBOUND"
 */

export async function exclusionCheckRecent(_input: unknown): Promise<never> {
    throw new Error("pure_function exclusion_check_recent: stub not implemented");
}
