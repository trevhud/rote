/**
 * Stub for pure_function node: exclusion_check_dnc
 *
 * For each contact, look up "BDR do not contact" list memberships.
 *
 * Replace this stub with the deterministic API call. Direct vendor SDKs
 * are preferred over MCP wrappers — the rote compiler removes the MCP
 * layer at emit time, so production code calls Salesforce / HubSpot /
 * ZoomInfo / etc. directly.
 *
 * MANDATORY: this node was marked mandatory in the source skill.
 * The workflow always calls it; do not make it conditional.
 */

export async function exclusionCheckDnc(_input: unknown): Promise<never> {
    throw new Error("pure_function exclusion_check_dnc: stub not implemented");
}
