/**
 * Stub for external_call node: hubspot_upsert
 *
 * Batch upsert contacts to HubSpot (create or update by email). Hard
 *
 * Replace this stub with the deterministic API call. Direct vendor SDKs
 * are preferred over MCP wrappers — the rote graduator removes the MCP
 * layer at emit time, so production code calls Salesforce / HubSpot /
 * ZoomInfo / etc. directly.
 *
 * Constants from the source skill (lifted into the IR):
 *   batch_size = 100
 */

export async function hubspotUpsert(_input: unknown) {
    throw new Error("external_call hubspot_upsert: stub not implemented");
}
