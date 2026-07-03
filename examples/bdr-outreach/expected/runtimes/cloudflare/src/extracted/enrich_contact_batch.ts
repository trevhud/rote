/**
 * Stub for external_call node: enrich_contact_batch
 *
 * Batch enrich contacts via ZoomInfo. Always requests employmentHistory
 *
 * Replace this stub with the deterministic API call. Direct vendor SDKs
 * are preferred over MCP wrappers — the rote graduator removes the MCP
 * layer at emit time, so production code calls Salesforce / HubSpot /
 * ZoomInfo / etc. directly.
 *
 * Constants from the source skill (lifted into the IR):
 *   batch_size = 10
 *   output_fields = ["firstName", "lastName", "jobTitle", "email", "phone", "mobilePhone", "contactAccuracyScore", "companyName", "externalUrls", "employmentHistory", "directPhoneDoNotCall", "mobilePhoneDoNotCall", "validDate"]
 */

export async function enrichContactBatch(_input: unknown): Promise<never> {
    throw new Error("external_call enrich_contact_batch: stub not implemented");
}
