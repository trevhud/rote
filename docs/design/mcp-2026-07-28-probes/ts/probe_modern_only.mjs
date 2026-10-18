// TS SDK v1 1.32.1 (what emitted _roteMcp.ts pins, ^1.29) and v2 default/auto against the modern-only server.
import { Client as ClientV1 } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport as TransportV1 } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { Client as ClientV2, StreamableHTTPClientTransport as TransportV2 } from "@modelcontextprotocol/client";

const url = new URL(process.argv[2]);
const cases = [
  ["v1 1.32.1", () => new ClientV1({ name: "p", version: "0" }), () => new TransportV1(url)],
  ["v2 default (legacy)", () => new ClientV2({ name: "p", version: "0" }), () => new TransportV2(url)],
  ["v2 auto", () => new ClientV2({ name: "p", version: "0" }, { versionNegotiation: { mode: "auto" } }), () => new TransportV2(url)],
];
for (const [label, mk, tr] of cases) {
  try {
    const c = mk();
    await c.connect(tr());
    const r = await c.callTool({ name: "echo", arguments: { text: "hi" } });
    console.log(`${label}: OK structuredContent=${JSON.stringify(r.structuredContent)}`);
    await c.close();
  } catch (e) {
    console.log(`${label}: FAILED ${e?.name}: ${String(e?.message).slice(0, 250)}`);
  }
}
