// TS SDK v1 (what rote's emitted helpers pin, ^1.29) and v2 client (default + auto mode)
// against public no-auth servers. Logs the protocol headers on every request via a fetch wrapper.
import { Client as ClientV1 } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport as TransportV1 } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { Client as ClientV2, StreamableHTTPClientTransport as TransportV2 } from "@modelcontextprotocol/client";

const TARGETS = [
  ["https://docs.mcp.cloudflare.com/mcp", "search_cloudflare_documentation", { query: "Workflows" }],
  ["https://mcp.deepwiki.com/mcp", "read_wiki_structure", { repoName: "modelcontextprotocol/python-sdk" }],
];

function loggingFetch() {
  return async (input, init) => {
    const h = new Headers(init?.headers ?? {});
    let rpc = null;
    try { rpc = JSON.parse(init?.body ?? "{}").method ?? null; } catch {}
    console.log(`    -> ${init?.method ?? "GET"} rpc=${rpc} MCP-Protocol-Version=${h.get("mcp-protocol-version")} Mcp-Method=${h.get("mcp-method")} Mcp-Name=${h.get("mcp-name")}`);
    const r = await fetch(input, init);
    console.log(`    <- ${r.status}`);
    return r;
  };
}

async function run(label, makeClient, makeTransport) {
  for (const [url, tool, args] of TARGETS) {
    console.log(`=== ${label} ${url}`);
    try {
      const client = makeClient();
      await client.connect(makeTransport(new URL(url)));
      const v = client.getNegotiatedProtocolVersion?.() ?? client.transport?.protocolVersion ?? "(n/a)";
      console.log(`  negotiated=${v}`);
      const res = await client.callTool({ name: tool, arguments: args });
      console.log(`  callTool OK isError=${res.isError ?? false} resultType=${res.resultType ?? "(absent)"}`);
      await client.close();
    } catch (e) {
      console.log(`  FAILED ${e?.name}: ${String(e?.message).slice(0, 300)}`);
    }
  }
}

await run("v1 1.32.1", () => new ClientV1({ name: "rote-probe", version: "0.1" }),
  (u) => new TransportV1(u, { fetch: loggingFetch() }));
await run("v2 2.3.1 default", () => new ClientV2({ name: "rote-probe", version: "0.1" }),
  (u) => new TransportV2(u, { fetch: loggingFetch() }));
await run("v2 2.3.1 auto", () => new ClientV2({ name: "rote-probe", version: "0.1" }, { versionNegotiation: { mode: "auto" } }),
  (u) => new TransportV2(u, { fetch: loggingFetch() }));
