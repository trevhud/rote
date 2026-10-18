// TS SDK v2 client in auto mode (the mode Claude Code's v2 runtime negotiates with) against `rote serve --http`.
import { Client, StreamableHTTPClientTransport } from "@modelcontextprotocol/client";

const c = new Client({ name: "p", version: "0" }, { versionNegotiation: { mode: "auto" } });
await c.connect(new StreamableHTTPClientTransport(new URL(process.argv[2])));
console.log("negotiated:", c.getNegotiatedProtocolVersion?.());
const { tools } = await c.listTools();
console.log("tools:", tools.map((t) => t.name));
await c.close();
