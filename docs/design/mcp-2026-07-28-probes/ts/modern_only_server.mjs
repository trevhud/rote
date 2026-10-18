// A 2026-07-28 MODERN-ONLY MCP server (TS SDK v2 createMcpHandler, legacy: 'reject') on 127.0.0.1:$PORT/mcp.
// One tool, `echo`, declared WITH an outputSchema so structured content is returned.
import http from "node:http";
import { McpServer, createMcpHandler } from "@modelcontextprotocol/server";
import * as z from "zod";

const port = Number(process.env.PORT ?? 8799);

const handler = createMcpHandler(() => {
  const server = new McpServer({ name: "modern-only-probe", version: "0.1.0" });
  server.registerTool(
    "echo",
    {
      description: "Echo the text back",
      inputSchema: z.object({ text: z.string() }),
      outputSchema: z.object({ echoed: z.string(), length: z.number() }),
    },
    async ({ text }) => ({
      content: [{ type: "text", text: JSON.stringify({ echoed: text, length: text.length }) }],
      structuredContent: { echoed: text, length: text.length },
    }),
  );
  return server;
}, { legacy: "reject" });

http.createServer(async (req, res) => {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  const body = Buffer.concat(chunks);
  const request = new Request(`http://127.0.0.1:${port}${req.url}`, {
    method: req.method,
    headers: req.headers,
    body: ["GET", "HEAD"].includes(req.method) ? undefined : body,
  });
  const response = await handler.fetch(request);
  res.writeHead(response.status, Object.fromEntries(response.headers));
  if (response.body) {
    for await (const chunk of response.body) res.write(chunk);
  }
  res.end();
}).listen(port, "127.0.0.1", () => console.log(`modern-only server on http://127.0.0.1:${port}/mcp`));
