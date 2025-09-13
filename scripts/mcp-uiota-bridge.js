#!/usr/bin/env node
// Minimal MCP server bridging UIOTA HTTP API to Claude Code via stdio
// Exposes tools: get_env(node_id), report_status(metrics, epistemic_state)

const BASE = process.env.UIOTA_API_BASE || 'http://127.0.0.1:9090/api/v1';
const SECRET = process.env.UIOTA_API_SECRET || '';

const tools = [
  {
    name: 'uiota.get_env',
    description: 'Fetch environment context for a node_id',
    inputSchema: {
      type: 'object',
      properties: { node_id: { type: 'string' } },
      required: ['node_id']
    }
  },
  {
    name: 'uiota.report_status',
    description: 'Report model metrics and epistemic state',
    inputSchema: {
      type: 'object',
      properties: {
        node_id: { type: 'string' },
        metrics: { type: 'object' },
        epistemic_state: { type: 'object' }
      },
      required: ['node_id']
    }
  }
];

function write(msg) {
  const buf = Buffer.from(JSON.stringify(msg));
  const header = Buffer.from(`Content-Length: ${buf.length}\r\n\r\n`);
  process.stdout.write(header);
  process.stdout.write(buf);
}

async function handleCall(id, name, args = {}) {
  if (!BASE) throw new Error('UIOTA_API_BASE not set');
  if (!SECRET) throw new Error('UIOTA_API_SECRET not set');
  try {
    if (name === 'uiota.get_env') {
      const node_id = args.node_id;
      const res = await fetch(`${BASE.replace(/\/$/, '')}/env/${encodeURIComponent(node_id)}`, {
        headers: { 'Authorization': `Bearer ${SECRET}` }
      });
      const data = await res.json();
      write({ jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: JSON.stringify(data) }], isError: false } });
      return;
    }
    if (name === 'uiota.report_status') {
      const { node_id, metrics = {}, epistemic_state = {} } = args;
      const res = await fetch(`${BASE.replace(/\/$/, '')}/status`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${SECRET}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_id, model_metrics: metrics, epistemic_state })
      });
      const data = await res.json().catch(() => ({}));
      write({ jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: JSON.stringify(data) }], isError: false } });
      return;
    }
    write({ jsonrpc: '2.0', id, error: { code: -32601, message: `Unknown tool ${name}` } });
  } catch (e) {
    write({ jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: String(e.message || e) }], isError: true } });
  }
}

function handle(msg) {
  const { id, method } = msg;
  if (method === 'initialize') {
    write({ jsonrpc: '2.0', id, result: { protocolVersion: '2024-11-05', capabilities: { tools: {} } } });
    return;
  }
  if (method === 'tools/list') {
    write({ jsonrpc: '2.0', id, result: { tools } });
    return;
  }
  if (method === 'tools/call') {
    const { name, arguments: args } = msg.params || {};
    handleCall(id, name, args);
    return;
  }
  if (method === 'ping') { write({ jsonrpc: '2.0', id, result: {} }); return; }
  if (method === 'shutdown') { write({ jsonrpc: '2.0', id, result: {} }); process.exit(0); }
  write({ jsonrpc: '2.0', id, error: { code: -32601, message: `Unknown method ${method}` } });
}

let buf = Buffer.alloc(0);
process.stdin.on('data', chunk => { buf = Buffer.concat([buf, chunk]); pump(); });

function pump() {
  while (true) {
    const headerEnd = buf.indexOf('\r\n\r\n');
    if (headerEnd === -1) return;
    const header = buf.slice(0, headerEnd).toString('utf8');
    const m = /Content-Length:\s*(\d+)/i.exec(header);
    if (!m) { buf = buf.slice(headerEnd + 4); continue; }
    const len = parseInt(m[1], 10);
    const total = headerEnd + 4 + len;
    if (buf.length < total) return;
    const body = buf.slice(headerEnd + 4, total).toString('utf8');
    buf = buf.slice(total);
    try { const msg = JSON.parse(body); handle(msg); } catch (e) { /* ignore parse errors */ }
  }
}

