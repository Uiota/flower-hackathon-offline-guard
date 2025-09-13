Link Claude Code to your API (MCP)
=================================

Goal: let Claude Code call your local API so agents can share tools and code together.

What’s included
- scripts/mcp-uiota-bridge.js — Minimal MCP server that bridges to your HTTP API
- .claude.mcp.example.json — Example Claude Code MCP config mapping the bridge
- scripts/claude-install.ps1 — Installs Claude Code on Windows and sets Git Bash path

Steps (Windows)
1) Install Claude Code
   powershell -ExecutionPolicy Bypass -File scripts/claude-install.ps1 -Channel stable

2) Configure MCP for Claude Code
   - Copy .claude.mcp.example.json to your Claude Code config location.
     Common locations (pick one depending on installer):
       %USERPROFILE%\.claude\config.json
       %APPDATA%\Claude\config.json
   - If a file already exists, merge the "mcpServers" section.

3) Set API credentials
   - Edit the env in the config you just copied:
       UIOTA_API_BASE: http://127.0.0.1:9090/api/v1
       UIOTA_API_SECRET: change-me

4) Start Claude Code in this repo
   - Open PowerShell in the repo root and run:
       claude doctor
       claude
   - When prompted, enable the server "uiota-http".

5) Try a tool call
   - Ask: "Call tool uiota.get_env with node_id=\"mirabilis\"".
   - Or: "Report status for node mirabilis with metrics {...}".

Notes
- The MCP bridge reads env vars and calls:
  GET  {BASE}/env/{node_id}
  POST {BASE}/status  with { node_id, model_metrics, epistemic_state }
- Update the example config to point to your API host or port.
- The pre-push hook prevents GitHub pushes; use scripts/configure-gitea.ps1 to set Gitea as origin.

Run inside VS Code
- Open this repo in VS Code and use Terminal → Run Task…
  - "Claude: Verify Install (PS1)" runs version + doctor checks
  - "Claude: Doctor" shows install details
  - "MCP: Run UIOTA Bridge" starts the API bridge (leave running)
- Open a separate terminal and run `claude` to start a chat session with the bridge available.
