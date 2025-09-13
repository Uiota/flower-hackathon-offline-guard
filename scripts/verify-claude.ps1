Param(
  [switch]$McpBridge
)
$ErrorActionPreference = 'Stop'

Write-Host "=== Verifying Claude Code installation ===" -ForegroundColor Cyan
$cmd = Get-Command claude -ErrorAction SilentlyContinue
if (-not $cmd) {
  Write-Error "'claude' was not found in PATH. Run scripts/claude-install.ps1 first."; exit 1
}
Write-Host ("claude path: {0}" -f $cmd.Source)

try { claude --version } catch { Write-Error $_; exit 1 }
try { claude doctor } catch { Write-Error $_; exit 1 }

if ($McpBridge) {
  Write-Host "`n=== Testing MCP bridge (node required) ===" -ForegroundColor Cyan
  $node = Get-Command node -ErrorAction SilentlyContinue
  if (-not $node) { Write-Warning "Node.js not found; skipping MCP test."; exit 0 }
  $env:UIOTA_API_BASE = $env:UIOTA_API_BASE -as [string]
  if (-not $env:UIOTA_API_BASE) { $env:UIOTA_API_BASE = 'http://127.0.0.1:9090/api/v1' }
  if (-not $env:UIOTA_API_SECRET) { Write-Warning "UIOTA_API_SECRET not set; bridge will error on real calls." }
  Write-Host ("node: {0}" -f $node.Source)
  Write-Host ("UIOTA_API_BASE={0}" -f $env:UIOTA_API_BASE)
  $p = Start-Process -NoNewWindow -PassThru -FilePath node -ArgumentList 'scripts/mcp-uiota-bridge.js'
  Start-Sleep -Milliseconds 500
  try { $p.Kill() } catch {}
}

Write-Host "`nOK - Claude Code appears installed and responding." -ForegroundColor Green