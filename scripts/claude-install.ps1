Param(
  [ValidateSet('stable','latest')]
  [string]$Channel = 'stable',
  [string]$GitBashPath = 'C:\\Program Files\\Git\\bin\\bash.exe'
)
$ErrorActionPreference = 'Stop'
Write-Host "Installing Claude Code ($Channel) and configuring Git Bash path..." -ForegroundColor Cyan

if ($Channel -eq 'latest') {
  & ([scriptblock]::Create((irm https://claude.ai/install.ps1))) latest
} else {
  irm https://claude.ai/install.ps1 | iex
}

if (Test-Path $GitBashPath) {
  [Environment]::SetEnvironmentVariable('CLAUDE_CODE_GIT_BASH_PATH', $GitBashPath, 'User')
  $env:CLAUDE_CODE_GIT_BASH_PATH = $GitBashPath
  Write-Host "Set CLAUDE_CODE_GIT_BASH_PATH -> $GitBashPath" -ForegroundColor Green
} else {
  Write-Warning "Git Bash not found at: $GitBashPath. If using portable Git, set env var manually."
}

Write-Host "Running 'claude doctor'..." -ForegroundColor Cyan
claude doctor

