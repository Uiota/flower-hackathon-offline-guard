Param(
  [Parameter(Mandatory=$true)] [string]$GiteaHost,         # e.g. git.example.com
  [Parameter(Mandatory=$true)] [string]$Owner,             # e.g. uiota
  [Parameter(Mandatory=$true)] [string]$Repo,              # e.g. offline-ai-landing
  [string]$Protocol = 'ssh'                                # 'ssh' or 'https'
)

# Fail on errors
$ErrorActionPreference = 'Stop'

Write-Host "Configuring Gitea remote + hooks for: $Owner/$Repo on $GiteaHost" -ForegroundColor Cyan

# Ensure we are in a git repo
git rev-parse --is-inside-work-tree | Out-Null

# Enable repo-local hooks in .githooks
git config core.hooksPath .githooks

# Make pre-push executable on Git Bash (if present)
if (Test-Path .githooks\pre-push) {
  try { & git update-index --chmod=+x .githooks/pre-push | Out-Null } catch {}
}

# Build remote URL
if ($Protocol -eq 'ssh') {
  $remote = "git@$GiteaHost:$Owner/$Repo.git"
} else {
  $remote = "https://$GiteaHost/$Owner/$Repo.git"
}

# If an origin exists and points to GitHub, switch it. Otherwise add/set.
$existing = try { git remote get-url origin } catch { '' }
if ($existing) {
  if ($existing -match 'github\.com') {
    Write-Host "Origin points to GitHub. Updating to Gitea..." -ForegroundColor Yellow
    git remote set-url origin $remote
    git remote set-url --push origin $remote
  } else {
    Write-Host "Origin exists (non-GitHub). Updating to: $remote" -ForegroundColor Yellow
    git remote set-url origin $remote
    git remote set-url --push origin $remote
  }
} else {
  git remote add origin $remote
}

Write-Host "Done. Current remotes:" -ForegroundColor Green
git remote -v

Write-Host "\nProtection active: pushes to GitHub are blocked by .githooks/pre-push" -ForegroundColor Green

