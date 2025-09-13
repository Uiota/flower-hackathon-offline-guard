Gitea-only setup (no GitHub pushes)
===================================

What’s included in this repo
- .githooks/pre-push: blocks any push where the remote URL contains github.com
- scripts/configure-gitea.ps1: sets origin to your Gitea repo and enables the hook

One-time steps (Windows PowerShell)
1) From the repo root, run:
   powershell -ExecutionPolicy Bypass -File scripts/configure-gitea.ps1 -GiteaHost YOUR_GITEA_HOST -Owner YOUR_ORG -Repo offline-ai-landing -Protocol ssh

   Example:
   powershell -ExecutionPolicy Bypass -File scripts/configure-gitea.ps1 -GiteaHost gitea.example.com -Owner uiota -Repo offline-ai-landing -Protocol ssh

2) Verify remotes:
   git remote -v
   # origin should point to your Gitea instance

3) Test protection:
   git push origin HEAD:main --dry-run
   # should succeed to Gitea; any GitHub remote would be blocked by the hook

Notes
- The hook is enabled via `git config core.hooksPath .githooks` inside the script. If you prefer, run it manually.
- To apply the GitHub block to all new repos, set a global template with this pre-push hook and run:
  git config --global init.templateDir "%USERPROFILE%/.git-templates"

Claude Code (Windows quick install)
- PowerShell: `powershell -ExecutionPolicy Bypass -File scripts/claude-install.ps1 -Channel stable`
- For portable Git, pass your Bash path:
  `powershell -ExecutionPolicy Bypass -File scripts/claude-install.ps1 -GitBashPath "C:\\PortableGit\\bin\\bash.exe"`
- Verify: `claude doctor`
