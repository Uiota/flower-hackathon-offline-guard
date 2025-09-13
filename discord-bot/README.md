Discord Agent Factory Bot

Overview

- Purpose: Manage your Discord server and provide a factory for "agents" (developer helpers/bots) that can be defined via simple YAML or JSON and deployed as slash-command driven helpers.
- Highlights:
  - Slash commands to create/list/delete agents and spin up workspaces (channels, roles).
  - Pluggable agent definitions in `agents/` (no code changes required to add new ones).
  - Simple JSON-based persistence in `data/agents.json` (swap for a DB later).
  - Safe defaults; only admins (configurable) can manage agents.

Quick Start

1) Create a Discord application + bot

- Go to https://discord.com/developers/applications
- Create Application → Bot → Reset Token → copy token.
- Under Bot → Privileged Gateway Intents: enable "Server Members" if you want the bot to manage roles.
- Under OAuth2 → URL Generator: Scopes `bot applications.commands`, Permissions at least:
  - Manage Roles, Manage Channels, Send Messages, Create Public Threads, Create Private Threads.
  - Copy the invite URL and add the bot to your server.

2) Configure environment

- Copy `.env.example` to `.env` and fill out:
  - `DISCORD_TOKEN`: your bot token.
  - `DISCORD_CLIENT_ID`: application ID.
  - `DISCORD_GUILD_ID`: your test guild/server ID (for fast command deploys).

3) Install + run

- Requires Node.js 18+
- In `discord-bot/` run:
  - `npm install`
  - `npm run deploy:commands` (deploy guild commands for fast iteration)
  - `npm start`

Slash Commands

- `/agent create name:<string> template:<string> private:<bool>`: Create an agent from a template and optionally create a private channel + role.
- `/agent list`: List all registered agents in this guild.
- `/agent delete name:<string>`: Remove an agent (and optionally its channel/role if managed).
- `/agent run name:<string> topic:<string>`: Spin up a new thread under the agent's channel to handle a task. Stub handler ready for LLM integration.

Agent Definitions

- Folder: `agents/`
- Format: YAML or JSON. Example `agents/example.yaml`:

```
name: helper
description: "General dev helper for the consortium"
capabilities:
  - code-review
  - docs-search
  - repo-scaffold
channel:
  prefix: agent-
  topic: "Agent workspace for {{name}}"
permissions:
  adminOnly: true
```

- Add more templates; they appear as options in `/agent create`.

Extending

- Hook your LLM/tools in `src/agentRuntime.js` and call them from `/agent run`.
- Replace JSON storage in `src/store.js` with a DB.
- Add more commands under `src/commands/` — they auto-load at startup.

Notes

- This is a single bot that hosts many agents. If you truly need separate Discord bot identities (multiple tokens), run multiple processes with different `.env` files. Most use cases are served by a single host bot with multi-agent channels/threads.

