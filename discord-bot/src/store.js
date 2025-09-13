const fs = require('fs');
const path = require('path');

const dataDir = path.join(__dirname, '..', 'data');
const agentsFile = path.join(dataDir, 'agents.json');

function ensure() {
  if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });
  if (!fs.existsSync(agentsFile)) fs.writeFileSync(agentsFile, JSON.stringify({ agents: [] }, null, 2));
}

function readAgents() {
  ensure();
  const raw = fs.readFileSync(agentsFile, 'utf8');
  return JSON.parse(raw).agents || [];
}

function writeAgents(list) {
  ensure();
  fs.writeFileSync(agentsFile, JSON.stringify({ agents: list }, null, 2));
}

function addAgent(agent) {
  const list = readAgents();
  list.push(agent);
  writeAgents(list);
}

function removeAgentByName(guildId, name) {
  const list = readAgents();
  const next = list.filter(a => !(a.guildId === guildId && a.name.toLowerCase() === name.toLowerCase()));
  writeAgents(next);
  return next.length !== list.length;
}

function getAgentsByGuild(guildId) {
  return readAgents().filter(a => a.guildId === guildId);
}

function getAgent(guildId, name) {
  return getAgentsByGuild(guildId).find(a => a.name.toLowerCase() === name.toLowerCase()) || null;
}

module.exports = {
  ensure,
  addAgent,
  removeAgentByName,
  getAgentsByGuild,
  getAgent
};

