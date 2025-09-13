const fs = require('fs');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });
const { REST, Routes } = require('discord.js');

const commands = [];
const commandsPath = path.join(__dirname, 'commands');
for (const file of fs.readdirSync(commandsPath).filter(f => f.endsWith('.js'))) {
  const cmd = require(path.join(commandsPath, file));
  if (cmd && cmd.data) {
    commands.push(cmd.data.toJSON());
  }
}

const clientId = process.env.DISCORD_CLIENT_ID;
const guildId = process.env.DISCORD_GUILD_ID;
const token = process.env.DISCORD_TOKEN;

if (!clientId || !guildId || !token) {
  console.error('Missing required env: DISCORD_CLIENT_ID, DISCORD_GUILD_ID, DISCORD_TOKEN');
  process.exit(1);
}

const rest = new REST({ version: '10' }).setToken(token);

(async () => {
  try {
    console.log(`Deploying ${commands.length} commands to guild ${guildId}...`);
    await rest.put(Routes.applicationGuildCommands(clientId, guildId), { body: commands });
    console.log('Command deployment complete.');
  } catch (error) {
    console.error(error);
  }
})();

