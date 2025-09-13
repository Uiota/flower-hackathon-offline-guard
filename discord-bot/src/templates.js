const fs = require('fs');
const path = require('path');
const YAML = require('yaml');

let cache = null;

function loadTemplates() {
  const dir = path.join(__dirname, '..', 'agents');
  const templates = {};
  if (!fs.existsSync(dir)) return templates;
  for (const file of fs.readdirSync(dir)) {
    const full = path.join(dir, file);
    if (!fs.statSync(full).isFile()) continue;
    const ext = path.extname(file).toLowerCase();
    const raw = fs.readFileSync(full, 'utf8');
    try {
      let data;
      if (ext === '.yaml' || ext === '.yml') data = YAML.parse(raw);
      else if (ext === '.json') data = JSON.parse(raw);
      else continue;
      if (data && data.name) {
        templates[data.name] = data;
      }
    } catch (e) {
      console.error(`Failed to parse template ${file}:`, e.message);
    }
  }
  cache = templates;
  return templates;
}

function getTemplates() {
  if (!cache) return loadTemplates();
  return cache;
}

function getTemplateNames() {
  return Object.keys(getTemplates());
}

function getTemplate(name) {
  return getTemplates()[name] || null;
}

module.exports = { loadTemplates, getTemplates, getTemplateNames, getTemplate };

