#!/usr/bin/env node

const { exec } = require('child_process');

console.log('🐳 Starting Podman container...');

// Check if container is running
exec('podman ps --filter name=offline-ai-web --format "{{.Status}}"', (error, stdout, stderr) => {
  if (stdout.trim()) {
    console.log('✅ Podman container is running at: http://localhost:8080');
    console.log('🌐 Website should open automatically in your browser!');
  } else {
    console.log('❌ Container not running. Check the terminal output.');
    return;
  }
  
  // Keep the process running for debugging
  console.log('\n📝 Logs: Run "podman logs offline-ai-web" to view logs');
  console.log('🛑 To stop: Run "podman stop offline-ai-web" or Ctrl+C');
  
  // Keep process alive and show status
  setInterval(() => {
    exec('podman ps --filter name=offline-ai-web --format "{{.Status}}"', (err, out) => {
      if (!err && out.trim()) {
        process.stdout.write(`\r🟢 Container Status: ${out.trim()} | URL: http://localhost:8080`);
      } else {
        process.stdout.write(`\r🔴 Container stopped`);
      }
    });
  }, 5000);
});