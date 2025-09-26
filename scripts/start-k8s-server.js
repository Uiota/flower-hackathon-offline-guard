#!/usr/bin/env node

const { exec } = require('child_process');

const kubectl = process.env.HOME + '/.local/bin/kubectl';
const minikube = process.env.HOME + '/.local/bin/minikube';

console.log('🚀 Starting Kubernetes deployment...');

// Get minikube service URL
exec(`${minikube} service offline-ai-webserver-service --url`, (error, stdout) => {
  if (error) {
    console.error('❌ Error getting service URL:', error.message);
    console.log('💡 Try running: minikube service offline-ai-webserver-service --url');
    return;
  }
  
  const serviceUrl = stdout.trim();
  console.log('');
  console.log('🎉 ===== KUBERNETES DEPLOYMENT READY =====');
  console.log(`🌐 Website URL: ${serviceUrl}`);
  console.log('📋 Copy this URL to your browser if it doesn\'t open automatically');
  console.log('🚀 Kubernetes cluster: minikube');
  console.log('📦 Container: nginx:alpine');
  console.log('');
  
  // Keep the process running for debugging
  console.log('📝 Logs: Use VS Code Kubernetes extension to view pod logs');
  console.log('🛑 To stop: Use "K8s: Delete Deployment" task or Ctrl+C');
  console.log('');
  
  // Keep process alive and show status
  let counter = 0;
  setInterval(() => {
    // Check if service is still running
    exec(`${kubectl} get pods -l app=offline-ai-webserver --no-headers`, (err, out) => {
      if (!err && out.trim()) {
        const status = out.split(/\s+/)[2];
        const time = new Date().toLocaleTimeString();
        process.stdout.write(`\r🟢 [${time}] Pod: ${status} | URL: ${serviceUrl} | Uptime: ${Math.floor(counter/12)}min`);
        counter++;
      } else {
        process.stdout.write(`\r🔴 Pod stopped or not found`);
      }
    });
  }, 5000);
});