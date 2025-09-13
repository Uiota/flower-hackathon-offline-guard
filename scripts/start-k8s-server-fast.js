#!/usr/bin/env node

const { exec } = require('child_process');

const kubectl = process.env.HOME + '/.local/bin/kubectl';
const minikube = process.env.HOME + '/.local/bin/minikube';

console.log('🚀 Starting Kubernetes website (Fast Mode)...');

// Get minikube IP and service port directly
Promise.all([
  new Promise((resolve, reject) => {
    exec(`${minikube} ip`, (error, stdout) => {
      if (error) reject(error);
      else resolve(stdout.trim());
    });
  }),
  new Promise((resolve, reject) => {
    exec(`${kubectl} get service offline-ai-webserver-service -o jsonpath="{.spec.ports[0].nodePort}"`, (error, stdout) => {
      if (error) reject(error);
      else resolve(stdout.trim());
    });
  })
]).then(([minikubeIP, nodePort]) => {
  const serviceUrl = `http://${minikubeIP}:${nodePort}`;
  
  console.log('');
  console.log('🎉 ===== KUBERNETES WEBSITE READY =====');
  console.log(`🌐 Website URL: ${serviceUrl}`);
  console.log('📋 Copy this URL to your browser');
  console.log('🚀 Kubernetes cluster: minikube');
  console.log('📦 Container: nginx:alpine');
  console.log('');
  
  // Open browser automatically
  exec(`xdg-open "${serviceUrl}"`, (err) => {
    if (!err) {
      console.log('🌐 Browser opened automatically!');
    } else {
      console.log('💡 Manual: Copy the URL above to your browser');
    }
  });
  
  console.log('📝 Logs: Use VS Code Kubernetes extension to view pod logs');
  console.log('🛑 To stop: Use "K8s: Delete Deployment" task or Ctrl+C');
  console.log('');
  
  // Keep process alive and show status
  let counter = 0;
  setInterval(() => {
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
  
}).catch((error) => {
  console.error('❌ Error getting service info:', error.message);
  console.log('💡 Fallback: Your website should be accessible at http://localhost:30080');
});