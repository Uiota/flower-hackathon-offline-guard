#!/usr/bin/env node

const { exec, spawn } = require('child_process');

console.log('🚀 Starting Kubernetes website on localhost...');

// Use kubectl port-forward for localhost access
const kubectl = process.env.HOME + '/.local/bin/kubectl';
const portForwardProcess = spawn(kubectl, [
  'port-forward', 
  'service/offline-ai-webserver-service', 
  '8081:80'
], {
  stdio: ['ignore', 'pipe', 'pipe']
});

let isReady = false;

portForwardProcess.stdout.on('data', (data) => {
  const output = data.toString();
  if (output.includes('Forwarding from') && !isReady) {
    isReady = true;
    
    const serviceUrl = 'http://localhost:8081';
    
    console.log('');
    console.log('🎉 ===== KUBERNETES WEBSITE READY =====');
    console.log(`🌐 Website URL: ${serviceUrl}`);
    console.log('📋 Perfect for local development!');
    console.log('🚀 Kubernetes cluster: minikube');
    console.log('📦 Container: nginx:alpine');
    console.log('🔀 Port-forward: service → localhost:8081');
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
    console.log('🛑 To stop: Press Ctrl+C (will stop port-forward)');
    console.log('');
    
    // Keep process alive and show status
    let counter = 0;
    const statusInterval = setInterval(() => {
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
    
    // Cleanup on exit
    process.on('SIGINT', () => {
      console.log('\n🛑 Stopping port-forward...');
      clearInterval(statusInterval);
      portForwardProcess.kill();
      process.exit(0);
    });
  }
});

portForwardProcess.stderr.on('data', (data) => {
  const error = data.toString();
  if (error.includes('error') || error.includes('Error')) {
    console.error('❌ Port-forward error:', error);
  }
});

portForwardProcess.on('close', (code) => {
  if (code !== 0) {
    console.error('❌ Port-forward failed. Make sure the service is running.');
    console.log('💡 Try: kubectl get services');
  }
});