#!/bin/bash

echo "🔧 Fixing Kubernetes startup issues after reboot..."

# 1. Start required services
echo "Starting podman services..."
systemctl --user start podman.socket
systemctl --user enable podman.socket

# 2. Clean up any stuck processes
echo "Cleaning up stuck processes..."
pkill -f minikube 2>/dev/null || true
minikube delete 2>/dev/null || true

# 3. Try to start minikube with retry logic
echo "Starting minikube (may take a few attempts)..."
for i in {1..3}; do
    echo "Attempt $i/3..."
    if minikube start --driver=podman --force --wait=false; then
        echo "✅ Minikube started successfully!"
        break
    else
        echo "❌ Attempt $i failed, retrying in 10 seconds..."
        minikube delete 2>/dev/null || true
        sleep 10
    fi
done

# 4. Check if cluster is ready
echo "Checking cluster status..."
if kubectl cluster-info &>/dev/null; then
    echo "✅ Kubernetes cluster is ready!"
    echo "Now you can run: ./start-k8s.sh"
else
    echo "❌ Cluster still not ready. Try manual steps:"
    echo "1. Update podman: sudo apt update && sudo apt install podman"
    echo "2. Or use Docker: minikube start --driver=docker (requires sudo docker setup)"
    echo "3. Check logs: minikube logs"
fi