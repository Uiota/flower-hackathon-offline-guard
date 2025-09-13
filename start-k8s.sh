#!/bin/bash

cd /media/uiota/USB31FD/offline-ai-landing

echo "Starting Kubernetes webserver with auto-updates..."

# Apply/start the deployment
kubectl apply -f k8s-webserver-deployment.yaml
kubectl apply -f k8s-webserver-service.yaml

# Run initial update
echo "Running initial update..."
./update-k8s.sh

# Wait for pod to be ready
echo "Waiting for pod to be ready..."
kubectl wait --for=condition=ready pod -l app=offline-ai-webserver --timeout=60s

# Start port-forward to 8081
echo "Starting port-forward to 8081..."
kubectl port-forward service/offline-ai-webserver-service 8081:80 &
PORT_FORWARD_PID=$!

# Start auto-updater in background
echo "Starting auto-updater (updates every 30 seconds)..."
./auto-update.sh &
AUTO_UPDATE_PID=$!

echo ""
echo "================================"
echo "🚀 Server started successfully!"
echo "📝 Website: http://localhost:8081"
echo "🔄 Auto-updates: ENABLED (every 30s)"
echo "⏹️  Press Ctrl+C to stop"
echo "================================"
echo ""

# Wait for interrupt
trap "echo 'Stopping services...'; kill $AUTO_UPDATE_PID $PORT_FORWARD_PID 2>/dev/null; exit 0" INT
wait