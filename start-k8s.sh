#!/bin/bash

cd /media/uiota/USB31FD/offline-ai-landing

echo "🔧 Starting Kubernetes webserver with auto-updates..."

# Check and fix restart issues
echo "🔍 Checking for restart issues..."

# 1. Check if podman socket is running (common restart issue)
if ! systemctl --user is-active --quiet podman.socket; then
    echo "🔧 Starting podman.socket service..."
    systemctl --user start podman.socket
    systemctl --user enable podman.socket
fi

# 2. Check if minikube cluster is running
if ! kubectl cluster-info &>/dev/null; then
    echo "🔧 Kubernetes cluster not responding, checking minikube..."
    
    # Check minikube status
    if ! minikube status &>/dev/null; then
        echo "🚀 Starting minikube cluster..."
        minikube start --driver=podman --force
        
        # Wait a bit for cluster to be ready
        echo "⏳ Waiting for cluster to be ready..."
        sleep 10
    else
        echo "🔄 Minikube running but cluster not responding, restarting..."
        minikube stop
        minikube start --driver=podman --force
        sleep 10
    fi
fi

# 3. Verify cluster is now accessible
echo "✅ Verifying cluster access..."
if ! kubectl cluster-info &>/dev/null; then
    echo "❌ Failed to connect to Kubernetes cluster. Please check:"
    echo "   - Run: systemctl --user status podman.socket"
    echo "   - Run: minikube status"
    echo "   - Run: minikube logs"
    exit 1
fi

echo "✅ Kubernetes cluster is ready!"

# Apply/start the deployment
echo "📦 Applying Kubernetes manifests..."
kubectl apply -f k8s-webserver-deployment.yaml
kubectl apply -f k8s-webserver-service.yaml

# Run initial update (create ConfigMaps)
echo "📄 Running initial update (creating ConfigMaps)..."
chmod +x update-k8s.sh 2>/dev/null || true
bash update-k8s.sh

# Wait for pod to be ready
echo "⏳ Waiting for pod to be ready..."
if ! kubectl wait --for=condition=ready pod -l app=offline-ai-webserver --timeout=120s; then
    echo "❌ Pod failed to become ready. Checking pod status..."
    kubectl get pods -l app=offline-ai-webserver
    kubectl describe pod -l app=offline-ai-webserver
    echo "💡 Common fixes:"
    echo "   - ConfigMaps missing (should be fixed by update script)"
    echo "   - Image pull issues (check: kubectl describe pod)"
    echo "   - Resource constraints (check: kubectl describe nodes)"
    exit 1
fi

# Kill any existing port-forward processes
pkill -f "kubectl port-forward.*offline-ai-webserver-service" 2>/dev/null || true

# Start port-forward to 8081
echo "🔗 Starting port-forward to 8081..."
kubectl port-forward service/offline-ai-webserver-service 8081:80 &
PORT_FORWARD_PID=$!

# Start auto-updater in background
echo "🔄 Starting auto-updater (updates every 30 seconds)..."
chmod +x auto-update.sh 2>/dev/null || true
./auto-update.sh &
AUTO_UPDATE_PID=$!

echo ""
echo "================================"
echo "🚀 Server started successfully!"
echo "📝 Website: http://localhost:8081 (port-forward)"
echo "📝 Website: http://localhost:30080 (NodePort)"
echo "🔄 Auto-updates: ENABLED (every 30s)"
echo "⏹️  Press Ctrl+C to stop"
echo ""
echo "💡 If you get connection issues after restart:"
echo "   Just run this script again - it will auto-fix!"
echo "================================"
echo ""

# Wait for interrupt
trap "echo 'Stopping services...'; kill $AUTO_UPDATE_PID $PORT_FORWARD_PID 2>/dev/null; exit 0" INT
wait