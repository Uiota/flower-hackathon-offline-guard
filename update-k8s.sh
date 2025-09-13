#!/bin/bash

# Update Kubernetes website deployment with ALL files in directory
echo "Updating ConfigMaps with ALL files in current directory..."

# Delete old configmaps
kubectl delete configmap website-content --ignore-not-found=true
kubectl delete configmap nginx-config --ignore-not-found=true

# Create new configmap with all files (excluding directories and hidden files)
kubectl create configmap website-content --from-file=.

# Create nginx config configmap
kubectl create configmap nginx-config --from-file=nginx.conf

echo "Restarting deployment..."
kubectl rollout restart deployment/offline-ai-webserver

echo "Waiting for deployment to complete..."
kubectl rollout status deployment/offline-ai-webserver

echo "Website updated successfully with ALL directory files and no-cache headers!"
echo "Access at: http://localhost:30080 (cache disabled for development)"