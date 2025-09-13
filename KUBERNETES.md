# Kubernetes Development Setup

## Overview
Your Offline.ai website is now deployed on a local Kubernetes cluster using minikube and Podman.

## Access URLs
- **Podman Container**: http://localhost:8080
- **Kubernetes Service**: http://127.0.0.1:32829 (or run `minikube service offline-ai-webserver-service --url`)

## VS Code Integration

### Available Tasks (Ctrl+Shift+P → "Tasks: Run Task")
- **K8s: Apply Deployment** - Deploy/update the website
- **K8s: Delete Deployment** - Remove the deployment
- **K8s: Get Service URL** - Get the current service URL
- **K8s: Update ConfigMap** - Update website files in Kubernetes

### Kubernetes Extension
With the Kubernetes extension installed, you can:
1. View cluster resources in the sidebar
2. Right-click on pods/services to inspect logs, terminal, etc.
3. Edit YAML files with syntax highlighting and validation

## Manual Commands

### Basic Operations
```bash
# Check cluster status
~/.local/bin/kubectl cluster-info

# View pods
~/.local/bin/kubectl get pods

# View services
~/.local/bin/kubectl get services

# Get service URL
~/.local/bin/minikube service offline-ai-webserver-service --url
```

### Update Website Content
```bash
# Update ConfigMap with new files
~/.local/bin/kubectl create configmap website-content \
  --from-file=index.html \
  --from-file=styles.css \
  --from-file=site.js \
  --dry-run=client -o yaml | ~/.local/bin/kubectl apply -f -

# Restart deployment to pick up changes
~/.local/bin/kubectl rollout restart deployment offline-ai-webserver
```

### Cleanup
```bash
# Stop the Podman container
podman stop offline-ai-web

# Delete Kubernetes deployment
~/.local/bin/kubectl delete -f k8s-webserver-deployment.yaml -f k8s-webserver-service.yaml

# Stop minikube
~/.local/bin/minikube stop
```

## Files
- `k8s-webserver-deployment.yaml` - Kubernetes deployment configuration
- `k8s-webserver-service.yaml` - Kubernetes service configuration
- `.vscode/settings.json` - VS Code Kubernetes extension settings
- `.vscode/tasks.json` - VS Code tasks for Kubernetes operations
- `.vscode/launch.json` - VS Code debugging configuration

## Notes
- The deployment uses nginx:alpine to serve static files
- Website files are stored in a ConfigMap for easy updates
- Service is exposed via NodePort on port 30080
- Both Podman and Kubernetes deployments serve the same content