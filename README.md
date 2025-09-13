# Offline.ai Website - Local Development Setup

A complete local development environment for the Offline.ai website with support for Podman containers, Kubernetes deployment, and VS Code integration.

## 🚀 Quick Start

### Prerequisites
- **Linux** (Debian/Ubuntu tested)
- **VS Code** with Kubernetes extension
- **curl** and **bash**
- **Node.js** (for dev server option)

### Option 1: One-Click Setup (Recommended)
1. **Clone this repository**
2. **Open in VS Code**
3. **Press `F5`** and select "🚀 Start Kubernetes Website (Localhost)"
4. **Website opens at**: `http://localhost:8081`

### Option 2: Manual Setup
Follow the detailed instructions below.

---

## 📋 Detailed Installation Guide

### Step 1: Install Podman
```bash
sudo apt update && sudo apt install -y podman
```

### Step 2: Install kubectl
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
mkdir -p ~/.local/bin
mv kubectl ~/.local/bin/
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
export PATH=$HOME/.local/bin:$PATH
```

### Step 3: Install minikube
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
chmod +x minikube-linux-amd64
mv minikube-linux-amd64 ~/.local/bin/minikube
```

### Step 4: Start Kubernetes Cluster
```bash
minikube config set rootless true
minikube start --driver=podman
```

### Step 5: Deploy Website
```bash
kubectl create configmap website-content --from-file=index.html --from-file=styles.css --from-file=site.js
kubectl apply -f k8s-webserver-deployment.yaml
kubectl apply -f k8s-webserver-service.yaml
```

---

## 🎯 Development Options

You have **4 ways** to run the website:

### 1. 🚀 Kubernetes (Localhost) - **Recommended**
- **URL**: `http://localhost:8081`
- **Method**: Port-forwarding to localhost
- **Use case**: Production-like environment, localhost URL

### 2. 🌐 Kubernetes (Minikube IP)
- **URL**: `http://192.168.49.2:30080` (varies)
- **Method**: Direct minikube service access
- **Use case**: Testing external access patterns

### 3. 🐳 Podman Container
- **URL**: `http://localhost:8080`
- **Method**: Direct container deployment
- **Use case**: Fast, simple container testing

### 4. 🔧 Dev Server (Node.js)
- **URL**: `http://localhost:3000`
- **Method**: Simple HTTP server
- **Use case**: Quick development, no containers

---

## 🎮 VS Code Integration

### Run Configurations (Press F5)
- **🚀 Start Kubernetes Website (Localhost)** - Best for development
- **🌐 Start Kubernetes Website (Minikube IP)** - External IP access
- **🐳 Start Podman Website** - Fast container option
- **🔧 Dev Server (Simple HTTP)** - Lightweight option

### Available Tasks (Ctrl+Shift+P → "Tasks: Run Task")
- **K8s: Apply Deployment** - Deploy/update to Kubernetes
- **K8s: Delete Deployment** - Remove from Kubernetes  
- **K8s: Get Service URL** - Get current service URL
- **K8s: Update ConfigMap** - Update website files in Kubernetes
- **Podman: Start Container** - Start Podman container
- **Podman: Open Browser** - Open Podman website

### Kubernetes Extension Features
- **Cluster Explorer**: View pods, services, deployments in sidebar
- **Resource Management**: Right-click resources for logs, terminal, etc.
- **YAML Editing**: Syntax highlighting and validation
- **Live Monitoring**: Real-time cluster status

---

## 📁 Project Structure

```
offline-ai-landing/
├── index.html                    # Main website file
├── styles.css                    # Website styles  
├── site.js                       # Website JavaScript
├── k8s-webserver-deployment.yaml # Kubernetes deployment config
├── k8s-webserver-service.yaml    # Kubernetes service config
├── scripts/                      # Launch scripts
│   ├── start-k8s-server-localhost.js   # K8s localhost launcher
│   ├── start-k8s-server-fast.js        # K8s minikube IP launcher
│   ├── start-podman-server.js          # Podman launcher  
│   └── start-dev-server.js             # Dev server launcher
├── .vscode/                      # VS Code configuration
│   ├── launch.json              # Run configurations
│   ├── tasks.json               # Task definitions
│   └── settings.json            # Kubernetes extension settings
├── README.md                     # This file
├── KUBERNETES.md                 # Detailed Kubernetes guide
└── [other website files...]
```

---

## 🔄 Development Workflow

### Making Changes
1. **Edit HTML/CSS/JS files**
2. **Update Kubernetes**: Run "K8s: Update ConfigMap" task
3. **Refresh browser** to see changes

### Updating Containers
```bash
# For Kubernetes
kubectl create configmap website-content --from-file=index.html --from-file=styles.css --from-file=site.js --dry-run=client -o yaml | kubectl apply -f -
kubectl rollout restart deployment offline-ai-webserver

# For Podman  
podman stop offline-ai-web
podman run -d --name offline-ai-web -p 8080:80 -v "$(pwd)":/usr/share/nginx/html:ro,Z docker.io/nginx:alpine
```

---

## 🧹 Cleanup Commands

### Stop Everything
```bash
# Stop Kubernetes deployment
kubectl delete -f k8s-webserver-deployment.yaml -f k8s-webserver-service.yaml

# Stop Podman container
podman stop offline-ai-web && podman rm offline-ai-web

# Stop minikube cluster
minikube stop
```

### Complete Reset
```bash
# Delete minikube cluster
minikube delete

# Remove containers
podman system prune -a

# Remove binaries (optional)
rm ~/.local/bin/kubectl ~/.local/bin/minikube
```

---

## 🔧 Troubleshooting

### Common Issues

**1. "kubectl: command not found"**
```bash
# Add to PATH
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

**2. "Permission denied" on scripts**
```bash
chmod +x scripts/*.js
```

**3. "Podman version too old" warning**
- Warning is safe to ignore
- For latest Podman: follow [official installation](https://podman.io/getting-started/installation.html)

**4. "Port already in use"**
```bash
# Find and kill process using port
sudo lsof -i :8080
sudo kill -9 [PID]
```

**5. VS Code tasks not working**
- Ensure `$HOME/.local/bin` is in your PATH
- Restart VS Code after installing kubectl/minikube

### Verification Commands
```bash
# Check installations
kubectl version --client
minikube version  
podman version

# Check cluster status
kubectl cluster-info
kubectl get pods
kubectl get services

# Check containers
podman ps
```

---

## 🌐 Access URLs Summary

| Method | URL | Use Case |
|--------|-----|----------|
| **Kubernetes (Localhost)** | `http://localhost:8081` | **Best for development** |
| **Kubernetes (Minikube IP)** | `http://192.168.49.2:30080` | External access testing |
| **Podman Container** | `http://localhost:8080` | Fast container testing |
| **Dev Server** | `http://localhost:3000` | Quick prototyping |

---

## 🎯 Recommended Development Flow

1. **Daily Development**: Use "🚀 Start Kubernetes Website (Localhost)"
2. **Quick Testing**: Use "🐳 Start Podman Website"  
3. **Prototyping**: Use "🔧 Dev Server"
4. **Production Testing**: Use "🌐 Start Kubernetes Website (Minikube IP)"

---

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Podman Documentation](https://docs.podman.io/)
- [Minikube Documentation](https://minikube.sigs.k8s.io/docs/)
- [VS Code Kubernetes Extension](https://marketplace.visualstudio.com/items?itemName=ms-kubernetes-tools.vscode-kubernetes-tools)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test with all deployment methods
4. Submit a pull request

---

**Ready to start? Press `F5` in VS Code and select your preferred option!** 🚀
