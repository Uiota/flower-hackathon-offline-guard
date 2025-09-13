# Container & Orchestration Guardian Guide
## The Complete Container Architecture for Offline Guard

Welcome, Container Guardian! You're stepping into the role of managing a sovereign, offline-first container architecture designed for the Flower AI hackathon. This guide will help you understand and operate the complete container ecosystem.

## 🛡️ Your Guardian Mission

As a Container/Orchestration Guardian, you're responsible for:
- **Rootless Security**: Implementing defense-in-depth container security
- **Multi-Platform Support**: Ensuring containers work across Intel, ARM, and Apple Silicon
- **Offline-First Operation**: Managing containers without internet dependency
- **Team Collaboration**: Enabling distributed hackathon team development
- **Production Readiness**: Scaling from laptop to cloud deployment

## 🏗️ Architecture Overview

### Core Services Stack

```
┌─────────────────────────────────────────────────────────┐
│                   OFFLINE GUARD                         │
├─────────────────┬─────────────────┬─────────────────────┤
│   Frontend      │   Backend       │   Infrastructure    │
├─────────────────┼─────────────────┼─────────────────────┤
│ • web-demo      │ • guardian-svc  │ • redis             │
│ • nginx         │ • node.js       │ • gitea             │
│ • static sites  │ • REST API      │ • local registry   │
├─────────────────┼─────────────────┼─────────────────────┤
│   ML/AI         │   Team Tools    │   Monitoring        │
├─────────────────┼─────────────────┼─────────────────────┤
│ • ml-toolkit    │ • discord-bot   │ • prometheus        │
│ • jupyter lab   │ • python        │ • grafana           │
│ • flower AI     │ • team coord    │ • jaeger            │
└─────────────────┴─────────────────┴─────────────────────┘
```

### Security Architecture

- **Rootless Containers**: No privileged access required
- **Read-Only Root Filesystems**: Immutable container bases
- **Capability Dropping**: Minimal privileges (principle of least privilege)
- **Security Contexts**: Non-root users, security profiles
- **Network Policies**: Segmented communication paths

## 🚀 Quick Start Commands

### Basic Operations

```bash
# Start the complete stack
./containers/offline-dev.sh start

# Check status
./containers/offline-dev.sh status

# View logs
podman-compose -f containers/podman-compose.yml logs -f

# Stop everything
./containers/offline-dev.sh stop
```

### Build & Deploy

```bash
# Build all containers for multiple architectures
./containers/build-multiarch.sh build

# Start with security-hardened configuration
podman-compose -f containers/podman-security-compose.yml up -d

# Export for offline distribution
./containers/export-bundle.sh hackathon-$(date +%Y%m%d)
```

### Monitoring

```bash
# Start full monitoring stack
./containers/start-monitoring.sh start

# Access dashboards
echo "Grafana: http://localhost:3002 (guardian/hackathon2025)"
echo "Prometheus: http://localhost:9090"
echo "Jaeger: http://localhost:16686"
```

## 📁 Container File Structure

```
containers/
├── podman-compose.yml              # Main development setup
├── podman-security-compose.yml     # Security-hardened setup
├── monitoring-stack.yml            # Complete observability
├── build-multiarch.sh             # Multi-architecture builds
├── offline-registry-setup.sh      # Offline distribution
├── start-monitoring.sh            # Monitoring control
│
├── web-demo/
│   └── Containerfile              # Nginx + static sites
├── discord-bot/
│   └── Containerfile              # Python team coordination
├── ml-toolkit/
│   └── Containerfile              # Jupyter + Flower AI
├── guardian-service/
│   └── Containerfile              # Node.js gamification API
│
├── kubernetes/
│   ├── offline-guard-namespace.yaml    # Basic K8s setup
│   ├── production-security.yaml        # Hardened production
│   └── production-services.yaml        # Production networking
│
└── .github/workflows/
    └── containers.yml             # CI/CD pipeline
```

## 🔧 Container Services Deep Dive

### 1. Web Demo (`web-demo`)
**Purpose**: Static web frontend with offline API simulation  
**Base**: `nginx:alpine`  
**Ports**: `8080:80`  
**Features**:
- Serves multiple demo sites
- Offline API responses
- Security headers
- Guardian branding

```bash
# Build and test
podman build -f containers/web-demo/Containerfile -t offline-guard-web .
podman run -p 8080:80 offline-guard-web
```

### 2. Discord Bot (`discord-bot`)
**Purpose**: Team coordination and hackathon management  
**Base**: `python:3.11-alpine`  
**Features**:
- Team formation commands
- Guardian role assignment
- Flower AI hackathon integration
- Demo mode fallback

```bash
# Run with Discord token
podman run -e DISCORD_BOT_TOKEN=your_token offline-guard-bot
```

### 3. ML Toolkit (`ml-toolkit`)
**Purpose**: Jupyter Lab with Flower AI federated learning  
**Base**: `python:3.11-slim` (multi-stage)  
**Ports**: `8888:8888`  
**Features**:
- Complete ML stack (PyTorch, TensorFlow, scikit-learn)
- Flower AI integration
- Offline-capable notebooks
- Security hardening

```bash
# Access Jupyter Lab
echo "http://localhost:8888/lab"
echo "Token: ${JUPYTER_TOKEN:-offline-guard-ml}"
```

### 4. Guardian Service (`guardian-service`)
**Purpose**: Character gamification and team management API  
**Base**: `node:18-alpine` (multi-stage)  
**Ports**: `3001:3001`  
**Features**:
- Guardian character creation
- Team matching algorithms
- Badge and XP system
- Hackathon leaderboards

```bash
# Test Guardian API
curl http://localhost:3001/api/guardians/roles
curl http://localhost:3001/api/flower/hackathon
```

## 🔒 Security Features

### Container Security Hardening

1. **Rootless Execution**
   ```yaml
   security_opt:
     - no-new-privileges=true
   user: "1001:1001"
   read_only: true
   ```

2. **Capability Management**
   ```yaml
   cap_drop:
     - ALL
   cap_add:
     - NET_BIND_SERVICE  # Only when needed
   ```

3. **Resource Limits**
   ```yaml
   deploy:
     resources:
       limits:
         memory: 2G
         cpus: '2.0'
   ```

4. **Filesystem Security**
   ```yaml
   tmpfs:
     - /tmp:rw,noexec,nosuid,size=100m
   volumes:
     - source:/target:ro,Z  # SELinux labels
   ```

### Network Security

```yaml
networks:
  frontend:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/24
  backend:
    driver: bridge
    internal: true  # No external access
```

## 🌐 Multi-Architecture Support

### Supported Platforms
- `linux/amd64` - Intel/AMD laptops
- `linux/arm64` - Apple Silicon, modern Pi
- `linux/arm/v7` - Older Raspberry Pi

### Build Commands
```bash
# Build for all architectures
./containers/build-multiarch.sh build

# Build for specific platform
buildah build --platform=linux/arm64 -t offline-guard-web:arm64 .

# Quick development build (current arch only)
./containers/build-dev.sh web-demo
```

## 📦 Offline-First Distribution

### Local Registry
```bash
# Setup offline registry
./containers/offline-registry-setup.sh full-setup

# Share with team
./containers/share-registry.sh

# Access registry
echo "http://localhost:5000"
```

### Bundle Creation
```bash
# Create offline bundle
./containers/export-bundle.sh hackathon-bundle

# Import on another system
./containers/import-bundle.sh hackathon-bundle.tar.gz
```

### Team Collaboration
```bash
# Setup as team member
./containers/team-setup.sh

# Sync with peer
./containers/offline-dev.sh sync 192.168.1.100
```

## ☸️ Kubernetes Deployment

### Development Deployment
```bash
# Apply basic manifests
kubectl apply -f containers/kubernetes/offline-guard-namespace.yaml

# Check deployment
kubectl get pods -n offline-guard
```

### Production Deployment
```bash
# Apply security-hardened manifests
kubectl apply -f containers/kubernetes/production-security.yaml
kubectl apply -f containers/kubernetes/production-services.yaml

# Monitor deployment
kubectl get all -n offline-guard-prod
```

### Scaling
```bash
# Scale web frontend
kubectl scale deployment/web-demo-prod --replicas=5 -n offline-guard-prod

# Check HPA
kubectl get hpa -n offline-guard-prod
```

## 📊 Monitoring & Observability

### Access Points
- **Grafana**: `http://localhost:3002` (guardian/hackathon2025)
- **Prometheus**: `http://localhost:9090`
- **Jaeger**: `http://localhost:16686`
- **AlertManager**: `http://localhost:9093`
- **Uptime Kuma**: `http://localhost:3003`

### Key Metrics
```bash
# Container health
curl http://localhost:9090/api/v1/query?query=up

# Resource usage
curl http://localhost:9090/api/v1/query?query=container_memory_usage_bytes

# Guardian service metrics
curl http://localhost:3001/metrics
```

### Alert Management
- Service availability monitoring
- Resource usage thresholds
- Security incident detection
- Guardian team notifications

## 🎯 Hackathon-Specific Features

### Guardian Character System
```bash
# Create Guardian character
curl -X POST http://localhost:3001/api/guardian/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ContainerGuardian",
    "skills": ["Docker", "Kubernetes", "Security"],
    "location": "SF Bay Area",
    "preferences": ["offline-first", "privacy-focused"]
  }'
```

### Flower AI Integration
```bash
# Check Flower AI hackathon info
curl http://localhost:3001/api/flower/hackathon

# Access ML toolkit
echo "Jupyter Lab: http://localhost:8888/lab"
```

### Demo Modes
```bash
# Start in demo mode (no external dependencies)
DISCORD_BOT_TOKEN=demo_mode ./containers/offline-dev.sh start

# Check offline functionality
curl http://localhost:8080/api/
```

## 🚨 Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   podman ps --format "table {{.Names}}\t{{.Ports}}"
   netstat -tulpn | grep :8080
   ```

2. **Storage Issues**
   ```bash
   # Check disk space
   df -h ./data/
   
   # Clean up old images
   podman image prune -a
   ```

3. **Permission Problems**
   ```bash
   # Fix SELinux labels
   restorecon -Rv ./data/
   
   # Check container user
   podman exec -it container-name id
   ```

4. **Network Connectivity**
   ```bash
   # Test container networking
   podman network ls
   podman inspect bridge offline-guard
   ```

### Performance Optimization

1. **Resource Tuning**
   ```bash
   # Monitor resource usage
   podman stats
   
   # Adjust limits in compose file
   mem_limit: 2g
   cpus: 2.0
   ```

2. **Build Optimization**
   ```bash
   # Use build cache
   buildah build --cache-from=localhost/offline-guard-web:latest
   
   # Multi-stage builds for smaller images
   # See Containerfiles for examples
   ```

## 📚 Guardian Resources

### Learning Materials
- [Podman Documentation](https://docs.podman.io/)
- [Buildah User Guide](https://buildah.io/)
- [Container Security Best Practices](https://cloud.google.com/architecture/best-practices-for-operating-containers)
- [Kubernetes Security](https://kubernetes.io/docs/concepts/security/)

### Community
- **Discord**: Join the Offline Guard team channel
- **GitHub**: Contribute to the container architecture
- **Flower AI**: Connect with the federated learning community

### Hackathon Tips
1. **Start with offline-first mindset**
2. **Test on multiple architectures**
3. **Monitor resource usage on laptops**
4. **Plan for unstable internet**
5. **Document Guardian character skills**

## 🌟 Advanced Guardian Techniques

### Custom Container Images
```bash
# Create specialized ML image
FROM offline-guard-ml-toolkit:latest
RUN pip install your-special-ml-library
COPY your-notebooks/ /app/notebooks/
```

### Service Mesh Integration
```bash
# Install Istio for advanced networking
istioctl install --set values.defaultRevision=default

# Apply service mesh policies
kubectl apply -f service-mesh-config.yaml
```

### GitOps Deployment
```bash
# Use ArgoCD for automated deployments
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

---

## 🎊 Congratulations, Container Guardian!

You now have a complete, production-ready container architecture that supports:
- ✅ Rootless security hardening
- ✅ Multi-architecture builds
- ✅ Offline-first operation
- ✅ Team collaboration tools
- ✅ Complete monitoring stack
- ✅ Kubernetes production deployment
- ✅ CI/CD automation
- ✅ Guardian character gamification

**Your mission**: Help hackathon teams build sovereign communication systems that work anywhere, anytime, without dependency on centralized infrastructure.

**Guardian Motto**: "Secure containers, sovereign code, unstoppable teams!"

Ready to lead your team to victory in the Flower AI hackathon? The containers are at your command! 🛡️🚀