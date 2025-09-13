# 🛡️ UIOTA Offline Guard

**Sovereign AI ecosystem with specialized Guardian agents for offline-first collaboration**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Podman](https://img.shields.io/badge/Container-Podman-892CA0.svg)](https://podman.io/)
[![CPU Only](https://img.shields.io/badge/ML-CPU%20Only-green.svg)](https://pytorch.org/)

## 🚀 Quick Start

```bash
# Clone and start all agents
git clone <your-repo-url>
cd offline-guard
chmod +x start-demos.sh
./start-demos.sh

# Access running agents
# 🌐 Web Demo: http://localhost:8080
# 🧠 ML Toolkit: http://localhost:8888
# 🤖 Discord Bot: podman logs discord-agent
```

## 🤖 Guardian Agent Architecture

### Active Agents
1. **🌐 Web Agent** - Frontend dashboard and landing pages
2. **🤖 Discord Agent** - Team coordination and communication
3. **🧠 ML Agent** - CPU-only federated learning (NO NVIDIA)
4. **🛡️ Security Agent** - Threat monitoring and protection
5. **🌍 DNS Agent** - Decentralized service discovery
6. **📱 Mobile Agent** - Cross-platform offline apps

### Specialized Functions
- **Guardian Classes**: 5 unique Guardian types with specialized skills
- **Federated Learning**: CPU-only ML training without NVIDIA dependencies
- **Offline-First**: All agents work without internet connectivity
- **P2P Mesh**: Guardian-to-Guardian direct communication
- **QR Proofs**: Cryptographic verification via QR codes

## 📁 Project Structure

```
offline-guard/
├── 🌐 frontend/              # React dashboard
├── 🤖 team-building/         # Discord bots & P2P tools
├── 🧠 uiota-federation/      # ML & federated learning
├── 📱 android/               # Android APK (in development)
├── 📱 ios-sideload/          # iOS sideloading support
├── 🐳 containers/            # Podman deployment configs
├── 🌍 monitoring/            # System monitoring stack
├── 🏠 landing-website/       # Marketing landing page
├── ⚖️ judge-showcase/        # Guardian judge demos
├── 🌐 web-demo/              # Live web demonstrations
├── 📚 UIOTA_*.md            # Complete specifications
└── 🚀 start-demos.sh        # One-click agent launcher
```

## 🛡️ Guardian Ecosystem

### Guardian Classes
- **💎 Crypto Guardian** - Blockchain & security specialist
- **🤖 AI Guardian** - Machine learning expert
- **📱 Mobile Master** - Cross-platform app developer
- **🌐 Network Guardian** - Infrastructure & networking
- **👻 Ghost Verifier** - Security & verification

### Core Features
- **XP System**: Guardians gain experience through contributions
- **Team Formation**: Skill-based Guardian collaboration
- **Federated Learning**: Decentralized AI training
- **Offline Verification**: QR-based cryptographic proofs
- **P2P Mesh**: Direct Guardian-to-Guardian communication

## 🔧 Installation & Setup

### Prerequisites
- **Podman** (NOT Docker - we use Podman for security)
- **Python 3.11+**
- **Node.js 18+** (for frontend)
- **Git**

### System Requirements
- **CPU**: Multi-core recommended
- **Memory**: 8GB+ for ML agents
- **Storage**: 20GB+ for containers
- **Network**: Offline-capable

### Installation Steps

1. **Clone Repository**
```bash
git clone <your-repo-url>
cd offline-guard
```

2. **Install Dependencies**
```bash
# Install Podman (Ubuntu/Debian)
sudo apt update && sudo apt install podman

# Install Podman (macOS)
brew install podman
```

3. **Start All Agents**
```bash
chmod +x start-demos.sh
./start-demos.sh
```

4. **Verify Agent Status**
```bash
podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

## 🌐 Agent Access Points

| Agent | URL | Purpose |
|-------|-----|---------|
| **Web Dashboard** | http://localhost:8080 | Main interface |
| **ML Toolkit** | http://localhost:8888 | Jupyter notebooks |
| **Landing Page** | http://localhost:8080/landing | Project info |
| **Judge Demo** | http://localhost:8080/judges | Guardian showcase |

## 🚫 Important Rules

### ❌ NO NVIDIA/CUDA
- All ML uses **CPU-only** packages
- PyTorch CPU edition: `torch==2.0.1+cpu`
- TensorFlow CPU: `tensorflow-cpu`
- No GPU dependencies

### ❌ NO DOCKER
- **Podman only** for security and rootless operation
- All containers use `podman` commands
- No Docker Desktop or Docker Engine

### ✅ Offline-First Design
- All agents work without internet
- Local data storage and caching
- P2P communication protocols
- Offline QR verification

## 🤝 Collaboration Guide

### For Contributors
1. **Fork the repository**
2. **Choose a Guardian class** that matches your skills
3. **Work on agent specializations** from `UIOTA_SUB_AGENT_SPECIFICATIONS.md`
4. **Test with CPU-only, Podman-based setup**
5. **Submit PR with Guardian context**

### Agent Development
- Follow the 6 specialized agent specifications
- Implement Guardian authentication
- Use CPU-only ML libraries
- Deploy with Podman containers
- Maintain offline-first functionality

### Guardian Character System
- Contributors get Guardian NFT characters
- Characters reflect technical specializations
- XP tracking for contributions
- Team formation based on complementary skills

## 📚 Documentation

- **`UIOTA_SUB_AGENT_SPECIFICATIONS.md`** - Complete agent architecture
- **`UIOTA_FRAMEWORK_ARCHITECTURE.md`** - System design
- **`UIOTA_API_SPECIFICATIONS.md`** - API documentation
- **`UIOTA_SECURITY_MODEL.md`** - Security framework
- **`UIOTA_DEPLOYMENT_ARCHITECTURE.md`** - Deployment guide

## 🎯 Use Cases

### 👥 Team Collaboration
- **Classmate coordination** for group projects
- **Travel team management** for hackathons
- **P2P skill sharing** without central servers

### 🏆 Hackathon Ready
- **One-click deployment** with `start-demos.sh`
- **Offline demonstration** capability
- **Guardian character** team building
- **QR proof** verification system

### 🛡️ Digital Sovereignty
- **No big tech dependencies** (no Google, Microsoft, etc.)
- **Offline-first** operation
- **Decentralized** Guardian governance
- **Cryptographic** verification

## 🛑 Stop All Agents

```bash
./stop-demos.sh
```

## 🔧 Troubleshooting

### Container Issues
```bash
# Check container status
podman ps -a

# View logs
podman logs web-agent
podman logs ml-agent
podman logs discord-agent

# Restart agents
./stop-demos.sh && ./start-demos.sh
```

### Port Conflicts
- Web Agent: Port 8080
- ML Agent: Port 8888
- Ensure ports are available before starting

### Performance Issues
- ML Agent requires 4GB+ RAM for optimal performance
- Use `podman stats` to monitor resource usage

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

We welcome Guardian contributors! See CONTRIBUTING.md for guidelines.

**Every contributor gets their own Guardian character NFT reflecting their technical contributions to digital sovereignty.**

---

**🛡️ Built by Guardians, for Guardians. Protecting digital sovereignty one commit at a time.**