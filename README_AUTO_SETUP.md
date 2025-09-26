# 🛡️ UIOTA Offline Guard - Auto Setup

## One-Click Demo Startup 🚀

This project now includes **complete automation** - just download and run! No manual setup required.

### 🎯 Three Ways to Start (Pick Any!)

#### Option 1: Ultimate Starter (Recommended)
```bash
./start_everything.sh
```
This handles **everything** automatically and has multiple fallback methods.

#### Option 2: Python Auto-Launcher
```bash
python3 run.py
```
Simple Python script that runs the full auto-launcher.

#### Option 3: Direct Auto-Launcher
```bash
python3 auto_demo_launcher.py
```
The main orchestrator with all advanced features.

### 🤖 What Gets Automated

#### Auto-Save Agent
- **Continuous saving** of all progress and configurations
- **Automatic backup** system with rollback capability
- **State synchronization** across all other agents
- **File watching** for important configuration changes
- **Recovery** from crashes or interruptions

#### Smooth Setup Agent
- **Dependency detection** and automatic installation
- **Parallel downloads** for faster setup
- **System requirement** checking and validation
- **Container building** and configuration
- **Network setup** and port management
- **Comprehensive testing** of all components

#### Demo Launcher
- **Service orchestration** and health monitoring
- **Automatic startup** of all demo components
- **Real-time monitoring** of service health
- **Graceful shutdown** and cleanup
- **Progress tracking** and status reporting

### 🎉 What You Get

After running any of the startup options:

- **🌐 Web Demo**: http://localhost:8080
- **🧠 ML Toolkit**: http://localhost:8888 (Jupyter notebook)
- **🤖 Discord Bot**: Running in background
- **📊 Monitoring**: Auto-save and health checking
- **🛡️ Security**: Off-Guard zero-trust security
- **📱 Mobile Access**: QR code for easy sharing

### 🔧 Behind the Scenes

The system automatically:

1. **Checks system requirements** (Python, disk space, network)
2. **Installs missing dependencies** (Podman, Python packages)
3. **Downloads ML assets** and datasets
4. **Builds containers** for all services
5. **Configures networking** and firewalls
6. **Starts all services** in correct order
7. **Monitors health** and saves progress continuously
8. **Provides fallbacks** if any step fails

### 🛑 Stopping Everything

```bash
./stop_everything.sh
```
or
```bash
./stop-demos.sh
```

### 📋 Logs and Debugging

- **Auto-save logs**: `~/.uiota/auto_saves/`
- **Setup logs**: `logs/startup_YYYYMMDD_HHMMSS.log`
- **Service logs**: `logs/`
- **Container logs**: `podman logs <container-name>`

### 🌟 Perfect For

- **📚 Classmates**: Share localhost:8080 for instant collaboration
- **✈️ Travel Teams**: Works completely offline once set up
- **🏆 Hackathons**: Zero-config demo setup in minutes
- **🛡️ Security**: Offline-first with cryptographic verification
- **🎓 Learning**: Full ML and web development stack

### 🔄 Automatic Features

#### Continuous Auto-Save
- Saves every 15-30 seconds
- Maintains 100 backup copies
- Checksums for data integrity
- Automatic recovery on restart

#### Smart Setup
- Detects your operating system
- Installs only what's needed
- Parallel processing for speed
- Graceful fallbacks for failures

#### Health Monitoring
- Checks service status every 30 seconds
- Automatically restarts failed services
- Network connectivity monitoring
- Resource usage tracking

### 🎯 Zero Configuration Required

Just run one command and everything happens automatically:
- No need to install dependencies manually
- No need to configure ports or networks
- No need to build containers manually
- No need to start services individually
- No need to manage state or recovery

The system handles everything for you!

### 🏗️ Architecture

```
start_everything.sh
       ↓
auto_demo_launcher.py
       ↓
┌─────────────────┬─────────────────┐
│  auto_save_agent │ smooth_setup_agent │
│  • Continuous   │ • Dependencies  │
│    saving       │ • Downloads     │
│  • Backups      │ • Containers    │
│  • Recovery     │ • Testing       │
└─────────────────┴─────────────────┘
       ↓
   All Services Running
```

### 🚀 Get Started Now

1. **Download** this project
2. **Run** `./start_everything.sh`
3. **Wait** for automatic setup (first time: ~5-10 minutes)
4. **Visit** http://localhost:8080
5. **Share** with your team!

That's it! Everything else is automatic. 🎉