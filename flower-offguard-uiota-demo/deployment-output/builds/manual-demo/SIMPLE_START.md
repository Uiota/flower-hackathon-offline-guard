# 🚀 Simple Start Guide

## Immediate Demo (No Setup Required)

From your current location (`/home/uiota/projects/offline-guard`):

### Method 1: Use the Environment Script
```bash
cd flower-offguard-uiota-demo
./run_environment.sh
```

### Method 2: Direct Basic Demo
```bash
cd flower-offguard-uiota-demo
export OFFLINE_MODE=1
python3 demo_basic.py
```

### Method 3: Use the Parent Start Script
```bash
./start_demo.sh
```

## What Each Method Does

### 🎯 `run_environment.sh` (Recommended)
- **Option 1**: Basic demo (always works)
- **Option 2**: Try to install dependencies automatically
- **Option 3**: Docker demo (if Docker available)
- **Option 4**: Full Docker Compose cluster
- **Option 5**: Show environment info

### 🎯 `demo_basic.py`
- Simulates complete federated learning workflow
- Shows security verification, client training, server aggregation
- No external dependencies required
- Demonstrates all core concepts

### 🎯 `start_demo.sh`
- Handles navigation automatically
- Provides guided setup options
- Fallback to basic demo if dependencies fail

## Expected Output

You should see:
```
🚀 Starting Basic Federated Learning Demo
==================================================
🔒 Running Security Checks:
  ✅ Offline mode enabled
  ✅ Python version 3.11 verified
🔑 Generating Server Keypair:
  ✅ Server keypair generated
🧠 Initializing Global Model:
  ✅ Small CNN model initialized
==================== ROUND 1 ====================
👤 Client 0 Training (Round 1):
  📚 Loading MNIST data partition for client 0
  🎯 Training for 1 epoch...
  📉 Loss: 2.5273 → 1.9824
  ✍️  Parameters signed: 8ad27f18c886ae4d
...
🏁 FEDERATED LEARNING DEMO COMPLETE
✅ All 3 rounds completed successfully
🛡️  Security: All updates cryptographically verified
📊 Final estimated accuracy: ~90% (MNIST)
```

## If You Want Full Dependencies

### Option A: Try Automatic Installation
```bash
cd flower-offguard-uiota-demo
./run_environment.sh
# Choose option 2
```

### Option B: Manual Installation
```bash
cd flower-offguard-uiota-demo
python3 -m pip install --user -r requirements.txt
export OFFLINE_MODE=1
./run_server.sh &
./run_clients.sh 5
```

### Option C: Docker (If Available)
```bash
cd flower-offguard-uiota-demo
docker build -t fl-demo .
docker run --rm -it fl-demo
```

## Troubleshooting

- **"No such file"**: Make sure you're in the right directory
- **"Permission denied"**: Run `chmod +x *.sh`
- **"pip not found"**: Use the basic demo (option 1)
- **Import errors**: Dependencies missing, use basic demo

The basic demo (`demo_basic.py`) **always works** and shows the complete federated learning process!