# Quick Start Commands

## Step 1: Navigate to Demo Directory
```bash
cd /home/uiota/projects/offline-guard/flower-offguard-uiota-demo
```

## Step 2: Run the Basic Demo (No Dependencies Required)
```bash
export OFFLINE_MODE=1
python3 demo_basic.py
```

## Step 3: For Full Demo with Dependencies

### Option A: Try with system Python3
```bash
export OFFLINE_MODE=1
python3 -m pip install --user flwr torch torchvision numpy cryptography pydantic pytest scipy
```

### Option B: Use the basic demo (Recommended)
The `demo_basic.py` shows the complete workflow without requiring external dependencies.

## Current Location Commands
From `/home/uiota/projects/offline-guard/`:
```bash
# Navigate to demo
cd flower-offguard-uiota-demo

# Run basic demo
export OFFLINE_MODE=1
python3 demo_basic.py

# Or try the shell scripts (after installing deps)
./run_server.sh    # Terminal 1
./run_clients.sh 5 # Terminal 2
```

## Troubleshooting
- If `pip` not found: Use `python3 -m pip` instead
- If permission denied: Use `chmod +x run_*.sh`
- If imports fail: Use `demo_basic.py` which works without dependencies