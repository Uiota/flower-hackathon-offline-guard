# Federated Learning Demo: Flower + Off-Guard + UIOTA Mesh

A complete federated learning demonstration combining Flower FL framework with Zero-Trust security (Off-Guard) and offline mesh synchronization (UIOTA mock).

## Quick Start

```bash
# Setup environment
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
export OFFLINE_MODE=1  # Enable safe mode

# Start server (terminal 1)
bash run_server.sh

# Start simulated clients (terminal 2)
bash run_clients.sh 10

# Optional: Add real client from another machine
export OFFLINE_MODE=1
python -m src.client --server_address 192.168.1.10:8080 --cid real1
```

## Architecture Overview

**Flower Framework**: Orchestrates federated learning across distributed clients
**Off-Guard Security**: Zero-trust verification with cryptographic signatures and environment checks
**UIOTA Mesh Mock**: Simulates offline/delayed synchronization via local queue system

## Key Features

- **Security-First**: All model updates cryptographically signed and verified
- **Offline Capable**: Mock mesh transport works without internet connectivity
- **Differential Privacy**: Optional client-side privacy protection via Opacus
- **Fault Tolerant**: Handles client dropouts, network delays, and Byzantine failures
- **Configurable**: Extensive CLI options for research experimentation

## Demo Scenarios

1. **Basic Federation**: `bash run_server.sh && bash run_clients.sh 5`
2. **With Privacy**: Add `--dp on --dp-noise 0.8` to client commands
3. **Network Issues**: Use `--dropout-pct 0.2 --latency-ms 200 --jitter-ms 100`
4. **Multi-Machine**: Run server on one machine, clients connect via IP address

## Expected Results

- **Accuracy**: >90% on MNIST within 5 rounds (2-3 minutes on CPU)
- **Security**: Tampered updates rejected with signature verification failure
- **Resilience**: Training continues despite 20%+ client dropouts
- **Privacy**: DP adds noise while maintaining reasonable accuracy

## File Structure

```
flower-offguard-uiota-demo/
├── README.md              # This file
├── requirements.txt       # Pinned dependencies
├── run_server.sh         # Server startup script
├── run_clients.sh        # Multi-client startup script
├── src/
│   ├── server.py         # Flower server with Off-Guard integration
│   ├── client.py         # Flower client with DP and security
│   ├── strategy_custom.py # Custom aggregation strategy
│   ├── datasets.py       # MNIST/CIFAR-10 data handling
│   ├── models.py         # CNN model definitions
│   ├── guard.py          # Zero-trust security module
│   ├── mesh_sync.py      # UIOTA mesh transport mock
│   └── utils.py          # Shared utilities and logging
└── tests/
    ├── test_guard.py     # Security module tests
    └── test_mesh_sync.py # Mesh transport tests
```