#!/bin/bash
set -e

echo "Starting Federated Learning Server..."
echo "Server will run on localhost:8080"
echo "Press Ctrl+C to stop"

export OFFLINE_MODE=1

python -m src.server \
  --rounds 5 \
  --clients-per-round 10 \
  --strategy fedavg \
  --dataset mnist \
  --dp off \
  --latency-ms 50 \
  --jitter-ms 25 \
  --dropout-pct 0.1 \
  --server-address localhost:8080