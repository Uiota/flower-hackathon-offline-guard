#!/bin/bash
set -e

if [ $# -eq 0 ]; then
    NUM_CLIENTS=10
else
    NUM_CLIENTS=$1
fi

echo "Starting $NUM_CLIENTS federated learning clients..."
echo "Each client will connect to localhost:8080"
echo "Press Ctrl+C to stop all clients"

export OFFLINE_MODE=1

# Start clients in background
for i in $(seq 0 $((NUM_CLIENTS-1))); do
    echo "Starting client $i"
    python -m src.client \
        --cid $i \
        --server-address localhost:8080 \
        --dataset mnist \
        --dp off \
        --epochs 1 \
        --batch-size 32 \
        --lr 0.01 &

    # Small delay to prevent race conditions
    sleep 0.5
done

echo "All $NUM_CLIENTS clients started. Waiting for completion..."
wait
echo "All clients finished."