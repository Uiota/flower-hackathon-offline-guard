#!/bin/bash

echo "Starting required services after reboot..."

# Start podman socket
systemctl --user start podman.socket
systemctl --user enable podman.socket

# Start minikube with offline mode and skip pull
echo "Starting minikube..."
minikube start --driver=podman --pull-policy=Never --force

echo "Services started successfully!"