#!/bin/bash

# Auto-update script that runs periodically
cd /media/uiota/USB31FD/offline-ai-landing

echo "$(date): Starting auto-update..."

# Run the update script
./update-k8s.sh

echo "$(date): Auto-update completed"

# Schedule to run again in 30 seconds
(sleep 30 && $0) &