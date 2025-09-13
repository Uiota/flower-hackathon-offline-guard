#!/bin/bash
echo "🛑 Stopping Offline Guard demos..."
podman stop offline-guard-web offline-guard-bot offline-guard-ml 2>/dev/null
podman rm offline-guard-web offline-guard-bot offline-guard-ml 2>/dev/null
echo "✅ All demos stopped and cleaned up!"
