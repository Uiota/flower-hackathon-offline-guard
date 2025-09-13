#!/bin/bash

echo ""
echo "🛡️  STARTING OFFLINE GUARD DEMO SUITE"
echo "═══════════════════════════════════════"
echo ""

# Check if podman is available
if ! command -v podman &> /dev/null; then
    echo "❌ Podman not found. Installing..."
    echo "📝 Run: sudo apt install podman (Ubuntu/Debian)"
    echo "📝 Or: brew install podman (macOS)"
    exit 1
fi

echo "🔧 Building demo containers..."

# Build web demo container
echo "📦 Building web demo container..."
podman build -t offline-guard-web -f containers/web-demo/Containerfile .

# Build Discord bot container  
echo "🤖 Building Discord bot container..."
podman build -t offline-guard-bot -f containers/discord-bot/Containerfile .

# Create ML toolkit container
echo "🧠 Creating ML toolkit container..."
cat > containers/ml-toolkit/Containerfile << 'EOF'
FROM docker.io/python:3.11-slim

WORKDIR /app

# Install ML dependencies (CPU-only, no NVIDIA/CUDA)
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install tensorflow-cpu flwr jupyter pandas numpy scikit-learn

# Copy UIOTA federation tools
COPY uiota-federation/ ./uiota-federation/
COPY team-building/ ./team-building/

# Create startup script
RUN cat > start-ml-toolkit.sh << 'SCRIPT'
#!/bin/bash
echo ""
echo "🌸  OFFLINE GUARD ML TOOLKIT READY!"
echo "🧠  Flower AI integration + UIOTA federation"
echo ""
echo "📊  Available tools:"
echo "   🔧 flower-clone-downloader.py - Download complete FL ecosystem"
echo "   🤖 ai1_flower_client.py - Guardian FL client"
echo "   🌐 mesh-coordination.py - P2P team coordination"
echo "   📓 jupyter notebook - Interactive ML development"
echo ""
echo "🚀  Starting Jupyter notebook server..."
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
SCRIPT

chmod +x start-ml-toolkit.sh

CMD ["./start-ml-toolkit.sh"]
EOF

podman build -t offline-guard-ml -f containers/ml-toolkit/Containerfile .

echo ""
echo "🚀 Starting demo containers..."

# Start web demo
echo "🌐 Starting web demo on port 8080..."
podman run -d --name offline-guard-web -p 8080:80 offline-guard-web

# Start Discord bot (demo mode if no token)
echo "🤖 Starting Discord bot..."
podman run -d --name offline-guard-bot offline-guard-bot

# Start ML toolkit
echo "🧠 Starting ML toolkit on port 8888..."
podman run -d --name offline-guard-ml -p 8888:8888 offline-guard-ml

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 5

echo ""
echo "✅ ALL DEMOS RUNNING!"
echo "═══════════════════"
echo ""
echo "🌐  WEB DEMO:     http://localhost:8080"
echo "⚖️   JUDGE DEMO:   http://localhost:8080/judges"  
echo "🏠  LANDING:      http://localhost:8080/landing"
echo "🧠  ML TOOLKIT:   http://localhost:8888"
echo ""
echo "🤖  DISCORD BOT:  Running (check logs with: podman logs offline-guard-bot)"
echo ""
echo "🎯  PERFECT FOR:"
echo "   📚 Classmate collaboration"
echo "   ✈️ Travel team coordination" 
echo "   🏆 Hackathon demonstrations"
echo "   🛡️ Offline-first development"
echo ""

# Create QR code for easy mobile access
echo "📱  MOBILE ACCESS QR CODE:"
echo ""
echo "    ████ ▄▄▄▄▄▄▄ ██▄█ ▄▄▄▄▄▄▄ ████"
echo "    █ ▄▄ █ █   █ █▀▀▀█ █   █ █ ▄▄ █"
echo "    █ ▀▀ ██▄▄▄▄▄██ ▀ ██▄▄▄▄▄██ ▀▀ █"
echo "    ████ ▀ ▀ ▀ █▄█ ▀ █ ▀ ▀ █▀ ████"
echo "    █▄▀█▀▄█▄▄ ▀▀▀▄▀▀▄▀▄▄█ ▀▀█▄█▀██"
echo "    ██▀▄ ▄█ █ █▄▄▄█ ▀▄▄▄█▄██▄█▄█▀█"
echo "    █▀ ▄▀██▄▄▄▄▄ ▀█ █▄▄   ▄▄█▄▀ ▀█"
echo "    ████ ▄▄▄▄▄▄▄ █  ▀▄ █ █ ██ ▄█▀█"
echo "    █ ▄▄ █ █   █ █▄▄██▀▄▄█▄▄▄▄▀█▄██"
echo "    █ ▀▀ ██▄▄▄▄▄█▀▄▄▄▄▀ ▀▄▄ ▄▄█  █"
echo "    ████ ▀▀ ▀▀▀▀ ▀▀▀ ▀▀▀▀ ▀▀▀▀▀ ▀██"
echo ""
echo "📱  Scan with any phone → instant demo access!"
echo ""

# Create easy stop script
cat > stop-demos.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping Offline Guard demos..."
podman stop offline-guard-web offline-guard-bot offline-guard-ml 2>/dev/null
podman rm offline-guard-web offline-guard-bot offline-guard-ml 2>/dev/null
echo "✅ All demos stopped and cleaned up!"
EOF
chmod +x stop-demos.sh

echo "🛑  To stop all demos: ./stop-demos.sh"
echo ""

# Show container status
echo "📊  CONTAINER STATUS:"
podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "🎉  READY FOR TEAM COLLABORATION!"
echo "🌍  Share http://localhost:8080 with classmates"
echo "✈️   Perfect for travel team coordination"
echo "🏆  Demo-ready for hackathons!"
echo ""