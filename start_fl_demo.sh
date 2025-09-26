#!/bin/bash

# Federated Learning Demo Startup Script
# Launches the complete federated learning demonstration

echo "🤖 Federated Learning Demo Launcher"
echo "====================================="
echo ""
echo "🎯 Features:"
echo "  • Real-time federated learning simulation"
echo "  • Interactive dashboard with live metrics"
echo "  • Network visualization and client monitoring"
echo "  • OpenAI-style interface design"
echo "  • LangGraph workflow integration"
echo "  • Advanced graphics and animations"
echo ""

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/flower-offguard-uiota-demo/src:$(pwd)/agents"
export OFFLINE_MODE=1

# Create logs directory
mkdir -p logs

# Set default port
PORT=${1:-8888}

echo "🔧 Starting FL demo on port $PORT..."
echo "🌐 Dashboard will be available at: http://localhost:$PORT"
echo ""

# Try advanced launcher first, fall back to simple launcher
if python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "✅ Using advanced launcher with FastAPI"
    python3 complete_demo_launcher.py --host localhost --port $PORT
else
    echo "📦 Using simple launcher (no external dependencies)"
    python3 simple_demo_launcher.py --host localhost --port $PORT
fi

echo ""
echo "🔄 Demo stopped"
echo "✅ Cleanup complete"