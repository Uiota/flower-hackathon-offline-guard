#!/bin/bash

# Advanced FL Demo Startup Script
# Launches the complete federated learning demonstration with enhanced graphics

echo "🚀 Advanced Federated Learning Demo Launcher"
echo "============================================="
echo ""
echo "🤖 Initializing advanced FL demonstration..."
echo "📊 Setting up OpenAI-style dashboard interface"
echo "💻 Preparing integrated terminal functionality"
echo "🌐 Starting federated learning simulation"
echo ""

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/flower-offguard-uiota-demo/src:$(pwd)/agents"
export OFFLINE_MODE=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Check for Python dependencies
echo "🔧 Checking dependencies..."

# Check if required packages are installed
python3 -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing required web dependencies..."
    python3 -m pip install fastapi uvicorn websockets python-multipart jinja2
fi

# Check for additional ML dependencies
python3 -c "import psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing system monitoring dependencies..."
    python3 -m pip install psutil
fi

echo "✅ Dependencies check complete"
echo ""

# Display startup information
echo "🌟 Advanced FL Demo Features:"
echo "   • Real-time federated learning simulation"
echo "   • Interactive terminal with command processing"
echo "   • OpenAI-style dashboard interface"
echo "   • LangGraph workflow visualization"
echo "   • Advanced network topology display"
echo "   • Live metrics and performance monitoring"
echo "   • Model export and client management"
echo ""

# Set default port if not specified
PORT=${1:-8888}

echo "🌐 Starting demo on port $PORT..."
echo "📱 Dashboard will be available at: http://localhost:$PORT"
echo ""

# Start the advanced demo launcher
python3 complete_demo_launcher.py --host localhost --port $PORT

echo ""
echo "🔄 Demo launcher stopped"
echo "✅ Cleanup complete"