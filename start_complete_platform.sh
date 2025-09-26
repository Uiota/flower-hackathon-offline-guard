#!/bin/bash

# Complete FL Platform Startup Script
# Launches the integrated federated learning platform with AI assistant

clear

echo "🤖 Federated Learning Guardian Platform"
echo "========================================="
echo ""
echo "🎯 Complete AI-Powered Federated Learning System"
echo ""
echo "✨ Features:"
echo "  • ChatGPT-like AI assistant explaining FL concepts"
echo "  • Real-world federated learning with PyTorch"
echo "  • Live AI integration (OpenAI, Anthropic, etc.)"
echo "  • Privacy-first offline mode option"
echo "  • MCP server for AI service coordination"
echo "  • LangGraph workflow visualization"
echo "  • Interactive network topology"
echo "  • Advanced graphics and animations"
echo ""
echo "🛡️ Privacy & Security:"
echo "  • Request offline mode for air-gapped environments"
echo "  • GDPR/HIPAA compliance features"
echo "  • End-to-end encryption"
echo "  • Local data processing only"
echo ""

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/flower-offguard-uiota-demo/src:$(pwd)/agents"
export OFFLINE_MODE=0  # Can be changed via web interface

# Create necessary directories
mkdir -p logs
mkdir -p ~/.uiota

# Parse command line arguments
OFFLINE_MODE=false
PORT=8888
HOST="localhost"

while [[ $# -gt 0 ]]; do
    case $1 in
        --offline)
            OFFLINE_MODE=true
            export OFFLINE_MODE=1
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --offline     Start in offline mode (maximum privacy)"
            echo "  --port PORT   Specify port (default: 8888)"
            echo "  --host HOST   Specify host (default: localhost)"
            echo "  -h, --help    Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$OFFLINE_MODE" = true ]; then
    echo "🛡️ OFFLINE MODE ENABLED"
    echo "  • Maximum privacy and security"
    echo "  • No external network connections"
    echo "  • All processing happens locally"
    echo "  • Perfect for sensitive environments"
    echo ""
fi

echo "🔧 Checking system requirements..."

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  ✅ Python $PYTHON_VERSION detected"

# Check for required Python packages
echo "📦 Checking dependencies..."

# Core dependencies (always required)
python3 -c "import http.server, json, threading, webbrowser" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✅ Core Python libraries available"
else
    echo "  ❌ Missing core Python libraries"
    exit 1
fi

# Optional ML dependencies
ML_AVAILABLE=false
python3 -c "import torch, numpy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✅ PyTorch available - real FL training enabled"
    ML_AVAILABLE=true
else
    echo "  ⚠️ PyTorch not available - using simulation mode"
    echo "     Install with: pip install torch numpy"
fi

# Optional AI dependencies
AI_AVAILABLE=false
python3 -c "import openai" 2>/dev/null && python3 -c "import anthropic" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✅ AI libraries available - live AI integration enabled"
    AI_AVAILABLE=true
else
    echo "  ⚠️ AI libraries not available - using mock responses"
    echo "     Install with: pip install openai anthropic langchain"
fi

echo ""
echo "🚀 Starting Federated Learning Platform..."
echo ""
echo "📊 System Configuration:"
echo "  • Host: $HOST"
echo "  • Port: $PORT"
echo "  • Offline Mode: $OFFLINE_MODE"
echo "  • ML Training: $([ "$ML_AVAILABLE" = true ] && echo "Real (PyTorch)" || echo "Simulation")"
echo "  • AI Integration: $([ "$AI_AVAILABLE" = true ] && echo "Live APIs" || echo "Mock Responses")"
echo ""

# Determine which launcher to use
if [ -f "integrated_fl_launcher.py" ]; then
    LAUNCHER="integrated_fl_launcher.py"
    echo "🎯 Using integrated launcher with AI assistant"
elif [ -f "enhanced_live_demo.py" ]; then
    LAUNCHER="enhanced_live_demo.py"
    echo "🎯 Using enhanced live demo launcher"
elif [ -f "simple_demo_launcher.py" ]; then
    LAUNCHER="simple_demo_launcher.py"
    echo "🎯 Using simple demo launcher"
else
    echo "❌ No launcher found! Please ensure the launcher files exist."
    exit 1
fi

echo "🌐 Platform will be available at: http://$HOST:$PORT"
echo "📱 Browser will open automatically"
echo ""
echo "🤖 AI Assistant Features:"
echo "  • Explains federated learning concepts"
echo "  • Guides you through the platform"
echo "  • Helps configure privacy settings"
echo "  • Provides real-time insights"
echo ""
echo "🛡️ Privacy Controls:"
echo "  • Request offline mode from the web interface"
echo "  • Configure API keys securely"
echo "  • Control data sharing preferences"
echo ""

# Add delay to let user read the information
echo "Starting in 3 seconds... (Press Ctrl+C to cancel)"
sleep 1
echo "Starting in 2 seconds..."
sleep 1
echo "Starting in 1 second..."
sleep 1

echo ""
echo "🚀 LAUNCHING PLATFORM..."
echo "========================"

# Build launch command
LAUNCH_CMD="python3 $LAUNCHER --host $HOST --port $PORT"

if [ "$OFFLINE_MODE" = true ]; then
    LAUNCH_CMD="$LAUNCH_CMD --offline"
fi

# Launch the platform
echo "📡 Executing: $LAUNCH_CMD"
echo ""

# Execute the launch command
exec $LAUNCH_CMD

# This should not be reached, but just in case
echo ""
echo "🔄 Platform stopped"
echo "✅ Cleanup complete"