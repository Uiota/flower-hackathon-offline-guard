#!/bin/bash
#
# Memory Guardian Quick Start Script
# Launches the cognitive health and property protection application
#

set -e

echo "================================================================================"
echo "🧠 MEMORY GUARDIAN - Quick Start"
echo "================================================================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check dependencies
echo ""
echo "📦 Checking dependencies..."

MISSING_DEPS=()

if ! python3 -c "import cryptography" &> /dev/null; then
    MISSING_DEPS+=("cryptography")
fi

if ! python3 -c "import flask" &> /dev/null; then
    MISSING_DEPS+=("flask")
fi

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo ""
    echo "❌ Missing required dependencies:"
    for dep in "${MISSING_DEPS[@]}"; do
        echo "   - $dep"
    done
    echo ""
    echo "📦 Install with:"
    echo "   pip3 install ${MISSING_DEPS[*]}"
    echo ""
    read -p "Install now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip3 install "${MISSING_DEPS[@]}"
        echo "✅ Dependencies installed!"
    else
        echo "Please install dependencies manually and run this script again."
        exit 1
    fi
else
    echo "✅ All dependencies installed"
fi

# Set offline mode
export OFFLINE_MODE=1

echo ""
echo "🔒 Offline Mode: ENABLED"
echo "🔐 Quantum-Safe Encryption: ACTIVE"
echo "🌐 Federated Learning: READY"
echo ""

# Parse arguments
MODE="web"
PORT=8090

while [[ $# -gt 0 ]]; do
    case $1 in
        --cli)
            MODE="cli"
            shift
            ;;
        --agents)
            MODE="agents"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cli        Launch in CLI mode"
            echo "  --agents     Launch agent system"
            echo "  --port PORT  Custom port for web server (default: 8090)"
            echo "  --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Launch web interface"
            echo "  $0 --cli              # Launch CLI mode"
            echo "  $0 --agents           # Launch agent system"
            echo "  $0 --port 8080        # Web interface on port 8080"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Launch application
echo "🚀 Launching Memory Guardian..."
echo ""

if [ "$MODE" == "cli" ]; then
    echo "💻 Starting CLI mode..."
    python3 launch_memory_guardian.py --cli
elif [ "$MODE" == "agents" ]; then
    echo "🤖 Starting agent system..."
    python3 launch_memory_guardian.py --agents
else
    echo "🌐 Starting web interface on port $PORT..."
    echo "   URL: http://localhost:$PORT"
    echo ""
    python3 launch_memory_guardian.py --port "$PORT"
fi