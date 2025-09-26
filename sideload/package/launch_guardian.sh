#!/bin/bash
# UIOTA Offline Guard - Edge Device Launcher
# Auto-configures and starts the Guardian Agent system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🛡️  UIOTA Offline Guard - Edge Device Deployment"
echo "================================================="
echo ""

# Check system requirements
echo "🔍 Checking system requirements..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $python_version detected"

# Check available memory
mem_available=$(free -m | awk 'NR==2{printf "%.0f", $7}')
if [ "$mem_available" -lt 512 ]; then
    echo "⚠️  Warning: Low memory ($mem_available MB). 512MB+ recommended"
else
    echo "✅ Memory: $mem_available MB available"
fi

# Check disk space
disk_available=$(df . | awk 'NR==2{print $4}')
if [ "$disk_available" -lt 2097152 ]; then # 2GB in KB
    echo "⚠️  Warning: Low disk space. 2GB+ recommended"
else
    echo "✅ Disk space: $(($disk_available / 1024))MB available"
fi

echo ""

# Install Python dependencies if needed
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    python3 -m pip install --user -q -r requirements.txt
    echo "✅ Dependencies installed"
fi

# Make scripts executable
echo "🔧 Setting up permissions..."
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x scripts/*.py 2>/dev/null || true

# Start the system
echo ""
echo "🚀 Starting UIOTA Offline Guard system..."
echo ""

# Check what interface to use
if [ "$1" = "web" ]; then
    echo "🌐 Starting web dashboard..."
    cd scripts && python3 simple_web_dashboard.py
elif [ "$1" = "demo" ]; then
    echo "🎬 Starting live demonstration..."
    cd scripts && python3 show_agent_demo.py
elif [ "$1" = "test" ]; then
    echo "🧪 Running system tests..."
    cd scripts && python3 test_agent_system.py
else
    echo "🎛️  Starting interactive control interface..."
    cd scripts && python3 interactive_control.py
fi
