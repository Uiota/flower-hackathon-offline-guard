#!/bin/bash
set -e

echo "🚀 Starting Federated Learning Demo"
echo "=================================="

# Navigate to demo directory
cd flower-offguard-uiota-demo

echo "📍 Current directory: $(pwd)"
echo "📁 Available files:"
ls -1 *.py *.sh *.md 2>/dev/null | head -10

echo ""
echo "🎯 Choose demo option:"
echo "1. Basic Demo (no dependencies) - RECOMMENDED"
echo "2. Try to install dependencies and run full demo"
echo "3. Just show project structure"

read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🎬 Running Basic Demo..."
        export OFFLINE_MODE=1
        python3 demo_basic.py
        ;;
    2)
        echo ""
        echo "📦 Attempting to install dependencies..."
        export OFFLINE_MODE=1
        if command -v pip3 &> /dev/null; then
            pip3 install --user -r requirements.txt
        elif python3 -m pip --version &> /dev/null; then
            python3 -m pip install --user -r requirements.txt
        else
            echo "❌ pip not available. Running basic demo instead..."
            python3 demo_basic.py
            exit 0
        fi

        echo "🚀 Starting full demo..."
        echo "🖥️  Starting server in background..."
        ./run_server.sh &
        SERVER_PID=$!

        echo "⏳ Waiting for server to start..."
        sleep 5

        echo "👥 Starting clients..."
        ./run_clients.sh 5

        # Clean up
        kill $SERVER_PID 2>/dev/null || true
        ;;
    3)
        echo ""
        echo "📁 Project Structure:"
        find . -type f -name "*.py" -o -name "*.sh" -o -name "*.md" -o -name "*.txt" | sort
        echo ""
        echo "📊 Statistics:"
        echo "- Total files: $(find . -type f | wc -l)"
        echo "- Python files: $(find . -name '*.py' | wc -l)"
        echo "- Documentation: $(find . -name '*.md' | wc -l)"
        ;;
    *)
        echo "❌ Invalid choice. Running basic demo..."
        export OFFLINE_MODE=1
        python3 demo_basic.py
        ;;
esac

echo ""
echo "✅ Demo completed!"