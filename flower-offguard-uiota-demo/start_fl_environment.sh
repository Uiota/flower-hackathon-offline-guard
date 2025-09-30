#!/bin/bash
set -e

echo "🚀 Starting Federated Learning Environment"
echo "=========================================="

# Load environment
if [ -f .env ]; then
    source .env
    echo "✅ Environment loaded"
else
    echo "⚠️  No .env file found, using defaults"
    export OFFLINE_MODE=1
fi

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found"
    echo "Run: python3 setup_environment.py"
    exit 1
fi

# Check dependencies
echo "🔍 Checking dependencies..."
python -c "import sys; print(f'Python: {sys.version}')"

# Start the complete demo
echo "🎯 Starting complete FL demo..."
exec ./run_all_demos.sh
