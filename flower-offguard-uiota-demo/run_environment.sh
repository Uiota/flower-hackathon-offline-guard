#!/bin/bash
set -e

echo "🌟 Federated Learning Environment Setup"
echo "======================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run basic demo
run_basic_demo() {
    echo "🚀 Running Basic Demo (No Dependencies)"
    echo "======================================="
    export OFFLINE_MODE=1
    python3 demo_basic.py
}

# Function to try pip installation
try_pip_install() {
    echo "📦 Attempting Package Installation"
    echo "=================================="

    # Try different pip commands
    if command_exists pip3; then
        echo "Using pip3..."
        pip3 install --user -r requirements.txt
        return 0
    elif command_exists pip; then
        echo "Using pip..."
        pip install --user -r requirements.txt
        return 0
    elif python3 -m pip --version >/dev/null 2>&1; then
        echo "Using python3 -m pip..."
        python3 -m pip install --user -r requirements.txt
        return 0
    else
        echo "❌ No pip found"
        return 1
    fi
}

# Function to run full demo
run_full_demo() {
    echo "🚀 Running Full Federated Learning Demo"
    echo "======================================="
    export OFFLINE_MODE=1

    echo "🖥️  Starting FL Server..."
    ./run_server.sh &
    SERVER_PID=$!

    echo "⏳ Waiting for server startup..."
    sleep 5

    echo "👥 Starting FL Clients..."
    ./run_clients.sh 3

    # Cleanup
    echo "🧹 Cleaning up..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true

    echo "✅ Full demo completed!"
}

# Function to run Docker demo
run_docker_demo() {
    echo "🐳 Running Docker Demo"
    echo "===================="

    if ! command_exists docker; then
        echo "❌ Docker not found"
        return 1
    fi

    echo "🏗️  Building Docker image..."
    docker build -t fl-demo .

    echo "🚀 Running containerized demo..."
    docker run --rm -it fl-demo

    echo "✅ Docker demo completed!"
}

# Function to run Docker Compose demo
run_docker_compose_demo() {
    echo "🐳 Running Docker Compose Full Demo"
    echo "==================================="

    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        echo "❌ Docker Compose not found"
        return 1
    fi

    echo "🏗️  Starting federated learning cluster..."
    if command_exists docker-compose; then
        docker-compose up --build fl-server fl-client-1 fl-client-2 fl-client-3
    else
        docker compose up --build fl-server fl-client-1 fl-client-2 fl-client-3
    fi

    echo "✅ Docker Compose demo completed!"
}

# Main menu
echo "🎯 Choose Environment Option:"
echo "1. Basic Demo (No Dependencies) - Always Works"
echo "2. Try to Install Dependencies + Full Demo"
echo "3. Docker Demo (if Docker available)"
echo "4. Docker Compose Full Demo (if Docker Compose available)"
echo "5. Show Environment Info"

read -p "Enter choice (1-5): " choice

case $choice in
    1)
        run_basic_demo
        ;;
    2)
        if try_pip_install; then
            echo "✅ Dependencies installed successfully!"
            run_full_demo
        else
            echo "❌ Failed to install dependencies. Running basic demo instead..."
            run_basic_demo
        fi
        ;;
    3)
        if run_docker_demo; then
            echo "✅ Docker demo successful!"
        else
            echo "❌ Docker not available. Running basic demo..."
            run_basic_demo
        fi
        ;;
    4)
        if run_docker_compose_demo; then
            echo "✅ Docker Compose demo successful!"
        else
            echo "❌ Docker Compose not available. Trying basic demo..."
            run_basic_demo
        fi
        ;;
    5)
        echo "🔍 Environment Information:"
        echo "=========================="
        echo "Python: $(python3 --version 2>&1)"
        echo "Python Path: $(which python3 2>&1)"
        echo "Pip3: $(command_exists pip3 && echo "✅ Available" || echo "❌ Not found")"
        echo "Docker: $(command_exists docker && echo "✅ Available" || echo "❌ Not found")"
        echo "Docker Compose: $(command_exists docker-compose && echo "✅ Available" || echo "❌ Not found")"
        echo "Current Dir: $(pwd)"
        echo "OFFLINE_MODE: ${OFFLINE_MODE:-Not set}"
        echo ""
        echo "📁 Available Files:"
        ls -1 *.py *.sh 2>/dev/null | head -10
        ;;
    *)
        echo "❌ Invalid choice. Running basic demo..."
        run_basic_demo
        ;;
esac

echo ""
echo "🎉 Environment demo completed!"
echo "💡 The basic demo always works and shows the complete FL workflow."
echo "📚 For production use, install the dependencies or use Docker."