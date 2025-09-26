#!/bin/bash

echo "🛡️ UIOTA.Space Portal Launcher"
echo "================================"
echo ""

# Check if server is running
if curl -s http://localhost:8080/api/status > /dev/null 2>&1; then
    echo "✅ Server is running on port 8080"
    echo ""
    echo "🌐 Available interfaces:"
    echo "   Main Portal: http://localhost:8080"
    echo "   Offline AI:  http://localhost:8080/offline-ai/index.html"
    echo "   FL Dashboard: http://localhost:8080/flower-offguard-uiota-demo/dashboard.html"
    echo "   Guardian Portal: http://localhost:8080/web-demo/portal.html"
    echo ""

    # Try to open in browser
    if command -v xdg-open > /dev/null; then
        echo "🚀 Opening portal in browser..."
        xdg-open http://localhost:8080
    elif command -v firefox > /dev/null; then
        echo "🚀 Opening portal in Firefox..."
        firefox http://localhost:8080 &
    else
        echo "📱 Please open http://localhost:8080 in your web browser"
    fi
else
    echo "❌ Server not running. Starting it now..."
    echo ""
    python3 uiota_space_integration_simple.py &
    SERVER_PID=$!
    echo "⏳ Waiting for server to start..."
    sleep 3

    if curl -s http://localhost:8080/api/status > /dev/null 2>&1; then
        echo "✅ Server started successfully!"
        echo "🌐 Portal: http://localhost:8080"

        if command -v xdg-open > /dev/null; then
            xdg-open http://localhost:8080
        else
            echo "📱 Please open http://localhost:8080 in your web browser"
        fi
    else
        echo "❌ Failed to start server"
    fi
fi