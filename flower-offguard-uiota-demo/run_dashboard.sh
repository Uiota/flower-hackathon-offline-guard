#!/bin/bash
set -e

echo "🚀 Starting State-of-the-Art Federated Learning Dashboard"
echo "========================================================="
echo ""
echo "🌟 Features:"
echo "   • Real-time training visualization"
echo "   • Live accuracy and loss charts"
echo "   • Client status monitoring"
echo "   • Modern glass morphism UI"
echo "   • Animated particle background"
echo "   • Responsive design"
echo ""
echo "🔧 Dashboard will start on: http://localhost:8080"
echo "📊 Press Ctrl+C to stop the server"
echo ""

export OFFLINE_MODE=1

# Make executable
chmod +x dashboard_server.py

# Start the dashboard server
python3 dashboard_server.py