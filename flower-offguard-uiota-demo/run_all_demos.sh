#!/bin/bash
set -e

echo "🎯 COMPLETE FEDERATED LEARNING DEMO SUITE"
echo "🔒 Off-Guard Security • 🤖 Real Agents • 📊 Live Dashboard"
echo "============================================================="

export OFFLINE_MODE=1

# Clean up any existing processes
echo "🧹 Cleaning up any existing demo processes..."
pkill -f "python.*demo\|python.*dashboard\|python.*fl_agent" 2>/dev/null || true
sleep 2

# Create logs directory
mkdir -p logs

echo ""
echo "🚀 Starting all demo components..."
echo ""

# 1. Enhanced Dashboard with FL Agents (Main Interface)
echo "📊 Starting Enhanced Dashboard + FL Agents..."
python3 dashboard_with_agents.py > logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "   PID: $DASHBOARD_PID"
sleep 3

# 2. Fast Real Demo (Standalone FL Training)
echo "⚡ Starting Fast Real FL Demo..."
python3 demo_fast_real.py > logs/fast_real.log 2>&1 &
FAST_REAL_PID=$!
echo "   PID: $FAST_REAL_PID"
sleep 2

# 3. Complete Working FL (Advanced Simulation)
echo "🔬 Starting Complete Working FL..."
python3 final-working-fl.py > logs/working_fl.log 2>&1 &
WORKING_FL_PID=$!
echo "   PID: $WORKING_FL_PID"
sleep 2

# 4. Network FL Simulator (Network Analysis)
echo "🌐 Starting Network FL Simulator..."
python3 network-fl-simulator.py > logs/network_sim.log 2>&1 &
NETWORK_SIM_PID=$!
echo "   PID: $NETWORK_SIM_PID"
sleep 2

# 5. Basic Demo (Quick Reference)
echo "🎮 Starting Basic FL Demo..."
python3 demo_basic.py > logs/basic_demo.log 2>&1 &
BASIC_DEMO_PID=$!
echo "   PID: $BASIC_DEMO_PID"

echo ""
echo "⏳ Waiting for components to initialize..."
sleep 8

echo ""
echo "✅ ALL DEMO COMPONENTS LAUNCHED!"
echo "============================================================="
echo ""
echo "🔗 ACCESS INFORMATION:"
echo "📊 Main Dashboard: http://localhost:8081"
echo "   • Real-time FL monitoring"
echo "   • Live agent metrics"
echo "   • Security status"
echo "   • Interactive controls"
echo ""
echo "🖥️  Running Processes:"
echo "   1. Enhanced Dashboard + Agents: PID $DASHBOARD_PID"
echo "   2. Fast Real FL Demo: PID $FAST_REAL_PID"
echo "   3. Complete Working FL: PID $WORKING_FL_PID"
echo "   4. Network FL Simulator: PID $NETWORK_SIM_PID"
echo "   5. Basic FL Demo: PID $BASIC_DEMO_PID"
echo ""
echo "📁 Log Files:"
echo "   ls logs/  # View all log files"
echo "   tail -f logs/dashboard.log  # Monitor dashboard"
echo ""
echo "🧪 Test Commands:"
echo "   python3 test_full_system.py  # Comprehensive test"
echo "   curl http://localhost:8081/api/metrics | jq  # Live metrics"
echo ""
echo "⏹️  To Stop All:"
echo "   pkill -f \"python.*demo\\|python.*dashboard\\|python.*fl_agent\""
echo ""

# Wait and monitor
echo "🎮 MONITORING MODE - Press Ctrl+C to stop all components"
echo "============================================================="

# Function to check if processes are still running
check_processes() {
    local running=0

    for pid in $DASHBOARD_PID $FAST_REAL_PID $WORKING_FL_PID $NETWORK_SIM_PID $BASIC_DEMO_PID; do
        if kill -0 $pid 2>/dev/null; then
            ((running++))
        fi
    done

    echo "Status: $running/5 components running"
}

# Monitor every 30 seconds
while true; do
    sleep 30
    check_processes
done

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 Stopping all demo components..."

    for pid in $DASHBOARD_PID $FAST_REAL_PID $WORKING_FL_PID $NETWORK_SIM_PID $BASIC_DEMO_PID; do
        if kill -0 $pid 2>/dev/null; then
            echo "Stopping PID $pid..."
            kill $pid 2>/dev/null || true
        fi
    done

    # Force cleanup
    sleep 3
    pkill -f "python.*demo\|python.*dashboard\|python.*fl_agent" 2>/dev/null || true

    echo "✅ All components stopped"
    exit 0
}

# Trap signals
trap cleanup SIGINT SIGTERM