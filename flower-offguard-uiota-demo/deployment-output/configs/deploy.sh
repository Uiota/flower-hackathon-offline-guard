#!/bin/bash
set -e

echo "🚀 Deploying Federated Learning System to Production"
echo "=================================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "❌ Do not run as root"
    exit 1
fi

# Check environment
echo "🔍 Checking deployment environment..."
if [ ! -f "/home/uiota/projects/offline-guard/flower-offguard-uiota-demo/deployment-output/configs/production.env" ]; then
    echo "❌ Production configuration not found"
    echo "Run: python3 deploy_production.py"
    exit 1
fi

# Load configuration
source /home/uiota/projects/offline-guard/flower-offguard-uiota-demo/deployment-output/configs/production.env

# Create directories
echo "📁 Creating system directories..."
sudo mkdir -p /opt/fl-system /var/log/fl-system
sudo chown -R $USER:$USER /opt/fl-system /var/log/fl-system

# Copy application
echo "📦 Deploying application..."
cp -r . /opt/fl-system/
cd /opt/fl-system

# Setup virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-full.txt

# Install systemd services
if command -v systemctl &> /dev/null; then
    echo "🔧 Installing systemd services..."
    sudo cp /home/uiota/projects/offline-guard/flower-offguard-uiota-demo/deployment-output/configs/systemd/*.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable fl-dashboard fl-agents
    echo "   ✅ Services installed"
fi

# Start services
echo "🚀 Starting services..."
if command -v systemctl &> /dev/null; then
    sudo systemctl start fl-dashboard
    sleep 5
    sudo systemctl start fl-agents
    echo "   ✅ Services started"
else
    echo "   ⚠️  Systemd not available, manual start required"
fi

echo ""
echo "✅ Deployment complete!"
echo "🌐 Dashboard: http://$(hostname):8081"
echo "📊 Monitoring: http://$(hostname):9090"
echo ""
echo "📋 Management commands:"
echo "   sudo systemctl status fl-dashboard"
echo "   sudo systemctl logs -f fl-dashboard"
echo "   sudo systemctl restart fl-dashboard"
