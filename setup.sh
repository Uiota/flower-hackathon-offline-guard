#!/bin/bash

echo "🛡️  UIOTA OFFLINE GUARD SETUP"
echo "==============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check system requirements
echo "🔍 Checking system requirements..."
echo ""

# Check OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

print_info "Operating System: $MACHINE"

# Check for required commands
check_command() {
    if command -v "$1" &> /dev/null; then
        print_status "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

echo ""
echo "📋 Checking prerequisites..."

# Critical requirements
REQUIREMENTS_MET=true

if ! check_command "podman"; then
    print_error "Podman is required but not installed"
    echo "   Install with: sudo apt install podman (Ubuntu) or brew install podman (Mac)"
    REQUIREMENTS_MET=false
fi

if ! check_command "python3"; then
    print_error "Python 3 is required but not installed"
    REQUIREMENTS_MET=false
fi

if ! check_command "git"; then
    print_error "Git is required but not installed"
    REQUIREMENTS_MET=false
fi

# Optional but recommended
check_command "node" || print_warning "Node.js recommended for frontend development"
check_command "npm" || print_warning "npm recommended for frontend development"

echo ""

# Check for forbidden tools
echo "🚫 Checking for forbidden dependencies..."

if command -v "docker" &> /dev/null; then
    print_warning "Docker detected - Remember: Use Podman only for UIOTA development"
fi

if command -v "nvidia-smi" &> /dev/null; then
    print_warning "NVIDIA drivers detected - Remember: Use CPU-only ML for UIOTA"
fi

echo ""

if [ "$REQUIREMENTS_MET" = false ]; then
    print_error "Missing required dependencies. Please install them and run setup again."
    exit 1
fi

# Create necessary directories
echo "📁 Creating project structure..."
mkdir -p logs
mkdir -p data
mkdir -p .guardian
print_status "Project directories created"

# Set up Guardian configuration
echo ""
echo "🛡️ Setting up Guardian configuration..."

# Create Guardian config file
cat > .guardian/config.yaml << EOF
# UIOTA Guardian Configuration
guardian:
  class: "unassigned"  # Will be set during first contribution
  level: 1
  xp: 0
  specializations: []

agents:
  web:
    enabled: true
    port: 8080
  
  discord:
    enabled: true
    demo_mode: true
  
  ml:
    enabled: true
    port: 8888
    cpu_only: true  # NO NVIDIA/CUDA
  
system:
  container_engine: "podman"  # NO DOCKER
  offline_first: true
  security_level: "high"

# Development settings
development:
  auto_restart: true
  debug_mode: false
  log_level: "info"
EOF

print_status "Guardian configuration created"

# Make scripts executable
echo ""
echo "🔧 Setting up executable scripts..."
chmod +x start-demos.sh
chmod +x stop-demos.sh 2>/dev/null || echo "stop-demos.sh will be created when starting demos"
chmod +x setup.sh
print_status "Scripts made executable"

# Validate Podman setup
echo ""
echo "🐳 Validating Podman setup..."

if podman --version &> /dev/null; then
    PODMAN_VERSION=$(podman --version)
    print_status "Podman validation: $PODMAN_VERSION"
else
    print_error "Podman validation failed"
    exit 1
fi

# Test Podman without sudo
if podman ps &> /dev/null; then
    print_status "Podman rootless mode working"
else
    print_warning "Podman may require configuration for rootless mode"
    echo "   Run: sudo usermod -aG podman \$USER && newgrp podman"
fi

echo ""
echo "🧪 Running system validation..."

# Create a simple test container
print_info "Testing container creation..."
if podman run --rm docker.io/alpine:latest echo "Container test successful" &> /dev/null; then
    print_status "Container creation test passed"
else
    print_error "Container creation test failed"
    echo "   Check Podman installation and permissions"
    exit 1
fi

# Check port availability
print_info "Checking port availability..."
check_port() {
    if ! nc -z localhost "$1" 2>/dev/null; then
        print_status "Port $1 is available"
        return 0
    else
        print_warning "Port $1 is in use"
        return 1
    fi
}

check_port 8080
check_port 8888

echo ""
echo "📊 System Summary"
echo "=================="
echo "OS: $MACHINE"
echo "Container Engine: Podman (✅ Correct - No Docker)"
echo "ML Backend: CPU-only (✅ Correct - No NVIDIA)"
echo "Guardian Config: Created"
echo "Project Structure: Ready"
echo ""

print_status "UIOTA Offline Guard setup complete!"
echo ""
echo "🚀 Next steps:"
echo "   1. Run: ./start-demos.sh"
echo "   2. Visit: http://localhost:8080"
echo "   3. Check: http://localhost:8888 for ML toolkit"
echo "   4. Read: CONTRIBUTING.md to choose your Guardian class"
echo ""
echo "🛡️ Welcome to the Guardian ecosystem!"
echo "   Your journey toward digital sovereignty begins now."