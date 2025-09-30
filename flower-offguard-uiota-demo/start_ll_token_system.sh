#!/bin/bash

# LL TOKEN OFFLINE - Quantum-Resistant Tokenized Federated Learning System
# Startup script for the complete tokenization infrastructure

set -e

echo "🪙 LL TOKEN OFFLINE - Starting Tokenization Phase"
echo "🔒 Quantum-Resistant • Offline Ledger • Agent-Based Architecture"
echo "======================================================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
AGENTS=6
ROUNDS=5
BASE_PATH="./ll_token_system"
PYTHON_CMD="python3"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --agents)
            AGENTS="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --base-path)
            BASE_PATH="$2"
            shift 2
            ;;
        --python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --agents N      Number of LL TOKEN agents (default: 6)"
            echo "  --rounds N      Number of FL rounds (default: 5)"
            echo "  --base-path P   Base path for system files (default: ./ll_token_system)"
            echo "  --python CMD    Python command to use (default: python3)"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  OFFLINE_MODE=1  Required for secure operation (automatically set)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Configuration:${NC}"
echo "  Agents: $AGENTS"
echo "  Rounds: $ROUNDS"
echo "  Base path: $BASE_PATH"
echo "  Python: $PYTHON_CMD"
echo ""

# Check Python installation
echo -e "${BLUE}Checking Python environment...${NC}"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}❌ Python not found: $PYTHON_CMD${NC}"
    echo "Please install Python 3.10+ or specify correct path with --python"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"

# Check required dependencies
echo -e "${BLUE}Checking dependencies...${NC}"

# Function to check if Python package is installed
check_package() {
    if $PYTHON_CMD -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✅ $1${NC}"
        return 0
    else
        echo -e "${RED}❌ $1${NC}"
        return 1
    fi
}

# Check core packages
missing_packages=false
for pkg in flwr torch numpy cryptography; do
    if ! check_package $pkg; then
        missing_packages=true
    fi
done

if [ "$missing_packages" = true ]; then
    echo -e "${YELLOW}⚠️  Some dependencies are missing.${NC}"
    echo "Install with: pip install -r requirements-full.txt"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation aborted"
        exit 1
    fi
fi

# Set environment variables
export OFFLINE_MODE=1
export PYTHONPATH="$PWD:$PYTHONPATH"

echo -e "${GREEN}✅ Environment configured${NC}"
echo ""

# Create base directory if it doesn't exist
mkdir -p "$BASE_PATH"

# Check for existing system
if [ -d "$BASE_PATH/fl_system" ]; then
    echo -e "${YELLOW}⚠️  Existing LL TOKEN system found at $BASE_PATH${NC}"
    read -p "Continue with existing system? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Cleaning up existing system...${NC}"
        rm -rf "$BASE_PATH"
        mkdir -p "$BASE_PATH"
        echo -e "${GREEN}✅ Clean slate prepared${NC}"
    fi
fi

# Start the system
echo -e "${BLUE}🚀 Starting LL TOKEN OFFLINE system...${NC}"
echo ""

# Run the tokenization system
$PYTHON_CMD tokenization_system.py \
    --agents $AGENTS \
    --rounds $ROUNDS \
    --base-path "$BASE_PATH"

# Check if system completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 LL TOKEN OFFLINE tokenization phase completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}System artifacts:${NC}"
    if [ -d "$BASE_PATH" ]; then
        echo "  Base path: $BASE_PATH"
        if [ -f "$BASE_PATH/tokenization_proof.json" ]; then
            echo "  Ledger proof: $BASE_PATH/tokenization_proof.json"
        fi
        if [ -d "$BASE_PATH/fl_system" ]; then
            echo "  FL system: $BASE_PATH/fl_system/"
        fi
        if [ -d "$BASE_PATH/agents" ]; then
            echo "  Agents: $BASE_PATH/agents/"
        fi
    fi
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Review tokenization proof: cat $BASE_PATH/tokenization_proof.json"
    echo "  2. Examine agent wallets in: $BASE_PATH/agents/"
    echo "  3. Check FL ledger: $BASE_PATH/fl_system/fl_token_ledger/"
    echo ""
    echo -e "${GREEN}✅ LL TOKEN OFFLINE ready for production deployment!${NC}"
else
    echo ""
    echo -e "${RED}❌ LL TOKEN system encountered an error${NC}"
    echo "Check the logs above for details"
    exit 1
fi