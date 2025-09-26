#!/bin/bash

# Quick Start Script for Functional Federated Learning Application

set -e

echo "======================================"
echo "Functional Federated Learning App"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed${NC}"
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "run_demo.py" ]]; then
    echo -e "${RED}Error: Please run this script from the functional-app directory${NC}"
    exit 1
fi

echo -e "${BLUE}Checking dependencies...${NC}"

# Install dependencies if requirements file exists
if [[ -f "containers/requirements.txt" ]]; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip3 install -r containers/requirements.txt --quiet || {
        echo -e "${RED}Failed to install dependencies. Please install manually:${NC}"
        echo "pip3 install -r containers/requirements.txt"
        exit 1
    }
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}Warning: requirements.txt not found${NC}"
fi

# Create necessary directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p data logs config artifacts
echo -e "${GREEN}✓ Directories created${NC}"

# Test basic functionality
echo -e "${BLUE}Running functionality tests...${NC}"
python3 test_functionality.py > test_output.log 2>&1 || {
    echo -e "${RED}Tests failed. Check test_output.log for details${NC}"
    echo "You can still try running the demo, but some features may not work."
}
echo -e "${GREEN}✓ Basic tests completed${NC}"

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Ready to start! Choose an option:${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "1. Quick Demo (3 clients, 5 rounds, MNIST)"
echo "2. Interactive Demo (customizable options)"
echo "3. Docker Deployment"
echo "4. Manual Component Start"
echo "5. Run Tests Only"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo -e "${BLUE}Starting Quick Demo...${NC}"
        echo "Dashboard will be available at: http://localhost:5000"
        echo "Press Ctrl+C to stop"
        echo ""
        python3 run_demo.py --dataset mnist --num-clients 3 --num-rounds 5
        ;;
    2)
        echo -e "${BLUE}Interactive Demo Configuration${NC}"
        echo ""

        # Get dataset choice
        echo "Select dataset:"
        echo "1. MNIST (handwritten digits)"
        echo "2. CIFAR-10 (color images)"
        echo "3. Synthetic (generated data)"
        read -p "Choice (1-3): " dataset_choice

        case $dataset_choice in
            1) dataset="mnist" ;;
            2) dataset="cifar10" ;;
            3) dataset="synthetic" ;;
            *) dataset="mnist" ;;
        esac

        # Get model choice
        echo ""
        echo "Select model:"
        echo "1. CNN (Convolutional Neural Network)"
        echo "2. Linear (Simple Linear Classifier)"
        read -p "Choice (1-2): " model_choice

        case $model_choice in
            1) model="cnn" ;;
            2) model="linear" ;;
            *) model="cnn" ;;
        esac

        # Get number of clients
        read -p "Number of clients (2-10) [3]: " num_clients
        num_clients=${num_clients:-3}

        # Get number of rounds
        read -p "Number of training rounds (1-20) [5]: " num_rounds
        num_rounds=${num_rounds:-5}

        echo ""
        echo -e "${BLUE}Starting Custom Demo...${NC}"
        echo "Configuration:"
        echo "  Dataset: $dataset"
        echo "  Model: $model"
        echo "  Clients: $num_clients"
        echo "  Rounds: $num_rounds"
        echo ""
        echo "Dashboard will be available at: http://localhost:5000"
        echo "Press Ctrl+C to stop"
        echo ""

        python3 run_demo.py --dataset $dataset --model $model --num-clients $num_clients --num-rounds $num_rounds
        ;;
    3)
        echo -e "${BLUE}Starting Docker Deployment...${NC}"
        if command -v docker-compose &> /dev/null; then
            cd containers
            echo "Building and starting containers..."
            docker-compose up --build
        else
            echo -e "${RED}Error: docker-compose is required for this option${NC}"
            echo "Please install Docker and docker-compose, then run:"
            echo "cd containers && docker-compose up --build"
        fi
        ;;
    4)
        echo -e "${BLUE}Manual Component Start${NC}"
        echo ""
        echo "To start components manually, open separate terminals and run:"
        echo ""
        echo -e "${YELLOW}Terminal 1 (FL Server):${NC}"
        echo "python3 run_server.py --dataset mnist --num-rounds 10"
        echo ""
        echo -e "${YELLOW}Terminal 2 (Client 1):${NC}"
        echo "python3 run_client.py --client-id client-1 --dataset mnist"
        echo ""
        echo -e "${YELLOW}Terminal 3 (Client 2):${NC}"
        echo "python3 run_client.py --client-id client-2 --dataset mnist"
        echo ""
        echo -e "${YELLOW}Terminal 4 (Client 3 with mesh):${NC}"
        echo "python3 run_client.py --client-id client-3 --dataset mnist --enable-mesh"
        echo ""
        echo "Dashboard: http://localhost:5000"
        ;;
    5)
        echo -e "${BLUE}Running comprehensive tests...${NC}"
        python3 test_functionality.py
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Thanks for using the FL Application!${NC}"
echo -e "${GREEN}======================================${NC}"