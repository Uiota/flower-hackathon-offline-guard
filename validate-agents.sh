#!/bin/bash

echo "🔍 UIOTA AGENT VALIDATION"
echo "========================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_pass() { echo -e "${GREEN}✅ PASS${NC} $1"; }
print_fail() { echo -e "${RED}❌ FAIL${NC} $1"; }
print_warn() { echo -e "${YELLOW}⚠️  WARN${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ️  INFO${NC} $1"; }

TOTAL_TESTS=0
PASSED_TESTS=0

run_test() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if eval "$2"; then
        print_pass "$1"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        print_fail "$1"
        return 1
    fi
}

echo "🚫 Validating NO NVIDIA/DOCKER policy..."
echo ""

# Test 1: No Docker processes
run_test "No Docker daemon running" "! pgrep -f docker > /dev/null 2>&1"

# Test 2: No NVIDIA runtime in containers
run_test "No NVIDIA runtime in containers" "! podman ps --format '{{.Command}}' 2>/dev/null | grep -i nvidia > /dev/null"

# Test 3: Podman is primary container engine
run_test "Podman is available" "command -v podman > /dev/null"

echo ""
echo "🐳 Container Status Validation..."
echo ""

# Test 4: Check running containers
CONTAINERS=$(podman ps --format "{{.Names}}" 2>/dev/null)
if [ -n "$CONTAINERS" ]; then
    print_info "Running containers detected:"
    echo "$CONTAINERS" | while read -r container; do
        echo "    📦 $container"
    done
    run_test "Containers are running" "[ -n '$CONTAINERS' ]"
else
    print_warn "No containers currently running"
    echo "    Run ./start-demos.sh to start agents"
fi

echo ""
echo "🌐 Network Validation..."
echo ""

# Test 5: Web agent port
run_test "Port 8080 accessible" "curl -s http://localhost:8080 > /dev/null 2>&1 || nc -z localhost 8080 2>/dev/null"

# Test 6: ML agent port  
run_test "Port 8888 accessible" "curl -s http://localhost:8888 > /dev/null 2>&1 || nc -z localhost 8888 2>/dev/null"

echo ""
echo "🧠 ML Agent CPU-Only Validation..."
echo ""

# Test 7: No CUDA in ML container
if podman ps --filter "name=ml-agent" --format "{{.Names}}" | grep -q ml-agent; then
    # Check if container has CUDA libraries
    CUDA_CHECK=$(podman exec ml-agent python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        print('CUDA_AVAILABLE')
    else:
        print('CPU_ONLY')
except:
    print('NO_TORCH')
" 2>/dev/null)
    
    case "$CUDA_CHECK" in
        "CPU_ONLY")
            print_pass "ML Agent is CPU-only (correct)"
            ;;
        "CUDA_AVAILABLE")
            print_fail "ML Agent has CUDA available (violates NO NVIDIA rule)"
            ;;
        "NO_TORCH")
            print_warn "PyTorch not available in ML Agent"
            ;;
        *)
            print_warn "Could not validate ML Agent configuration"
            ;;
    esac
else
    print_warn "ML Agent container not running"
fi

echo ""
echo "🛡️ Guardian System Validation..."
echo ""

# Test 8: Guardian config exists
run_test "Guardian configuration exists" "[ -f .guardian/config.yaml ]"

# Test 9: Project structure
run_test "UIOTA specifications exist" "[ -f UIOTA_SUB_AGENT_SPECIFICATIONS.md ]"

# Test 10: Required scripts are executable
run_test "start-demos.sh is executable" "[ -x start-demos.sh ]"

echo ""
echo "🔒 Security Validation..."
echo ""

# Test 11: No privileged containers
PRIVILEGED_CONTAINERS=$(podman ps --filter "label=privileged=true" --format "{{.Names}}" 2>/dev/null)
if [ -z "$PRIVILEGED_CONTAINERS" ]; then
    print_pass "No privileged containers running"
else
    print_fail "Privileged containers detected: $PRIVILEGED_CONTAINERS"
fi

# Test 12: Containers running as non-root
ROOT_CONTAINERS=$(podman ps --format "{{.Names}}" 2>/dev/null | xargs -I {} podman inspect {} --format "{{.Config.User}}" 2>/dev/null | grep -c "^$\|^0$\|^root$" || echo "0")
if [ "$ROOT_CONTAINERS" -eq 0 ]; then
    print_pass "No containers running as root"
else
    print_warn "$ROOT_CONTAINERS containers may be running as root"
fi

echo ""
echo "📊 VALIDATION SUMMARY"
echo "===================="
echo ""

SCORE=$((PASSED_TESTS * 100 / TOTAL_TESTS))

echo "Tests Passed: $PASSED_TESTS/$TOTAL_TESTS"
echo "Score: $SCORE%"
echo ""

if [ "$SCORE" -ge 80 ]; then
    print_pass "UIOTA Offline Guard validation: EXCELLENT"
    echo "🛡️ Your Guardian agents are ready for collaboration!"
elif [ "$SCORE" -ge 60 ]; then
    print_warn "UIOTA Offline Guard validation: GOOD"
    echo "⚡ Some improvements needed but agents are functional"
else
    print_fail "UIOTA Offline Guard validation: NEEDS WORK"
    echo "🔧 Several issues need to be addressed"
fi

echo ""
echo "🎯 Guardian Compliance Checklist:"
echo ""

if ! pgrep -f docker > /dev/null 2>&1; then
    print_pass "NO DOCKER policy compliance"
else
    print_fail "DOCKER detected - violates NO DOCKER policy"
fi

if ! podman ps --format '{{.Command}}' 2>/dev/null | grep -i nvidia > /dev/null; then
    print_pass "NO NVIDIA policy compliance"
else
    print_fail "NVIDIA detected in containers - violates NO NVIDIA policy"
fi

if command -v podman > /dev/null; then
    print_pass "Podman container engine compliance"
else
    print_fail "Podman not available"
fi

echo ""
echo "🚀 Ready for Guardian collaboration!"
echo "    Share this validation report with your team."
echo "    All contributors can run this script to verify compliance."