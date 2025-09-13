#!/bin/bash
# Multi-Architecture Container Build Script for Offline Guard
# Supports hackathon scenarios across different hardware platforms

set -euo pipefail

# Configuration
PROJECT_NAME="offline-guard"
REGISTRY=${REGISTRY:-"localhost:5000"}
VERSION=${VERSION:-"latest"}
PUSH_REGISTRY=${PUSH_REGISTRY:-false}

# Supported architectures for hackathon scenarios
PLATFORMS=(
    "linux/amd64"    # Intel/AMD laptops (most hackathon participants)
    "linux/arm64"    # Apple Silicon Macs, modern Pi
    "linux/arm/v7"   # Older Raspberry Pi models
)

# Container services to build
SERVICES=(
    "web-demo"
    "discord-bot"  
    "ml-toolkit"
    "guardian-service"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "\n${PURPLE}=== $1 ===${NC}\n"
}

# Check prerequisites
check_prerequisites() {
    log_section "Checking Prerequisites"
    
    # Check if buildah is installed
    if ! command -v buildah &> /dev/null; then
        log_error "buildah is required but not installed"
        exit 1
    fi
    
    # Check if podman is installed
    if ! command -v podman &> /dev/null; then
        log_error "podman is required but not installed" 
        exit 1
    fi
    
    # Check if we can create multi-arch manifests
    if ! buildah manifest --help &> /dev/null; then
        log_error "buildah manifest support is required"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Setup buildah for multi-arch
setup_buildah() {
    log_section "Setting up Multi-Architecture Build Environment"
    
    # Enable QEMU emulation for cross-platform builds
    if [ -f /proc/sys/fs/binfmt_misc/qemu-aarch64 ] || [ -f /proc/sys/fs/binfmt_misc/qemu-arm ]; then
        log_success "QEMU emulation already enabled"
    else
        log_warning "QEMU emulation may not be available - cross-arch builds might fail"
        log_info "Install qemu-user-static package if needed"
    fi
    
    # Create data directories if they don't exist
    mkdir -p data/{ml-data,guardian-data,redis-data,gitea-data,gitea-config}
    log_success "Data directories created"
}

# Build single service for multiple architectures
build_service() {
    local service=$1
    local manifest_name="${REGISTRY}/${PROJECT_NAME}-${service}:${VERSION}"
    
    log_section "Building ${service} for Multiple Architectures"
    
    # Create manifest
    buildah manifest create "${manifest_name}" || {
        log_warning "Manifest already exists, removing and recreating"
        buildah manifest rm "${manifest_name}" 2>/dev/null || true
        buildah manifest create "${manifest_name}"
    }
    
    # Build for each platform
    for platform in "${PLATFORMS[@]}"; do
        log_info "Building ${service} for ${platform}"
        
        local arch_tag="${manifest_name}-${platform//\//-}"
        
        # Build the container
        buildah build \
            --platform="${platform}" \
            --tag="${arch_tag}" \
            --file="containers/${service}/Containerfile" \
            --format=oci \
            --layers \
            --build-arg "BUILDPLATFORM=${platform}" \
            --build-arg "TARGETPLATFORM=${platform}" \
            . || {
            log_error "Failed to build ${service} for ${platform}"
            continue
        }
        
        # Add to manifest
        buildah manifest add "${manifest_name}" "${arch_tag}"
        log_success "Built ${service} for ${platform}"
    done
    
    log_success "Multi-arch manifest created for ${service}"
}

# Push manifests to registry
push_manifests() {
    if [ "${PUSH_REGISTRY}" != "true" ]; then
        log_info "Skipping registry push (PUSH_REGISTRY=${PUSH_REGISTRY})"
        return
    fi
    
    log_section "Pushing Multi-Architecture Manifests"
    
    for service in "${SERVICES[@]}"; do
        local manifest_name="${REGISTRY}/${PROJECT_NAME}-${service}:${VERSION}"
        
        log_info "Pushing ${service} manifest to registry"
        buildah manifest push \
            --all \
            "${manifest_name}" \
            "docker://${manifest_name}" || {
            log_error "Failed to push ${service} manifest"
            continue
        }
        
        log_success "Pushed ${service} manifest"
    done
}

# Create a local registry for offline development
setup_local_registry() {
    log_section "Setting up Local Container Registry"
    
    # Check if registry is already running
    if podman ps | grep -q "registry:2"; then
        log_info "Local registry already running"
        return
    fi
    
    # Start local registry
    podman run -d \
        --name offline-guard-registry \
        --restart=unless-stopped \
        -p 5000:5000 \
        -v registry-data:/var/lib/registry \
        docker.io/registry:2 || {
        log_warning "Could not start local registry (may already exist)"
    }
    
    log_success "Local registry available at localhost:5000"
}

# Create build optimization scripts
create_build_scripts() {
    log_section "Creating Build Helper Scripts"
    
    # Fast development build (current architecture only)
    cat > containers/build-dev.sh << 'EOF'
#!/bin/bash
# Fast development build for current architecture only
set -e

SERVICE=${1:-"all"}
CURRENT_ARCH=$(uname -m)

case $CURRENT_ARCH in
    x86_64) PLATFORM="linux/amd64" ;;
    aarch64) PLATFORM="linux/arm64" ;;
    armv7l) PLATFORM="linux/arm/v7" ;;
    *) PLATFORM="linux/amd64" ;;
esac

echo "🔨 Fast build for $PLATFORM"

if [ "$SERVICE" = "all" ]; then
    for svc in web-demo discord-bot ml-toolkit guardian-service; do
        echo "Building $svc..."
        buildah build \
            --platform="$PLATFORM" \
            --tag="offline-guard-$svc:dev" \
            --file="containers/$svc/Containerfile" \
            --layers \
            .
    done
else
    buildah build \
        --platform="$PLATFORM" \
        --tag="offline-guard-$SERVICE:dev" \
        --file="containers/$SERVICE/Containerfile" \
        --layers \
        .
fi

echo "✅ Development build complete"
EOF

    chmod +x containers/build-dev.sh
    
    # Platform-specific optimized builds
    cat > containers/build-raspberry-pi.sh << 'EOF'
#!/bin/bash
# Optimized build for Raspberry Pi deployment
set -e

echo "🥧 Building for Raspberry Pi (ARM64/ARM7)"

# Lightweight services for Pi constraints
SERVICES=("web-demo" "guardian-service")

for service in "${SERVICES[@]}"; do
    echo "Building $service for Raspberry Pi..."
    
    # Use smaller base images and optimize for Pi
    buildah build \
        --platform="linux/arm64,linux/arm/v7" \
        --tag="offline-guard-$service:pi" \
        --file="containers/$service/Containerfile" \
        --build-arg="BASE_IMAGE=alpine:3.18" \
        --layers \
        .
done

echo "✅ Raspberry Pi build complete"
EOF

    chmod +x containers/build-raspberry-pi.sh
    
    log_success "Build helper scripts created"
}

# Main build orchestration
main() {
    local command=${1:-"build"}
    
    case $command in
        "check")
            check_prerequisites
            ;;
        "setup")
            check_prerequisites
            setup_buildah
            setup_local_registry
            create_build_scripts
            ;;
        "build")
            check_prerequisites
            setup_buildah
            
            # Build each service
            for service in "${SERVICES[@]}"; do
                if [ -f "containers/${service}/Containerfile" ]; then
                    build_service "$service"
                else
                    log_warning "Containerfile missing for ${service}, skipping"
                fi
            done
            
            log_success "All multi-architecture builds complete!"
            ;;
        "push")
            PUSH_REGISTRY=true
            main build
            push_manifests
            ;;
        "registry")
            setup_local_registry
            ;;
        "clean")
            log_section "Cleaning Build Artifacts"
            
            # Remove manifests
            for service in "${SERVICES[@]}"; do
                buildah manifest rm "${REGISTRY}/${PROJECT_NAME}-${service}:${VERSION}" 2>/dev/null || true
            done
            
            # Remove images
            buildah rmi --all 2>/dev/null || true
            
            log_success "Build artifacts cleaned"
            ;;
        *)
            echo "Usage: $0 {check|setup|build|push|registry|clean}"
            echo ""
            echo "Commands:"
            echo "  check    - Verify prerequisites"
            echo "  setup    - Setup build environment and local registry"
            echo "  build    - Build all services for multiple architectures"
            echo "  push     - Build and push to registry"
            echo "  registry - Start local registry only"
            echo "  clean    - Clean build artifacts"
            echo ""
            echo "Environment variables:"
            echo "  REGISTRY - Container registry (default: localhost:5000)"
            echo "  VERSION  - Image version tag (default: latest)"
            echo "  PUSH_REGISTRY - Push to registry (default: false)"
            exit 1
            ;;
    esac
}

# Hackathon-friendly startup message
echo -e "${PURPLE}"
cat << 'EOF'
  ____   __  __ _ _              ____                     _ 
 / __ \ / _|/ _| (_)            / ___|_   _  __ _ _ __ __| |
| |  | | |_| |_| |_ _ __   ___  | |  _| | | |/ _` | '__/ _` |
| |  | |  _|  _| | | '_ \ / _ \ | |_| | |_| | (_| | | | (_| |
 \____/|_| |_| |_|_|_| |_|_____| \____|\__,_|\__,_|_|  \__,_|

Multi-Architecture Container Builds
🌸 Perfect for Flower AI Hackathon Teams! 🌸

EOF
echo -e "${NC}"

# Run main function
main "$@"