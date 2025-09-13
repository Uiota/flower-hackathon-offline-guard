#!/bin/bash
# Offline-First Container Registry and Distribution System
# Perfect for hackathon teams with unstable internet or travel scenarios

set -euo pipefail

# Configuration
REGISTRY_NAME="offline-guard-registry"
REGISTRY_PORT="5000"
REGISTRY_DATA_DIR="./data/registry"
REGISTRY_CONFIG_DIR="./data/registry-config"
MIRROR_CACHE_DIR="./data/registry-mirror-cache"
OFFLINE_BUNDLE_DIR="./data/offline-bundles"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

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

# Create directory structure
setup_directories() {
    log_section "Setting up Offline Registry Directories"
    
    mkdir -p "$REGISTRY_DATA_DIR"
    mkdir -p "$REGISTRY_CONFIG_DIR"
    mkdir -p "$MIRROR_CACHE_DIR"
    mkdir -p "$OFFLINE_BUNDLE_DIR"
    
    log_success "Directory structure created"
}

# Create registry configuration
create_registry_config() {
    log_section "Creating Registry Configuration"
    
    # Main registry config
    cat > "$REGISTRY_CONFIG_DIR/config.yml" << 'EOF'
version: 0.1
log:
  fields:
    service: registry
storage:
  cache:
    blobdescriptor: inmemory
  filesystem:
    rootdirectory: /var/lib/registry
  delete:
    enabled: true
http:
  addr: :5000
  headers:
    X-Content-Type-Options: [nosniff]
    Access-Control-Allow-Origin: ['*']
    Access-Control-Allow-Methods: ['HEAD,GET,OPTIONS,DELETE']
    Access-Control-Allow-Headers: ['Authorization,Accept,Cache-Control']
health:
  storagedriver:
    enabled: true
    interval: 10s
    threshold: 3
proxy:
  remoteurl: https://registry-1.docker.io
  username: ''
  password: ''
  ttl: 168h
EOF

    # Authentication config (optional for development)
    cat > "$REGISTRY_CONFIG_DIR/auth-config.yml" << 'EOF'
version: 0.1
log:
  fields:
    service: registry
storage:
  filesystem:
    rootdirectory: /var/lib/registry
http:
  addr: :5000
auth:
  htpasswd:
    realm: basic-realm
    path: /auth/htpasswd
EOF

    # Create htpasswd for authentication (offline-guard:hackathon2025)
    if command -v htpasswd &> /dev/null; then
        htpasswd -Bbn offline-guard hackathon2025 > "$REGISTRY_CONFIG_DIR/htpasswd"
    else
        # Fallback: create simple auth file
        echo 'offline-guard:$2y$10$8K1p/AdDkh5nH4dHFwDgV.6m.8i1Og1I2yN2gHwn9s4d4u8RH9r3a' > "$REGISTRY_CONFIG_DIR/htpasswd"
    fi
    
    log_success "Registry configuration created"
}

# Start local registry
start_registry() {
    log_section "Starting Local Container Registry"
    
    # Check if registry is already running
    if podman ps | grep -q "$REGISTRY_NAME"; then
        log_warning "Registry already running, stopping first..."
        podman stop "$REGISTRY_NAME" || true
        podman rm "$REGISTRY_NAME" || true
    fi
    
    # Start registry container
    podman run -d \
        --name "$REGISTRY_NAME" \
        --restart=unless-stopped \
        -p "$REGISTRY_PORT:5000" \
        -v "$PWD/$REGISTRY_DATA_DIR:/var/lib/registry:Z" \
        -v "$PWD/$REGISTRY_CONFIG_DIR:/etc/docker/registry:Z" \
        -e REGISTRY_CONFIG_PATH=/etc/docker/registry/config.yml \
        docker.io/registry:2 || {
        log_error "Failed to start registry"
        return 1
    }
    
    # Wait for registry to be ready
    log_info "Waiting for registry to be ready..."
    timeout 60 bash -c "until curl -f http://localhost:$REGISTRY_PORT/v2/; do sleep 2; done" || {
        log_error "Registry failed to start properly"
        return 1
    }
    
    log_success "Local registry started at http://localhost:$REGISTRY_PORT"
}

# Create container bundle export functions
create_bundle_scripts() {
    log_section "Creating Container Bundle Scripts"
    
    # Export script for creating offline bundles
    cat > containers/export-bundle.sh << 'EOF'
#!/bin/bash
# Export Offline Guard container bundle for offline distribution

set -e

BUNDLE_NAME=${1:-"offline-guard-$(date +%Y%m%d)"}
BUNDLE_DIR="./data/offline-bundles/$BUNDLE_NAME"
SERVICES=("web-demo" "discord-bot" "ml-toolkit" "guardian-service" "redis:7-alpine" "registry:2")

echo "🎁 Creating Offline Guard container bundle: $BUNDLE_NAME"

mkdir -p "$BUNDLE_DIR"

# Export each container image
for service in "${SERVICES[@]}"; do
    if [[ "$service" == *":"* ]]; then
        # External image
        image_name="$service"
        export_name=$(echo "$service" | sed 's/:/-/g')
    else
        # Our built images
        image_name="localhost:5000/offline-guard-$service:latest"
        export_name="offline-guard-$service"
    fi
    
    echo "📦 Exporting $image_name..."
    
    # Export to OCI archive
    podman save --format oci-archive -o "$BUNDLE_DIR/$export_name.tar" "$image_name" || {
        echo "⚠️  Failed to export $image_name, skipping..."
        continue
    }
done

# Copy configuration files
cp -r containers/ "$BUNDLE_DIR/"
cp README.md "$BUNDLE_DIR/"

# Create import script
cat > "$BUNDLE_DIR/import-bundle.sh" << 'IMPORT_EOF'
#!/bin/bash
# Import Offline Guard container bundle

echo "📥 Importing Offline Guard container bundle..."

# Load all container images
for tar_file in *.tar; do
    if [ -f "$tar_file" ]; then
        echo "Loading $tar_file..."
        podman load -i "$tar_file"
    fi
done

echo "✅ Container bundle imported successfully!"
echo "🚀 Start with: podman-compose -f containers/podman-compose.yml up -d"
IMPORT_EOF

chmod +x "$BUNDLE_DIR/import-bundle.sh"

# Create bundle archive
echo "🗜️  Creating compressed bundle..."
tar -czf "$BUNDLE_DIR.tar.gz" -C "./data/offline-bundles" "$BUNDLE_NAME"

echo "✅ Bundle created: $BUNDLE_DIR.tar.gz"
echo "📏 Bundle size: $(du -h "$BUNDLE_DIR.tar.gz" | cut -f1)"
echo ""
echo "📋 To use on another system:"
echo "1. Copy $BUNDLE_DIR.tar.gz to target system"
echo "2. tar -xzf $BUNDLE_NAME.tar.gz"
echo "3. cd $BUNDLE_NAME && ./import-bundle.sh"
echo "4. podman-compose -f containers/podman-compose.yml up -d"
EOF

    chmod +x containers/export-bundle.sh
    
    # Import script for loading bundles
    cat > containers/import-bundle.sh << 'EOF'
#!/bin/bash
# Import container bundle from archive

BUNDLE_FILE=${1:-""}

if [ -z "$BUNDLE_FILE" ]; then
    echo "Usage: $0 <bundle-file.tar.gz>"
    echo "Available bundles:"
    ls -la ./data/offline-bundles/*.tar.gz 2>/dev/null || echo "No bundles found"
    exit 1
fi

if [ ! -f "$BUNDLE_FILE" ]; then
    echo "❌ Bundle file not found: $BUNDLE_FILE"
    exit 1
fi

echo "📥 Importing container bundle: $BUNDLE_FILE"

# Extract bundle
TEMP_DIR=$(mktemp -d)
tar -xzf "$BUNDLE_FILE" -C "$TEMP_DIR"

# Find the bundle directory
BUNDLE_DIR=$(find "$TEMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n1)

if [ -z "$BUNDLE_DIR" ]; then
    echo "❌ Invalid bundle structure"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Import containers
cd "$BUNDLE_DIR"
./import-bundle.sh

# Cleanup
cd - > /dev/null
rm -rf "$TEMP_DIR"

echo "✅ Bundle imported successfully!"
EOF

    chmod +x containers/import-bundle.sh
    
    log_success "Bundle scripts created"
}

# Create registry proxy setup
setup_registry_proxy() {
    log_section "Setting up Registry Proxy and Mirror"
    
    # Create mirror cache registry for Docker Hub
    cat > "$REGISTRY_CONFIG_DIR/mirror-config.yml" << 'EOF'
version: 0.1
log:
  fields:
    service: registry-mirror
storage:
  filesystem:
    rootdirectory: /var/lib/registry
http:
  addr: :5001
proxy:
  remoteurl: https://registry-1.docker.io
  username: ''
  password: ''
  ttl: 168h
EOF

    # Start mirror registry
    if ! podman ps | grep -q "registry-mirror"; then
        podman run -d \
            --name offline-guard-registry-mirror \
            --restart=unless-stopped \
            -p 5001:5001 \
            -v "$PWD/$MIRROR_CACHE_DIR:/var/lib/registry:Z" \
            -v "$PWD/$REGISTRY_CONFIG_DIR:/etc/docker/registry:Z" \
            -e REGISTRY_CONFIG_PATH=/etc/docker/registry/mirror-config.yml \
            docker.io/registry:2 || {
            log_warning "Failed to start mirror registry"
        }
    fi
    
    log_success "Registry mirror configured"
}

# Create offline development workflow
create_offline_workflow() {
    log_section "Creating Offline Development Workflow"
    
    # Main offline development script
    cat > containers/offline-dev.sh << 'EOF'
#!/bin/bash
# Offline Development Workflow for Hackathon Teams
# Works completely offline after initial setup

set -e

ACTION=${1:-"status"}

case $ACTION in
    "start")
        echo "🚀 Starting Offline Guard in offline mode..."
        
        # Start local registry first
        if ! podman ps | grep -q "offline-guard-registry"; then
            echo "📦 Starting local container registry..."
            ./offline-registry-setup.sh start-registry
        fi
        
        # Use offline compose configuration
        podman-compose -f podman-compose.yml \
            --env-file .env.offline up -d
            
        echo "✅ Offline Guard started in offline mode"
        echo "🌐 Access points:"
        echo "   Web Demo: http://localhost:8080"
        echo "   ML Toolkit: http://localhost:8888"
        echo "   Guardian API: http://localhost:3001"
        echo "   Local Registry: http://localhost:5000"
        ;;
        
    "stop")
        echo "⏹️  Stopping Offline Guard..."
        podman-compose -f podman-compose.yml down
        podman stop offline-guard-registry || true
        echo "✅ Offline Guard stopped"
        ;;
        
    "status")
        echo "📊 Offline Guard Status:"
        echo ""
        echo "🔧 Containers:"
        podman ps --filter "label=app=offline-guard" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        echo "💾 Storage:"
        du -sh ./data/* 2>/dev/null || echo "No data directories found"
        echo ""
        echo "📦 Local Images:"
        podman images | grep -E "(offline-guard|localhost:5000)" || echo "No local images found"
        ;;
        
    "export")
        echo "📤 Exporting for offline distribution..."
        ./containers/export-bundle.sh "hackathon-$(date +%Y%m%d)"
        ;;
        
    "sync")
        PEER_IP=${2:-""}
        if [ -z "$PEER_IP" ]; then
            echo "Usage: $0 sync <peer-ip-address>"
            echo "Example: $0 sync 192.168.1.100"
            exit 1
        fi
        
        echo "🔄 Syncing with peer at $PEER_IP..."
        
        # Simple rsync-based synchronization
        rsync -avz --progress \
            ./data/ \
            "$PEER_IP:~/offline-guard/data/" || {
            echo "❌ Sync failed. Ensure SSH access to $PEER_IP"
            exit 1
        }
        
        echo "✅ Sync completed with $PEER_IP"
        ;;
        
    *)
        echo "Usage: $0 {start|stop|status|export|sync}"
        echo ""
        echo "Commands:"
        echo "  start  - Start all services in offline mode"
        echo "  stop   - Stop all services"
        echo "  status - Show current status"
        echo "  export - Create offline distribution bundle"
        echo "  sync   - Sync data with another team member"
        ;;
esac
EOF

    chmod +x containers/offline-dev.sh
    
    # Create offline environment configuration
    cat > .env.offline << 'EOF'
# Offline Development Environment Configuration
REGISTRY=localhost:5000
DISCORD_BOT_TOKEN=demo_mode
JUPYTER_TOKEN=offline-hackathon-ml
REDIS_PASSWORD=offline-redis-pass
GITEA_DOMAIN=localhost
GITEA_SECRET_KEY=offline-gitea-secret
GITEA_INTERNAL_TOKEN=offline-gitea-internal

# Offline-specific settings
PULL_POLICY=never
RESTART_POLICY=unless-stopped
NETWORK_MODE=offline-guard
HEALTH_CHECK_TIMEOUT=30s

# Resource limits for laptop development
MEMORY_LIMIT=2G
CPU_LIMIT=2.0
EOF

    log_success "Offline development workflow created"
}

# Create team collaboration tools
create_collaboration_tools() {
    log_section "Creating Team Collaboration Tools"
    
    # QR code generator for sharing registry access
    cat > containers/share-registry.sh << 'EOF'
#!/bin/bash
# Generate QR code for easy registry sharing

REGISTRY_URL="http://$(hostname -I | awk '{print $1}'):5000"

echo "📱 Registry sharing information:"
echo "URL: $REGISTRY_URL"
echo ""

# Try to generate QR code if qrencode is available
if command -v qrencode &> /dev/null; then
    echo "📲 QR Code for registry URL:"
    qrencode -t ANSI "$REGISTRY_URL"
    echo ""
fi

# Create connection instructions
cat << INFO_EOF
🔗 Connection Instructions:

1. For Podman users:
   echo "$REGISTRY_URL" | sudo tee -a /etc/containers/registries.conf.d/offline-guard.conf

2. For Docker users:
   Add to daemon.json:
   {
     "insecure-registries": ["$REGISTRY_URL"]
   }

3. For other team members:
   podman pull $REGISTRY_URL/offline-guard-web-demo:latest
   
📋 Available images:
INFO_EOF

# List available images
curl -s "$REGISTRY_URL/v2/_catalog" | jq -r '.repositories[]' 2>/dev/null || echo "Registry not accessible"
EOF

    chmod +x containers/share-registry.sh
    
    # Team setup script
    cat > containers/team-setup.sh << 'EOF'
#!/bin/bash
# Quick team member onboarding

echo "👥 Offline Guard Team Setup"
echo "Perfect for hackathon collaboration!"
echo ""

# Detect system
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo "❌ Neither podman nor docker found"
    echo "Please install podman or docker first"
    exit 1
fi

echo "✅ Using $CONTAINER_CMD"

# Check if we're joining existing setup or creating new
read -p "Are you joining an existing team setup? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Joining existing setup
    read -p "Enter team lead's IP address: " LEAD_IP
    read -p "Enter registry port (default 5000): " REGISTRY_PORT
    REGISTRY_PORT=${REGISTRY_PORT:-5000}
    
    REGISTRY_URL="$LEAD_IP:$REGISTRY_PORT"
    
    echo "🔄 Connecting to team registry at $REGISTRY_URL..."
    
    # Test connection
    if curl -f "http://$REGISTRY_URL/v2/" &>/dev/null; then
        echo "✅ Registry accessible"
        
        # Configure container runtime
        if [ "$CONTAINER_CMD" = "podman" ]; then
            echo "Configuring podman for team registry..."
            mkdir -p ~/.config/containers/
            echo "[[registry]]" >> ~/.config/containers/registries.conf
            echo "location = \"$REGISTRY_URL\"" >> ~/.config/containers/registries.conf
            echo "insecure = true" >> ~/.config/containers/registries.conf
        fi
        
        # Pull latest images
        echo "📥 Pulling latest team images..."
        $CONTAINER_CMD pull "$REGISTRY_URL/offline-guard-web-demo:latest" || true
        $CONTAINER_CMD pull "$REGISTRY_URL/offline-guard-guardian-service:latest" || true
        
        echo "✅ Team setup complete!"
        echo "🚀 Start developing with: ./offline-dev.sh start"
        
    else
        echo "❌ Cannot connect to registry at $REGISTRY_URL"
        echo "Please check the IP address and ensure the lead has started their registry"
    fi
else
    # Creating new setup
    echo "🎯 Setting up as team lead..."
    
    # Start full setup
    ./offline-registry-setup.sh full-setup
    
    echo "✅ Team lead setup complete!"
    echo "📱 Share registry access with: ./share-registry.sh"
fi
EOF

    chmod +x containers/team-setup.sh
    
    log_success "Team collaboration tools created"
}

# Main setup orchestration
full_setup() {
    log_section "Full Offline Registry Setup"
    
    setup_directories
    create_registry_config
    start_registry
    create_bundle_scripts
    setup_registry_proxy
    create_offline_workflow
    create_collaboration_tools
    
    log_success "🎉 Offline Guard registry setup complete!"
    echo ""
    echo "🌟 Ready for hackathon teams!"
    echo "📦 Local registry: http://localhost:$REGISTRY_PORT"
    echo "🔧 Control script: ./containers/offline-dev.sh"
    echo "👥 Team setup: ./containers/team-setup.sh"
    echo "📱 Share access: ./containers/share-registry.sh"
    echo ""
    echo "🚀 Next steps:"
    echo "1. Build your containers: ./containers/build-multiarch.sh build"
    echo "2. Start offline mode: ./containers/offline-dev.sh start"
    echo "3. Share with team: ./containers/share-registry.sh"
}

# Handle different commands
case "${1:-full-setup}" in
    "directories")
        setup_directories
        ;;
    "config")
        create_registry_config
        ;;
    "start-registry")
        start_registry
        ;;
    "proxy")
        setup_registry_proxy
        ;;
    "scripts")
        create_bundle_scripts
        create_offline_workflow
        create_collaboration_tools
        ;;
    "full-setup")
        full_setup
        ;;
    *)
        echo "Usage: $0 {directories|config|start-registry|proxy|scripts|full-setup}"
        exit 1
        ;;
esac