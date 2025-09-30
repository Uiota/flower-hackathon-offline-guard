#!/bin/bash

################################################################################
# Offline AI Operating System - Master Installation Script
# Complete automated deployment for air-gapped environments
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables
INSTALL_DIR="/opt/offline-ai"
MODELS_DIR="${INSTALL_DIR}/models"
CONFIG_DIR="${INSTALL_DIR}/config"
DATA_DIR="${INSTALL_DIR}/data"
LOGS_DIR="${INSTALL_DIR}/logs"
BACKUP_DIR="${INSTALL_DIR}/backups"

# Log function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

################################################################################
# Pre-flight Checks
################################################################################

preflight_checks() {
    log "Running pre-flight checks..."

    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
        exit 1
    fi

    # Check OS
    if [[ ! -f /etc/os-release ]]; then
        error "Cannot detect OS version"
        exit 1
    fi

    source /etc/os-release
    if [[ "$ID" != "ubuntu" ]] && [[ "$ID" != "debian" ]]; then
        warning "This script is optimized for Ubuntu/Debian"
    fi

    # Check available disk space (need at least 100GB)
    available_space=$(df /opt | tail -1 | awk '{print $4}')
    required_space=$((100 * 1024 * 1024))  # 100GB in KB

    if [[ $available_space -lt $required_space ]]; then
        error "Insufficient disk space. Need at least 100GB free in /opt"
        exit 1
    fi

    # Check RAM (need at least 16GB)
    total_ram=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $total_ram -lt 16 ]]; then
        warning "System has less than 16GB RAM. Performance may be impacted."
    fi

    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        info "NVIDIA GPU detected"
        GPU_AVAILABLE=1
    else
        warning "No NVIDIA GPU detected. Will use CPU-only mode."
        GPU_AVAILABLE=0
    fi

    log "✓ Pre-flight checks passed"
}

################################################################################
# System Preparation
################################################################################

prepare_system() {
    log "Preparing system..."

    # Update package lists
    log "Updating package lists..."
    apt-get update -qq

    # Install essential packages
    log "Installing essential packages..."
    apt-get install -y -qq \
        build-essential \
        git \
        curl \
        wget \
        vim \
        htop \
        net-tools \
        python3 \
        python3-pip \
        python3-venv \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        ufw \
        fail2ban \
        aide \
        auditd \
        apparmor \
        cryptsetup

    log "✓ System prepared"
}

################################################################################
# Security Hardening
################################################################################

harden_system() {
    log "Applying security hardening..."

    # Configure UFW firewall
    log "Configuring firewall..."
    ufw --force enable
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow 22/tcp    # SSH
    ufw allow 443/tcp   # HTTPS
    ufw allow 6443/tcp  # Kubernetes API

    # Configure fail2ban
    log "Configuring fail2ban..."
    systemctl enable fail2ban
    systemctl start fail2ban

    # Initialize AIDE
    log "Initializing AIDE file integrity monitoring..."
    aideinit || true

    # Configure auditd
    log "Configuring audit daemon..."
    cat > /etc/audit/rules.d/ai-system.rules << EOF
# AI System Audit Rules
-w ${INSTALL_DIR}/ -p wa -k ai_system_changes
-w /etc/docker/ -p wa -k docker_config
-a always,exit -F arch=b64 -S execve -k process_execution
EOF

    systemctl restart auditd

    # Disable unnecessary services
    log "Disabling unnecessary services..."
    systemctl disable bluetooth.service || true
    systemctl disable cups.service || true

    log "✓ Security hardening complete"
}

################################################################################
# Docker Installation
################################################################################

install_docker() {
    log "Installing Docker..."

    if command -v docker &> /dev/null; then
        info "Docker already installed"
        return
    fi

    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh

    # Install Docker Compose
    DOCKER_COMPOSE_VERSION="2.20.0"
    curl -L "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" \
        -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose

    # Configure Docker daemon
    mkdir -p /etc/docker
    cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2"
}
EOF

    systemctl enable docker
    systemctl start docker

    log "✓ Docker installed"
}

################################################################################
# GPU Support (NVIDIA)
################################################################################

install_gpu_support() {
    if [[ $GPU_AVAILABLE -eq 0 ]]; then
        info "Skipping GPU support installation"
        return
    fi

    log "Installing NVIDIA GPU support..."

    # Install NVIDIA driver
    log "Installing NVIDIA driver..."
    apt-get install -y -qq nvidia-driver-535

    # Install NVIDIA Container Toolkit
    log "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        tee /etc/apt/sources.list.d/nvidia-docker.list

    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit

    # Configure Docker for GPU
    cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
EOF

    systemctl restart docker

    log "✓ GPU support installed"
}

################################################################################
# Directory Structure
################################################################################

create_directories() {
    log "Creating directory structure..."

    mkdir -p "${INSTALL_DIR}"
    mkdir -p "${MODELS_DIR}"
    mkdir -p "${CONFIG_DIR}"
    mkdir -p "${DATA_DIR}"
    mkdir -p "${LOGS_DIR}"
    mkdir -p "${BACKUP_DIR}"
    mkdir -p "${INSTALL_DIR}/ssl"
    mkdir -p "${INSTALL_DIR}/agent_system"
    mkdir -p "${INSTALL_DIR}/monitoring"
    mkdir -p "${INSTALL_DIR}/security_tools"

    # Set permissions
    chmod 755 "${INSTALL_DIR}"
    chmod 700 "${CONFIG_DIR}"
    chmod 700 "${DATA_DIR}"

    log "✓ Directories created"
}

################################################################################
# Python Environment
################################################################################

setup_python_environment() {
    log "Setting up Python environment..."

    # Create virtual environment
    python3 -m venv "${INSTALL_DIR}/venv"
    source "${INSTALL_DIR}/venv/bin/activate"

    # Upgrade pip
    pip install --upgrade pip wheel setuptools

    # Install core packages
    log "Installing core Python packages..."
    pip install \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
        transformers \
        langchain langchain-community langgraph \
        llama-cpp-python \
        sentence-transformers \
        faiss-cpu \
        chromadb \
        qdrant-client \
        fastapi uvicorn \
        pydantic \
        asyncio \
        aiohttp \
        redis \
        psycopg2-binary \
        pymongo \
        python-dotenv \
        pyyaml \
        prometheus-client \
        pytest pytest-asyncio

    log "✓ Python environment configured"
}

################################################################################
# Security Tools Installation
################################################################################

install_security_tools() {
    log "Installing security tools..."

    # Suricata
    log "Installing Suricata..."
    add-apt-repository -y ppa:oisf/suricata-stable
    apt-get update -qq
    apt-get install -y -qq suricata

    # Update Suricata rules
    suricata-update || true

    # YARA
    log "Installing YARA..."
    apt-get install -y -qq yara
    pip install yara-python

    # ClamAV
    log "Installing ClamAV..."
    apt-get install -y -qq clamav clamav-daemon

    # Stop ClamAV daemon for initial update
    systemctl stop clamav-freshclam
    freshclam
    systemctl start clamav-freshclam
    systemctl enable clamav-daemon
    systemctl start clamav-daemon

    # OSSEC/Wazuh (optional - commented out for speed)
    # log "Installing Wazuh agent..."
    # curl -s https://packages.wazuh.com/key/GPG-KEY-WAZUH | apt-key add -
    # echo "deb https://packages.wazuh.com/4.x/apt/ stable main" | \
    #     tee /etc/apt/sources.list.d/wazuh.list
    # apt-get update -qq
    # apt-get install -y -qq wazuh-agent

    log "✓ Security tools installed"
}

################################################################################
# Download Models
################################################################################

download_models() {
    log "Downloading AI models..."

    # Check if models already exist
    if [[ -d "${MODELS_DIR}/llama-3.2-3b" ]]; then
        info "Models already downloaded, skipping..."
        return
    fi

    # Install huggingface-cli
    pip install huggingface-hub

    # Download models (this requires internet - do offline if air-gapped)
    log "Downloading Llama 3.2 3B..."
    huggingface-cli download \
        meta-llama/Llama-3.2-3B-Instruct \
        --local-dir "${MODELS_DIR}/llama-3.2-3b" \
        --quiet || warning "Could not download Llama model"

    log "Downloading embedding model..."
    huggingface-cli download \
        sentence-transformers/all-MiniLM-L6-v2 \
        --local-dir "${MODELS_DIR}/embeddings" \
        --quiet || warning "Could not download embedding model"

    log "✓ Models downloaded"
}

################################################################################
# SSL Certificates
################################################################################

generate_ssl_certificates() {
    log "Generating SSL certificates..."

    if [[ -f "${INSTALL_DIR}/ssl/certificate.crt" ]]; then
        info "SSL certificates already exist"
        return
    fi

    openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
        -keyout "${INSTALL_DIR}/ssl/private.key" \
        -out "${INSTALL_DIR}/ssl/certificate.crt" \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=offline-ai.local"

    chmod 600 "${INSTALL_DIR}/ssl/private.key"
    chmod 644 "${INSTALL_DIR}/ssl/certificate.crt"

    log "✓ SSL certificates generated"
}

################################################################################
# Configuration Files
################################################################################

create_configuration_files() {
    log "Creating configuration files..."

    # .env file
    cat > "${INSTALL_DIR}/.env" << EOF
# Database Passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)
MONGO_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)

# MinIO Credentials
MINIO_USER=ai_admin
MINIO_PASSWORD=$(openssl rand -base64 32)

# RabbitMQ Credentials
RABBITMQ_USER=ai_admin
RABBITMQ_PASSWORD=$(openssl rand -base64 32)

# Grafana Admin Password
GRAFANA_PASSWORD=$(openssl rand -base64 32)

# Encryption Keys
VAULT_ENCRYPTION_KEY=$(openssl rand -hex 32)

# Resource Limits
MAX_AGENTS=50
MAX_MEMORY_PER_AGENT=2G
GPU_MEMORY_FRACTION=0.8
EOF

    chmod 600 "${INSTALL_DIR}/.env"

    # Agent configuration
    cat > "${CONFIG_DIR}/agents.yaml" << EOF
agent_factory:
  max_agents: 50
  resource_limits:
    cpu_cores: 16
    memory_gb: 32
    gpu_memory_gb: 16

models:
  default: llama-3.2-3b
  available:
    - llama-3.2-3b
    - mistral-7b
    - phi-3-mini

security_tools:
  suricata:
    enabled: true
    eve_log_path: /var/log/suricata/eve.json
  yara:
    enabled: true
    rules_dir: /etc/yara/rules
  clamav:
    enabled: true
  ossec:
    enabled: false
EOF

    log "✓ Configuration files created"
}

################################################################################
# Setup Backup System
################################################################################

setup_backup_system() {
    log "Setting up backup system..."

    # Create backup script
    cat > /usr/local/bin/backup-offline-ai.sh << 'EOF'
#!/bin/bash
set -e

BACKUP_DIR="/opt/offline-ai/backups"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/opt/offline-ai/logs/backup_${DATE}.log"

echo "Starting backup at $(date)" | tee -a $LOG_FILE

# Backup databases
if docker ps | grep -q offline_ai_postgres; then
    echo "Backing up PostgreSQL..." | tee -a $LOG_FILE
    docker exec offline_ai_postgres pg_dumpall -U ai_admin | \
        gzip > $BACKUP_DIR/postgres_$DATE.sql.gz
fi

if docker ps | grep -q offline_ai_mongodb; then
    echo "Backing up MongoDB..." | tee -a $LOG_FILE
    docker exec offline_ai_mongodb mongodump --archive | \
        gzip > $BACKUP_DIR/mongo_$DATE.archive.gz
fi

# Backup configurations
echo "Backing up configurations..." | tee -a $LOG_FILE
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /opt/offline-ai/config

# Clean old backups (keep 7 days)
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.archive.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed at $(date)" | tee -a $LOG_FILE
EOF

    chmod +x /usr/local/bin/backup-offline-ai.sh

    # Add cron job
    (crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup-offline-ai.sh") | crontab -

    log "✓ Backup system configured"
}

################################################################################
# Create Summary
################################################################################

create_summary() {
    log "Creating deployment summary..."

    # Get system info
    HOSTNAME=$(hostname)
    IP_ADDRESS=$(hostname -I | awk '{print $1}')

    # Read passwords from .env
    source "${INSTALL_DIR}/.env"

    cat > "${INSTALL_DIR}/INSTALLATION_SUMMARY.txt" << EOF
================================================================================
Offline AI Operating System - Installation Summary
================================================================================

Installation Date: $(date)
Hostname: $HOSTNAME
IP Address: $IP_ADDRESS

Installation Directory: ${INSTALL_DIR}

Services Deployed:
- PostgreSQL (port 5432)
- MongoDB (port 27017)
- Redis (port 6379)
- Qdrant (port 6333)
- MinIO (ports 9000, 9001)
- RabbitMQ (ports 5672, 15672)
- Prometheus (port 9090)
- Grafana (port 3000)

Access Credentials:
- PostgreSQL: ai_admin / ${POSTGRES_PASSWORD}
- MongoDB: ai_admin / ${MONGO_PASSWORD}
- Redis: ${REDIS_PASSWORD}
- MinIO: ai_admin / ${MINIO_PASSWORD}
- RabbitMQ: ai_admin / ${RABBITMQ_PASSWORD}
- Grafana: admin / ${GRAFANA_PASSWORD}

Access URLs:
- Grafana: https://${IP_ADDRESS}:3000
- Prometheus: http://${IP_ADDRESS}:9090
- MinIO Console: http://${IP_ADDRESS}:9001
- RabbitMQ Management: http://${IP_ADDRESS}:15672

Security Features:
- UFW Firewall: Enabled
- Fail2ban: Active
- AIDE: Initialized
- Auditd: Monitoring system changes
- SSL Certificates: Generated

Backup System:
- Schedule: Daily at 2:00 AM
- Location: ${BACKUP_DIR}
- Retention: 7 days

Next Steps:
1. Review and save credentials above
2. Configure Grafana dashboards
3. Test backup/restore procedures
4. Review security settings

For support: https://github.com/your-org/offline-ai-os

================================================================================
EOF

    chmod 600 "${INSTALL_DIR}/INSTALLATION_SUMMARY.txt"

    log "✓ Summary created at ${INSTALL_DIR}/INSTALLATION_SUMMARY.txt"
}

################################################################################
# Main Installation Flow
################################################################################

main() {
    clear
    echo "================================================================================"
    echo "  Offline AI Operating System - Automated Installation"
    echo "================================================================================"
    echo ""

    log "Starting installation..."

    # Run installation steps
    preflight_checks
    prepare_system
    harden_system
    install_docker
    install_gpu_support
    create_directories
    setup_python_environment
    install_security_tools
    download_models
    generate_ssl_certificates
    create_configuration_files
    setup_backup_system
    create_summary

    # Display summary
    echo ""
    echo "================================================================================"
    log "Installation Complete!"
    echo "================================================================================"
    echo ""
    cat "${INSTALL_DIR}/INSTALLATION_SUMMARY.txt"
    echo ""
    log "Please save the credentials shown above in a secure location"
    echo ""
}

# Run main installation
main "$@"