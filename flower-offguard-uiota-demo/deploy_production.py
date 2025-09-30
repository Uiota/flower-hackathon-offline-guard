#!/usr/bin/env python3
"""
Production Deployment Script for Federated Learning with Off-Guard Security
Handles production deployment with security hardening and monitoring
"""

import os
import sys
import subprocess
import platform
import shutil
import json
import time
import socket
from pathlib import Path
import secrets
import hashlib

class ProductionDeployment:
    """Production deployment manager for FL system."""

    def __init__(self):
        self.project_root = Path.cwd()
        self.deployment_dir = self.project_root / "deployment-output"
        self.build_dir = self.deployment_dir / "builds"
        self.config_dir = self.deployment_dir / "configs"

        # Create deployment structure
        self.deployment_dir.mkdir(exist_ok=True)
        self.build_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)

    def generate_security_keys(self):
        """Generate production security keys."""
        print("🔐 Generating production security keys...")

        # Create secure secrets directory
        secrets_dir = self.project_root / ".secrets"
        secrets_dir.mkdir(mode=0o700, exist_ok=True)

        # Generate server keys
        server_private_key = secrets.token_hex(32)
        server_public_key = hashlib.sha256(server_private_key.encode()).hexdigest()

        # API keys
        admin_api_key = secrets.token_urlsafe(32)
        monitoring_key = secrets.token_urlsafe(24)

        # SSL/TLS certificates (self-signed for demo)
        self.generate_ssl_certificates()

        # Save keys securely
        keys_data = {
            "server_private_key": server_private_key,
            "server_public_key": server_public_key,
            "admin_api_key": admin_api_key,
            "monitoring_key": monitoring_key,
            "generated_at": time.time(),
            "environment": "production"
        }

        keys_file = secrets_dir / "production_keys.json"
        with open(keys_file, "w") as f:
            json.dump(keys_data, f, indent=2)
        keys_file.chmod(0o600)

        print("   ✅ Security keys generated")
        return keys_data

    def generate_ssl_certificates(self):
        """Generate self-signed SSL certificates for demo."""
        print("   🔒 Generating SSL certificates...")

        try:
            # Create certificate directory
            cert_dir = self.project_root / ".secrets" / "ssl"
            cert_dir.mkdir(parents=True, exist_ok=True)

            # Generate private key and certificate using openssl
            key_file = cert_dir / "server.key"
            cert_file = cert_dir / "server.crt"

            # OpenSSL command to generate self-signed certificate
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:4096",
                "-keyout", str(key_file),
                "-out", str(cert_file),
                "-days", "365", "-nodes",
                "-subj", "/C=US/ST=Demo/L=Demo/O=FL-Demo/CN=localhost"
            ], check=True, capture_output=True)

            # Set secure permissions
            key_file.chmod(0o600)
            cert_file.chmod(0o644)

            print("   ✅ SSL certificates generated")

        except subprocess.CalledProcessError:
            print("   ⚠️  OpenSSL not available, skipping SSL certificate generation")
        except Exception as e:
            print(f"   ⚠️  SSL certificate generation failed: {e}")

    def create_production_config(self):
        """Create production configuration files."""
        print("⚙️  Creating production configuration...")

        # Load security keys
        keys_file = self.project_root / ".secrets" / "production_keys.json"
        if keys_file.exists():
            with open(keys_file) as f:
                keys_data = json.load(f)
        else:
            keys_data = self.generate_security_keys()

        # Production environment variables
        prod_env = f"""# Production Environment Configuration
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

# Security (Production)
OFFLINE_MODE=1
SECURITY_LEVEL=enterprise
PRODUCTION_READY=true
ADMIN_API_KEY={keys_data.get('admin_api_key', 'change-me')}

# SSL/TLS
TLS_ENABLED=true
SSL_CERT_PATH=.secrets/ssl/server.crt
SSL_KEY_PATH=.secrets/ssl/server.key

# Federated Learning (Production)
FL_SERVER_HOST=0.0.0.0
FL_SERVER_PORT=8080
FL_DASHBOARD_PORT=8081
FL_ROUNDS=20
FL_CLIENTS_PER_ROUND=10

# Security Hardening
SIGNATURE_REQUIRED=true
DP_ENABLED=true
DP_NOISE_MULTIPLIER=1.0
NETWORK_ENCRYPTION=true

# Logging (Production)
LOG_LEVEL=WARNING
LOG_FILE_ENABLED=true
LOG_ROTATION=daily
LOG_RETENTION_DAYS=90

# Performance (Production)
MAX_WORKERS=16
MEMORY_LIMIT_GB=8
CPU_LIMIT_CORES=8

# Monitoring
METRICS_ENABLED=true
PROMETHEUS_ENABLED=true
HEALTH_CHECK_INTERVAL=60

# Paths (Production)
DATA_DIR=/opt/fl-system/data
MODELS_DIR=/opt/fl-system/models
LOGS_DIR=/var/log/fl-system
ARTIFACTS_DIR=/opt/fl-system/artifacts

# Development Features (Disabled)
DEBUG_MODE=false
VERBOSE_LOGGING=false
AUTO_START_AGENTS=false
AUTO_OPEN_BROWSER=false
"""

        # Save production environment
        prod_env_file = self.config_dir / "production.env"
        with open(prod_env_file, "w") as f:
            f.write(prod_env)
        prod_env_file.chmod(0o600)

        print("   ✅ Production environment configuration created")

    def create_systemd_services(self):
        """Create systemd service files for Linux production deployment."""
        print("🔧 Creating systemd service files...")

        # FL Dashboard service
        dashboard_service = f"""[Unit]
Description=Federated Learning Dashboard
After=network.target
Wants=network.target

[Service]
Type=simple
User=fl-system
Group=fl-system
WorkingDirectory={self.project_root}
Environment=OFFLINE_MODE=1
EnvironmentFile={self.config_dir}/production.env
ExecStart={self.project_root}/.venv/bin/python dashboard_with_agents.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={self.project_root}/logs {self.project_root}/data

[Install]
WantedBy=multi-user.target
"""

        # FL Agent service
        agent_service = f"""[Unit]
Description=Federated Learning Agents
After=network.target fl-dashboard.service
Wants=network.target
Requires=fl-dashboard.service

[Service]
Type=simple
User=fl-system
Group=fl-system
WorkingDirectory={self.project_root}
Environment=OFFLINE_MODE=1
EnvironmentFile={self.config_dir}/production.env
ExecStart={self.project_root}/.venv/bin/python fl_agent_system.py
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={self.project_root}/logs {self.project_root}/data

[Install]
WantedBy=multi-user.target
"""

        # Save service files
        services_dir = self.config_dir / "systemd"
        services_dir.mkdir(exist_ok=True)

        with open(services_dir / "fl-dashboard.service", "w") as f:
            f.write(dashboard_service)

        with open(services_dir / "fl-agents.service", "w") as f:
            f.write(agent_service)

        print("   ✅ Systemd service files created")

    def create_docker_production(self):
        """Create production Docker configuration."""
        print("🐳 Creating production Docker configuration...")

        # Production Dockerfile
        dockerfile_prod = """FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    openssl \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r fl-system && useradd -r -g fl-system fl-system

# Set working directory
WORKDIR /opt/fl-system

# Copy and install Python dependencies
COPY requirements-full.txt .
RUN pip install --no-cache-dir -r requirements-full.txt

# Copy application code
COPY . .
RUN chown -R fl-system:fl-system /opt/fl-system

# Create necessary directories
RUN mkdir -p data logs models artifacts && \\
    chown -R fl-system:fl-system data logs models artifacts

# Switch to non-root user
USER fl-system

# Set environment variables
ENV OFFLINE_MODE=1
ENV PRODUCTION_READY=true
ENV PYTHONPATH=/opt/fl-system

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:8081/api/status || exit 1

# Default command
CMD ["python", "dashboard_with_agents.py"]
"""

        # Production Docker Compose
        compose_prod = """version: '3.8'

services:
  fl-dashboard:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8080:8080"
      - "8081:8081"
    environment:
      - OFFLINE_MODE=1
      - PRODUCTION_READY=true
      - FL_SERVER_HOST=0.0.0.0
    env_file:
      - deployment-output/configs/production.env
    volumes:
      - fl-data:/opt/fl-system/data
      - fl-logs:/opt/fl-system/logs
      - fl-models:/opt/fl-system/models
    networks:
      - fl-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/api/status"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  fl-agents:
    build:
      context: .
      dockerfile: Dockerfile.prod
    command: python fl_agent_system.py
    environment:
      - OFFLINE_MODE=1
      - FL_SERVER_HOST=fl-dashboard
    env_file:
      - deployment-output/configs/production.env
    volumes:
      - fl-data:/opt/fl-system/data
      - fl-logs:/opt/fl-system/logs
    networks:
      - fl-network
    depends_on:
      fl-dashboard:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 6G
        reservations:
          cpus: '2.0'
          memory: 3G

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./deployment-output/configs/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - fl-network
    restart: unless-stopped

volumes:
  fl-data:
  fl-logs:
  fl-models:
  prometheus-data:

networks:
  fl-network:
    driver: bridge
"""

        # Save production Docker files
        with open(self.project_root / "Dockerfile.prod", "w") as f:
            f.write(dockerfile_prod)

        with open(self.config_dir / "docker-compose.prod.yml", "w") as f:
            f.write(compose_prod)

        print("   ✅ Production Docker configuration created")

    def create_monitoring_config(self):
        """Create monitoring and observability configuration."""
        print("📊 Creating monitoring configuration...")

        # Prometheus configuration
        prometheus_config = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fl-dashboard'
    static_configs:
      - targets: ['fl-dashboard:8081']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'fl-agents'
    static_configs:
      - targets: ['fl-agents:8082']
    metrics_path: '/metrics'
    scrape_interval: 30s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
"""

        # Alert rules
        alert_rules = """groups:
- name: fl_system_alerts
  rules:
  - alert: FL_Dashboard_Down
    expr: up{job="fl-dashboard"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "FL Dashboard is down"

  - alert: FL_Agents_Low
    expr: fl_active_clients < 3
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Low number of active FL clients"

  - alert: FL_Training_Stalled
    expr: increase(fl_training_rounds[10m]) == 0
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "FL training appears to be stalled"
"""

        # Save monitoring configuration
        with open(self.config_dir / "prometheus.yml", "w") as f:
            f.write(prometheus_config)

        with open(self.config_dir / "alert_rules.yml", "w") as f:
            f.write(alert_rules)

        print("   ✅ Monitoring configuration created")

    def create_deployment_scripts(self):
        """Create deployment and management scripts."""
        print("📜 Creating deployment scripts...")

        # Production deployment script
        deploy_script = f"""#!/bin/bash
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
if [ ! -f "{self.config_dir}/production.env" ]; then
    echo "❌ Production configuration not found"
    echo "Run: python3 deploy_production.py"
    exit 1
fi

# Load configuration
source {self.config_dir}/production.env

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
    sudo cp {self.config_dir}/systemd/*.service /etc/systemd/system/
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
"""

        deploy_path = self.config_dir / "deploy.sh"
        with open(deploy_path, "w") as f:
            f.write(deploy_script)
        deploy_path.chmod(0o755)

        print("   ✅ Deployment scripts created")

    def run_production_setup(self):
        """Run complete production setup."""
        print("🏭 PRODUCTION DEPLOYMENT SETUP")
        print("🔒 Enterprise Security • 📊 Monitoring • 🐳 Containers")
        print("=" * 60)

        steps = [
            ("Generating security keys", self.generate_security_keys),
            ("Creating production config", self.create_production_config),
            ("Creating systemd services", self.create_systemd_services),
            ("Creating Docker production config", self.create_docker_production),
            ("Creating monitoring config", self.create_monitoring_config),
            ("Creating deployment scripts", self.create_deployment_scripts),
        ]

        success_count = 0
        for step_name, step_function in steps:
            print(f"\n{step_name}...")
            try:
                step_function()
                success_count += 1
            except Exception as e:
                print(f"   ❌ {step_name} failed: {e}")

        print("\n" + "=" * 60)
        print(f"🏁 PRODUCTION SETUP: {success_count}/{len(steps)} steps successful")

        if success_count == len(steps):
            print("✅ Production setup completed successfully!")
            self.show_production_next_steps()
        else:
            print("⚠️  Setup completed with some issues")

    def show_production_next_steps(self):
        """Show production deployment next steps."""
        print("\n🎯 PRODUCTION DEPLOYMENT:")
        print("=" * 40)
        print("1. Review configuration:")
        print(f"   cat {self.config_dir}/production.env")

        print("\n2. Deploy with Docker:")
        print("   docker-compose -f deployment-output/configs/docker-compose.prod.yml up -d")

        print("\n3. Deploy with systemd (Linux):")
        print(f"   bash {self.config_dir}/deploy.sh")

        print("\n4. Access production dashboard:")
        print("   https://your-server:8081")

        print("\n5. Monitor with Prometheus:")
        print("   http://your-server:9090")

        print("\n🔒 SECURITY NOTES:")
        print("   • Change default passwords")
        print("   • Configure firewall rules")
        print("   • Set up TLS certificates")
        print("   • Review security configurations")

def main():
    """Main entry point."""
    deployment = ProductionDeployment()
    deployment.run_production_setup()

if __name__ == "__main__":
    main()