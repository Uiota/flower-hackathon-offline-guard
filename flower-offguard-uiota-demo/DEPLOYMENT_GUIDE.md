# Offline AI Operating System - Production Deployment Guide 🚀

**Complete deployment guide for production-ready Offline AI OS**

---

## 📋 Overview

This guide covers **production deployment** of the Offline AI Operating System with:

- ✅ **Docker Compose** multi-container orchestration
- ✅ **Ansible** automated infrastructure provisioning
- ✅ **Security hardening** (UFW, fail2ban, AIDE, auditd)
- ✅ **GPU support** (NVIDIA Container Toolkit)
- ✅ **Monitoring** (Prometheus + Grafana)
- ✅ **Automated backups** (PostgreSQL, MongoDB, Redis)
- ✅ **High availability** (multi-node clustering)

---

## 🏗️ Architecture

### Infrastructure Stack

```
┌─────────────────────────────────────────────────────────┐
│                    NGINX API GATEWAY                    │
│                      (Ports 80, 443)                    │
└────────────────────┬────────────────────────────────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
│  Agent    │ │ Threat    │ │ Incident  │
│Controller │ │ Detectors │ │ Responders│
│  (Master) │ │  (x3)     │ │    (x2)   │
└─────┬─────┘ └─────┬─────┘ └─────┬─────┘
      │             │              │
      └─────────────┼──────────────┘
                    │
      ┌─────────────┴─────────────┐
      │                           │
┌─────▼──────┐          ┌────────▼────────┐
│ Data Layer │          │ Message Layer   │
│            │          │                 │
│ PostgreSQL │          │ RabbitMQ        │
│ MongoDB    │          │ Redis (Queue)   │
│ Qdrant     │          │                 │
│ MinIO      │          │                 │
└────────────┘          └─────────────────┘
      │
┌─────▼──────────────────┐
│  Monitoring Stack      │
│                        │
│  Prometheus            │
│  Grafana               │
└────────────────────────┘
```

---

## 🚦 Prerequisites

### Hardware Requirements

**Minimum (Development)**:
- CPU: 8 cores
- RAM: 16 GB
- Disk: 100 GB SSD
- GPU: Optional (NVIDIA with 8GB VRAM)

**Recommended (Production)**:
- CPU: 16+ cores
- RAM: 64 GB
- Disk: 500 GB NVMe SSD
- GPU: NVIDIA RTX 3080+ (16GB+ VRAM)

**High-Availability Cluster**:
- 3+ nodes (1 master, 2+ workers)
- Each with production specs above
- Dedicated storage node (optional)

### Software Requirements

- **OS**: Ubuntu 22.04 LTS (recommended)
- **Python**: 3.10+
- **Ansible**: 2.10+
- **Docker**: 24.0+
- **Docker Compose**: 2.20+

---

## 📦 Installation Methods

### Method 1: Docker Compose (Single Node)

**Best for**: Development, testing, small deployments

**Steps**:

1. **Clone repository**:
```bash
cd /opt
git clone https://github.com/your-org/offline-ai-os.git
cd offline-ai-os
```

2. **Configure environment**:
```bash
cp deployment/.env.example deployment/.env
nano deployment/.env  # Edit passwords and settings
```

3. **Generate SSL certificates**:
```bash
mkdir -p deployment/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
  -keyout deployment/nginx/ssl/private.key \
  -out deployment/nginx/ssl/certificate.crt
```

4. **Pull images**:
```bash
cd deployment
docker-compose pull
```

5. **Start services**:
```bash
docker-compose up -d
```

6. **Verify deployment**:
```bash
docker-compose ps
docker-compose logs -f agent_controller
```

7. **Access services**:
- Grafana: `https://localhost:3000` (admin / <GRAFANA_PASSWORD>)
- Prometheus: `http://localhost:9090`
- MinIO Console: `http://localhost:9001`
- RabbitMQ Management: `http://localhost:15672`

---

### Method 2: Ansible (Multi-Node Cluster)

**Best for**: Production, high-availability, scale-out deployments

**Steps**:

#### 1. Prepare Control Node

```bash
# Install Ansible
sudo apt update
sudo apt install -y ansible python3-pip

# Install required Ansible collections
ansible-galaxy collection install community.docker
ansible-galaxy collection install ansible.posix

# Install Python dependencies
pip3 install docker docker-compose
```

#### 2. Configure Inventory

```bash
cd deployment/ansible

# Edit inventory file
nano inventory/hosts.yml
```

**Update with your node IPs**:
```yaml
ai_nodes:
  hosts:
    ai-node-01:
      ansible_host: 192.168.1.10  # Change to your IP
      node_role: master

    ai-node-02:
      ansible_host: 192.168.1.11  # Change to your IP
      node_role: worker

    ai-node-03:
      ansible_host: 192.168.1.12  # Change to your IP
      node_role: worker
```

#### 3. Setup SSH Access

```bash
# Generate SSH key (if not exists)
ssh-keygen -t ed25519 -C "ansible@offline-ai"

# Copy to all nodes
ssh-copy-id admin@192.168.1.10
ssh-copy-id admin@192.168.1.11
ssh-copy-id admin@192.168.1.12

# Test connectivity
ansible all -i inventory/hosts.yml -m ping
```

#### 4. Run Pre-Deployment Checks

```bash
# Check Ansible syntax
ansible-playbook deploy_offline_ai.yml -i inventory/hosts.yml --syntax-check

# Dry run (check mode)
ansible-playbook deploy_offline_ai.yml -i inventory/hosts.yml --check

# List tasks
ansible-playbook deploy_offline_ai.yml -i inventory/hosts.yml --list-tasks
```

#### 5. Deploy System

```bash
# Full deployment
ansible-playbook deploy_offline_ai.yml -i inventory/hosts.yml

# Deploy to specific node
ansible-playbook deploy_offline_ai.yml -i inventory/hosts.yml --limit ai-node-01

# Deploy specific phases
ansible-playbook deploy_offline_ai.yml -i inventory/hosts.yml --tags "system_hardening,container_runtime"
```

#### 6. Verify Deployment

```bash
# Check deployment status
ansible all -i inventory/hosts.yml -m command -a "docker ps"

# View deployment summary
ansible all -i inventory/hosts.yml -m command -a "cat /opt/offline-ai/DEPLOYMENT_SUMMARY.txt"

# Check service health
ansible all -i inventory/hosts.yml -m command -a "systemctl status docker"
```

---

## 🔐 Security Configuration

### 1. Firewall Rules

**UFW** (Uncomplicated Firewall) is configured automatically by Ansible.

**Manual configuration**:
```bash
# Enable firewall
sudo ufw enable

# Allow SSH (IMPORTANT: Do this first!)
sudo ufw allow 22/tcp

# Allow application ports
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 3000/tcp  # Grafana
sudo ufw allow 9090/tcp  # Prometheus

# Check status
sudo ufw status verbose
```

### 2. Fail2ban Configuration

**Fail2ban** protects against brute-force attacks.

**Configuration** (`/etc/fail2ban/jail.local`):
```ini
[DEFAULT]
bantime = 3600      # 1 hour ban
findtime = 600      # 10 minute window
maxretry = 3        # 3 failed attempts

[sshd]
enabled = true
port = 22

[nginx-http-auth]
enabled = true
port = http,https
```

**Check banned IPs**:
```bash
sudo fail2ban-client status sshd
```

### 3. AppArmor Profiles

**AppArmor** provides mandatory access control.

**Check status**:
```bash
sudo aa-status
```

**Create custom profile** for Docker containers:
```bash
sudo nano /etc/apparmor.d/docker-ai-controller

# Reload profiles
sudo apparmor_parser -r /etc/apparmor.d/docker-ai-controller
```

### 4. Audit Logging

**Auditd** monitors system changes.

**View audit logs**:
```bash
# Recent audit events
sudo ausearch -ts today

# AI system changes
sudo ausearch -k ai_system_changes

# Process executions
sudo ausearch -k process_execution
```

---

## 📊 Monitoring & Observability

### Prometheus

**Access**: `http://localhost:9090`

**Metrics Available**:
- Container resource usage (CPU, memory)
- Agent performance metrics
- Database query performance
- Network traffic statistics

**Query Examples**:
```promql
# CPU usage by container
rate(container_cpu_usage_seconds_total[5m])

# Memory usage
container_memory_usage_bytes

# Agent message throughput
rate(agent_messages_sent_total[1m])
```

### Grafana

**Access**: `https://localhost:3000`
**Default Credentials**: `admin` / `<GRAFANA_PASSWORD>` (from `.env`)

**Pre-configured Dashboards**:
1. **System Overview** - Node resources, container health
2. **Agent Performance** - Message rates, task completion
3. **Database Metrics** - Query performance, connection pools
4. **Network Security** - Threat detections, incident responses

**Add Data Source**:
1. Settings → Data Sources → Add data source
2. Select "Prometheus"
3. URL: `http://prometheus:9090`
4. Save & Test

---

## 💾 Backup & Recovery

### Automated Backups

**Schedule**: Daily at 2:00 AM (configured via cron)

**Backup Script**: `/usr/local/bin/backup-ai-system.sh`

**Backup Location**: `/opt/offline-ai/backups/`

**Components Backed Up**:
- PostgreSQL (full database dump)
- MongoDB (archive dump)
- Redis (RDB snapshot)
- Configuration files
- SSL certificates

**Retention**: 7 days (older backups automatically deleted)

### Manual Backup

```bash
# Run backup script manually
sudo /usr/local/bin/backup-ai-system.sh

# List backups
ls -lh /opt/offline-ai/backups/

# Create custom backup
DATE=$(date +%Y%m%d_%H%M%S)
docker exec offline_ai_postgres pg_dumpall -U ai_admin > backup_$DATE.sql
```

### Restore from Backup

**PostgreSQL**:
```bash
# Stop application containers
cd /opt/offline-ai/deployment
docker-compose stop agent_controller threat_detector

# Restore database
gunzip -c /opt/offline-ai/backups/postgres_20250929_020000.sql.gz | \
  docker exec -i offline_ai_postgres psql -U ai_admin

# Restart containers
docker-compose up -d
```

**MongoDB**:
```bash
gunzip -c /opt/offline-ai/backups/mongo_20250929_020000.archive.gz | \
  docker exec -i offline_ai_mongodb mongorestore --archive
```

**Redis**:
```bash
# Copy RDB file to Redis data volume
docker cp /opt/offline-ai/backups/redis_20250929_020000.rdb \
  offline_ai_redis:/data/dump.rdb

# Restart Redis
docker restart offline_ai_redis
```

---

## 🔧 Configuration

### Environment Variables

**File**: `/opt/offline-ai/.env`

**Critical Variables**:
```bash
# Database passwords (CHANGE THESE!)
POSTGRES_PASSWORD=<strong_password_here>
MONGO_PASSWORD=<strong_password_here>
REDIS_PASSWORD=<strong_password_here>

# Service credentials
MINIO_USER=ai_admin
MINIO_PASSWORD=<strong_password_here>
RABBITMQ_USER=ai_admin
RABBITMQ_PASSWORD=<strong_password_here>

# Grafana admin password
GRAFANA_PASSWORD=<strong_password_here>

# Resource limits
MAX_AGENTS=50
MAX_MEMORY_PER_AGENT=2G
GPU_MEMORY_FRACTION=0.8
```

**Apply Changes**:
```bash
cd /opt/offline-ai/deployment
docker-compose down
docker-compose up -d
```

### Agent Configuration

**File**: `/opt/offline-ai/config/agent_config.yml`

```yaml
agents:
  threat_detector:
    count: 3
    threshold: 0.75
    scan_interval: 60

  incident_responder:
    count: 2
    auto_mitigation: false
    response_timeout: 300

  coordinator:
    enabled: true
    max_managed_agents: 50
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Container Won't Start

**Check logs**:
```bash
docker-compose logs -f <service_name>
```

**Common causes**:
- Port already in use
- Insufficient memory
- Database not initialized

**Solutions**:
```bash
# Check port usage
sudo netstat -tulpn | grep <port>

# Restart service
docker-compose restart <service_name>

# Rebuild container
docker-compose up -d --force-recreate <service_name>
```

#### 2. Database Connection Failed

**Test connection**:
```bash
# PostgreSQL
docker exec offline_ai_postgres pg_isready -U ai_admin

# MongoDB
docker exec offline_ai_mongodb mongosh --eval "db.adminCommand('ping')"

# Redis
docker exec offline_ai_redis redis-cli -a $REDIS_PASSWORD ping
```

**Reset database**:
```bash
# WARNING: This deletes all data!
docker-compose down -v
docker-compose up -d
```

#### 3. GPU Not Detected

**Check NVIDIA driver**:
```bash
nvidia-smi
```

**Test GPU in Docker**:
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

**Reinstall NVIDIA Container Toolkit**:
```bash
sudo apt-get install --reinstall nvidia-container-toolkit
sudo systemctl restart docker
```

#### 4. Out of Memory

**Check memory usage**:
```bash
docker stats
```

**Increase Docker memory limit** (`/etc/docker/daemon.json`):
```json
{
  "default-ulimits": {
    "memlock": {
      "Hard": -1,
      "Name": "memlock",
      "Soft": -1
    }
  }
}
```

**Restart Docker**:
```bash
sudo systemctl restart docker
```

---

## 📈 Scaling

### Horizontal Scaling (Add More Nodes)

**1. Add node to inventory**:
```yaml
ai-node-04:
  ansible_host: 192.168.1.13
  node_role: worker
```

**2. Run deployment**:
```bash
ansible-playbook deploy_offline_ai.yml -i inventory/hosts.yml --limit ai-node-04
```

**3. Verify node joined cluster**:
```bash
ansible all -i inventory/hosts.yml -m command -a "docker node ls"
```

### Vertical Scaling (Increase Resources)

**1. Update `docker-compose.yml`**:
```yaml
agent_controller:
  deploy:
    resources:
      limits:
        cpus: '8'      # Increase from 4
        memory: 16G    # Increase from 8G
```

**2. Restart services**:
```bash
docker-compose up -d --force-recreate agent_controller
```

### Agent Scaling

**Scale threat detectors**:
```bash
docker-compose up -d --scale threat_detector=5
```

**Update configuration**:
```yaml
# agent_config.yml
agents:
  threat_detector:
    count: 5  # Increased from 3
```

---

## 🧪 Testing

### Health Checks

**Script**: `deployment/scripts/health_check.sh`

```bash
#!/bin/bash

# Check all services
services=("postgres" "mongodb" "redis" "qdrant" "minio" "rabbitmq")

for service in "${services[@]}"; do
  if docker exec offline_ai_$service echo "OK" > /dev/null 2>&1; then
    echo "✓ $service: Healthy"
  else
    echo "✗ $service: Unhealthy"
  fi
done
```

**Run health checks**:
```bash
chmod +x deployment/scripts/health_check.sh
./deployment/scripts/health_check.sh
```

### Load Testing

**Install Apache Bench**:
```bash
sudo apt install apache2-utils
```

**Test agent controller API**:
```bash
ab -n 1000 -c 10 http://localhost:8000/api/health
```

---

## 📚 Additional Resources

- **Agent System Documentation**: `OFFLINE_AI_OS_PHASE1.md`
- **Agent Comparison Guide**: `AGENT_SYSTEM_COMPARISON.md`
- **Memory Guardian**: `MEMORY_GUARDIAN_README.md`
- **Secure Metrics**: `SECURE_METRICS_README.md`
- **LL TOKEN System**: `LL_TOKEN_SPECIFICATIONS.md`

---

## 🎉 Deployment Complete!

Your Offline AI Operating System is now running in production!

**Next Steps**:
1. ✅ Review security hardening
2. ✅ Configure Grafana dashboards
3. ✅ Test backup/restore procedures
4. ✅ Setup monitoring alerts
5. ✅ Train team on system operations

---

**🚀 Offline AI Operating System** - *Production deployment made simple*

*Enterprise-grade cybersecurity with multi-agent AI*