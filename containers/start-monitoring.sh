#!/bin/bash
# Start the complete monitoring stack for Offline Guard

set -euo pipefail

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

log_section() {
    echo -e "\n${PURPLE}=== $1 ===${NC}\n"
}

# Banner
echo -e "${PURPLE}"
cat << 'EOF'
  ____   __  __ _ _              ____                     _ 
 / __ \ / _|/ _| (_)            / ___|_   _  __ _ _ __ __| |
| |  | | |_| |_| |_ _ __   ___  | |  _| | | |/ _` | '__/ _` |
| |  | |  _|  _| | | '_ \ / _ \ | |_| | |_| | (_| | | | (_| |
 \____/|_| |_| |_|_|_| |_|_____| \____|\__,_|\__,_|_|  \__,_|

🔍 MONITORING & OBSERVABILITY STACK
🌸 Perfect for Flower AI Hackathon Teams! 🌸

EOF
echo -e "${NC}"

# Setup monitoring data directories
setup_monitoring_directories() {
    log_section "Setting up Monitoring Directories"
    
    local dirs=(
        "data/prometheus-data"
        "data/grafana-data"
        "data/jaeger-data"
        "data/loki-data"
        "data/alertmanager-data"
        "data/uptime-data"
        "monitoring/grafana/dashboards/hackathon"
        "monitoring/grafana/dashboards/guardian"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        log_info "Created directory: $dir"
    done
    
    log_success "Monitoring directories ready"
}

# Create additional monitoring configurations
create_monitoring_configs() {
    log_section "Creating Additional Monitoring Configurations"
    
    # Loki configuration
    cat > monitoring/loki/loki-config.yaml << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /tmp/loki
  storage:
    filesystem:
      chunks_directory: /tmp/loki/chunks
      rules_directory: /tmp/loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory

query_range:
  results_cache:
    cache:
      embedded_cache:
        enabled: true
        max_size_mb: 100

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

ruler:
  alertmanager_url: http://alertmanager:9093

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
  ingestion_rate_mb: 16
  ingestion_burst_size_mb: 32
  max_streams_per_user: 10000
  max_line_size: 256kb

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s
EOF

    # Promtail configuration
    cat > monitoring/promtail/promtail-config.yaml << 'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: containerlogs
          __path__: /var/log/containers/*log

  - job_name: podman
    static_configs:
      - targets:
          - localhost
        labels:
          job: podmanlogs
          __path__: /var/log/podman/*log

  - job_name: system
    static_configs:
      - targets:
          - localhost
        labels:
          job: systemlogs
          __path__: /var/log/syslog
EOF

    # AlertManager configuration
    cat > monitoring/alertmanager/alertmanager.yml << 'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@offline-guard.dev'
  slack_api_url: ''

route:
  group_by: ['alertname', 'team']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'guardian-team'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      team: guardian
    receiver: 'guardian-team'
  - match:
      team: ml-guardian
    receiver: 'ml-team'
  - match:
      team: container-guardian
    receiver: 'container-team'

receivers:
- name: 'guardian-team'
  webhook_configs:
  - url: 'http://guardian-service:3001/api/alerts/webhook'
    title: 'Offline Guard Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

- name: 'critical-alerts'
  webhook_configs:
  - url: 'http://guardian-service:3001/api/alerts/critical'
    title: 'CRITICAL: Offline Guard Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}\nImpact: {{ .Annotations.hackathon_impact }}{{ end }}'

- name: 'ml-team'
  webhook_configs:
  - url: 'http://guardian-service:3001/api/alerts/ml'
    title: 'ML Guardian Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

- name: 'container-team'
  webhook_configs:
  - url: 'http://guardian-service:3001/api/alerts/container'
    title: 'Container Guardian Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF
    
    log_success "Additional monitoring configurations created"
}

# Create Guardian-themed dashboard
create_guardian_dashboard() {
    log_section "Creating Guardian-Themed Dashboard"
    
    cat > monitoring/grafana/dashboards/guardian/guardian-overview.json << 'EOF'
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": {
          "type": "grafana",
          "uid": "-- Grafana --"
        },
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 0,
  "id": 1,
  "links": [],
  "liveNow": false,
  "panels": [
    {
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "drawStyle": "line",
            "fillOpacity": 10,
            "gradientMode": "none",
            "hideFrom": {
              "legend": false,
              "tooltip": false,
              "vis": false
            },
            "lineInterpolation": "linear",
            "lineWidth": 1,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "never",
            "spanNulls": false,
            "stacking": {
              "group": "A",
              "mode": "none"
            },
            "thresholdsStyle": {
              "mode": "off"
            }
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          },
          "unit": "short"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "id": 1,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "mode": "single",
          "sort": "none"
        }
      },
      "targets": [
        {
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
          },
          "expr": "up{job=~\".*guard.*\"}",
          "refId": "A"
        }
      ],
      "title": "Guardian Services Status",
      "type": "timeseries"
    }
  ],
  "refresh": "30s",
  "schemaVersion": 37,
  "style": "dark",
  "tags": [
    "guardian",
    "hackathon",
    "offline-guard"
  ],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-1h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Guardian Overview Dashboard",
  "uid": "guardian-overview",
  "version": 1,
  "weekStart": ""
}
EOF
    
    log_success "Guardian dashboard created"
}

# Start monitoring stack
start_monitoring() {
    log_section "Starting Monitoring Stack"
    
    # Check if main services are running
    if ! podman ps | grep -q "offline-guard"; then
        log_warning "Main Offline Guard services not detected"
        log_info "You may want to start them first with: podman-compose -f podman-compose.yml up -d"
    fi
    
    # Start monitoring stack
    log_info "Starting monitoring services..."
    podman-compose -f monitoring-stack.yml up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    
    local services=(
        "prometheus:9090"
        "grafana:3000"
        "jaeger:16686"
        "loki:3100"
        "alertmanager:9093"
        "uptime-kuma:3001"
    )
    
    for service in "${services[@]}"; do
        name=$(echo "$service" | cut -d: -f1)
        port=$(echo "$service" | cut -d: -f2)
        
        log_info "Checking $name on port $port..."
        timeout 120 bash -c "until curl -f http://localhost:$port/ &>/dev/null; do sleep 2; done" || {
            log_warning "$name may not be ready yet"
        }
    done
    
    log_success "Monitoring stack started successfully!"
}

# Show access information
show_access_info() {
    log_section "Monitoring Stack Access Information"
    
    echo "🎯 MONITORING DASHBOARDS:"
    echo "   📊 Grafana:         http://localhost:3002 (guardian/hackathon2025)"
    echo "   📈 Prometheus:      http://localhost:9090"
    echo "   🔍 Jaeger Tracing:  http://localhost:16686"
    echo "   ⚠️  AlertManager:    http://localhost:9093"
    echo "   ✅ Uptime Monitor:  http://localhost:3003"
    echo ""
    echo "🔧 METRICS ENDPOINTS:"
    echo "   📊 Node Metrics:    http://localhost:9100/metrics"
    echo "   🐳 Container Metrics: http://localhost:8081"
    echo "   📝 Logs (Loki):     http://localhost:3100"
    echo ""
    echo "🎮 GUARDIAN FEATURES:"
    echo "   • Real-time service monitoring"
    echo "   • Guardian role-based alerts"
    echo "   • Hackathon performance metrics"
    echo "   • Team collaboration insights"
    echo "   • Offline-first monitoring"
    echo ""
    echo "🌸 Perfect for Flower AI Hackathon teams!"
    echo "   Track your Guardian characters' performance"
    echo "   Monitor federated learning experiments"
    echo "   Observe offline-first architecture health"
}

# Main execution
main() {
    local action="${1:-start}"
    
    case "$action" in
        "start")
            setup_monitoring_directories
            create_monitoring_configs
            create_guardian_dashboard
            start_monitoring
            show_access_info
            ;;
        "stop")
            log_section "Stopping Monitoring Stack"
            podman-compose -f monitoring-stack.yml down
            log_success "Monitoring stack stopped"
            ;;
        "status")
            log_section "Monitoring Stack Status"
            podman-compose -f monitoring-stack.yml ps
            ;;
        "restart")
            log_section "Restarting Monitoring Stack"
            podman-compose -f monitoring-stack.yml restart
            log_success "Monitoring stack restarted"
            ;;
        "logs")
            service="${2:-}"
            if [ -n "$service" ]; then
                podman-compose -f monitoring-stack.yml logs -f "$service"
            else
                podman-compose -f monitoring-stack.yml logs
            fi
            ;;
        *)
            echo "Usage: $0 {start|stop|status|restart|logs [service]}"
            echo ""
            echo "Commands:"
            echo "  start   - Setup and start the complete monitoring stack"
            echo "  stop    - Stop all monitoring services"
            echo "  status  - Show status of monitoring services"
            echo "  restart - Restart the monitoring stack"
            echo "  logs    - Show logs for all or specific service"
            echo ""
            echo "Examples:"
            echo "  $0 start"
            echo "  $0 logs grafana"
            echo "  $0 status"
            ;;
    esac
}

# Run main function with all arguments
main "$@"