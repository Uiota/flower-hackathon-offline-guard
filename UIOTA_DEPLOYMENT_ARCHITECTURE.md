# UIOTA Container & Kubernetes Deployment Architecture
## Complete Infrastructure Specifications for UIOTA Framework

**Version**: 1.0.0  
**Date**: September 11, 2025  
**Foundation**: Building on existing `/containers/kubernetes/` infrastructure  
**Architecture**: Based on UIOTA Framework Architecture

---

## 🏗️ DEPLOYMENT OVERVIEW

The UIOTA deployment architecture extends the existing offline-guard Kubernetes infrastructure to support the complete UIOTA ecosystem. All new components integrate seamlessly with the existing production-ready environment.

**Foundation Analysis** (from `/containers/kubernetes/`):
- ✅ **Namespace**: `offline-guard-prod` already configured
- ✅ **Ingress**: TLS termination and load balancing ready
- ✅ **Monitoring**: Prometheus, Grafana, and AlertManager deployed
- ✅ **Security**: NetworkPolicies, PodSecurityPolicies, and RBAC configured
- ✅ **Storage**: PVC templates and storage classes defined
- ✅ **Secrets**: External secrets management patterns established

**UIOTA Extensions Required**:
- 🔄 **New Services**: offline.ai, MCP server, offline DNS, backend API
- 🔄 **Enhanced Monitoring**: Guardian-specific metrics and dashboards  
- 🔄 **Extended Security**: Guardian authentication and authorization
- 🔄 **Additional Storage**: Model storage, Guardian data, DNS cache
- 🔄 **Service Mesh**: Enhanced inter-service communication

---

## 🐳 CONTAINER SPECIFICATIONS

### **1. offline.ai Inference Engine**
```yaml
# /containers/kubernetes/offline-ai-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: offline-ai-deployment
  namespace: offline-guard-prod
  labels:
    app: offline-guard
    component: offline-ai
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: offline-guard
      component: offline-ai
  template:
    metadata:
      labels:
        app: offline-guard
        component: offline-ai
        version: v1.0.0
    spec:
      serviceAccountName: offline-ai-service-account
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: offline-ai
        image: uiota/offline-ai:1.0.0
        ports:
        - containerPort: 11434
          name: ollama-api
        - containerPort: 8080
          name: http-api
        - containerPort: 9090
          name: metrics
        env:
        - name: GUARDIAN_AUTH_ENDPOINT
          value: "http://backend-api-service:8000/api/v1/auth"
        - name: MODEL_CACHE_DIR
          value: "/app/models"
        - name: FLOWER_SERVER_ADDRESS
          value: "flower-server.offline-guard-prod.svc.cluster.local:8080"
        - name: PROMETHEUS_METRICS_ENABLED
          value: "true"
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        - name: guardian-config
          mountPath: /app/config
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: offline-ai-models-pvc
      - name: guardian-config
        configMap:
          name: guardian-config
      nodeSelector:
        node-type: gpu-enabled
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

---
# Model Storage PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: offline-ai-models-pvc
  namespace: offline-guard-prod
  labels:
    app: offline-guard
    component: storage
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 200Gi

---
# Service Account with RBAC
apiVersion: v1
kind: ServiceAccount
metadata:
  name: offline-ai-service-account
  namespace: offline-guard-prod

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: offline-ai-role
  namespace: offline-guard-prod
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: offline-ai-rolebinding
  namespace: offline-guard-prod
subjects:
- kind: ServiceAccount
  name: offline-ai-service-account
  namespace: offline-guard-prod
roleRef:
  kind: Role
  name: offline-ai-role
  apiGroup: rbac.authorization.k8s.io
```

### **2. MCP Server**
```yaml
# /containers/kubernetes/mcp-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server-deployment
  namespace: offline-guard-prod
  labels:
    app: offline-guard
    component: mcp-server
    version: v1.0.0
spec:
  replicas: 2
  selector:
    matchLabels:
      app: offline-guard
      component: mcp-server
  template:
    metadata:
      labels:
        app: offline-guard
        component: mcp-server
        version: v1.0.0
    spec:
      serviceAccountName: mcp-server-service-account
      containers:
      - name: mcp-server
        image: uiota/mcp-server:1.0.0
        ports:
        - containerPort: 8080
          name: http-api
        - containerPort: 8081
          name: websocket
        - containerPort: 11434
          name: ollama-compat
        - containerPort: 9090
          name: metrics
        env:
        - name: GUARDIAN_AUTH_ENDPOINT
          valueFrom:
            configMapKeyRef:
              name: uiota-config
              key: guardian-auth-endpoint
        - name: OFFLINE_AI_ENDPOINT
          value: "http://offline-ai-service:11434"
        - name: BACKEND_API_ENDPOINT
          value: "http://backend-api-service:8000"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secrets
              key: url
        resources:
          requests:
            memory: "1Gi"
            cpu: "0.5"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mcp-server-service-account
  namespace: offline-guard-prod
```

### **3. Offline DNS Server**
```yaml
# /containers/kubernetes/offline-dns-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: offline-dns-deployment
  namespace: offline-guard-prod
  labels:
    app: offline-guard
    component: offline-dns
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: offline-guard
      component: offline-dns
  template:
    metadata:
      labels:
        app: offline-guard
        component: offline-dns
        version: v1.0.0
    spec:
      serviceAccountName: offline-dns-service-account
      containers:
      - name: offline-dns
        image: uiota/offline-dns:1.0.0
        ports:
        - containerPort: 53
          name: dns-udp
          protocol: UDP
        - containerPort: 53
          name: dns-tcp
          protocol: TCP
        - containerPort: 853
          name: dns-tls
        - containerPort: 443
          name: dns-https
        - containerPort: 9090
          name: metrics
        env:
        - name: GUARDIAN_AUTH_ENDPOINT
          value: "http://backend-api-service:8000/api/v1/auth"
        - name: DNS_CACHE_SIZE
          value: "1000000"
        - name: MESH_COORDINATION_ENABLED
          value: "true"
        resources:
          requests:
            memory: "512Mi"
            cpu: "0.25"
          limits:
            memory: "1Gi"
            cpu: "0.5"
        volumeMounts:
        - name: dns-cache
          mountPath: /app/cache
        - name: guardian-dns-config
          mountPath: /app/config
        livenessProbe:
          tcpSocket:
            port: 53
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          tcpSocket:
            port: 53
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: dns-cache
        persistentVolumeClaim:
          claimName: offline-dns-cache-pvc
      - name: guardian-dns-config
        configMap:
          name: guardian-dns-config

---
# DNS Cache Storage
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: offline-dns-cache-pvc
  namespace: offline-guard-prod
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 10Gi

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: offline-dns-service-account
  namespace: offline-guard-prod
```

### **4. Backend API Server**
```yaml
# /containers/kubernetes/backend-api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-api-deployment
  namespace: offline-guard-prod
  labels:
    app: offline-guard
    component: backend-api
    version: v1.0.0
spec:
  replicas: 5
  selector:
    matchLabels:
      app: offline-guard
      component: backend-api
  template:
    metadata:
      labels:
        app: offline-guard
        component: backend-api
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: backend-api-service-account
      containers:
      - name: backend-api
        image: uiota/backend-api:1.0.0
        ports:
        - containerPort: 8000
          name: http-api
        - containerPort: 9090
          name: metrics
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secrets
              key: url
        - name: GUARDIAN_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: guardian-secrets
              key: secret-key
        - name: FLOWER_COORDINATION_ENABLED
          value: "true"
        - name: DISCORD_BOT_INTEGRATION
          value: "true"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: guardian-data
          mountPath: /app/guardian-data
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: guardian-data
        persistentVolumeClaim:
          claimName: guardian-data-pvc

---
# Guardian Data Storage (already exists, extended)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: guardian-data-pvc-extended
  namespace: offline-guard-prod
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 50Gi

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: backend-api-service-account
  namespace: offline-guard-prod
```

### **5. Security Monitoring Stack**
```yaml
# /containers/kubernetes/security-monitor-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: security-monitor-deployment
  namespace: offline-guard-prod
  labels:
    app: offline-guard
    component: security-monitor
    version: v1.0.0
spec:
  replicas: 2
  selector:
    matchLabels:
      app: offline-guard
      component: security-monitor
  template:
    metadata:
      labels:
        app: offline-guard
        component: security-monitor
        version: v1.0.0
    spec:
      serviceAccountName: security-monitor-service-account
      containers:
      - name: security-monitor
        image: uiota/security-monitor:1.0.0
        ports:
        - containerPort: 8080
          name: http-api
        - containerPort: 9090
          name: metrics
        env:
        - name: GUARDIAN_AUTH_ENDPOINT
          value: "http://backend-api-service:8000/api/v1/auth"
        - name: PROMETHEUS_URL
          value: "http://prometheus-server:9090"
        - name: ALERT_MANAGER_URL
          value: "http://alertmanager:9093"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: security-data
          mountPath: /app/security-data
        - name: threat-intelligence
          mountPath: /app/threat-intel
      volumes:
      - name: security-data
        persistentVolumeClaim:
          claimName: security-data-pvc
      - name: threat-intelligence
        configMap:
          name: threat-intelligence-feeds

---
# Security Data Storage
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: security-data-pvc
  namespace: offline-guard-prod
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 20Gi

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: security-monitor-service-account
  namespace: offline-guard-prod
```

---

## 🌐 SERVICE DEFINITIONS & NETWORKING

### **Extended Service Configuration**
```yaml
# /containers/kubernetes/uiota-services-extension.yaml
# Extends existing production-services.yaml

# offline.ai Service
---
apiVersion: v1
kind: Service
metadata:
  name: offline-ai-service
  namespace: offline-guard-prod
  labels:
    app: offline-guard
    component: offline-ai
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
spec:
  type: ClusterIP
  selector:
    app: offline-guard
    component: offline-ai
  ports:
  - name: ollama-api
    port: 11434
    targetPort: 11434
  - name: http-api
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090

# MCP Server Service
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server-service
  namespace: offline-guard-prod
  labels:
    app: offline-guard
    component: mcp-server
spec:
  type: ClusterIP
  selector:
    app: offline-guard
    component: mcp-server
  ports:
  - name: http-api
    port: 8080
    targetPort: 8080
  - name: websocket
    port: 8081
    targetPort: 8081
  - name: ollama-compat
    port: 11434
    targetPort: 11434

# Backend API Service
---
apiVersion: v1
kind: Service
metadata:
  name: backend-api-service
  namespace: offline-guard-prod
  labels:
    app: offline-guard
    component: backend-api
spec:
  type: ClusterIP
  selector:
    app: offline-guard
    component: backend-api
  ports:
  - name: http-api
    port: 8000
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090

# Offline DNS Service
---
apiVersion: v1
kind: Service
metadata:
  name: offline-dns-service
  namespace: offline-guard-prod
  labels:
    app: offline-guard
    component: offline-dns
spec:
  type: LoadBalancer
  selector:
    app: offline-guard
    component: offline-dns
  ports:
  - name: dns-udp
    port: 53
    targetPort: 53
    protocol: UDP
  - name: dns-tcp
    port: 53
    targetPort: 53
    protocol: TCP
  - name: dns-tls
    port: 853
    targetPort: 853
  - name: dns-https
    port: 443
    targetPort: 443

# Security Monitor Service
---
apiVersion: v1
kind: Service
metadata:
  name: security-monitor-service
  namespace: offline-guard-prod
  labels:
    app: offline-guard
    component: security-monitor
spec:
  type: ClusterIP
  selector:
    app: offline-guard
    component: security-monitor
  ports:
  - name: http-api
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
```

### **Extended Ingress Configuration**
```yaml
# /containers/kubernetes/uiota-ingress-extension.yaml
# Extends existing offline-guard-prod-ingress

apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: uiota-api-ingress
  namespace: offline-guard-prod
  labels:
    app: offline-guard
    component: ingress
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    
    # Enhanced security for UIOTA APIs
    nginx.ingress.kubernetes.io/auth-response-headers: "X-Guardian-ID, X-Guardian-Class"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://offline.ai, https://app.uiota.dev"
    
    # Rate limiting specific to UIOTA services
    nginx.ingress.kubernetes.io/rate-limit-rps: "100"
    nginx.ingress.kubernetes.io/rate-limit-connections: "50"
    
spec:
  tls:
  - hosts:
    - api.uiota.dev
    - offline.ai
    - mcp.uiota.dev
    - dns.uiota.dev
    - security.uiota.dev
    secretName: uiota-tls-cert
  rules:
  
  # Backend API
  - host: api.uiota.dev
    http:
      paths:
      - path: /api/v1
        pathType: Prefix
        backend:
          service:
            name: backend-api-service
            port:
              number: 8000
  
  # offline.ai Inference
  - host: offline.ai
    http:
      paths:
      - path: /api/v1
        pathType: Prefix
        backend:
          service:
            name: offline-ai-service
            port:
              number: 8080
      - path: /v1 # OpenAI compatibility
        pathType: Prefix
        backend:
          service:
            name: offline-ai-service
            port:
              number: 11434
  
  # MCP Server
  - host: mcp.uiota.dev
    http:
      paths:
      - path: /mcp
        pathType: Prefix
        backend:
          service:
            name: mcp-server-service
            port:
              number: 8080
      - path: /ws # WebSocket endpoint
        pathType: Prefix
        backend:
          service:
            name: mcp-server-service
            port:
              number: 8081
  
  # Security Dashboard
  - host: security.uiota.dev
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: security-monitor-service
            port:
              number: 8080
```

---

## 🔐 ENHANCED SECURITY CONFIGURATION

### **Network Policies**
```yaml
# /containers/kubernetes/uiota-network-policies.yaml
# Guardian-specific network segmentation

apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: offline-ai-network-policy
  namespace: offline-guard-prod
spec:
  podSelector:
    matchLabels:
      component: offline-ai
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          component: backend-api
    - podSelector:
        matchLabels:
          component: mcp-server
    ports:
    - protocol: TCP
      port: 11434
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          component: backend-api
    ports:
    - protocol: TCP
      port: 8000
  - to: [] # DNS and external model downloads
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 443

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: guardian-data-protection-policy
  namespace: offline-guard-prod
spec:
  podSelector:
    matchLabels:
      component: backend-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          component: frontend
    - podSelector:
        matchLabels:
          component: mcp-server
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          component: cache # Redis
    - podSelector:
        matchLabels:
          component: database # PostgreSQL
    - podSelector:
        matchLabels:
          component: offline-ai
    ports:
    - protocol: TCP
      port: 6379
    - protocol: TCP
      port: 5432
    - protocol: TCP
      port: 11434
```

### **Pod Security Policies**
```yaml
# /containers/kubernetes/uiota-pod-security-policies.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: uiota-restricted-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  supplementalGroups:
    rule: 'MustRunAs'
    ranges:
      - min: 1000
        max: 65535
  fsGroup:
    rule: 'MustRunAs'
    ranges:
      - min: 1000
        max: 65535
  readOnlyRootFilesystem: false
  seLinux:
    rule: 'RunAsAny'

---
# Guardian-specific security context
apiVersion: v1
kind: SecurityContextConstraints
metadata:
  name: guardian-scc
allowHostDirVolumePlugin: false
allowHostIPC: false
allowHostNetwork: false
allowHostPID: false
allowHostPorts: false
allowPrivilegedContainer: false
allowedCapabilities: null
defaultAddCapabilities: null
requiredDropCapabilities:
- KILL
- MKNOD
- SETUID
- SETGID
runAsUser:
  type: MustRunAsRange
  uidRangeMin: 1000
  uidRangeMax: 65535
seLinuxContext:
  type: MustRunAs
supplementalGroups:
  type: RunAsAny
users:
- system:serviceaccount:offline-guard-prod:offline-ai-service-account
- system:serviceaccount:offline-guard-prod:backend-api-service-account
- system:serviceaccount:offline-guard-prod:mcp-server-service-account
volumes:
- configMap
- downwardAPI
- emptyDir
- persistentVolumeClaim
- projected
- secret
```

---

## 📊 MONITORING & OBSERVABILITY

### **Enhanced Monitoring Configuration**
```yaml
# /containers/kubernetes/uiota-monitoring-extension.yaml
# Extends existing monitoring-stack.yml

# Guardian-specific ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: uiota-components-monitor
  namespace: offline-guard-prod
  labels:
    app: offline-guard
    monitoring: prometheus
spec:
  selector:
    matchLabels:
      app: offline-guard
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
    scheme: http
  namespaceSelector:
    matchNames:
    - offline-guard-prod

---
# Guardian Activity Dashboard ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: guardian-dashboard-config
  namespace: offline-guard-prod
data:
  guardian-activity.json: |
    {
      "dashboard": {
        "id": null,
        "title": "Guardian Activity Dashboard",
        "tags": ["guardian", "uiota"],
        "timezone": "UTC",
        "panels": [
          {
            "title": "Active Guardians by Class",
            "type": "piechart",
            "targets": [
              {
                "expr": "guardian_active_count_by_class",
                "legendFormat": "{{guardian_class}}"
              }
            ]
          },
          {
            "title": "Federated Learning Rounds",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(federated_learning_rounds_total[5m])",
                "legendFormat": "FL Rounds per minute"
              }
            ]
          },
          {
            "title": "API Response Times by Component",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                "legendFormat": "{{component}} 95th percentile"
              }
            ]
          }
        ]
      }
    }

---
# Enhanced PrometheusRule for Guardian metrics
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: uiota-guardian-alerts
  namespace: offline-guard-prod
  labels:
    app: offline-guard
    prometheus: kube-prometheus
spec:
  groups:
  - name: guardian.rules
    rules:
    - alert: GuardianAuthenticationFailures
      expr: rate(guardian_auth_failures_total[5m]) > 0.1
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High Guardian authentication failure rate"
        description: "Guardian authentication failures are occurring at {{ $value }} per second"
    
    - alert: FederatedLearningStalled
      expr: time() - federated_learning_last_successful_round > 3600
      for: 10m
      labels:
        severity: critical
      annotations:
        summary: "Federated learning has stalled"
        description: "No successful federated learning rounds in the last hour"
    
    - alert: OfflineAIHighMemoryUsage
      expr: container_memory_usage_bytes{container="offline-ai"} / container_spec_memory_limit_bytes{container="offline-ai"} > 0.9
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "offline.ai container high memory usage"
        description: "offline.ai container is using {{ $value | humanizePercentage }} of available memory"
    
    - alert: GuardianXPSystemDown
      expr: up{job="backend-api"} == 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Guardian XP system is down"
        description: "Backend API responsible for Guardian XP tracking is unavailable"
```

---

## 🚀 DEPLOYMENT AUTOMATION

### **Helm Chart Structure**
```yaml
# /containers/helm/uiota-chart/Chart.yaml
apiVersion: v2
name: uiota-framework
description: Complete UIOTA sovereign AI framework deployment
version: 1.0.0
appVersion: "1.0.0"
dependencies:
- name: postgresql
  version: "11.9.13"
  repository: https://charts.bitnami.com/bitnami
- name: redis
  version: "17.3.7"
  repository: https://charts.bitnami.com/bitnami
- name: prometheus
  version: "15.5.3"
  repository: https://prometheus-community.github.io/helm-charts
```

### **Values Configuration**
```yaml
# /containers/helm/uiota-chart/values.yaml
global:
  namespace: offline-guard-prod
  imageRegistry: docker.io/uiota
  storageClass: fast-ssd

# Component-specific configurations
offlineAI:
  enabled: true
  image:
    repository: uiota/offline-ai
    tag: "1.0.0"
  replicas: 3
  resources:
    requests:
      memory: "8Gi"
      cpu: "2"
      nvidia.com/gpu: "1"
    limits:
      memory: "16Gi"
      cpu: "4"
      nvidia.com/gpu: "1"
  storage:
    models: "200Gi"

mcpServer:
  enabled: true
  image:
    repository: uiota/mcp-server
    tag: "1.0.0"
  replicas: 2
  resources:
    requests:
      memory: "1Gi"
      cpu: "0.5"
    limits:
      memory: "2Gi"
      cpu: "1"

backendAPI:
  enabled: true
  image:
    repository: uiota/backend-api
    tag: "1.0.0"
  replicas: 5
  resources:
    requests:
      memory: "2Gi"
      cpu: "1"
    limits:
      memory: "4Gi"
      cpu: "2"
  storage:
    guardianData: "50Gi"

offlineDNS:
  enabled: true
  image:
    repository: uiota/offline-dns
    tag: "1.0.0"
  replicas: 3
  resources:
    requests:
      memory: "512Mi"
      cpu: "0.25"
    limits:
      memory: "1Gi"
      cpu: "0.5"
  storage:
    cache: "10Gi"

securityMonitor:
  enabled: true
  image:
    repository: uiota/security-monitor
    tag: "1.0.0"
  replicas: 2
  resources:
    requests:
      memory: "4Gi"
      cpu: "2"
    limits:
      memory: "8Gi"
      cpu: "4"
  storage:
    securityData: "20Gi"

# Guardian-specific configurations
guardian:
  authentication:
    enabled: true
    secretKey: "change-in-production"
  classes:
    - CryptoGuardian
    - FederatedLearner
    - MobileMaster
    - GhostVerifier
    - TeamCoordinator
  xpSystem:
    enabled: true
    baseXpPerActivity: 10

# External dependencies
postgresql:
  enabled: true
  auth:
    postgresPassword: "change-in-production"
    database: "uiota"
  primary:
    persistence:
      size: "100Gi"
      storageClass: "fast-ssd"

redis:
  enabled: true
  auth:
    password: "change-in-production"
  master:
    persistence:
      size: "10Gi"
      storageClass: "fast-ssd"

# Monitoring configuration
monitoring:
  prometheus:
    enabled: true
    retention: "30d"
    storage: "50Gi"
  grafana:
    enabled: true
    adminPassword: "change-in-production"
  alertmanager:
    enabled: true
    config:
      slack:
        apiURL: "change-in-production"
        channel: "#uiota-alerts"
```

### **Deployment Scripts**
```bash
#!/bin/bash
# /containers/deploy-uiota.sh
# Complete UIOTA deployment automation

set -e

echo "🛡️ Starting UIOTA Framework Deployment"

# Check prerequisites
echo "📋 Checking prerequisites..."
kubectl cluster-info
helm version

# Create namespace if it doesn't exist
echo "🏗️ Creating namespace..."
kubectl create namespace offline-guard-prod --dry-run=client -o yaml | kubectl apply -f -

# Deploy external secrets (if using external secret manager)
echo "🔐 Setting up secrets..."
kubectl apply -f /containers/kubernetes/secrets/

# Deploy UIOTA framework with Helm
echo "🚀 Deploying UIOTA components..."
helm upgrade --install uiota-framework /containers/helm/uiota-chart \
  --namespace offline-guard-prod \
  --values /containers/helm/uiota-chart/values.yaml \
  --wait --timeout=600s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n offline-guard-prod
kubectl get services -n offline-guard-prod
kubectl get ingress -n offline-guard-prod

# Run health checks
echo "🏥 Running health checks..."
kubectl wait --for=condition=ready pod -l app=offline-guard -n offline-guard-prod --timeout=300s

# Display access information
echo "🌐 Access Information:"
echo "- API: https://api.uiota.dev"
echo "- offline.ai: https://offline.ai"
echo "- MCP Server: https://mcp.uiota.dev"
echo "- DNS: dns.uiota.dev"
echo "- Security: https://security.uiota.dev"
echo "- Monitoring: https://monitor.uiota.dev"

echo "🎉 UIOTA Framework deployment complete!"
```

---

## 🔄 AUTOSCALING & RESOURCE MANAGEMENT

### **Horizontal Pod Autoscaler**
```yaml
# /containers/kubernetes/uiota-hpa.yaml
# Dynamic scaling based on Guardian activity

# Backend API HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-api-hpa
  namespace: offline-guard-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend-api-deployment
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: guardian_active_sessions
      target:
        type: AverageValue
        averageValue: "50"

---
# offline.ai HPA (GPU-aware)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: offline-ai-hpa
  namespace: offline-guard-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: offline-ai-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  - type: Pods
    pods:
      metric:
        name: inference_queue_length
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 600
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
```

### **Vertical Pod Autoscaler**
```yaml
# /containers/kubernetes/uiota-vpa.yaml
# Memory and CPU optimization

apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: offline-ai-vpa
  namespace: offline-guard-prod
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: offline-ai-deployment
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: offline-ai
      maxAllowed:
        memory: 32Gi
        cpu: 8
        nvidia.com/gpu: 2
      minAllowed:
        memory: 4Gi
        cpu: 1
        nvidia.com/gpu: 1
```

---

## 📋 DEPLOYMENT CHECKLIST

### **Pre-Deployment Verification**
```yaml
prerequisites:
  - ✅ Kubernetes cluster (1.24+)
  - ✅ Helm 3.8+
  - ✅ Container registry access
  - ✅ Storage classes configured
  - ✅ GPU nodes available (for offline.ai)
  - ✅ External DNS configured
  - ✅ SSL certificates ready
  - ✅ Secrets management configured

infrastructure_validation:
  - ✅ Existing offline-guard namespace accessible
  - ✅ Production services running
  - ✅ Monitoring stack operational
  - ✅ Ingress controller configured
  - ✅ Network policies applied
  - ✅ RBAC permissions set
```

### **Post-Deployment Validation**
```yaml
service_health_checks:
  - offline.ai: "GET /health returns 200"
  - backend-api: "Guardian auth endpoint functional"
  - mcp-server: "WebSocket connections working"
  - offline-dns: "DNS queries resolving"
  - security-monitor: "Metrics collection active"

integration_tests:
  - Guardian authentication flow
  - Federated learning round coordination
  - Cross-component communication
  - Monitoring and alerting
  - Security policy enforcement

performance_validation:
  - API response times within SLA
  - Autoscaling triggers working
  - Resource utilization optimized
  - Guardian XP tracking functional
```

---

This comprehensive deployment architecture extends the existing offline-guard Kubernetes infrastructure to support the complete UIOTA ecosystem. All components are designed to integrate seamlessly with the current environment while providing the scalability and security required for sovereign AI operations.

**Ready for production deployment with Guardian-powered coordination.** 🛡️⚡🚀