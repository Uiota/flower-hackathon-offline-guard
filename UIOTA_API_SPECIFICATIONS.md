# UIOTA API Specifications & Integration Protocols
## Component APIs and Inter-Service Communication

**Version**: 1.0.0  
**Date**: September 11, 2025  
**Architecture**: Based on UIOTA Framework Architecture

---

## 🔗 API OVERVIEW

The UIOTA ecosystem uses multiple API layers for different purposes:
- **REST APIs**: External interfaces and Guardian interactions
- **gRPC APIs**: High-performance inter-service communication  
- **WebSocket APIs**: Real-time Guardian coordination and federated learning
- **Guardian Protocols**: Specialized authentication and authorization

---

## 🛡️ GUARDIAN AUTHENTICATION API

### **Guardian Token Authentication**
```yaml
endpoint: "/api/v1/auth/guardian"
method: "POST"
purpose: "Authenticate Guardian and receive access token"

request_body:
  guardian_id: "guardian_crypto_001"
  guardian_class: "CryptoGuardian" 
  signature: "0x..." # Guardian cryptographic signature
  timestamp: "2025-09-11T17:30:00Z"
  nonce: "unique_random_string"

response:
  access_token: "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
  refresh_token: "refresh_token_string"
  expires_in: 3600
  guardian_profile:
    id: "guardian_crypto_001"
    class: "CryptoGuardian"
    level: 7
    permissions: ["cryptography", "verification", "security"]
    specializations: ["QR_generation", "offline_proofs"]
```

### **Guardian Permission Check**
```yaml
endpoint: "/api/v1/auth/permissions"
method: "GET"
headers:
  Authorization: "Bearer {guardian_token}"

response:
  guardian_id: "guardian_crypto_001"
  permissions:
    - cryptography
    - verification
    - security
  specializations:
    - QR_generation
    - offline_proofs
    - signatures
  access_level: "master"
  can_access:
    - "/api/v1/crypto/*"
    - "/api/v1/verification/*"
    - "/api/v1/security/*"
```

---

## 🤖 BACKEND API SPECIFICATIONS

### **Guardian Management Endpoints**

#### **Get Guardian Profile**
```yaml
endpoint: "/api/v1/guardians/{guardian_id}"
method: "GET"
authentication: "Guardian Bearer Token"

response:
  guardian_id: "guardian_crypto_001"
  display_name: "Alice CryptoMaster"
  guardian_class: "CryptoGuardian"
  level: 7
  xp: 1250
  xp_to_next_level: 250
  specializations:
    - name: "QR Generation"
      level: "Expert"
    - name: "Cryptographic Proofs"
      level: "Master"
  contributions:
    - timestamp: "2025-09-11T15:00:00Z"
      type: "commit"
      description: "Added Guardian evolution system"
      xp_gained: 50
  team_memberships:
    - team_id: "team_hackathon_001"
      role: "Security Lead"
  achievements:
    - "Lightning Commits" 
    - "Cipher Master"
    - "Team Player"
```

#### **Update Guardian Profile**
```yaml
endpoint: "/api/v1/guardians/{guardian_id}"
method: "PUT"
authentication: "Guardian Bearer Token (self or admin)"

request_body:
  display_name: "Alice CryptoMaster Supreme"
  specializations:
    - "Advanced Cryptography"
    - "Hardware Verification"
  preferences:
    notifications: true
    public_profile: true

response:
  success: true
  updated_fields: ["display_name", "specializations", "preferences"]
  new_xp: 25 # XP for profile improvement
```

### **Team Management Endpoints**

#### **Create Team**
```yaml
endpoint: "/api/v1/teams"
method: "POST"
authentication: "Guardian Bearer Token"

request_body:
  team_name: "Hackathon Legends"
  description: "Building sovereign AI systems"
  required_skills: ["android", "blockchain", "federated_learning"]
  max_members: 5
  team_lead: "guardian_crypto_001"

response:
  team_id: "team_hackathon_002"
  invite_code: "HACK2025"
  creation_timestamp: "2025-09-11T17:30:00Z"
  status: "recruiting"
```

#### **Join Team**
```yaml
endpoint: "/api/v1/teams/{team_id}/join"
method: "POST"
authentication: "Guardian Bearer Token"

request_body:
  invite_code: "HACK2025"
  role_preference: "ML Engineer"

response:
  success: true
  team_role: "Fed Learning Specialist"
  team_info:
    name: "Hackathon Legends"
    members_count: 3
    skills_covered: ["android", "blockchain", "federated_learning"]
    missing_skills: ["ui/ux"]
```

### **Federated Learning Coordination**

#### **Start Federated Training Round**
```yaml
endpoint: "/api/v1/federation/rounds"
method: "POST"
authentication: "Guardian Bearer Token (FederatedLearner class)"

request_body:
  model_name: "guardian_classifier_v1"
  participating_guardians: 
    - "guardian_fed_001"
    - "guardian_fed_002"  
    - "guardian_fed_003"
  training_config:
    rounds: 10
    epochs_per_round: 5
    learning_rate: 0.01

response:
  round_id: "fl_round_20250911_001"
  status: "initializing"
  participants:
    - guardian_id: "guardian_fed_001"
      status: "ready"
    - guardian_id: "guardian_fed_002" 
      status: "downloading_model"
  estimated_duration: "15 minutes"
  coordinator: "guardian_fed_001"
```

#### **Get Federated Learning Status**
```yaml
endpoint: "/api/v1/federation/rounds/{round_id}"
method: "GET"
authentication: "Guardian Bearer Token"

response:
  round_id: "fl_round_20250911_001"
  status: "training" # initializing, training, aggregating, completed, failed
  current_round: 3
  total_rounds: 10
  participants:
    - guardian_id: "guardian_fed_001"
      status: "training"
      progress: 0.6
      last_update: "2025-09-11T17:32:00Z"
  global_metrics:
    loss: 0.234
    accuracy: 0.876
    improvement: 0.023
  estimated_completion: "2025-09-11T17:45:00Z"
```

---

## 🌐 OFFLINE.AI INFERENCE API

### **Model Management**

#### **List Available Models**
```yaml
endpoint: "/api/v1/models"
method: "GET"
authentication: "Guardian Bearer Token"

response:
  models:
    - model_id: "llama2-7b-chat"
      name: "Llama 2 7B Chat"
      type: "conversational"
      size: "13.5GB"
      status: "ready"
      guardian_permissions: ["all"]
    - model_id: "mistral-7b-instruct"
      name: "Mistral 7B Instruct"
      type: "instruct"
      size: "14.2GB"
      status: "downloading"
      progress: 0.65
      guardian_permissions: ["FederatedLearner", "TeamCoordinator"]
    - model_id: "guardian-classifier-v1"
      name: "Guardian Skill Classifier"
      type: "classification"
      size: "1.2GB"
      status: "ready"
      guardian_permissions: ["TeamCoordinator"]
      federated_model: true
```

#### **Load Model for Inference**
```yaml
endpoint: "/api/v1/models/{model_id}/load"
method: "POST"
authentication: "Guardian Bearer Token with model access"

request_body:
  context_size: 4096
  temperature: 0.7
  max_tokens: 512
  guardian_context: "CryptoGuardian analyzing security patterns"

response:
  session_id: "session_guardian_001_20250911"
  model_loaded: true
  estimated_memory_usage: "8.2GB"
  context_window: 4096
  ready_for_inference: true
```

### **Inference Endpoints**

#### **Generate Text/Chat Completion**
```yaml
endpoint: "/api/v1/inference/chat"
method: "POST"
authentication: "Guardian Bearer Token + Session ID"

request_body:
  session_id: "session_guardian_001_20250911"
  messages:
    - role: "system"
      content: "You are a helpful assistant specializing in cryptography and security."
    - role: "user"
      content: "Explain how QR codes can be used for offline verification"
  guardian_context:
    guardian_id: "guardian_crypto_001"
    specializations: ["cryptography", "QR_generation"]
    access_level: "expert"

response:
  response_id: "resp_20250911_173245_001"
  message:
    role: "assistant"
    content: "QR codes can provide offline verification by encoding cryptographic signatures..."
  usage:
    prompt_tokens: 45
    completion_tokens: 287
    total_tokens: 332
  guardian_xp_awarded: 5 # XP for using AI assistance
```

#### **Embedding Generation**
```yaml
endpoint: "/api/v1/inference/embeddings"
method: "POST"
authentication: "Guardian Bearer Token"

request_body:
  text: "Guardian CryptoMaster specializes in offline verification systems"
  model: "all-MiniLM-L6-v2"
  guardian_context: "skill_analysis"

response:
  embedding: [0.123, -0.456, 0.789, ...] # 384-dimensional vector
  dimensions: 384
  processing_time_ms: 45
```

---

## 🌸 FLOWER FEDERATION PROTOCOL API

### **Federated Learning Client Registration**

#### **Register Guardian as FL Client**
```yaml
endpoint: "/api/v1/flower/clients/register"
method: "POST"
authentication: "Guardian Bearer Token (FederatedLearner class)"

request_body:
  guardian_id: "guardian_fed_001"
  client_capabilities:
    - "pytorch_training"
    - "tensorflow_inference"
    - "differential_privacy"
  available_datasets:
    - name: "guardian_interactions"
      size: 10000
      privacy_level: "high"
  hardware_info:
    cpu_cores: 8
    memory_gb: 16
    gpu: "NVIDIA GTX 1080"

response:
  client_id: "fl_client_guardian_fed_001"
  registration_token: "reg_token_12345"
  assigned_server: "fl-server.uiota.local"
  initial_model_url: "/api/v1/flower/models/guardian_classifier_v1/initial"
  status: "registered"
```

#### **Get Training Assignment**
```yaml
endpoint: "/api/v1/flower/training/assignment"
method: "GET"
authentication: "Guardian Bearer Token + Client Registration"
headers:
  X-Client-ID: "fl_client_guardian_fed_001"

response:
  assignment_id: "train_assign_20250911_001"
  round_number: 5
  model_url: "/api/v1/flower/models/guardian_classifier_v1/round_4"
  training_config:
    epochs: 3
    batch_size: 32
    learning_rate: 0.001
    privacy_budget: 1.0 # Differential privacy
  deadline: "2025-09-11T18:00:00Z"
  expected_participants: 15
```

### **Model Update Submission**

#### **Submit Local Training Results**
```yaml
endpoint: "/api/v1/flower/training/submit"
method: "POST"
authentication: "Guardian Bearer Token"
content_type: "application/octet-stream"

headers:
  X-Client-ID: "fl_client_guardian_fed_001"
  X-Assignment-ID: "train_assign_20250911_001"
  X-Model-Hash: "sha256:abc123..."

request_body: # Binary model weights
  [binary model parameters]

response:
  submission_id: "submit_20250911_173001"
  status: "accepted"
  validation_score: 0.92
  guardian_xp_awarded: 100 # XP for completing FL round
  next_round_eta: "2025-09-11T18:15:00Z"
```

---

## 📱 MOBILE API INTEGRATION

### **Offline Detection & QR Generation**

#### **Generate Offline Proof**
```yaml
endpoint: "/api/v1/mobile/proof/generate"
method: "POST"
authentication: "Guardian Bearer Token"

request_body:
  device_info:
    platform: "android"
    version: "1.0.0"
    device_id: "android_device_001"
  location_context:
    network_available: false
    last_online: "2025-09-11T16:30:00Z"
    offline_duration: 3600 # seconds
  guardian_id: "guardian_mobile_001"

response:
  proof_id: "offline_proof_20250911_001"
  qr_code_data: "UIOTA:PROOF:eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
  qr_code_image: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..." 
  expiry: "2025-09-11T19:30:00Z"
  verification_endpoint: "/api/v1/mobile/proof/verify"
```

#### **Verify Offline Proof (Ghost Verifier)**
```yaml
endpoint: "/api/v1/mobile/proof/verify"
method: "POST"
authentication: "Guardian Bearer Token (GhostVerifier class)"

request_body:
  qr_code_data: "UIOTA:PROOF:eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
  verifier_id: "guardian_ghost_001"
  verification_timestamp: "2025-09-11T17:35:00Z"
  location_context:
    verifier_location: "hackathon_venue"
    verification_method: "raspberry_pi_scanner"

response:
  verification_id: "verify_20250911_001"
  proof_valid: true
  original_guardian: "guardian_mobile_001"
  offline_duration_verified: 3600
  trust_score: 0.95
  guardian_xp_awarded: 25 # XP for successful verification
```

---

## 🔒 MCP SERVER PROTOCOL

### **Claude Code Integration**

#### **MCP Tool Registration**
```yaml
endpoint: "/mcp/tools/register"
method: "POST"
authentication: "MCP Server Token"

request_body:
  tool_name: "guardian_coordination"
  description: "Coordinate Guardian activities and federated learning"
  schema:
    type: "object"
    properties:
      action:
        type: "string"
        enum: ["create_team", "start_training", "get_status"]
      guardian_context:
        type: "object"
        properties:
          guardian_id: {"type": "string"}
          guardian_class: {"type": "string"}
  
response:
  tool_id: "tool_guardian_coordination_001"
  status: "registered"
  available_to: "claude_code_clients"
```

#### **Execute MCP Tool**
```yaml
endpoint: "/mcp/tools/{tool_id}/execute"
method: "POST"
authentication: "Claude Code Client Token"

request_body:
  parameters:
    action: "start_training"
    guardian_context:
      guardian_id: "guardian_fed_001"
      guardian_class: "FederatedLearner"
    training_config:
      model: "guardian_classifier_v1"
      participants: 5

response:
  execution_id: "exec_20250911_001"
  result:
    round_id: "fl_round_20250911_002"
    status: "initializing"
    participants: 3
    estimated_duration: "12 minutes"
  guardian_activity_logged: true
```

---

## 🌐 OFFLINE DNS PROTOCOL

### **Service Discovery**

#### **Register UIOTA Service**
```yaml
endpoint: "/dns/services/register"
method: "POST"
authentication: "Guardian Bearer Token"

request_body:
  service_name: "guardian-coordinator"
  service_type: "_guardian._tcp"
  domain: "local.uiota"
  port: 8080
  guardian_owner: "guardian_coord_001"
  metadata:
    guardian_class: "TeamCoordinator"
    capabilities: ["team_management", "coordination"]
    version: "1.0.0"

response:
  service_id: "svc_guardian_coord_001"
  full_domain: "guardian-coordinator.local.uiota"
  status: "registered"
  ttl: 3600
```

#### **Discover UIOTA Services**
```yaml
endpoint: "/dns/services/discover"
method: "GET"
authentication: "Guardian Bearer Token"
parameters:
  service_type: "_guardian._tcp"
  domain: "local.uiota"

response:
  services:
    - service_id: "svc_guardian_coord_001"
      name: "guardian-coordinator.local.uiota"
      address: "192.168.1.100"
      port: 8080
      guardian_owner: "guardian_coord_001"
      capabilities: ["team_management"]
      last_seen: "2025-09-11T17:30:00Z"
    - service_id: "svc_fl_server_001"
      name: "flower-server.local.uiota"
      address: "192.168.1.101"
      port: 8080
      guardian_owner: "guardian_fed_001"
      capabilities: ["federated_learning"]
```

---

## 🔐 SECURITY & MONITORING APIS

### **Security Event Reporting**

#### **Report Security Event**
```yaml
endpoint: "/api/v1/security/events"
method: "POST"
authentication: "Guardian Bearer Token (GhostVerifier or admin)"

request_body:
  event_type: "authentication_failure"
  severity: "medium"
  source_ip: "192.168.1.150"
  target_resource: "/api/v1/models/llama2-7b-chat"
  guardian_context:
    attempted_guardian_id: "unknown_guardian_999"
    guardian_class_claimed: "FederatedLearner"
  details:
    user_agent: "UIOTAMobile/1.0.0"
    failure_reason: "invalid_signature"

response:
  event_id: "sec_event_20250911_001"
  status: "logged"
  assigned_investigator: "guardian_ghost_001"
  alert_level: "watch"
  guardian_xp_awarded: 15 # XP for security reporting
```

#### **Get Security Dashboard**
```yaml
endpoint: "/api/v1/security/dashboard"
method: "GET"
authentication: "Guardian Bearer Token (GhostVerifier or admin)"

response:
  summary:
    total_events_24h: 23
    high_severity_events: 1
    blocked_attacks: 5
    active_threats: 0
  recent_events:
    - event_id: "sec_event_20250911_001"
      timestamp: "2025-09-11T17:30:00Z"
      type: "authentication_failure"
      severity: "medium"
      status: "investigating"
  guardian_security_score: 95
  recommendations:
    - "Enable 2FA for all CryptoGuardian accounts"
    - "Update firewall rules to block IP 192.168.1.150"
```

---

## 📊 MONITORING & METRICS APIS

### **Guardian Activity Metrics**

#### **Get Guardian Activity Stats**
```yaml
endpoint: "/api/v1/metrics/guardians/{guardian_id}"
method: "GET"
authentication: "Guardian Bearer Token (self or admin)"

response:
  guardian_id: "guardian_crypto_001"
  activity_summary:
    daily_active_days: 15
    total_commits: 47
    federated_rounds_participated: 12
    teams_led: 2
    security_events_handled: 5
  performance_metrics:
    average_response_time: "0.15s"
    api_success_rate: 0.99
    collaboration_score: 8.5
  recent_activity:
    - timestamp: "2025-09-11T17:30:00Z"
      action: "completed_federated_round"
      xp_gained: 100
    - timestamp: "2025-09-11T16:45:00Z"
      action: "created_team"
      xp_gained: 50
```

### **System Health Monitoring**

#### **Get Component Health Status**
```yaml
endpoint: "/api/v1/health/components"
method: "GET"
authentication: "Guardian Bearer Token (admin)"

response:
  timestamp: "2025-09-11T17:30:00Z"
  overall_status: "healthy"
  components:
    offline_ai:
      status: "healthy"
      response_time: "0.12s"
      memory_usage: "8.2GB / 16GB"
      active_sessions: 15
    mcp_server:
      status: "healthy" 
      response_time: "0.05s"
      active_tools: 12
      requests_per_minute: 45
    flower_federation:
      status: "training"
      active_rounds: 2
      connected_clients: 8
      average_round_time: "15min"
    offline_dns:
      status: "healthy"
      registered_services: 25
      queries_per_minute: 120
  alerts: []
```

---

## 🌐 WEBSOCKET REAL-TIME APIS

### **Guardian Coordination Channel**
```yaml
websocket_endpoint: "wss://api.uiota.dev/ws/guardian/{guardian_id}"
authentication: "Guardian Bearer Token as URL parameter"

message_types:
  team_invitation:
    type: "team_invitation"
    from_guardian: "guardian_coord_001"
    to_guardian: "guardian_crypto_001"
    team_id: "team_hackathon_002"
    message: "Join our Flower AI hackathon team!"
  
  federated_learning_update:
    type: "fl_update"
    round_id: "fl_round_20250911_001"
    status: "aggregating"
    progress: 0.8
    eta_completion: "2025-09-11T17:45:00Z"
  
  guardian_level_up:
    type: "level_up"
    guardian_id: "guardian_crypto_001"
    new_level: 8
    xp_gained: 150
    new_abilities: ["Master_Cryptography", "Advanced_Verification"]

  system_alert:
    type: "system_alert"
    severity: "warning"
    message: "High CPU usage detected in offline.ai component"
    affected_guardians: ["guardian_fed_001", "guardian_fed_002"]
```

---

## 🚀 INTEGRATION TESTING FRAMEWORK

### **API Integration Tests**
```yaml
test_suites:
  authentication_flow:
    - test: "Guardian login with valid signature"
      endpoint: "POST /api/v1/auth/guardian"
      expected_status: 200
      expected_response: "access_token present"
    
    - test: "Access protected endpoint with valid token"
      endpoint: "GET /api/v1/guardians/guardian_crypto_001"
      headers: "Authorization: Bearer {token}"
      expected_status: 200
  
  federated_learning_flow:
    - test: "Register FL client"
      endpoint: "POST /api/v1/flower/clients/register"
      guardian_class: "FederatedLearner"
      expected_status: 201
    
    - test: "Complete training round"
      steps:
        - "GET training assignment"
        - "Download model"
        - "Submit trained weights"
      expected_outcome: "XP awarded and next round scheduled"

  mobile_integration:
    - test: "Generate offline proof on mobile"
      endpoint: "POST /api/v1/mobile/proof/generate"
      device: "android"
      expected: "QR code generated"
    
    - test: "Verify proof with Ghost Verifier"
      endpoint: "POST /api/v1/mobile/proof/verify"
      guardian_class: "GhostVerifier"
      expected: "Proof validated and XP awarded"
```

### **Performance Benchmarks**
```yaml
performance_targets:
  api_response_times:
    authentication: "<200ms"
    guardian_profile: "<100ms"
    model_inference: "<2s"
    fl_coordination: "<500ms"
  
  throughput_targets:
    concurrent_guardians: 1000
    api_requests_per_second: 10000
    federated_learning_clients: 100
    websocket_connections: 5000
  
  availability_targets:
    uptime: "99.9%"
    maximum_downtime: "8.76 hours/year"
    recovery_time: "<5 minutes"
```

---

## 📚 DOCUMENTATION & SDK

### **API Documentation Standards**
```yaml
documentation_requirements:
  openapi_specification: "3.0.3"
  interactive_documentation: "Swagger UI"
  code_examples: "Python, JavaScript, curl"
  guardian_context_examples: "For each Guardian class"
  authentication_flows: "Complete examples with tokens"

sdk_languages:
  python:
    package_name: "uiota-sdk"
    features: ["Guardian auth", "FL client", "API wrapper"]
  
  javascript:
    package_name: "@uiota/sdk"
    features: ["Web integration", "WebSocket client", "React components"]
  
  mobile:
    android: "UIOTA Android SDK (Kotlin)"
    ios: "UIOTA iOS SDK (Swift)"
    react_native: "Cross-platform components"
```

---

## 🎯 DEPLOYMENT CONFIGURATION

### **Container API Configuration**
```yaml
api_containers:
  backend-api:
    image: "uiota/backend-api:1.0.0"
    ports: ["8000:8000"]
    environment:
      - "GUARDIAN_SECRET_KEY=production_secret"
      - "DATABASE_URL=postgresql://user:pass@db:5432/uiota"
      - "REDIS_URL=redis://redis:6379"
    volumes:
      - "guardian-data:/app/guardian-data"
    
  offline-ai:
    image: "uiota/offline-ai:1.0.0"
    ports: ["11434:11434"]
    environment:
      - "MODEL_CACHE_DIR=/app/models"
      - "GUARDIAN_INTEGRATION=enabled"
    volumes:
      - "model-storage:/app/models"
    resources:
      memory: "16Gi"
      cpu: "4"
```

---

This comprehensive API specification provides the foundation for implementing all UIOTA components with Guardian-centric authentication, federated learning coordination, and offline-first capabilities. Each API is designed to work seamlessly with the existing offline-guard infrastructure while enabling the full UIOTA ecosystem.

**Ready for implementation by specialized sub-agents.** 🛡️⚡🌸