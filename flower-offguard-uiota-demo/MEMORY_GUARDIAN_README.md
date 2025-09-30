# Memory Guardian 🧠

**Comprehensive Cognitive Health & Property Protection System**

Memory Guardian is a revolutionary application designed to help aging users preserve their cognitive health, detect early signs of Alzheimer's disease, and protect their property and assets. Built on the LL TOKEN ecosystem with quantum-safe encryption and federated learning capabilities.

---

## 🎯 Core Features

### 1. Cognitive Health Monitoring
- **Daily Assessments**: Interactive exercises for memory, pattern recognition, problem-solving, and reaction time
- **Baseline Tracking**: Establish personal cognitive baseline and track deviations
- **Early Detection**: AI-powered detection of cognitive decline patterns
- **Adaptive Difficulty**: Exercises automatically adjust to your performance level
- **Progress Visualization**: Beautiful charts and graphs showing your cognitive trends

### 2. Property Protection Vault
- **Quantum-Safe Encryption**: AES-256-GCM encryption for all sensitive documents
- **Document Management**: Securely store wills, deeds, financial documents, and medical records
- **Integrity Verification**: SHA-256 checksums ensure documents haven't been tampered with
- **Emergency Access**: Trusted contacts can access documents in emergencies
- **Offline Capable**: Full functionality without internet connection

### 3. Trusted Contact System
- **Tiered Access Levels**:
  - Level 1: Emergency notification only
  - Level 2: View-only access to selected documents
  - Level 3: Full access to all documents
- **Verification Codes**: Secure verification system for contact authentication
- **Reputation Scores**: LL-REP token integration for trust verification
- **Multi-Contact Support**: Add family, doctors, attorneys, and caregivers

### 4. Federated Learning Integration
- **Privacy-Preserving Research**: Contribute to Alzheimer's research without sharing personal data
- **Differential Privacy**: Advanced privacy techniques protect your identity
- **Model Training**: Help train global cognitive decline detection models
- **Token Rewards**: Earn LLT-REWARD and LLT-DATA tokens for contributions
- **Opt-In Participation**: Full control over data sharing

### 5. LL TOKEN Reward System
Earn tokens for healthy behaviors and participation:

| Token Type | Earned For | Amount |
|------------|------------|--------|
| **LLT-EXP** | Daily assessments | 50 per assessment |
| **LLT-EXP** | Baseline established | 200 (one-time) |
| **LLT-EXP** | Consistent participation | 100 per week |
| **LLT-EDU** | Educational content | 10 per assessment |
| **LLT-REWARD** | FL contributions | 50 per contribution |
| **LLT-DATA** | Data sharing | 25 per contribution |
| **LLT-REP** | Document security | 10 per document |
| **LLT-REP** | Trusted contact verified | 50 per contact |

---

## 📊 Cognitive Exercises

### Memory Exercises
1. **Sequence Memory**: Memorize and recall number sequences
2. **Image Grid Memory**: Remember positions of highlighted items
3. **Word Pair Memory**: Associate and recall word pairs

### Pattern Recognition
1. **Number Patterns**: Identify arithmetic and geometric sequences
2. **Shape Patterns**: Complete visual pattern sequences
3. **Matrix Puzzles**: Solve Raven's Progressive Matrices-style puzzles

### Problem Solving
1. **Math Puzzles**: Word problems and calculations
2. **Logic Puzzles**: Deductive reasoning exercises
3. **Strategy Puzzles**: Planning and sequential thinking

### Reaction Time
1. **Simple Reaction**: Basic stimulus-response tests
2. **Choice Reaction**: Multi-option decision making
3. **Go/No-Go**: Inhibitory control tests

---

## 🤖 Agent System

### Development Agent
**Responsibilities:**
- System health monitoring
- Database optimization
- Automated backups (weekly)
- Security updates
- Performance tuning
- Error detection and logging

**Features:**
- Database integrity checks
- File system health monitoring
- Encryption system verification
- Memory usage tracking
- Automated maintenance tasks

### Research Agent
**Responsibilities:**
- Cognitive trend analysis
- Early decline detection
- Personalized insights
- Exercise recommendations
- Federated learning preparation
- Health report generation

**Features:**
- Linear regression trend analysis
- Anomaly detection (2+ standard deviations)
- Risk level assessment (none/low/medium/high)
- Category-specific recommendations
- Privacy-preserving data anonymization

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd flower-offguard-uiota-demo

# Install dependencies
pip install cryptography flask

# Set offline mode
export OFFLINE_MODE=1

# Launch Memory Guardian
python launch_memory_guardian.py
```

The web interface will open automatically at `http://localhost:8090`

### CLI Mode

```bash
# Launch in command-line mode
python launch_memory_guardian.py --cli
```

### Agent System

```bash
# Run maintenance agents
python launch_memory_guardian.py --agents
```

---

## 💻 Usage Examples

### 1. Web Interface (Recommended)

```bash
# Default launch (opens browser)
python launch_memory_guardian.py

# Custom port
python launch_memory_guardian.py --port 8080

# Don't open browser
python launch_memory_guardian.py --no-browser

# Specify host (for network access)
python launch_memory_guardian.py --host 0.0.0.0 --port 8090
```

### 2. Python API

```python
from memory_guardian_system import MemoryGuardianApp

# Initialize app
app = MemoryGuardianApp(
    user_id="user_001",
    master_password="YourSecurePassword123!",
    ll_token_wallet="LL_YOUR_WALLET_ADDRESS"
)

# Run daily assessment
result = app.run_daily_assessment({
    'memory_score': 85.0,
    'reaction_time_ms': 450.0,
    'pattern_recognition_score': 88.0,
    'problem_solving_score': 82.0,
    'overall_score': 85.0
})

print(f"Status: {result['evaluation']['status']}")
print(f"Tokens Earned: {result['rewards']}")

# Secure a document
doc_result = app.secure_document(
    doc_type="will",
    title="Last Will and Testament",
    content="Document content here...",
    trusted_contacts=["contact_001", "contact_002"]
)

print(f"Document secured: {doc_result['record_id']}")

# Add trusted contact
contact_result = app.add_trusted_contact(
    name="Jane Smith",
    relationship="daughter",
    access_level=3,  # Full access
    phone="+1-555-0123",
    email="jane@example.com",
    ll_token_address="LL_JANE_WALLET"
)

print(f"Contact added: {contact_result['contact']['contact_id']}")

# Get dashboard summary
summary = app.get_dashboard_summary()
print(f"Total assessments: {summary['total_assessments']}")
print(f"Trend: {summary['trend']}")
print(f"Tokens earned: {summary['total_tokens_earned']}")
```

### 3. Agent System

```python
from memory_guardian_agents import AgentCoordinator

# Initialize coordinator
coordinator = AgentCoordinator()

# Run daily maintenance
results = coordinator.run_daily_maintenance(user_id="user_001")

# Development agent tasks
health_report = coordinator.dev_agent.system_health_check()
print(f"System status: {health_report['overall_status']}")

# Research agent analysis
analysis = coordinator.research_agent.analyze_cognitive_trends(
    user_id="user_001",
    days=90
)

print(f"Risk level: {analysis['risk_level']}")
for insight in analysis['insights']:
    print(f"  - {insight}")
```

---

## 🔐 Security Features

### Encryption
- **Algorithm**: AES-256-GCM (quantum-resistant)
- **Key Derivation**: PBKDF2 with 100,000 iterations
- **Random Salts**: Unique salt per user
- **Integrity Checks**: SHA-256 checksums for all documents

### Privacy
- **Offline First**: All data stored locally
- **No Cloud Storage**: Zero external data transmission
- **Differential Privacy**: FL contributions use noise injection
- **Anonymization**: Age groups, timestamp generalization

### Authentication
- **Master Password**: Required for document vault access
- **Biometric Support**: (Future implementation)
- **Trusted Contacts**: Multi-factor emergency access
- **Session Security**: Automatic timeout and re-authentication

---

## 📈 Cognitive Trend Analysis

### Risk Levels

| Risk Level | Criteria | Recommendations |
|------------|----------|-----------------|
| **None** | Stable or improving trends | Continue regular exercises |
| **Low** | Minor decline (< 5%) | Increase exercise frequency |
| **Medium** | Moderate decline (5-20%) | Consult healthcare provider |
| **High** | Significant decline (> 20%) | Immediate medical evaluation |

### Trend Calculation
- **Linear Regression**: Slope analysis over time
- **Direction**: Improving / Stable / Declining
- **Strength**: 0-100% confidence level
- **Anomaly Detection**: 2+ standard deviations from mean

---

## 🏥 Healthcare Integration

### Report Generation
Export comprehensive reports for healthcare providers:
- 30/60/90-day cognitive trends
- Category-specific breakdowns
- Anomaly highlights
- Risk assessment
- Recommendation summary

### Medical Records
Securely store and share:
- Medication lists
- Allergy information
- Emergency contacts
- Healthcare provider information
- Insurance details

---

## 🌐 Federated Learning Details

### How It Works
1. **Local Training**: Your device trains on your cognitive data
2. **Model Updates**: Only model weights are shared (not raw data)
3. **Aggregation**: Central server combines updates from many users
4. **Global Model**: Improved model distributed back to all users
5. **Privacy Preserved**: Your personal data never leaves your device

### Contribution Pipeline
```
Local Assessment → Anonymization → Differential Privacy →
Local Model Update → Encrypted Transmission → Global Aggregation →
Token Rewards → Updated Global Model
```

### Privacy Guarantees
- **k-Anonymity**: Data generalized to at least k=10 users
- **Differential Privacy**: ε-differential privacy with ε=0.1
- **Secure Aggregation**: Cryptographic multi-party computation
- **Zero Personal Data**: Only aggregated statistics transmitted

---

## 📱 Future Features (Roadmap)

### Phase 1: Enhanced Cognitive Tools (Q2 2025)
- [ ] Virtual reality exercises
- [ ] Multiplayer cognitive games
- [ ] Voice-based assessments
- [ ] Mobile app (iOS/Android)

### Phase 2: Advanced Analytics (Q3 2025)
- [ ] AI-powered personalized recommendations
- [ ] Predictive decline modeling
- [ ] Genetic risk factor integration
- [ ] Lifestyle correlation analysis

### Phase 3: Social Features (Q4 2025)
- [ ] Family portal for caregivers
- [ ] Support group integration
- [ ] Progress sharing (opt-in)
- [ ] Gamification and achievements

### Phase 4: Healthcare Integration (Q1 2026)
- [ ] EHR integration (FHIR standard)
- [ ] Telemedicine support
- [ ] Prescription reminders
- [ ] Appointment scheduling

---

## 🛠️ Technical Architecture

### Components
```
┌─────────────────────────────────────────────┐
│           Memory Guardian System            │
├─────────────────────────────────────────────┤
│  Frontend (HTML/CSS/JS)                     │
│  - Web Interface (TailwindCSS)              │
│  - Interactive Exercises                    │
│  - Real-time Visualizations                 │
├─────────────────────────────────────────────┤
│  Backend (Python)                           │
│  - memory_guardian_system.py (Core)         │
│  - cognitive_exercises.py (Exercises)       │
│  - memory_guardian_agents.py (Agents)       │
├─────────────────────────────────────────────┤
│  Storage (SQLite)                           │
│  - Cognitive assessments                    │
│  - Encrypted documents                      │
│  - Trusted contacts                         │
│  - Alert logs                               │
├─────────────────────────────────────────────┤
│  Integration                                │
│  - LL TOKEN system                          │
│  - Federated Learning network               │
│  - Flower FL framework                      │
└─────────────────────────────────────────────┘
```

### Database Schema
```sql
-- Cognitive assessments
CREATE TABLE cognitive_assessments (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    user_id TEXT NOT NULL,
    memory_score REAL,
    reaction_time_ms REAL,
    pattern_recognition_score REAL,
    problem_solving_score REAL,
    overall_score REAL,
    baseline_deviation REAL,
    tokens_earned REAL
);

-- Property records
CREATE TABLE property_records (
    id INTEGER PRIMARY KEY,
    record_id TEXT UNIQUE NOT NULL,
    record_type TEXT NOT NULL,
    title TEXT NOT NULL,
    encrypted_content TEXT NOT NULL,
    document_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    trusted_contacts TEXT
);

-- Trusted contacts
CREATE TABLE trusted_contacts (
    id INTEGER PRIMARY KEY,
    contact_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    relationship TEXT,
    access_level INTEGER,
    phone TEXT,
    email TEXT,
    verification_code TEXT,
    ll_token_address TEXT,
    reputation_score REAL,
    created_at TEXT NOT NULL
);

-- Alert logs
CREATE TABLE alert_logs (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    resolved INTEGER DEFAULT 0
);
```

---

## 🤝 Contributing

Memory Guardian is part of the LL TOKEN OFFLINE ecosystem. Contributions are welcome!

### Areas for Contribution
- New cognitive exercises
- Improved ML models
- UI/UX enhancements
- Localization/translations
- Healthcare integrations
- Security audits

---

## 📄 License

Part of the LL TOKEN OFFLINE ecosystem.
See main project LICENSE for details.

---

## 🆘 Support & Resources

### Documentation
- [LL TOKEN Specifications](LL_TOKEN_SPECIFICATIONS.md)
- [LL TOKEN README](LL_TOKEN_README.md)
- [Main Project README](README.md)

### Contact
- GitHub Issues: Report bugs and request features
- Email: support@lltoken.offline (via offline mesh)

### Medical Disclaimer
Memory Guardian is a cognitive health tool and not a medical device. It does not diagnose, treat, or prevent any disease. Always consult with qualified healthcare professionals for medical advice.

---

## 🏆 Acknowledgments

Built on:
- **Flower Framework**: Federated Learning infrastructure
- **LL TOKEN System**: Quantum-safe token economy
- **Off-Guard Security**: Zero-trust cryptographic verification
- **UIOTA Mesh**: Offline synchronization protocol

---

**🧠 Memory Guardian** - *Protecting minds and memories for generations to come*

*Part of the LL TOKEN OFFLINE ecosystem - Empowering users to age with dignity, security, and independence*