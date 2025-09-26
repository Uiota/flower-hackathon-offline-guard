# 🛡️ Off Guard - Demo Showcase

Three interactive demos showcasing the Off Guard security platform with AI-powered offline protection.

## 🎯 Live Demos

### 1. Off Guard - Quantum Security System
**File:** `web-demo/index.html`
- **Purpose:** Main security dashboard with quantum proof vault
- **Features:** Login system, QR proof generation, transaction history
- **Demo Credentials:**
  - Neural ID: `admin`
  - Quantum Passphrase: `quantum`

### 2. UIOTA Tactical AI Operations Platform
**File:** `web-demo/portal.html`
- **Purpose:** Enterprise-grade tactical operations center
- **Features:** ML toolkit, federation learning, device management, jamming simulation
- **Security:** Auto-redirects to airplane mode guardian if not in safe mode

### 3. Airplane Mode Guardian
**File:** `web-demo/airplane_mode_guardian.html`
- **Purpose:** Security gateway enforcing safe mode operation
- **Features:** Network state monitoring, QR relay demonstration, auto-activation

## 🚀 Quick Start

### For Team Collaboration

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd offline-guard
   ```

2. **Serve the demos locally:**
   ```bash
   # Python 3
   python -m http.server 8000

   # Node.js
   npx serve .

   # Or any local server
   ```

3. **Access demos:**
   - Main Demo: `http://localhost:8000/web-demo/index.html`
   - Tactical Portal: `http://localhost:8000/web-demo/portal.html`
   - Guardian Gateway: `http://localhost:8000/web-demo/airplane_mode_guardian.html`

### Demo Flow

1. **Start with Airplane Mode Guardian** - Ensures secure environment
2. **Access Tactical Portal** - Enterprise operations dashboard
3. **Use Quantum Security System** - Main user-facing application

## 🔧 Development

### File Structure
```
web-demo/
├── index.html                 # Main quantum security demo
├── portal.html               # Tactical operations platform
├── airplane_mode_guardian.html # Security gateway
├── ml-demo.js                # Machine learning simulator
├── jamming_simulator.js      # Network jamming simulation
├── enterprise-demo-controller.js # Enterprise features
├── deployable-ml-toolkit.js  # ML deployment tools
└── uiota-token-system.js     # Token system integration
```

### Key Features to Demo

#### 🔐 Security Features
- Offline-first operation
- QR code mesh relay
- Airplane mode enforcement
- Cryptographic proof generation

#### 🧠 AI/ML Features
- Federated learning simulation
- Model training and deployment
- Real-time performance metrics
- Multi-device coordination

#### 🌐 Network Features
- Jamming resistance testing
- Mesh network visualization
- Device discovery and connection
- Real-time status monitoring

## 📱 Mobile-Friendly

All demos are responsive and work on mobile devices. The airplane mode guardian automatically detects and responds to network state changes.

## 🎨 Customization

### Themes
- Quantum: Cyber-tech blue/purple theme
- Tactical: Military-grade green/orange theme
- Guardian: Warning red/orange theme

### Branding
- Logo: Configurable in each demo
- Colors: CSS custom properties
- Animations: Modular and toggleable

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes across all three demos
4. Submit a pull request

### Testing Checklist
- [ ] All demos load without errors
- [ ] Mobile responsiveness works
- [ ] Airplane mode detection functions
- [ ] QR relay simulation runs
- [ ] ML training simulator works
- [ ] Jamming simulation operates correctly

## 📄 License

See the main repository LICENSE file for details.

---

**Live Demo URLs:** Ready for deployment to any static hosting service (GitHub Pages, Netlify, Vercel, etc.)