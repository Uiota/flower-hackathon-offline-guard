# 🛡️ UIOTA.Space Local Access Guide

## How It Works Locally

Your UIOTA offline-guard system runs entirely on your local machine using:

### **Local Python HTTP Server**
- **Port**: 8080
- **Address**: localhost (127.0.0.1)
- **Protocol**: HTTP (no internet required)

### **File Structure**
```
/home/uiota/projects/offline-guard/
├── uiota_space_integration_simple.py    # Main server
├── unified_portal.html                  # Unified dashboard
├── offline-ai/index.html               # AI interface
├── flower-offguard-uiota-demo/         # FL dashboard
├── web-demo/                           # Guardian portals
└── open_portal.sh                      # Quick launcher
```

## 🚀 **3 Ways to Access:**

### **Method 1: Quick Launcher**
```bash
./open_portal.sh
```

### **Method 2: Manual Browser**
1. Open any web browser
2. Go to: `http://localhost:8080`
3. You'll see the unified portal

### **Method 3: Command Line**
```bash
# Start server if not running
python3 uiota_space_integration_simple.py &

# Open in browser
firefox http://localhost:8080 &
# OR
google-chrome http://localhost:8080 &
# OR
xdg-open http://localhost:8080
```

## 📱 **Direct Component Access:**

| Component | URL |
|-----------|-----|
| **Main Portal** | http://localhost:8080 |
| **Offline AI** | http://localhost:8080/offline-ai/index.html |
| **FL Dashboard** | http://localhost:8080/flower-offguard-uiota-demo/dashboard.html |
| **Guardian Portal** | http://localhost:8080/web-demo/portal.html |
| **Airplane Mode** | http://localhost:8080/web-demo/airplane_mode_guardian.html |

## 🔧 **Troubleshooting:**

### **Can't see the portal?**
1. Check server is running: `ps aux | grep python3 | grep uiota`
2. Test API: `curl http://localhost:8080/api/status`
3. Try alternative: `http://127.0.0.1:8080`

### **Port conflict?**
```bash
# Kill existing server
pkill -f uiota_space_integration

# Restart
python3 uiota_space_integration_simple.py &
```

### **Browser issues?**
- Clear browser cache
- Try incognito/private mode
- Use different browser

## 🛡️ **Offline-First Design:**

✅ **No Internet Required** - Everything runs locally
✅ **All Data Local** - No external servers
✅ **Privacy Protected** - Data never leaves your machine
✅ **CPU-Only ML** - No NVIDIA/GPU dependencies

## 🎯 **Perfect For:**
- Local development
- Offline demonstrations
- Privacy-focused AI work
- Team collaboration (local network)
- Educational purposes

---

**The system is designed to work completely offline on your local machine!**