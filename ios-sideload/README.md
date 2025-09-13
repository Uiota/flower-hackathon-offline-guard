# 📱 iOS Sideload Options for Offline Guard

## 🍎 **iOS Installation Methods**

### **Method 1: TestFlight (Easiest)**
```bash
# For judges and early testers
1. Join TestFlight Beta: https://testflight.apple.com/join/OfflineGuard
2. Install directly from App Store infrastructure
3. Get updates automatically
4. No technical knowledge required
```

### **Method 2: AltStore (Advanced)**
```bash
# For developers and power users
1. Install AltStore: https://altstore.io
2. Download OfflineGuard.ipa from our releases
3. Sideload via AltStore
4. Refresh every 7 days (free Apple ID) or yearly (paid)
```

### **Method 3: Sideloadly (Cross-platform)**
```bash
# For Windows/Mac users
1. Download Sideloadly: https://sideloadly.io
2. Get OfflineGuard.ipa from our GitHub releases
3. Connect iPhone via USB
4. Sideload with your Apple ID
```

### **Method 4: Xcode (Developers)**
```bash
# For iOS developers
1. Clone our iOS project from Gitea
2. Open in Xcode
3. Change bundle identifier
4. Build and run on your device
5. Trust developer certificate in Settings
```

---

## 🔧 **iOS Project Structure**

### **Swift/SwiftUI Implementation**
```swift
// OfflineGuard iOS App Structure
OfflineGuard-iOS/
├── OfflineGuard/
│   ├── Views/
│   │   ├── MainView.swift           // Main dashboard
│   │   ├── GuardianView.swift       // Guardian character display
│   │   ├── QRGeneratorView.swift    // QR proof generation
│   │   └── SettingsView.swift       // App settings
│   ├── Services/
│   │   ├── NetworkMonitor.swift     // iOS network monitoring
│   │   ├── OfflineGuardService.swift // Core offline logic
│   │   └── GuardianManager.swift    // Guardian character system
│   ├── Models/
│   │   ├── Guardian.swift           // Guardian data model
│   │   ├── OfflineProof.swift       // Cryptographic proof model
│   │   └── NetworkState.swift       // Network state model
│   └── Utils/
│       ├── CryptoUtils.swift        // Ed25519 signatures
│       ├── QRGenerator.swift        // QR code generation
│       └── FlowerClient.swift       // Federated learning client
├── OfflineGuard.xcodeproj
└── Info.plist                      // iOS app configuration
```

### **Key iOS Features**
- **Network Monitoring**: iOS NetworkMonitor for offline detection
- **Background Processing**: Keep Guardian active in background
- **Keychain Integration**: Secure private key storage
- **Camera Access**: QR scanning with AVFoundation
- **Local Storage**: Core Data for offline proof storage
- **Push Notifications**: Alert when back online

---

## 🚨 **iOS Sideload Challenges & Solutions**

### **Challenge 1: Apple Certificate Restrictions**
**Problem**: Apps expire every 7 days with free Apple ID
**Solutions**:
- Enterprise certificate (annual renewal)
- TestFlight beta distribution
- User instructions for weekly refresh
- AltStore automatic refresh

### **Challenge 2: iOS Security Sandbox**
**Problem**: Limited background processing and network monitoring
**Solutions**:
- Background App Refresh permissions
- Silent push notifications for updates
- Local notifications for offline detection
- Widget extension for quick status

### **Challenge 3: App Store Guidelines**
**Problem**: May not meet App Store distribution requirements
**Solutions**:
- Focus on sideload distribution initially
- TestFlight for beta testing
- Emphasize enterprise/developer use case
- Position as research/educational tool

---

## 📦 **iOS Build & Distribution**

### **Automated IPA Generation**
```bash
# iOS build script
#!/bin/bash
# build-ios.sh

echo "🍎 Building Offline Guard for iOS..."

# Clean and archive
xcodebuild clean archive \
  -project OfflineGuard.xcodeproj \
  -scheme OfflineGuard \
  -configuration Release \
  -archivePath build/OfflineGuard.xcarchive

# Export IPA
xcodebuild -exportArchive \
  -archivePath build/OfflineGuard.xcarchive \
  -exportPath build/ \
  -exportOptionsPlist ExportOptions.plist

echo "✅ IPA ready: build/OfflineGuard.ipa"
echo "📦 Size: $(ls -lh build/OfflineGuard.ipa | awk '{print $5}')"
```

### **Distribution Channels**
```bash
# Release distribution
releases/
├── OfflineGuard-iOS-v1.0.0.ipa     # Signed IPA for sideload
├── OfflineGuard-Enterprise.ipa     # Enterprise distribution
├── OfflineGuard-TestFlight.ipa     # TestFlight version
├── checksums.txt                   # File integrity verification
└── INSTALL_iOS.md                  # Installation instructions
```

---

## 🔐 **iOS Security Implementation**

### **Keychain Services Integration**
```swift
import Security

class SecureStorage {
    static func storePrivateKey(_ key: Data, for guardianID: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassKey,
            kSecAttrKeyClass as String: kSecAttrKeyClassPrivate,
            kSecAttrApplicationTag as String: "com.uiota.offlineguard.\(guardianID)",
            kSecValueData as String: key,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        
        let status = SecItemAdd(query as CFDictionary, nil)
        return status == errSecSuccess
    }
}
```

### **Background Network Monitoring**
```swift
import Network

class iOSNetworkMonitor: ObservableObject {
    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "NetworkMonitor")
    
    @Published var isConnected = false
    @Published var offlineSince: Date?
    
    func startMonitoring() {
        monitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                self?.isConnected = path.status == .satisfied
                
                if !self?.isConnected ?? false && self?.offlineSince == nil {
                    self?.offlineSince = Date()
                    // Trigger Guardian offline mode
                    GuardianManager.shared.activateOfflineMode()
                }
            }
        }
        
        monitor.start(queue: queue)
    }
}
```

---

## 📱 **iOS Installation Guide for Users**

### **For Non-Technical Users (TestFlight)**
```
1. 📧 Get TestFlight invitation link
2. 📱 Install TestFlight from App Store
3. 🔗 Tap invitation link on iPhone
4. ⬇️ Install Offline Guard beta
5. ✅ Ready to use!
```

### **For Technical Users (AltStore)**
```
1. 💻 Install AltStore on computer: https://altstore.io
2. 📱 Install AltStore app on iPhone
3. 🔗 Connect iPhone to computer via USB
4. ⬇️ Download OfflineGuard.ipa from our releases
5. 📂 Drag IPA into AltStore
6. 🔄 Refresh every 7 days (or enable auto-refresh)
```

### **For Developers (Xcode)**
```
1. 🔧 Install Xcode from Mac App Store
2. 📦 Clone iOS project from Gitea
3. 🔑 Change bundle identifier to unique value
4. 🏗️ Build and run on connected device
5. ⚙️ Trust developer certificate in iPhone Settings
```

---

## 🌐 **iOS Web Fallback (PWA)**

### **Advanced PWA for iOS**
```javascript
// iOS-optimized PWA features
// Saved to home screen = app-like experience

// iOS-specific features
if (navigator.platform.includes('iPhone') || navigator.platform.includes('iPad')) {
    // Enable iOS-specific optimizations
    enableiOSOptimizations();
    
    // Add to home screen prompt
    showiOSInstallPrompt();
    
    // iOS-style UI adjustments
    applyiOSDesign();
}

function enableiOSOptimizations() {
    // Prevent zoom on input focus
    document.addEventListener('touchstart', function(e) {
        if (e.touches.length > 1) {
            e.preventDefault();
        }
    });
    
    // Handle iOS safe areas
    document.documentElement.style.setProperty(
        '--safe-area-inset-top', 
        'env(safe-area-inset-top)'
    );
}
```

### **iOS PWA Manifest**
```json
{
  "name": "Offline Guard",
  "short_name": "OfflineGuard", 
  "display": "fullscreen",
  "orientation": "portrait",
  "theme_color": "#667eea",
  "background_color": "#667eea",
  "start_url": "/",
  "scope": "/",
  "apple-touch-icon": "assets/icon-180.png",
  "apple-mobile-web-app-capable": "yes",
  "apple-mobile-web-app-status-bar-style": "black-translucent"
}
```

---

## 🏆 **Judge Demo Strategy for iOS**

### **What to Show Judges**
1. **Multi-platform Support**: Same app works on Android + iOS
2. **Easy Distribution**: TestFlight link = instant install
3. **Security Focus**: Keychain integration, iOS sandbox
4. **Native Performance**: Swift/SwiftUI = fast, responsive

### **Live iOS Demo**
```bash
# If you have iPhone:
1. Show TestFlight installation (30 seconds)
2. Launch app and show native iOS UI
3. Demonstrate offline detection on iOS
4. Generate QR proof with iOS camera
5. Show Guardian evolution on iOS

# If no iPhone available:
1. Show iOS project in Xcode
2. Demonstrate iOS Simulator  
3. Show iOS-specific code features
4. Explain distribution strategy
```

### **iOS Value Proposition**
- **"Works on every platform"** - Android, iOS, Web
- **"No App Store required"** - Sideload distribution
- **"Judge-friendly"** - TestFlight = easy testing
- **"Enterprise ready"** - Works in secure iOS environments

This makes Offline Guard accessible to **every mobile device** including locked-down corporate iPhones! 📱🛡️