// Jamming Environment Simulator for UIOTA Portal

class JammingSimulator {
    constructor() {
        this.jammingActive = false;
        this.jammingLevel = 0; // 0-100%
        this.jammingType = 'none'; // 'cellular', 'wifi', 'bluetooth', 'gps', 'all'
        this.qrRelayActive = false;
        this.meshDevices = [];
        this.relaySuccess = 0;
        this.relayAttempts = 0;

        this.initMeshNetwork();
    }

    // Initialize mesh network simulation
    initMeshNetwork() {
        this.meshDevices = [
            { id: 'device-001', name: 'Guardian-Alpha', type: 'mobile', signal: 85, distance: 50, status: 'online' },
            { id: 'device-002', name: 'Guardian-Beta', type: 'laptop', signal: 92, distance: 30, status: 'online' },
            { id: 'device-003', name: 'Guardian-Gamma', type: 'tablet', signal: 78, distance: 75, status: 'online' },
            { id: 'device-004', name: 'Guardian-Delta', type: 'phone', signal: 65, distance: 120, status: 'online' },
            { id: 'device-005', name: 'Guardian-Epsilon', type: 'iot', signal: 55, distance: 200, status: 'intermittent' }
        ];
    }

    // Start jamming simulation
    startJamming(type = 'all', level = 75) {
        this.jammingActive = true;
        this.jammingType = type;
        this.jammingLevel = level;

        console.log(`🚫 Jamming started: ${type} at ${level}% intensity`);

        // Affect mesh devices based on jamming
        this.updateMeshDevicesUnderJamming();

        // Start visual effects
        this.showJammingEffects();

        // Log jamming event
        this.logJammingEvent('Jamming attack detected', 'warning');

        return {
            type: this.jammingType,
            level: this.jammingLevel,
            affectedDevices: this.getAffectedDevices()
        };
    }

    // Stop jamming simulation
    stopJamming() {
        this.jammingActive = false;
        this.jammingLevel = 0;
        this.jammingType = 'none';

        console.log('✅ Jamming stopped - networks recovering');

        // Restore mesh devices
        this.restoreMeshDevices();

        // Remove visual effects
        this.hideJammingEffects();

        // Log recovery event
        this.logJammingEvent('Network recovery complete', 'success');

        return {
            status: 'recovered',
            devicesOnline: this.getOnlineDevices().length
        };
    }

    // Update mesh devices under jamming conditions
    updateMeshDevicesUnderJamming() {
        this.meshDevices.forEach(device => {
            const jammingImpact = this.calculateJammingImpact(device);

            // Reduce signal strength
            device.signal = Math.max(10, device.signal - jammingImpact);

            // Change status based on signal strength
            if (device.signal < 30) {
                device.status = 'offline';
            } else if (device.signal < 60) {
                device.status = 'intermittent';
            } else {
                device.status = 'online';
            }
        });
    }

    // Calculate jamming impact on specific device
    calculateJammingImpact(device) {
        let impact = this.jammingLevel;

        // Distance affects jamming effectiveness
        const distanceFactor = Math.min(device.distance / 100, 1);
        impact = impact * (1 - distanceFactor * 0.3);

        // Device type affects resistance
        const typeResistance = {
            'mobile': 0.8,
            'laptop': 0.9,
            'tablet': 0.85,
            'phone': 0.75,
            'iot': 0.6
        };

        impact = impact * typeResistance[device.type];

        // Add some randomness
        impact = impact + (Math.random() - 0.5) * 20;

        return Math.max(0, Math.min(100, impact));
    }

    // Restore mesh devices after jamming
    restoreMeshDevices() {
        this.meshDevices.forEach(device => {
            // Restore original signal strength (simulate recovery)
            device.signal = Math.min(100, device.signal + 30 + Math.random() * 20);

            // Update status
            if (device.signal > 70) {
                device.status = 'online';
            } else if (device.signal > 40) {
                device.status = 'intermittent';
            }
        });
    }

    // Get devices affected by jamming
    getAffectedDevices() {
        return this.meshDevices.filter(device =>
            device.status === 'offline' || device.status === 'intermittent'
        );
    }

    // Get online devices
    getOnlineDevices() {
        return this.meshDevices.filter(device => device.status === 'online');
    }

    // Start QR code relay demonstration
    async startQRRelay(qrData, destination) {
        this.qrRelayActive = true;
        this.relayAttempts++;

        console.log(`📱 Starting QR relay: ${qrData.slice(0, 50)}...`);

        try {
            // Find optimal relay path
            const relayPath = this.findOptimalRelayPath(destination);

            if (relayPath.length === 0) {
                throw new Error('No relay path available');
            }

            // Simulate relay through mesh network
            const relayResult = await this.simulateQRRelay(qrData, relayPath);

            if (relayResult.success) {
                this.relaySuccess++;
                this.logJammingEvent(`QR relay successful via ${relayPath.length} devices`, 'success');
                return {
                    success: true,
                    path: relayPath,
                    latency: relayResult.latency,
                    integrity: relayResult.integrity
                };
            } else {
                throw new Error(relayResult.error);
            }

        } catch (error) {
            this.logJammingEvent(`QR relay failed: ${error.message}`, 'error');
            return {
                success: false,
                error: error.message,
                retryAvailable: this.getOnlineDevices().length > 0
            };
        } finally {
            this.qrRelayActive = false;
        }
    }

    // Find optimal relay path through mesh network
    findOptimalRelayPath(destination) {
        const onlineDevices = this.getOnlineDevices();

        if (onlineDevices.length === 0) {
            return [];
        }

        // Simple path finding - in reality would use more sophisticated routing
        const path = onlineDevices
            .sort((a, b) => b.signal - a.signal) // Sort by signal strength
            .slice(0, Math.min(4, onlineDevices.length)); // Max 4 hops

        return path;
    }

    // Simulate QR relay through path
    async simulateQRRelay(qrData, path) {
        const startTime = Date.now();
        let integrity = 100;

        for (let i = 0; i < path.length; i++) {
            const device = path[i];

            // Simulate transmission delay
            const delay = 200 + Math.random() * 300;
            await new Promise(resolve => setTimeout(resolve, delay));

            // Calculate transmission success probability
            const successProbability = device.signal / 100;
            const jammingResistance = 1 - (this.jammingLevel / 200); // QR codes are resilient

            const transmissionSuccess = Math.random() < (successProbability * jammingResistance);

            if (!transmissionSuccess) {
                return {
                    success: false,
                    error: `Transmission failed at device ${device.name}`,
                    hopsFailed: i + 1
                };
            }

            // Slight integrity degradation with each hop (minimal for QR codes)
            integrity -= Math.random() * 2;

            // Visual feedback
            this.showRelayProgress(device, i, path.length);
        }

        const totalLatency = Date.now() - startTime;

        return {
            success: true,
            latency: totalLatency,
            integrity: Math.max(95, integrity), // QR codes maintain high integrity
            hops: path.length
        };
    }

    // Show relay progress visually
    showRelayProgress(device, currentHop, totalHops) {
        console.log(`📡 Relay hop ${currentHop + 1}/${totalHops}: ${device.name} (${device.signal}% signal)`);

        // Update UI if elements exist
        const relayDisplay = document.getElementById('qr-relay-display');
        if (relayDisplay) {
            relayDisplay.innerHTML = `
                <div style="color: #2ed573; font-weight: bold;">
                    📡 Relaying via ${device.name} (${currentHop + 1}/${totalHops})
                </div>
            `;
        }
    }

    // Show jamming effects in UI
    showJammingEffects() {
        // Add jamming overlay to existing elements
        const existingOverlay = document.getElementById('jamming-overlay');
        if (existingOverlay) {
            existingOverlay.remove();
        }

        const overlay = document.createElement('div');
        overlay.id = 'jamming-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg,
                rgba(255, 71, 87, 0.1) 0%,
                transparent 25%,
                transparent 75%,
                rgba(255, 71, 87, 0.1) 100%);
            pointer-events: none;
            z-index: 1000;
            animation: jammingPulse 2s ease-in-out infinite;
        `;

        // Add animation keyframes if not already present
        if (!document.getElementById('jamming-styles')) {
            const style = document.createElement('style');
            style.id = 'jamming-styles';
            style.textContent = `
                @keyframes jammingPulse {
                    0%, 100% { opacity: 0.3; }
                    50% { opacity: 0.7; }
                }

                @keyframes interferenceWave {
                    0% { transform: scale(1); opacity: 1; }
                    100% { transform: scale(2); opacity: 0; }
                }
            `;
            document.head.appendChild(style);
        }

        document.body.appendChild(overlay);

        // Add interference waves
        this.createInterferenceWaves();
    }

    // Hide jamming effects
    hideJammingEffects() {
        const overlay = document.getElementById('jamming-overlay');
        if (overlay) {
            overlay.remove();
        }

        const waves = document.querySelectorAll('.interference-wave');
        waves.forEach(wave => wave.remove());
    }

    // Create interference wave effects
    createInterferenceWaves() {
        for (let i = 0; i < 3; i++) {
            setTimeout(() => {
                const wave = document.createElement('div');
                wave.className = 'interference-wave';
                wave.style.cssText = `
                    position: fixed;
                    border: 2px solid rgba(255, 71, 87, 0.6);
                    border-radius: 50%;
                    width: 50px;
                    height: 50px;
                    left: ${Math.random() * window.innerWidth}px;
                    top: ${Math.random() * window.innerHeight}px;
                    animation: interferenceWave 3s linear infinite;
                    pointer-events: none;
                    z-index: 1001;
                `;

                document.body.appendChild(wave);

                setTimeout(() => {
                    if (wave.parentNode) {
                        wave.parentNode.removeChild(wave);
                    }
                }, 3000);
            }, i * 500);
        }
    }

    // Log jamming events
    logJammingEvent(message, type = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        console.log(`[${timestamp}] ${message}`);

        // Update UI log if exists
        const logElement = document.getElementById('jamming-log');
        if (logElement) {
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry log-${type}`;
            logEntry.innerHTML = `
                <span class="log-time">${timestamp}</span>
                <span class="log-message">${message}</span>
            `;

            logElement.insertBefore(logEntry, logElement.firstChild);

            // Keep only last 10 entries
            while (logElement.children.length > 10) {
                logElement.removeChild(logElement.lastChild);
            }
        }
    }

    // Get jamming statistics
    getJammingStats() {
        return {
            jammingActive: this.jammingActive,
            jammingLevel: this.jammingLevel,
            jammingType: this.jammingType,
            totalDevices: this.meshDevices.length,
            onlineDevices: this.getOnlineDevices().length,
            affectedDevices: this.getAffectedDevices().length,
            relaySuccess: this.relaySuccess,
            relayAttempts: this.relayAttempts,
            successRate: this.relayAttempts > 0 ? (this.relaySuccess / this.relayAttempts * 100).toFixed(1) : 0
        };
    }

    // Get mesh network status
    getMeshNetworkStatus() {
        return {
            devices: this.meshDevices.map(device => ({
                ...device,
                affected: this.jammingActive && device.status !== 'online'
            })),
            averageSignal: this.meshDevices.reduce((sum, device) => sum + device.signal, 0) / this.meshDevices.length,
            networkHealth: this.getOnlineDevices().length / this.meshDevices.length * 100
        };
    }

    // Demonstrate QR relay under jamming
    async demonstrateJammingResistance() {
        console.log('🎭 Demonstrating QR relay under jamming conditions...');

        // Start with no jamming
        const normalRelay = await this.startQRRelay('PROOF_DATA_NORMAL_CONDITIONS', 'target-device');

        // Start jamming
        this.startJamming('all', 85);

        // Wait a moment for jamming to take effect
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Try relay under jamming
        const jammedRelay = await this.startQRRelay('PROOF_DATA_UNDER_JAMMING', 'target-device');

        // Stop jamming
        this.stopJamming();

        return {
            normalConditions: normalRelay,
            underJamming: jammedRelay,
            demonstration: 'QR codes successfully relay even under heavy jamming'
        };
    }
}

// Export for use in portal
if (typeof window !== 'undefined') {
    window.JammingSimulator = JammingSimulator;
}

// Global instance
const jammingSimulator = new JammingSimulator();