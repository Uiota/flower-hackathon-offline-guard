// ML Demo Functions for UIOTA Portal

class MLSimulator {
    constructor() {
        this.models = [
            { name: "ImageClassifier-v2.1", accuracy: 94.2, size: "47MB", status: "trained" },
            { name: "TextAnalyzer-v1.8", accuracy: 89.7, size: "23MB", status: "training" },
            { name: "AudioDetector-v3.0", accuracy: 96.1, size: "31MB", status: "deployed" },
            { name: "VideoSegment-v1.4", accuracy: 87.3, size: "72MB", status: "testing" }
        ];

        this.trainingProgress = 0;
        this.federationRounds = 152;
        this.connectedDevices = 23;
    }

    // Simulate training a new model
    async trainModel(modelName = "CustomModel-v1.0") {
        const terminal = document.getElementById('ml-terminal');

        const steps = [
            `$ python train.py --model ${modelName}`,
            `✓ Loading dataset: 50,000 samples`,
            `✓ Initializing ${modelName} architecture`,
            `⚡ Epoch 1/100 - Loss: 2.847 - Accuracy: 23.4%`,
            `⚡ Epoch 25/100 - Loss: 0.234 - Accuracy: 78.9%`,
            `⚡ Epoch 50/100 - Loss: 0.089 - Accuracy: 89.2%`,
            `⚡ Epoch 75/100 - Loss: 0.034 - Accuracy: 93.7%`,
            `⚡ Epoch 100/100 - Loss: 0.019 - Accuracy: 95.3%`,
            `✅ Training complete! Final accuracy: 95.3%`,
            `💾 Model saved: models/${modelName}.pth`
        ];

        for (let i = 0; i < steps.length; i++) {
            await this.addTerminalLine(steps[i], i * 800);

            // Update progress bar if it exists
            const progress = ((i + 1) / steps.length) * 100;
            this.updateTrainingProgress(progress);
        }

        // Add new model to the list
        this.models.push({
            name: modelName,
            accuracy: 95.3,
            size: "42MB",
            status: "trained"
        });

        this.showNotification("🎉 Model training completed successfully!");
        return true;
    }

    // Simulate federated learning round
    async joinFederationRound() {
        const terminal = document.getElementById('ml-terminal');
        this.federationRounds++;

        const steps = [
            `$ flower-client --server federation.uiota.network`,
            `✓ Connecting to federation server...`,
            `✓ Authentication with Guardian credentials`,
            `🌐 Joined federation round ${this.federationRounds}`,
            `📊 ${this.connectedDevices} devices participating`,
            `⬇️ Downloading global model (v${this.federationRounds - 1})`,
            `🧠 Running local training (5 epochs)`,
            `📈 Local accuracy improved: 94.2% → 95.1%`,
            `⬆️ Uploading model updates (2.3MB)`,
            `✅ Federation round ${this.federationRounds} complete`,
            `🏆 Global model accuracy: 96.4%`
        ];

        for (let i = 0; i < steps.length; i++) {
            await this.addTerminalLine(steps[i], i * 1200);
        }

        this.showNotification(`🌐 Federation round ${this.federationRounds} completed!`);
        return true;
    }

    // Simulate model export
    async exportModel(format = "onnx") {
        const terminal = document.getElementById('ml-terminal');

        const steps = [
            `$ python export.py --format ${format} --model latest`,
            `✓ Loading trained model: CustomModel-v1.0`,
            `🔄 Converting to ${format.toUpperCase()} format...`,
            `✓ Optimizing for mobile deployment`,
            `✓ Quantization applied (INT8)`,
            `📦 Model size reduced: 42MB → 12MB`,
            `💾 Exported: models/custom_model.${format}`,
            `✅ Model ready for deployment!`
        ];

        for (let i = 0; i < steps.length; i++) {
            await this.addTerminalLine(steps[i], i * 600);
        }

        this.showNotification(`📦 Model exported as ${format.toUpperCase()}`);
        return true;
    }

    // Simulate benchmark testing
    async runBenchmark() {
        const terminal = document.getElementById('ml-terminal');

        const steps = [
            `$ python benchmark.py --comprehensive`,
            `🚀 Starting comprehensive benchmark...`,
            `📊 Testing inference speed...`,
            `⚡ CPU inference: 247.3 FPS`,
            `⚡ Mobile optimized: 156.8 FPS`,
            `💾 Memory usage: 234MB peak`,
            `🔋 Battery impact: Low (3.2% per hour)`,
            `📐 Model size: 12MB compressed`,
            `✅ All benchmarks passed!`,
            `📈 Performance score: 94/100`
        ];

        for (let i = 0; i < steps.length; i++) {
            await this.addTerminalLine(steps[i], i * 700);
        }

        this.showNotification("📊 Benchmark completed - Excellent performance!");
        return true;
    }

    // Helper function to add terminal lines with delay
    addTerminalLine(text, delay = 0) {
        return new Promise(resolve => {
            setTimeout(() => {
                const terminal = document.getElementById('ml-terminal');
                if (!terminal) {
                    resolve();
                    return;
                }

                const div = document.createElement('div');
                div.className = 'terminal-line';

                if (text.startsWith('$')) {
                    div.innerHTML = `<span class="terminal-prompt">uiota@guardian:~</span> <span class="terminal-output">${text.substring(2)}</span>`;
                } else if (text.startsWith('✓') || text.startsWith('✅') || text.startsWith('🎉') || text.startsWith('🏆')) {
                    div.innerHTML = `<span class="terminal-success">${text}</span>`;
                } else if (text.startsWith('❌') || text.startsWith('⚠️')) {
                    div.innerHTML = `<span class="terminal-error">${text}</span>`;
                } else {
                    div.innerHTML = `<span class="terminal-output">${text}</span>`;
                }

                terminal.appendChild(div);
                terminal.scrollTop = terminal.scrollHeight;

                // Keep terminal clean (max 50 lines)
                const lines = terminal.querySelectorAll('.terminal-line');
                if (lines.length > 50) {
                    lines[0].remove();
                }

                resolve();
            }, delay);
        });
    }

    // Update training progress
    updateTrainingProgress(percentage) {
        const progressBar = document.querySelector('.training-progress');
        if (progressBar) {
            progressBar.style.width = percentage + '%';
        }
    }

    // Show notification
    showNotification(message) {
        const notification = document.getElementById('notification');
        if (notification) {
            notification.textContent = message;
            notification.classList.add('show');
            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }
    }

    // Get model list for display
    getModels() {
        return this.models;
    }

    // Get federation stats
    getFederationStats() {
        return {
            rounds: this.federationRounds,
            devices: this.connectedDevices,
            globalAccuracy: 96.4,
            dataProcessed: "1.2TB"
        };
    }
}

// Device Network Simulator
class DeviceNetwork {
    constructor() {
        this.devices = [
            { id: "guardian-001", name: "CryptoGuardian-Alpha", type: "Mobile", status: "online", battery: 89, models: 3 },
            { id: "guardian-002", name: "DataGuardian-Beta", type: "Desktop", status: "online", battery: 100, models: 7 },
            { id: "guardian-003", name: "EdgeGuardian-Gamma", type: "IoT", status: "offline", battery: 67, models: 2 },
            { id: "guardian-004", name: "CloudGuardian-Delta", type: "Server", status: "online", battery: 100, models: 12 },
            { id: "guardian-005", name: "MobileGuardian-Epsilon", type: "Mobile", status: "online", battery: 45, models: 4 }
        ];
    }

    // Scan for nearby devices
    async scanDevices() {
        const results = [];
        const scanSteps = [
            "🔍 Scanning for UIOTA devices...",
            "📡 Checking Bluetooth Low Energy...",
            "🌐 Checking WiFi Direct...",
            "📶 Checking mesh network...",
            "✅ Scan complete"
        ];

        for (let step of scanSteps) {
            await this.addScanResult(step);
            await new Promise(resolve => setTimeout(resolve, 800));
        }

        // Simulate finding devices
        const foundDevices = this.devices.filter(d => d.status === 'online').slice(0, 3);
        for (let device of foundDevices) {
            await this.addScanResult(`📱 Found: ${device.name} (${device.type})`);
            await new Promise(resolve => setTimeout(resolve, 500));
        }

        return foundDevices;
    }

    addScanResult(text) {
        return new Promise(resolve => {
            const terminal = document.getElementById('ml-terminal');
            if (terminal) {
                const div = document.createElement('div');
                div.className = 'terminal-line';
                div.innerHTML = `<span class="terminal-output">${text}</span>`;
                terminal.appendChild(div);
                terminal.scrollTop = terminal.scrollHeight;
            }
            resolve();
        });
    }

    getDevices() {
        return this.devices;
    }
}

// Blockchain Simulator
class BlockchainSimulator {
    constructor() {
        this.blocks = 15847;
        this.transactions = 234567;
        this.peers = 847;
        this.hashRate = "2.3 TH/s";
    }

    async syncBlockchain() {
        const terminal = document.getElementById('ml-terminal');

        const steps = [
            `$ uiota-blockchain sync`,
            `🔗 Connecting to UIOTA network...`,
            `✓ Connected to ${this.peers} peers`,
            `📦 Current block: ${this.blocks}`,
            `⬇️ Downloading blocks ${this.blocks + 1}-${this.blocks + 10}`,
            `🔐 Verifying cryptographic proofs...`,
            `✅ Block ${this.blocks + 1} verified`,
            `✅ Block ${this.blocks + 2} verified`,
            `✅ Block ${this.blocks + 3} verified`,
            `🎯 Blockchain sync complete`,
            `📊 Total transactions: ${this.transactions + 47}`
        ];

        for (let i = 0; i < steps.length; i++) {
            await this.addTerminalLine(steps[i], i * 600);
        }

        this.blocks += 10;
        this.transactions += 47;

        return true;
    }

    addTerminalLine(text, delay) {
        return new Promise(resolve => {
            setTimeout(() => {
                const terminal = document.getElementById('ml-terminal');
                if (terminal) {
                    const div = document.createElement('div');
                    div.className = 'terminal-line';

                    if (text.startsWith('$')) {
                        div.innerHTML = `<span class="terminal-prompt">uiota@blockchain:~</span> <span class="terminal-output">${text.substring(2)}</span>`;
                    } else if (text.startsWith('✅') || text.startsWith('🎯')) {
                        div.innerHTML = `<span class="terminal-success">${text}</span>`;
                    } else {
                        div.innerHTML = `<span class="terminal-output">${text}</span>`;
                    }

                    terminal.appendChild(div);
                    terminal.scrollTop = terminal.scrollHeight;
                }
                resolve();
            }, delay);
        });
    }

    getStats() {
        return {
            blocks: this.blocks,
            transactions: this.transactions,
            peers: this.peers,
            hashRate: this.hashRate
        };
    }
}

// Global instances
const mlSimulator = new MLSimulator();
const deviceNetwork = new DeviceNetwork();
const blockchainSim = new BlockchainSimulator();

// Export for use in portal
if (typeof window !== 'undefined') {
    window.mlSimulator = mlSimulator;
    window.deviceNetwork = deviceNetwork;
    window.blockchainSim = blockchainSim;
}