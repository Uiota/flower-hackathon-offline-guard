#!/usr/bin/env python3
"""
Off-Guard SDK Documentation Generator
Creates comprehensive SDK documentation with examples
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import json

def generate_sdk_docs():
    """Generate comprehensive SDK documentation"""

    sdk_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Off-Guard SDK Documentation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header {
            background: rgba(255,255,255,0.9);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.3);
        }
        .header h1 { color: #2c3e50; margin-bottom: 10px; font-size: 2.5rem; }
        .header p { color: #7f8c8d; font-size: 1.2rem; }

        .nav-menu {
            background: rgba(255,255,255,0.9);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.3);
        }
        .nav-menu ul { list-style: none; display: flex; flex-wrap: wrap; gap: 20px; }
        .nav-menu li a {
            text-decoration: none;
            color: #3498db;
            padding: 10px 15px;
            background: rgba(52,152,219,0.1);
            border-radius: 8px;
            transition: all 0.3s;
        }
        .nav-menu li a:hover { background: rgba(52,152,219,0.2); transform: translateY(-2px); }

        .section {
            background: rgba(255,255,255,0.9);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.3);
        }
        .section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 2rem;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        .section h3 {
            color: #34495e;
            margin: 25px 0 15px 0;
            font-size: 1.4rem;
        }

        .code-block {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            position: relative;
        }
        .code-block::before {
            content: 'Python';
            position: absolute;
            top: 5px;
            right: 10px;
            font-size: 0.8rem;
            color: #95a5a6;
        }

        .api-method {
            background: rgba(46,204,113,0.1);
            border-left: 4px solid #2ecc71;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }
        .api-method h4 {
            color: #27ae60;
            margin-bottom: 10px;
        }

        .example-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        .example-card {
            background: rgba(52,152,219,0.1);
            border-radius: 10px;
            padding: 20px;
            border: 1px solid rgba(52,152,219,0.2);
        }
        .example-card h4 {
            color: #2980b9;
            margin-bottom: 15px;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .feature-card {
            background: rgba(155,89,182,0.1);
            border-radius: 10px;
            padding: 20px;
            border: 1px solid rgba(155,89,182,0.2);
            text-align: center;
        }
        .feature-card .icon {
            font-size: 2.5rem;
            margin-bottom: 15px;
        }
        .feature-card h4 {
            color: #8e44ad;
            margin-bottom: 10px;
        }

        .install-steps {
            background: rgba(230,126,34,0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid rgba(230,126,34,0.2);
        }
        .install-steps ol {
            counter-reset: step-counter;
            list-style: none;
        }
        .install-steps li {
            counter-increment: step-counter;
            margin: 15px 0;
            padding: 15px;
            background: rgba(255,255,255,0.5);
            border-radius: 8px;
            position: relative;
            padding-left: 60px;
        }
        .install-steps li::before {
            content: counter(step-counter);
            position: absolute;
            left: 20px;
            top: 50%;
            transform: translateY(-50%);
            background: #e67e22;
            color: white;
            width: 25px;
            height: 25px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }

        @media (max-width: 768px) {
            .nav-menu ul { flex-direction: column; }
            .example-grid { grid-template-columns: 1fr; }
            .header h1 { font-size: 2rem; }
        }

        .copy-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            position: absolute;
            top: 10px;
            right: 50px;
            font-size: 0.8rem;
        }
        .copy-btn:hover { background: #2980b9; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Off-Guard SDK</h1>
            <p>Secure Federated Learning & AI Integration Framework</p>
        </div>

        <div class="nav-menu">
            <ul>
                <li><a href="#installation">📦 Installation</a></li>
                <li><a href="#quickstart">🚀 Quick Start</a></li>
                <li><a href="#features">✨ Features</a></li>
                <li><a href="#api">📚 API Reference</a></li>
                <li><a href="#examples">💡 Examples</a></li>
                <li><a href="#security">🔐 Security</a></li>
            </ul>
        </div>

        <div class="section" id="installation">
            <h2>📦 Installation</h2>
            <p>Get started with Off-Guard SDK in minutes. Follow these simple steps:</p>

            <div class="install-steps">
                <ol>
                    <li>
                        <strong>Install Python Dependencies</strong>
                        <div class="code-block">
                            <button class="copy-btn" onclick="copyCode(this)">Copy</button>
pip install off-guard-sdk flwr torch transformers fastapi anthropic openai
                        </div>
                    </li>
                    <li>
                        <strong>Clone the Repository</strong>
                        <div class="code-block">
                            <button class="copy-btn" onclick="copyCode(this)">Copy</button>
git clone https://github.com/uiota/offline-guard.git
cd offline-guard
                        </div>
                    </li>
                    <li>
                        <strong>Set Environment Variables</strong>
                        <div class="code-block">
                            <button class="copy-btn" onclick="copyCode(this)">Copy</button>
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export OFFGUARD_ENCRYPTION_KEY="your_32_byte_key"
                        </div>
                    </li>
                    <li>
                        <strong>Run the Demo</strong>
                        <div class="code-block">
                            <button class="copy-btn" onclick="copyCode(this)">Copy</button>
python web_interface.py
                        </div>
                    </li>
                </ol>
            </div>
        </div>

        <div class="section" id="quickstart">
            <h2>🚀 Quick Start</h2>
            <p>Start using Off-Guard SDK with this simple example:</p>

            <div class="code-block">
                <button class="copy-btn" onclick="copyCode(this)">Copy</button>
from offguard import SecureFLClient, AIManager, EncryptionManager

# Initialize Off-Guard components
encryption = EncryptionManager()
ai_manager = AIManager(providers=["openai", "anthropic"])
fl_client = SecureFLClient(
    client_id="device_1",
    encryption_key=encryption.key,
    ai_manager=ai_manager
)

# Start federated learning with AI integration
async def run_secure_fl():
    # Connect to FL server
    await fl_client.connect("localhost:8080")

    # Train model with encrypted communication
    await fl_client.train_model(rounds=5)

    # Use AI for model insights
    insights = await ai_manager.analyze_model_performance(
        fl_client.get_metrics()
    )

    print(f"Training completed with insights: {insights}")

# Run the demo
import asyncio
asyncio.run(run_secure_fl())
            </div>
        </div>

        <div class="section" id="features">
            <h2>✨ Key Features</h2>

            <div class="feature-grid">
                <div class="feature-card">
                    <div class="icon">🌸</div>
                    <h4>Flower Integration</h4>
                    <p>Native Flower framework support for scalable federated learning</p>
                </div>
                <div class="feature-card">
                    <div class="icon">🤖</div>
                    <h4>Multi-AI Support</h4>
                    <p>OpenAI, Anthropic, and custom model integration</p>
                </div>
                <div class="feature-card">
                    <div class="icon">🔐</div>
                    <h4>End-to-End Encryption</h4>
                    <p>Fernet encryption for all model communications</p>
                </div>
                <div class="feature-card">
                    <div class="icon">📱</div>
                    <h4>Cross-Platform</h4>
                    <p>Works on mobile, desktop, and edge devices</p>
                </div>
                <div class="feature-card">
                    <div class="icon">🌐</div>
                    <h4>Offline Capable</h4>
                    <p>Functions without internet connectivity</p>
                </div>
                <div class="feature-card">
                    <div class="icon">⚡</div>
                    <h4>Real-time UI</h4>
                    <p>WebSocket-based live dashboard</p>
                </div>
            </div>
        </div>

        <div class="section" id="api">
            <h2>📚 API Reference</h2>

            <div class="api-method">
                <h4>SecureFLClient</h4>
                <p>Main class for federated learning with encryption</p>
                <div class="code-block">
                    <button class="copy-btn" onclick="copyCode(this)">Copy</button>
class SecureFLClient:
    def __init__(self, client_id: str, encryption_key: bytes, ai_manager: AIManager)
    async def connect(self, server_address: str) -> bool
    async def train_model(self, rounds: int = 5) -> Dict
    async def evaluate_model(self) -> Dict
    def get_metrics(self) -> Dict
                </div>
            </div>

            <div class="api-method">
                <h4>AIManager</h4>
                <p>Manages multiple AI service integrations</p>
                <div class="code-block">
                    <button class="copy-btn" onclick="copyCode(this)">Copy</button>
class AIManager:
    def __init__(self, providers: List[str] = ["openai", "anthropic"])
    async def chat_completion(self, messages: List[Dict], provider: str) -> Dict
    async def analyze_model_performance(self, metrics: Dict) -> str
    async def generate_insights(self, data: Dict) -> str
                </div>
            </div>

            <div class="api-method">
                <h4>EncryptionManager</h4>
                <p>Handles all encryption/decryption operations</p>
                <div class="code-block">
                    <button class="copy-btn" onclick="copyCode(this)">Copy</button>
class EncryptionManager:
    def __init__(self, key: bytes = None)
    def encrypt_data(self, data: bytes) -> bytes
    def decrypt_data(self, encrypted_data: bytes) -> bytes
    def generate_key(self) -> bytes
    def save_key(self, filepath: str) -> None
                </div>
            </div>
        </div>

        <div class="section" id="examples">
            <h2>💡 Usage Examples</h2>

            <div class="example-grid">
                <div class="example-card">
                    <h4>🏥 Healthcare FL</h4>
                    <div class="code-block">
                        <button class="copy-btn" onclick="copyCode(this)">Copy</button>
# Medical data federated learning
client = SecureFLClient(
    client_id="hospital_1",
    model_type="medical_diagnosis",
    encryption_key=load_hipaa_key()
)

await client.train_with_privacy(
    data_path="/secure/medical_data",
    privacy_level="maximum"
)
                    </div>
                </div>

                <div class="example-card">
                    <h4>🏦 Financial FL</h4>
                    <div class="code-block">
                        <button class="copy-btn" onclick="copyCode(this)">Copy</button>
# Financial fraud detection
client = SecureFLClient(
    client_id="bank_branch_1",
    model_type="fraud_detection",
    compliance_mode="pci_dss"
)

results = await client.detect_anomalies(
    transaction_data=secure_data
)
                    </div>
                </div>

                <div class="example-card">
                    <h4>📱 Mobile FL</h4>
                    <div class="code-block">
                        <button class="copy-btn" onclick="copyCode(this)">Copy</button>
# Mobile device federated learning
mobile_client = MobileFLClient(
    device_id="mobile_001",
    battery_aware=True,
    bandwidth_limit="1MB"
)

await mobile_client.train_on_device(
    local_data=user_data,
    rounds=3
)
                    </div>
                </div>

                <div class="example-card">
                    <h4>🎯 Custom AI Integration</h4>
                    <div class="code-block">
                        <button class="copy-btn" onclick="copyCode(this)">Copy</button>
# Custom AI model integration
ai_manager = AIManager()
ai_manager.add_custom_provider(
    name="custom_llm",
    endpoint="https://api.custom.ai",
    auth_token="your_token"
)

response = await ai_manager.query(
    "Analyze FL performance",
    provider="custom_llm"
)
                    </div>
                </div>
            </div>
        </div>

        <div class="section" id="security">
            <h2>🔐 Security Features</h2>

            <h3>Encryption Standards</h3>
            <ul>
                <li><strong>Fernet Symmetric Encryption:</strong> 256-bit keys with AES 128 in CBC mode</li>
                <li><strong>Key Rotation:</strong> Automatic key rotation every 24 hours</li>
                <li><strong>Perfect Forward Secrecy:</strong> Each session uses unique encryption keys</li>
                <li><strong>Zero-Knowledge Architecture:</strong> Server never sees plaintext data</li>
            </ul>

            <h3>Privacy Protection</h3>
            <div class="code-block">
                <button class="copy-btn" onclick="copyCode(this)">Copy</button>
# Enable differential privacy
client = SecureFLClient(
    privacy_config={
        "differential_privacy": True,
        "epsilon": 1.0,  # Privacy budget
        "delta": 1e-5,   # Privacy parameter
        "noise_multiplier": 1.1
    }
)

# Add homomorphic encryption
client.enable_homomorphic_encryption(
    scheme="CKKS",
    poly_modulus_degree=8192
)
            </div>

            <h3>Compliance Features</h3>
            <ul>
                <li><strong>GDPR Compliance:</strong> Right to be forgotten, data portability</li>
                <li><strong>HIPAA Support:</strong> Healthcare data protection standards</li>
                <li><strong>SOC 2 Type II:</strong> Security and availability controls</li>
                <li><strong>Audit Logging:</strong> Complete audit trail of all operations</li>
            </ul>

            <div class="code-block">
                <button class="copy-btn" onclick="copyCode(this)">Copy</button>
# Enable compliance mode
client = SecureFLClient(
    compliance_mode="gdpr",
    audit_logging=True,
    data_retention_days=30
)

# Request data deletion (GDPR Article 17)
await client.request_data_deletion(
    user_id="user_123",
    reason="user_request"
)
            </div>
        </div>

        <div class="section">
            <h2>🚀 Advanced Configuration</h2>

            <div class="code-block">
                <button class="copy-btn" onclick="copyCode(this)">Copy</button>
# Complete Off-Guard configuration
config = {
    "server": {
        "host": "0.0.0.0",
        "port": 8080,
        "ssl_enabled": True,
        "cert_file": "/path/to/cert.pem",
        "key_file": "/path/to/key.pem"
    },
    "federated_learning": {
        "strategy": "FedAvg",
        "min_clients": 2,
        "min_available_clients": 2,
        "rounds": 10,
        "client_timeout": 300
    },
    "encryption": {
        "algorithm": "Fernet",
        "key_rotation_hours": 24,
        "backup_keys": 3
    },
    "ai_integration": {
        "providers": ["openai", "anthropic"],
        "fallback_model": "local_llm",
        "rate_limiting": True,
        "cache_responses": True
    },
    "privacy": {
        "differential_privacy": True,
        "epsilon": 1.0,
        "delta": 1e-5,
        "secure_aggregation": True
    }
}

# Initialize with advanced config
platform = OffGuardPlatform(config)
await platform.start()
            </div>
        </div>

        <div class="section">
            <h2>📞 Support & Community</h2>
            <p>Get help and connect with the Off-Guard community:</p>

            <div class="feature-grid">
                <div class="feature-card">
                    <div class="icon">📚</div>
                    <h4>Documentation</h4>
                    <p><a href="https://docs.offguard.ai">docs.offguard.ai</a></p>
                </div>
                <div class="feature-card">
                    <div class="icon">💬</div>
                    <h4>Discord Community</h4>
                    <p><a href="https://discord.gg/offguard">Join our Discord</a></p>
                </div>
                <div class="feature-card">
                    <div class="icon">🐛</div>
                    <h4>Bug Reports</h4>
                    <p><a href="https://github.com/uiota/offline-guard/issues">GitHub Issues</a></p>
                </div>
                <div class="feature-card">
                    <div class="icon">📧</div>
                    <h4>Email Support</h4>
                    <p><a href="mailto:support@offguard.ai">support@offguard.ai</a></p>
                </div>
            </div>
        </div>
    </div>

    <script>
        function copyCode(button) {
            const codeBlock = button.nextSibling;
            const text = codeBlock.textContent.trim();

            navigator.clipboard.writeText(text).then(() => {
                button.textContent = 'Copied!';
                button.style.background = '#27ae60';

                setTimeout(() => {
                    button.textContent = 'Copy';
                    button.style.background = '#3498db';
                }, 2000);
            });
        }

        // Smooth scrolling for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    </script>
</body>
</html>
    """
    return sdk_html

# Add SDK route to main web interface
def add_sdk_route(app):
    @app.get("/sdk", response_class=HTMLResponse)
    async def get_sdk_docs():
        return HTMLResponse(content=generate_sdk_docs())

    return app

if __name__ == "__main__":
    print("📚 SDK Documentation generated!")
    print("🌐 Access at: http://localhost:8000/sdk")