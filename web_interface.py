#!/usr/bin/env python3
"""
Off-Guard Web Interface with AI Integration
Comprehensive web interface with OpenAI, Anthropic, and Flower FL integration
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
import os
from datetime import datetime

# AI Client imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

app = FastAPI(title="Off-Guard AI Platform", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
connected_clients: List[WebSocket] = []
system_status = {
    "ai_services": {
        "openai": {"status": "available" if OPENAI_AVAILABLE else "unavailable", "models": []},
        "anthropic": {"status": "available" if ANTHROPIC_AVAILABLE else "unavailable", "models": []},
    },
    "fl_server": {"status": "stopped", "clients": 0, "rounds": 0},
    "encryption": {"status": "active", "algorithm": "Fernet"},
    "demo_mode": True
}

class AIManager:
    """Manages AI service integrations"""

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self._setup_clients()

    def _setup_clients(self):
        """Setup AI clients with API keys from environment"""
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                system_status["ai_services"]["openai"]["models"] = [
                    "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"
                ]

        if ANTHROPIC_AVAILABLE:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                system_status["ai_services"]["anthropic"]["models"] = [
                    "claude-3-sonnet", "claude-3-haiku", "claude-3-opus"
                ]

    async def chat_completion(self, messages: List[Dict], provider: str = "openai", model: str = None):
        """Get chat completion from specified AI provider"""
        try:
            if provider == "openai" and self.openai_client:
                model = model or "gpt-3.5-turbo"
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model=model,
                    messages=messages,
                    max_tokens=500
                )
                return {
                    "provider": "openai",
                    "model": model,
                    "response": response.choices[0].message.content,
                    "usage": response.usage._asdict() if response.usage else {}
                }

            elif provider == "anthropic" and self.anthropic_client:
                model = model or "claude-3-sonnet-20240229"
                # Convert messages to Anthropic format
                system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
                user_messages = [m for m in messages if m["role"] in ["user", "assistant"]]

                response = await asyncio.to_thread(
                    self.anthropic_client.messages.create,
                    model=model,
                    max_tokens=500,
                    system=system_msg,
                    messages=user_messages
                )
                return {
                    "provider": "anthropic",
                    "model": model,
                    "response": response.content[0].text,
                    "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
                }

            else:
                # Fallback demo response
                return {
                    "provider": "demo",
                    "model": "demo-model",
                    "response": "This is a demo response. Configure API keys to use real AI services.",
                    "usage": {"tokens": 0}
                }

        except Exception as e:
            logging.error(f"AI completion error: {e}")
            return {
                "provider": provider,
                "model": model or "unknown",
                "response": f"Error: {str(e)}",
                "usage": {"error": True}
            }

ai_manager = AIManager()

@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Serve the main web interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Off-Guard AI Platform</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 3rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { font-size: 1.2rem; opacity: 0.9; }

        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }
        .card:hover { transform: translateY(-5px); }
        .card h3 { margin-bottom: 15px; font-size: 1.5rem; }

        .chat-container {
            grid-column: 1 / -1;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.2);
        }

        .chat-messages {
            height: 300px;
            overflow-y: auto;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }

        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
        }
        .message.user {
            background: rgba(103,126,234,0.3);
            text-align: right;
            margin-left: 20%;
        }
        .message.ai {
            background: rgba(118,75,162,0.3);
            margin-right: 20%;
        }
        .message .provider {
            font-size: 0.8rem;
            opacity: 0.7;
            margin-bottom: 5px;
        }

        .chat-input {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .chat-input select,
        .chat-input input,
        .chat-input button {
            padding: 12px;
            border: none;
            border-radius: 8px;
            background: rgba(255,255,255,0.2);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.3);
        }
        .chat-input input { flex: 1; }
        .chat-input button {
            background: rgba(103,126,234,0.8);
            cursor: pointer;
            transition: background 0.3s;
        }
        .chat-input button:hover { background: rgba(103,126,234,1); }

        .status {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .status-active { background: #4CAF50; }
        .status-inactive { background: #f44336; }

        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .metric {
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }

        .actions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 10px 20px;
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
            display: inline-block;
        }
        .btn:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }
        .btn-primary { background: rgba(103,126,234,0.6); }
        .btn-success { background: rgba(76,175,80,0.6); }
        .btn-warning { background: rgba(255,152,0,0.6); }

        @media (max-width: 768px) {
            .dashboard { grid-template-columns: 1fr; }
            .header h1 { font-size: 2rem; }
            .chat-input { flex-direction: column; }
            .chat-input input,
            .chat-input button { width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Off-Guard AI Platform</h1>
            <p>Secure Federated Learning with AI Integration</p>
        </div>

        <div class="dashboard">
            <div class="card">
                <h3>🤖 AI Services</h3>
                <div class="status">
                    <div class="status-indicator status-active"></div>
                    <span>OpenAI Integration</span>
                </div>
                <div class="status">
                    <div class="status-indicator status-active"></div>
                    <span>Anthropic Integration</span>
                </div>
                <div class="actions">
                    <button class="btn btn-primary" onclick="testAI('openai')">Test OpenAI</button>
                    <button class="btn btn-primary" onclick="testAI('anthropic')">Test Anthropic</button>
                </div>
            </div>

            <div class="card">
                <h3>🌸 Federated Learning</h3>
                <div class="status">
                    <div class="status-indicator" id="fl-status"></div>
                    <span id="fl-status-text">Server Stopped</span>
                </div>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="fl-clients">0</div>
                        <div class="metric-label">Connected Clients</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="fl-rounds">0</div>
                        <div class="metric-label">Training Rounds</div>
                    </div>
                </div>
                <div class="actions">
                    <button class="btn btn-success" onclick="startFLServer()">Start FL Server</button>
                    <button class="btn btn-warning" onclick="startFLClient()">Start FL Client</button>
                </div>
            </div>

            <div class="card">
                <h3>🔐 Security Status</h3>
                <div class="status">
                    <div class="status-indicator status-active"></div>
                    <span>Encryption Active</span>
                </div>
                <div class="status">
                    <div class="status-indicator status-active"></div>
                    <span>Offline Mode Ready</span>
                </div>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">Fernet</div>
                        <div class="metric-label">Encryption</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">256-bit</div>
                        <div class="metric-label">Key Length</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>📊 System Metrics</h3>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="active-connections">0</div>
                        <div class="metric-label">Active Connections</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="uptime">0s</div>
                        <div class="metric-label">Uptime</div>
                    </div>
                </div>
                <div class="actions">
                    <a href="/docs" class="btn">API Docs</a>
                    <a href="/sdk" class="btn">SDK Guide</a>
                    <button class="btn" onclick="downloadDemo()">Download Demo</button>
                </div>
            </div>
        </div>

        <div class="chat-container">
            <h3>💬 AI Chat Interface</h3>
            <div class="chat-messages" id="chat-messages"></div>
            <div class="chat-input">
                <select id="ai-provider">
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="demo">Demo Mode</option>
                </select>
                <select id="ai-model">
                    <option value="auto">Auto Select</option>
                </select>
                <input type="text" id="chat-input" placeholder="Ask about federated learning, AI, or security..." />
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let startTime = Date.now();

        // Initialize WebSocket connection
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = function() {
                console.log('WebSocket connected');
                updateConnectionStatus();
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };

            ws.onclose = function() {
                console.log('WebSocket disconnected');
                setTimeout(initWebSocket, 3000); // Reconnect after 3 seconds
            };
        }

        function handleWebSocketMessage(data) {
            if (data.type === 'status_update') {
                updateSystemStatus(data.status);
            } else if (data.type === 'chat_response') {
                addChatMessage(data.response, 'ai', data.provider);
            }
        }

        function updateSystemStatus(status) {
            if (status.fl_server) {
                document.getElementById('fl-clients').textContent = status.fl_server.clients;
                document.getElementById('fl-rounds').textContent = status.fl_server.rounds;

                const statusEl = document.getElementById('fl-status');
                const statusTextEl = document.getElementById('fl-status-text');

                if (status.fl_server.status === 'running') {
                    statusEl.className = 'status-indicator status-active';
                    statusTextEl.textContent = 'Server Running';
                } else {
                    statusEl.className = 'status-indicator status-inactive';
                    statusTextEl.textContent = 'Server Stopped';
                }
            }
        }

        function updateConnectionStatus() {
            document.getElementById('active-connections').textContent = ws.readyState === WebSocket.OPEN ? '1' : '0';
        }

        function updateUptime() {
            const uptime = Math.floor((Date.now() - startTime) / 1000);
            document.getElementById('uptime').textContent = uptime + 's';
        }

        function sendMessage() {
            const input = document.getElementById('chat-input');
            const provider = document.getElementById('ai-provider').value;
            const model = document.getElementById('ai-model').value;
            const message = input.value.trim();

            if (!message) return;

            addChatMessage(message, 'user');
            input.value = '';

            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'chat',
                    message: message,
                    provider: provider,
                    model: model
                }));
            }
        }

        function addChatMessage(content, sender, provider = null) {
            const messagesEl = document.getElementById('chat-messages');
            const messageEl = document.createElement('div');
            messageEl.className = `message ${sender}`;

            let html = '';
            if (provider && sender === 'ai') {
                html += `<div class="provider">${provider.toUpperCase()}</div>`;
            }
            html += `<div>${content}</div>`;

            messageEl.innerHTML = html;
            messagesEl.appendChild(messageEl);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        function testAI(provider) {
            const testMessage = `Hello! Can you explain what ${provider} is and how it works with federated learning?`;
            document.getElementById('chat-input').value = testMessage;
            document.getElementById('ai-provider').value = provider;
            sendMessage();
        }

        function startFLServer() {
            fetch('/api/fl/start-server', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    addChatMessage(`FL Server started: ${JSON.stringify(data)}`, 'ai', 'system');
                });
        }

        function startFLClient() {
            fetch('/api/fl/start-client', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    addChatMessage(`FL Client started: ${JSON.stringify(data)}`, 'ai', 'system');
                });
        }

        function downloadDemo() {
            window.open('/api/download-demo', '_blank');
        }

        // Event listeners
        document.getElementById('chat-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Initialize
        initWebSocket();
        setInterval(updateUptime, 1000);
        setInterval(updateConnectionStatus, 5000);

        // Welcome message
        setTimeout(() => {
            addChatMessage('Welcome to Off-Guard AI Platform! Ask me about federated learning, AI integration, or security features.', 'ai', 'system');
        }, 1000);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            if message_data.get("type") == "chat":
                # Handle chat message
                response = await ai_manager.chat_completion(
                    messages=[{"role": "user", "content": message_data["message"]}],
                    provider=message_data.get("provider", "openai"),
                    model=message_data.get("model")
                )

                await websocket.send_text(json.dumps({
                    "type": "chat_response",
                    "response": response["response"],
                    "provider": response["provider"],
                    "model": response["model"]
                }))

    except WebSocketDisconnect:
        connected_clients.remove(websocket)

@app.get("/api/status")
async def get_system_status():
    """Get current system status"""
    return JSONResponse(content=system_status)

@app.post("/api/fl/start-server")
async def start_fl_server():
    """Start federated learning server"""
    try:
        # This would start the actual FL server in a real implementation
        system_status["fl_server"] = {
            "status": "running",
            "clients": 0,
            "rounds": 0,
            "address": "localhost:8080"
        }
        return {"status": "success", "message": "FL server started", "address": "localhost:8080"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/fl/start-client")
async def start_fl_client():
    """Start federated learning client"""
    try:
        # This would start the actual FL client in a real implementation
        system_status["fl_server"]["clients"] += 1
        return {"status": "success", "message": "FL client started", "client_id": f"client_{system_status['fl_server']['clients']}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-demo")
async def download_demo():
    """Download demo files"""
    return {"download_url": "/static/offguard-demo.zip", "size": "2.3MB"}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🚀 Starting Off-Guard AI Platform...")
    print("🌐 Web Interface: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")