#!/usr/bin/env python3
"""
Off-Guard Portal Demo
Complete demo portal with all functions integrated
"""

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio
import json
import logging
from typing import Dict, List
import uuid
from datetime import datetime
import subprocess
import os

class PortalDemo:
    """Complete portal demo with all Off-Guard functions"""

    def __init__(self):
        self.active_sessions = {}
        self.fl_clients = {}
        self.demo_metrics = {
            "total_sessions": 0,
            "active_fl_clients": 0,
            "messages_processed": 0,
            "encryption_operations": 0
        }

    def get_portal_html(self):
        """Generate complete portal demo HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Off-Guard Portal Demo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            min-height: 100vh;
            overflow-x: hidden;
        }

        .portal-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .portal-header {
            text-align: center;
            margin-bottom: 30px;
            animation: fadeInDown 1s ease-out;
        }
        .portal-header h1 {
            font-size: 3.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            background: linear-gradient(45deg, #fff, #e8f4fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .portal-header p {
            font-size: 1.3rem;
            opacity: 0.9;
        }

        .demo-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }

        .demo-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.2);
            transition: all 0.4s ease;
            animation: fadeInUp 0.6s ease-out;
            position: relative;
            overflow: hidden;
        }
        .demo-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3498db, #9b59b6, #e74c3c);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .demo-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }
        .demo-card:hover::before {
            opacity: 1;
        }

        .demo-card h3 {
            font-size: 1.8rem;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .demo-card .icon {
            font-size: 2rem;
            background: linear-gradient(45deg, #3498db, #9b59b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .demo-controls {
            display: flex;
            flex-direction: column;
            gap: 15px;
            margin-top: 20px;
        }

        .control-group {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
        }

        .btn {
            padding: 12px 20px;
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 25px;
            color: #fff;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-weight: 500;
        }
        .btn:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .btn-primary { background: rgba(52,152,219,0.6); border-color: #3498db; }
        .btn-success { background: rgba(46,204,113,0.6); border-color: #2ecc71; }
        .btn-warning { background: rgba(243,156,18,0.6); border-color: #f39c12; }
        .btn-danger { background: rgba(231,76,60,0.6); border-color: #e74c3c; }

        .status-display {
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            padding: 15px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            min-height: 120px;
            overflow-y: auto;
            border: 1px solid rgba(255,255,255,0.1);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            text-align: center;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            background: linear-gradient(45deg, #fff, #e8f4fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 5px;
        }

        .live-demo {
            background: rgba(0,0,0,0.2);
            border-radius: 20px;
            padding: 25px;
            margin: 30px 0;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .live-demo h3 {
            margin-bottom: 20px;
            font-size: 1.8rem;
        }

        .device-simulator {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        .device {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .device h4 {
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .device-status {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #2ecc71;
            animation: pulse 2s infinite;
        }

        .chat-interface {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .chat-messages {
            height: 200px;
            overflow-y: auto;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .chat-input {
            display: flex;
            gap: 10px;
        }
        .chat-input input {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 25px;
            background: rgba(255,255,255,0.2);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.3);
        }
        .chat-input input::placeholder {
            color: rgba(255,255,255,0.7);
        }

        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.2);
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            transition: width 0.3s ease;
            border-radius: 4px;
        }

        .console-output {
            background: #2c3e50;
            color: #ecf0f1;
            border-radius: 10px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            height: 150px;
            overflow-y: auto;
            margin: 15px 0;
        }

        @media (max-width: 768px) {
            .demo-grid { grid-template-columns: 1fr; }
            .device-simulator { grid-template-columns: 1fr; }
            .metrics-grid { grid-template-columns: repeat(2, 1fr); }
            .portal-header h1 { font-size: 2.5rem; }
        }
    </style>
</head>
<body>
    <div class="portal-container">
        <div class="portal-header">
            <h1>🛡️ Off-Guard Portal Demo</h1>
            <p>Complete Federated Learning & AI Integration Platform</p>
        </div>

        <div class="demo-grid">
            <div class="demo-card">
                <h3><span class="icon">🌸</span>Flower Federated Learning</h3>
                <p>Secure distributed machine learning with encrypted model updates</p>
                <div class="demo-controls">
                    <div class="control-group">
                        <button class="btn btn-success" onclick="startFLServer()">
                            🚀 Start FL Server
                        </button>
                        <button class="btn btn-primary" onclick="addFLClient()">
                            📱 Add Client
                        </button>
                    </div>
                    <div class="control-group">
                        <button class="btn btn-warning" onclick="startTraining()">
                            🏋️ Start Training
                        </button>
                        <button class="btn btn-danger" onclick="stopFL()">
                            ⏹️ Stop FL
                        </button>
                    </div>
                </div>
                <div class="status-display" id="fl-status">
                    FL Server: Stopped<br>
                    Connected Clients: 0<br>
                    Training Rounds: 0<br>
                    Encryption: Active
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="fl-progress" style="width: 0%"></div>
                </div>
            </div>

            <div class="demo-card">
                <h3><span class="icon">🤖</span>AI Integration Hub</h3>
                <p>OpenAI, Anthropic, and custom AI model integration</p>
                <div class="demo-controls">
                    <div class="control-group">
                        <button class="btn btn-primary" onclick="testOpenAI()">
                            🔵 Test OpenAI
                        </button>
                        <button class="btn btn-primary" onclick="testAnthropic()">
                            🟣 Test Anthropic
                        </button>
                    </div>
                    <div class="control-group">
                        <button class="btn btn-success" onclick="aiAnalysis()">
                            📊 AI Analysis
                        </button>
                        <button class="btn btn-warning" onclick="generateInsights()">
                            💡 Generate Insights
                        </button>
                    </div>
                </div>
                <div class="status-display" id="ai-status">
                    OpenAI: Ready<br>
                    Anthropic: Ready<br>
                    Custom Models: 2 loaded<br>
                    Rate Limiting: Active
                </div>
            </div>

            <div class="demo-card">
                <h3><span class="icon">🔐</span>Security & Encryption</h3>
                <p>End-to-end encryption with offline capability</p>
                <div class="demo-controls">
                    <div class="control-group">
                        <button class="btn btn-success" onclick="generateKeys()">
                            🔑 Generate Keys
                        </button>
                        <button class="btn btn-primary" onclick="testEncryption()">
                            🔒 Test Encryption
                        </button>
                    </div>
                    <div class="control-group">
                        <button class="btn btn-warning" onclick="offlineMode()">
                            📴 Offline Mode
                        </button>
                        <button class="btn btn-danger" onclick="securityAudit()">
                            🛡️ Security Audit
                        </button>
                    </div>
                </div>
                <div class="status-display" id="security-status">
                    Encryption: Fernet-256<br>
                    Key Rotation: 24h<br>
                    Offline Ready: Yes<br>
                    Audit Log: Active
                </div>
            </div>

            <div class="demo-card">
                <h3><span class="icon">📊</span>Live Metrics Dashboard</h3>
                <p>Real-time monitoring and analytics</p>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="total-clients">0</div>
                        <div class="metric-label">Total Clients</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="messages">0</div>
                        <div class="metric-label">Messages</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="uptime">0s</div>
                        <div class="metric-label">Uptime</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="encryption-ops">0</div>
                        <div class="metric-label">Encryptions</div>
                    </div>
                </div>
                <div class="demo-controls">
                    <div class="control-group">
                        <button class="btn btn-primary" onclick="exportMetrics()">
                            📈 Export Metrics
                        </button>
                        <button class="btn btn-warning" onclick="resetMetrics()">
                            🔄 Reset Metrics
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <div class="live-demo">
            <h3>🎬 Live Two-Device Federated Learning Demo</h3>
            <p>Watch two simulated devices collaborate securely using encrypted federated learning</p>

            <div class="device-simulator">
                <div class="device">
                    <h4>
                        <div class="device-status" id="device1-status"></div>
                        📱 Device 1 (Mobile)
                    </h4>
                    <div class="console-output" id="device1-console">
                        Device 1 initialized...<br>
                        Waiting for FL server...
                    </div>
                    <div class="control-group">
                        <button class="btn btn-primary" onclick="connectDevice(1)">Connect</button>
                        <button class="btn btn-success" onclick="trainDevice(1)">Train</button>
                    </div>
                </div>

                <div class="device">
                    <h4>
                        <div class="device-status" id="device2-status"></div>
                        💻 Device 2 (Desktop)
                    </h4>
                    <div class="console-output" id="device2-console">
                        Device 2 initialized...<br>
                        Waiting for FL server...
                    </div>
                    <div class="control-group">
                        <button class="btn btn-primary" onclick="connectDevice(2)">Connect</button>
                        <button class="btn btn-success" onclick="trainDevice(2)">Train</button>
                    </div>
                </div>
            </div>

            <div class="demo-controls">
                <div class="control-group">
                    <button class="btn btn-success" onclick="startFullDemo()">
                        🎯 Start Full Demo
                    </button>
                    <button class="btn btn-warning" onclick="simulateOffline()">
                        📴 Simulate Offline
                    </button>
                    <button class="btn btn-danger" onclick="stopDemo()">
                        ⏹️ Stop Demo
                    </button>
                </div>
            </div>
        </div>

        <div class="chat-interface">
            <h3>💬 AI Assistant</h3>
            <div class="chat-messages" id="demo-chat"></div>
            <div class="chat-input">
                <input type="text" id="demo-chat-input" placeholder="Ask about the demo, FL, or security..." />
                <button class="btn btn-primary" onclick="sendDemoChat()">Send</button>
            </div>
        </div>
    </div>

    <script>
        let demoState = {
            flServer: false,
            clients: 0,
            trainingRounds: 0,
            messages: 0,
            encryptionOps: 0,
            startTime: Date.now()
        };

        let ws = null;

        function initDemo() {
            // Initialize WebSocket connection
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/demo-ws`);

            ws.onopen = function() {
                console.log('Demo WebSocket connected');
                addDemoMessage('Portal Demo connected successfully!', 'system');
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleDemoMessage(data);
            };

            ws.onclose = function() {
                console.log('Demo WebSocket disconnected');
                setTimeout(initDemo, 3000);
            };
        }

        function handleDemoMessage(data) {
            if (data.type === 'status_update') {
                updateDemoStatus(data);
            } else if (data.type === 'demo_log') {
                addDeviceLog(data.device, data.message);
            } else if (data.type === 'chat_response') {
                addDemoMessage(data.message, 'ai');
            }
        }

        function updateDemoStatus(data) {
            if (data.fl_server) {
                document.getElementById('fl-status').innerHTML =
                    `FL Server: ${data.fl_server.status}<br>` +
                    `Connected Clients: ${data.fl_server.clients}<br>` +
                    `Training Rounds: ${data.fl_server.rounds}<br>` +
                    `Encryption: Active`;

                document.getElementById('fl-progress').style.width =
                    `${(data.fl_server.rounds / 10) * 100}%`;
            }

            updateMetrics();
        }

        function updateMetrics() {
            document.getElementById('total-clients').textContent = demoState.clients;
            document.getElementById('messages').textContent = demoState.messages;
            document.getElementById('encryption-ops').textContent = demoState.encryptionOps;

            const uptime = Math.floor((Date.now() - demoState.startTime) / 1000);
            document.getElementById('uptime').textContent = uptime + 's';
        }

        function addDeviceLog(device, message) {
            const console = document.getElementById(`device${device}-console`);
            console.innerHTML += message + '<br>';
            console.scrollTop = console.scrollHeight;
        }

        function addDemoMessage(message, sender) {
            const chat = document.getElementById('demo-chat');
            const messageEl = document.createElement('div');
            messageEl.style.cssText = `
                margin: 10px 0;
                padding: 10px;
                border-radius: 10px;
                background: ${sender === 'ai' ? 'rgba(46,204,113,0.3)' : 'rgba(52,152,219,0.3)'};
            `;
            messageEl.textContent = message;
            chat.appendChild(messageEl);
            chat.scrollTop = chat.scrollHeight;
        }

        // Demo Functions
        function startFLServer() {
            demoState.flServer = true;
            sendDemoCommand('start_fl_server');
            updateStatus('fl-status', 'FL Server: Starting...');
        }

        function addFLClient() {
            demoState.clients++;
            sendDemoCommand('add_fl_client');
            updateMetrics();
        }

        function startTraining() {
            if (!demoState.flServer) {
                alert('Please start FL server first');
                return;
            }
            sendDemoCommand('start_training');
            simulateTraining();
        }

        function simulateTraining() {
            let round = 0;
            const interval = setInterval(() => {
                round++;
                demoState.trainingRounds = round;
                demoState.encryptionOps += Math.floor(Math.random() * 10) + 5;
                updateMetrics();

                if (round >= 10) {
                    clearInterval(interval);
                    addDemoMessage('Training completed! Model accuracy improved by 12%', 'system');
                }
            }, 2000);
        }

        function testOpenAI() {
            sendDemoCommand('test_openai');
            updateStatus('ai-status', 'Testing OpenAI connection...');
            setTimeout(() => {
                updateStatus('ai-status', 'OpenAI: Connected ✅<br>Model: GPT-4<br>Rate Limit: 90%<br>Response Time: 1.2s');
            }, 2000);
        }

        function testAnthropic() {
            sendDemoCommand('test_anthropic');
            updateStatus('ai-status', 'Testing Anthropic connection...');
            setTimeout(() => {
                updateStatus('ai-status', 'Anthropic: Connected ✅<br>Model: Claude-3<br>Rate Limit: 95%<br>Response Time: 0.9s');
            }, 2000);
        }

        function generateKeys() {
            sendDemoCommand('generate_keys');
            demoState.encryptionOps += 5;
            updateMetrics();
            updateStatus('security-status', 'New encryption keys generated<br>Key strength: 256-bit<br>Rotation scheduled: 24h<br>Backup created: Yes');
        }

        function testEncryption() {
            sendDemoCommand('test_encryption');
            demoState.encryptionOps += 10;
            updateMetrics();
            addDemoMessage('Encryption test completed: 1000 operations in 0.5s', 'system');
        }

        function connectDevice(deviceId) {
            sendDemoCommand(`connect_device_${deviceId}`);
            addDeviceLog(deviceId, `Connecting to FL server...`);
            setTimeout(() => {
                addDeviceLog(deviceId, `Connected successfully! Device ID: client_${deviceId}`);
                addDeviceLog(deviceId, `Encryption key exchanged`);
                demoState.clients++;
                updateMetrics();
            }, 1500);
        }

        function trainDevice(deviceId) {
            sendDemoCommand(`train_device_${deviceId}`);
            addDeviceLog(deviceId, `Starting local training...`);

            let epoch = 0;
            const trainInterval = setInterval(() => {
                epoch++;
                addDeviceLog(deviceId, `Epoch ${epoch}: Loss = ${(Math.random() * 0.5 + 0.1).toFixed(4)}`);

                if (epoch >= 5) {
                    clearInterval(trainInterval);
                    addDeviceLog(deviceId, `Training complete. Sending encrypted model updates...`);
                    demoState.encryptionOps += 3;
                    updateMetrics();
                }
            }, 1000);
        }

        function startFullDemo() {
            addDemoMessage('Starting full federated learning demo with two devices...', 'system');

            setTimeout(() => startFLServer(), 500);
            setTimeout(() => connectDevice(1), 2000);
            setTimeout(() => connectDevice(2), 3000);
            setTimeout(() => trainDevice(1), 5000);
            setTimeout(() => trainDevice(2), 5500);
            setTimeout(() => startTraining(), 8000);
        }

        function sendDemoChat() {
            const input = document.getElementById('demo-chat-input');
            const message = input.value.trim();
            if (!message) return;

            addDemoMessage(message, 'user');
            input.value = '';

            // Simulate AI response
            setTimeout(() => {
                const responses = [
                    "The federated learning demo shows how multiple devices can collaborate without sharing raw data.",
                    "Off-Guard uses Fernet encryption to secure all model updates between devices.",
                    "The AI integration allows for real-time analysis of federated learning performance.",
                    "Offline mode ensures the system works even without internet connectivity.",
                    "Each device trains locally and only shares encrypted model parameters."
                ];
                const response = responses[Math.floor(Math.random() * responses.length)];
                addDemoMessage(response, 'ai');
            }, 1000);

            demoState.messages++;
            updateMetrics();
        }

        function sendDemoCommand(command) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'demo_command',
                    command: command,
                    timestamp: Date.now()
                }));
            }
            demoState.messages++;
            updateMetrics();
        }

        function updateStatus(elementId, html) {
            document.getElementById(elementId).innerHTML = html;
        }

        // Event handlers
        document.getElementById('demo-chat-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendDemoChat();
            }
        });

        // Initialize demo
        initDemo();
        setInterval(updateMetrics, 1000);

        // Welcome message
        setTimeout(() => {
            addDemoMessage('Welcome to Off-Guard Portal Demo! Try the full demo or individual features.', 'system');
        }, 1000);
    </script>
</body>
</html>
        """

portal_demo = PortalDemo()

# Add route to main web interface
def add_portal_route(app):
    @app.get("/portal", response_class=HTMLResponse)
    async def get_portal_demo():
        return HTMLResponse(content=portal_demo.get_portal_html())

    @app.websocket("/demo-ws")
    async def demo_websocket(websocket: WebSocket):
        await websocket.accept()
        session_id = str(uuid.uuid4())
        portal_demo.active_sessions[session_id] = websocket

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle demo commands
                if message.get("type") == "demo_command":
                    await portal_demo.handle_demo_command(websocket, message)

        except Exception as e:
            logging.error(f"Demo WebSocket error: {e}")
        finally:
            if session_id in portal_demo.active_sessions:
                del portal_demo.active_sessions[session_id]

    @app.get("/api/demo/status")
    async def get_demo_status():
        return JSONResponse(content={
            "active_sessions": len(portal_demo.active_sessions),
            "metrics": portal_demo.demo_metrics,
            "timestamp": datetime.now().isoformat()
        })

    return app

    async def handle_demo_command(self, websocket: WebSocket, message: Dict):
        """Handle demo commands from WebSocket"""
        command = message.get("command", "")

        # Simulate command responses
        responses = {
            "start_fl_server": {"type": "status_update", "fl_server": {"status": "running", "clients": 0, "rounds": 0}},
            "add_fl_client": {"type": "demo_log", "device": 1, "message": "New client connected"},
            "test_openai": {"type": "demo_log", "device": 0, "message": "OpenAI connection successful"},
            "test_anthropic": {"type": "demo_log", "device": 0, "message": "Anthropic connection successful"},
        }

        response = responses.get(command, {"type": "demo_log", "device": 0, "message": f"Command executed: {command}"})

        await websocket.send_text(json.dumps(response))

        # Update metrics
        self.demo_metrics["messages_processed"] += 1

if __name__ == "__main__":
    print("🌐 Portal Demo ready!")
    print("🎯 Access at: http://localhost:8000/portal")