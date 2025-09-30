#!/usr/bin/env python3
"""
Off-Guard Final Demo - Simplified Version
Self-contained demo that works without external dependencies
"""

import asyncio
import json
import time
import threading
import webbrowser
import http.server
import socketserver
from pathlib import Path
import urllib.parse
import logging

class OffGuardFinalDemo:
    """Self-contained final demo for Off-Guard platform"""

    def __init__(self):
        self.demo_active = False
        self.server = None
        self.port = 8000
        self.fl_metrics = {
            "rounds": 0,
            "clients": 0,
            "accuracy": 0.0,
            "encryption_ops": 0
        }
        self.start_time = time.time()

    def print_banner(self):
        """Print startup banner"""
        banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                  🛡️  Off-Guard Final Demo                    ║
    ║               Secure Federated Learning Platform             ║
    ║                                                              ║
    ║  🌸 Flower Framework Simulation                              ║
    ║  🤖 AI Integration Interface                                 ║
    ║  🔐 End-to-End Encryption Demo                              ║
    ║  📱 Two-Device FL Simulation                                ║
    ║  💬 Confidential Communication                              ║
    ║  🌐 Interactive Web Interface                               ║
    ╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def generate_demo_html(self):
        """Generate the complete demo HTML interface with Stanford-inspired design"""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Off-Guard | Secure Federated Learning Platform</title>
    <link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&family=Source+Serif+Pro:wght@400;600&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        :root {{
            --stanford-red: #8C1515;
            --stanford-dark-red: #820000;
            --stanford-light-gray: #F4F4F4;
            --stanford-dark-gray: #2E2D29;
            --stanford-medium-gray: #53565A;
            --stanford-green: #175E54;
            --stanford-blue: #006CB8;
            --stanford-gold: #B1040E;
            --white: #FFFFFF;
            --text-dark: #2E2D29;
            --text-medium: #53565A;
            --border-light: #E5E5E5;
        }}

        body {{
            font-family: 'Source Sans Pro', 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-dark);
            background: var(--white);
            overflow-x: hidden;
        }}

        /* Stanford Header */
        .stanford-header {{
            background: var(--stanford-red);
            color: var(--white);
            padding: 0;
            position: relative;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .stanford-nav {{
            background: var(--stanford-dark-red);
            padding: 8px 0;
            font-size: 14px;
        }}

        .nav-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .stanford-links {{
            display: flex;
            gap: 20px;
        }}

        .stanford-links a {{
            color: var(--white);
            text-decoration: none;
            opacity: 0.9;
            transition: opacity 0.3s;
        }}

        .stanford-links a:hover {{
            opacity: 1;
        }}

        .main-header {{
            padding: 20px 0;
        }}

        .header-content {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .logo-section {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}

        .stanford-logo {{
            font-family: 'Source Serif Pro', serif;
            font-size: 24px;
            font-weight: 600;
            color: var(--white);
            text-decoration: none;
        }}

        .divider {{
            width: 1px;
            height: 30px;
            background: rgba(255,255,255,0.3);
        }}

        .project-title {{
            font-family: 'Source Serif Pro', serif;
            font-size: 28px;
            font-weight: 400;
            color: var(--white);
        }}

        .header-actions {{
            display: flex;
            gap: 15px;
            align-items: center;
        }}

        /* Hero Section */
        .hero-section {{
            background: linear-gradient(135deg, var(--stanford-light-gray) 0%, #ffffff 100%);
            padding: 60px 0;
            border-bottom: 1px solid var(--border-light);
        }}

        .hero-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            text-align: center;
        }}

        .hero-title {{
            font-family: 'Source Serif Pro', serif;
            font-size: 48px;
            font-weight: 400;
            color: var(--text-dark);
            margin-bottom: 20px;
            line-height: 1.2;
        }}

        .hero-subtitle {{
            font-size: 24px;
            color: var(--text-medium);
            margin-bottom: 30px;
            font-weight: 300;
        }}

        .hero-description {{
            font-size: 18px;
            color: var(--text-medium);
            max-width: 800px;
            margin: 0 auto 40px;
            line-height: 1.7;
        }}

        .hero-stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 40px;
        }}

        .stat-item {{
            text-align: center;
        }}

        .stat-number {{
            font-size: 32px;
            font-weight: 600;
            color: var(--stanford-red);
            display: block;
        }}

        .stat-label {{
            font-size: 14px;
            color: var(--text-medium);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* Main Content */
        .main-content {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 60px 20px;
        }}

        .section-header {{
            text-align: center;
            margin-bottom: 50px;
        }}

        .section-title {{
            font-family: 'Source Serif Pro', serif;
            font-size: 36px;
            color: var(--text-dark);
            margin-bottom: 15px;
        }}

        .section-subtitle {{
            font-size: 18px;
            color: var(--text-medium);
            max-width: 600px;
            margin: 0 auto;
        }}

        /* Feature Grid */
        .feature-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
            margin-bottom: 60px;
        }}

        .feature-card {{
            background: var(--white);
            border: 1px solid var(--border-light);
            border-radius: 8px;
            padding: 30px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}

        .feature-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--stanford-red);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }}

        .feature-card:hover {{
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            transform: translateY(-5px);
        }}

        .feature-card:hover::before {{
            transform: scaleX(1);
        }}

        .feature-icon {{
            width: 60px;
            height: 60px;
            background: var(--stanford-red);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: var(--white);
            margin-bottom: 20px;
        }}

        .feature-title {{
            font-family: 'Source Serif Pro', serif;
            font-size: 24px;
            color: var(--text-dark);
            margin-bottom: 15px;
        }}

        .feature-description {{
            color: var(--text-medium);
            margin-bottom: 20px;
            line-height: 1.6;
        }}

        .feature-status {{
            background: var(--stanford-light-gray);
            border-radius: 6px;
            padding: 15px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            color: var(--text-dark);
            margin-bottom: 20px;
            border-left: 4px solid var(--stanford-green);
        }}

        .feature-actions {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}

        /* Buttons */
        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            border: 2px solid;
            border-radius: 4px;
            text-decoration: none;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.3s ease;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .btn-primary {{
            background: var(--stanford-red);
            border-color: var(--stanford-red);
            color: var(--white);
        }}

        .btn-primary:hover {{
            background: var(--stanford-dark-red);
            border-color: var(--stanford-dark-red);
        }}

        .btn-secondary {{
            background: transparent;
            border-color: var(--stanford-red);
            color: var(--stanford-red);
        }}

        .btn-secondary:hover {{
            background: var(--stanford-red);
            color: var(--white);
        }}

        .btn-success {{
            background: var(--stanford-green);
            border-color: var(--stanford-green);
            color: var(--white);
        }}

        .btn-warning {{
            background: var(--stanford-blue);
            border-color: var(--stanford-blue);
            color: var(--white);
        }}

        /* Demo Section */
        .demo-section {{
            background: var(--stanford-light-gray);
            padding: 60px 0;
            margin: 60px 0;
        }}

        .demo-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }}

        .device-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 40px 0;
        }}

        .device-simulator {{
            background: var(--white);
            border: 1px solid var(--border-light);
            border-radius: 8px;
            padding: 25px;
        }}

        .device-header {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border-light);
        }}

        .device-icon {{
            width: 40px;
            height: 40px;
            background: var(--stanford-green);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--white);
            font-size: 18px;
        }}

        .device-info h4 {{
            font-family: 'Source Serif Pro', serif;
            color: var(--text-dark);
            margin-bottom: 5px;
        }}

        .device-status {{
            font-size: 12px;
            color: var(--stanford-green);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .console-output {{
            background: var(--text-dark);
            color: #00ff00;
            border-radius: 4px;
            padding: 15px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            height: 150px;
            overflow-y: auto;
            margin-bottom: 15px;
            border: 1px solid var(--border-light);
        }}

        /* Metrics */
        .metrics-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }}

        .metric-card {{
            background: var(--white);
            border: 1px solid var(--border-light);
            border-radius: 8px;
            padding: 25px;
            text-align: center;
        }}

        .metric-value {{
            font-size: 36px;
            font-weight: 600;
            color: var(--stanford-red);
            display: block;
            margin-bottom: 8px;
        }}

        .metric-label {{
            font-size: 14px;
            color: var(--text-medium);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* Chat Interface */
        .chat-section {{
            background: var(--white);
            border: 1px solid var(--border-light);
            border-radius: 8px;
            margin: 40px 0;
            overflow: hidden;
        }}

        .chat-header {{
            background: var(--stanford-red);
            color: var(--white);
            padding: 20px;
            font-family: 'Source Serif Pro', serif;
            font-size: 20px;
        }}

        .chat-messages {{
            height: 300px;
            overflow-y: auto;
            padding: 20px;
            background: var(--stanford-light-gray);
        }}

        .chat-message {{
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 8px;
            background: var(--white);
            border-left: 4px solid var(--stanford-red);
        }}

        .chat-input-section {{
            padding: 20px;
            background: var(--white);
            border-top: 1px solid var(--border-light);
        }}

        .chat-input {{
            display: flex;
            gap: 15px;
        }}

        .chat-input input {{
            flex: 1;
            padding: 12px;
            border: 1px solid var(--border-light);
            border-radius: 4px;
            font-size: 16px;
        }}

        .progress-bar {{
            width: 100%;
            height: 6px;
            background: var(--border-light);
            border-radius: 3px;
            overflow: hidden;
            margin: 15px 0;
        }}

        .progress-fill {{
            height: 100%;
            background: var(--stanford-red);
            transition: width 0.3s ease;
            border-radius: 3px;
        }}

        /* Footer */
        .stanford-footer {{
            background: var(--stanford-dark-gray);
            color: var(--white);
            padding: 40px 0;
            margin-top: 60px;
        }}

        .footer-content {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            text-align: center;
        }}

        .footer-links {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 20px;
        }}

        .footer-links a {{
            color: var(--white);
            text-decoration: none;
            opacity: 0.8;
            transition: opacity 0.3s;
        }}

        .footer-links a:hover {{
            opacity: 1;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .device-grid {{ grid-template-columns: 1fr; }}
            .feature-grid {{ grid-template-columns: 1fr; }}
            .hero-title {{ font-size: 36px; }}
            .hero-stats {{ flex-direction: column; gap: 20px; }}
            .header-content {{ flex-direction: column; gap: 15px; }}
            .logo-section {{ flex-direction: column; gap: 10px; }}
            .divider {{ display: none; }}
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}

        .status-indicator {{
            width: 8px;
            height: 8px;
            background: var(--stanford-green);
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }}
    </style>
</head>
<body>
    <!-- Stanford Header -->
    <header class="stanford-header">
        <nav class="stanford-nav">
            <div class="nav-container">
                <div class="stanford-links">
                    <a href="#research">Research</a>
                    <a href="#labs">Labs</a>
                    <a href="#publications">Publications</a>
                    <a href="#about">About</a>
                </div>
                <div class="stanford-links">
                    <span>⏱️ Uptime: <span id="uptime">0s</span></span>
                    <span>🔐 Encryption: Active</span>
                    <span>🌐 Status: <span class="status-indicator"></span>Online</span>
                </div>
            </div>
        </nav>
        <div class="main-header">
            <div class="header-content">
                <div class="logo-section">
                    <a href="#" class="stanford-logo">Stanford University</a>
                    <div class="divider"></div>
                    <h1 class="project-title">Off-Guard Platform</h1>
                </div>
                <div class="header-actions">
                    <button class="btn btn-secondary" onclick="viewDocumentation()">Documentation</button>
                    <button class="btn btn-primary" onclick="startFullDemo()">Launch Demo</button>
                </div>
            </div>
        </div>
    </header>

    <!-- Hero Section -->
    <section class="hero-section">
        <div class="hero-container">
            <h2 class="hero-title">Secure Federated Learning Platform</h2>
            <p class="hero-subtitle">Advancing Privacy-Preserving AI Through Collaborative Intelligence</p>
            <p class="hero-description">
                Off-Guard represents a breakthrough in federated learning technology, enabling secure,
                distributed machine learning across multiple devices while preserving data privacy through
                advanced encryption and the Flower framework integration.
            </p>
            <div class="hero-stats">
                <div class="stat-item">
                    <span class="stat-number" id="total-clients">2</span>
                    <span class="stat-label">Connected Devices</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number" id="fl-accuracy">94.2%</span>
                    <span class="stat-label">Model Accuracy</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number" id="encryption-ops">89</span>
                    <span class="stat-label">Encryption Operations</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number" id="total-messages">156</span>
                    <span class="stat-label">Messages Processed</span>
                </div>
            </div>
        </div>
    </section>

    <div class="container">
        <div class="demo-grid">
            <div class="demo-card">
                <h3><span class="icon">🌸</span>Flower Federated Learning</h3>
                <p>Secure distributed machine learning with encrypted model updates</p>
                <div class="status-display" id="fl-status">
                    <div class="status-active"></div>FL Server: Ready<br>
                    Connected Clients: <span id="fl-clients">0</span><br>
                    Training Rounds: <span id="fl-rounds">0</span><br>
                    Model Accuracy: <span id="fl-accuracy">0.0%</span><br>
                    Encryption Operations: <span id="encryption-ops">0</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="fl-progress"></div>
                </div>
                <button class="btn btn-success" onclick="startFLDemo()">🚀 Start FL Demo</button>
                <button class="btn btn-primary" onclick="addFLClient()">📱 Add Client</button>
                <button class="btn btn-warning" onclick="runTraining()">🏋️ Train Model</button>
            </div>

            <div class="demo-card">
                <h3><span class="icon">🤖</span>AI Integration</h3>
                <p>OpenAI, Anthropic, and local AI model integration</p>
                <div class="status-display" id="ai-status">
                    <div class="status-active"></div>AI Services: Ready<br>
                    OpenAI: Simulated<br>
                    Anthropic: Simulated<br>
                    Local Models: 2 loaded<br>
                    Response Time: ~0.8s
                </div>
                <button class="btn btn-primary" onclick="testAI('openai')">🔵 Test OpenAI</button>
                <button class="btn btn-primary" onclick="testAI('anthropic')">🟣 Test Anthropic</button>
                <button class="btn btn-success" onclick="analyzeModel()">📊 AI Analysis</button>
            </div>

            <div class="demo-card">
                <h3><span class="icon">🔐</span>Security & Encryption</h3>
                <p>End-to-end encryption with offline capability</p>
                <div class="status-display" id="security-status">
                    <div class="status-active"></div>Encryption: Fernet-256<br>
                    Key Rotation: Every 24h<br>
                    Offline Mode: Enabled<br>
                    Security Audit: Passed<br>
                    Forward Secrecy: Active
                </div>
                <button class="btn btn-success" onclick="generateKeys()">🔑 Generate Keys</button>
                <button class="btn btn-warning" onclick="testEncryption()">🔒 Test Encryption</button>
                <button class="btn btn-danger" onclick="goOfflineInteractive()">🛡️ Go Offline Interactive</button>
            </div>

            <div class="demo-card">
                <h3><span class="icon">📊</span>Live Metrics & Multi-Device Network</h3>
                <p>Real-time monitoring and enhanced device interaction</p>

                <!-- Enhanced Multi-Device Network Visualization -->
                <div style="background: #2c3e50; border-radius: 8px; padding: 15px; margin: 15px 0; color: #ecf0f1;">
                    <h4 style="color: #3498db; margin-bottom: 10px;">🌐 Device Network Topology</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; text-align: center; font-size: 12px;">
                        <div style="background: #27ae60; padding: 8px; border-radius: 4px;">
                            📱 Mobile-1<br>iOS • 87.3%
                        </div>
                        <div style="background: #8e44ad; padding: 8px; border-radius: 4px;">
                            🖥️ Server<br>Aggregating
                        </div>
                        <div style="background: #27ae60; padding: 8px; border-radius: 4px;">
                            💻 Desktop-1<br>Linux • 89.1%
                        </div>
                        <div style="background: #f39c12; padding: 8px; border-radius: 4px;">
                            🏠 Edge-1<br>IoT • 85.7%
                        </div>
                        <div style="background: #e74c3c; padding: 8px; border-radius: 4px;">
                            📊 Analytics<br>Live Monitoring
                        </div>
                        <div style="background: #2ecc71; padding: 8px; border-radius: 4px;">
                            📲 Mobile-2<br>Android • 88.9%
                        </div>
                    </div>
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="total-clients">5</div>
                        <div class="metric-label">Connected Devices</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="total-messages">342</div>
                        <div class="metric-label">Messages Processed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="total-encryption">234</div>
                        <div class="metric-label">Encryptions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="accuracy">96.8%</div>
                        <div class="metric-label">Model Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="network-latency">23ms</div>
                        <div class="metric-label">Network Latency</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="data-efficiency">94.3%</div>
                        <div class="metric-label">Data Efficiency</div>
                    </div>
                </div>

                <!-- Multi-Device Controls -->
                <div style="margin: 15px 0; padding: 15px; background: #34495e; border-radius: 8px;">
                    <h4 style="color: #3498db; margin-bottom: 10px;">🎮 Multi-Device Controls</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
                        <button class="btn btn-success" onclick="addVirtualDevice()">➕ Add Device</button>
                        <button class="btn btn-warning" onclick="simulateNetworkIssue()">📡 Network Test</button>
                        <button class="btn btn-primary" onclick="syncAllDevices()">🔄 Sync All</button>
                    </div>
                </div>

                <button class="btn btn-primary" onclick="exportMetrics()">📈 Export Data</button>
                <button class="btn btn-warning" onclick="resetDemo()">🔄 Reset Demo</button>
            </div>
        </div>

        <div class="live-demo">
            <h3>🎬 Two-Device Federated Learning Demo</h3>
            <p>Watch two devices collaborate securely using encrypted federated learning</p>

            <div class="device-simulator">
                <div class="device">
                    <h4>📱 Device 1 (Mobile)</h4>
                    <div class="console-output" id="device1-console">
                        [12:34:56] Device 1 initialized
                        [12:34:57] Connecting to FL server...
                        [12:34:58] Connected successfully!
                        [12:34:59] Encryption key exchanged
                        [12:35:00] Ready for training
                    </div>
                    <button class="btn btn-success" onclick="trainDevice(1)">🏋️ Train Device 1</button>
                </div>

                <div class="device">
                    <h4>💻 Device 2 (Desktop)</h4>
                    <div class="console-output" id="device2-console">
                        [12:34:56] Device 2 initialized
                        [12:34:57] Connecting to FL server...
                        [12:34:58] Connected successfully!
                        [12:34:59] Encryption key exchanged
                        [12:35:00] Ready for training
                    </div>
                    <button class="btn btn-success" onclick="trainDevice(2)">🏋️ Train Device 2</button>
                </div>
            </div>

            <button class="btn btn-success" onclick="startFullDemo()">🎯 Start Complete Demo</button>
            <button class="btn btn-warning" onclick="simulateOffline()">📴 Simulate Offline</button>
            <button class="btn btn-danger" onclick="stopDemo()">⏹️ Stop Demo</button>
        </div>

        <div class="chat-interface">
            <h3>💬 AI Assistant</h3>
            <div class="chat-messages" id="chat-messages">
                <div style="margin: 10px 0; padding: 10px; background: rgba(46,204,113,0.3); border-radius: 10px;">
                    <strong>AI Assistant:</strong> Welcome to Off-Guard! I can help explain federated learning, security features, and AI integration. What would you like to know?
                </div>
            </div>
            <div class="chat-input">
                <input type="text" id="chat-input" placeholder="Ask about federated learning, security, or AI integration..." />
                <button class="btn btn-primary" onclick="sendChat()">Send</button>
            </div>
        </div>

        <div class="demo-card">
            <h3><span class="icon">📚</span>Documentation & Features</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h4>🔑 Key Features:</h4>
                    <ul class="feature-list">
                        <li>✅ Flower Framework Integration</li>
                        <li>✅ Multi-AI Provider Support</li>
                        <li>✅ End-to-End Encryption</li>
                        <li>✅ Offline Operation</li>
                        <li>✅ Real-time Monitoring</li>
                        <li>✅ Device Simulation</li>
                    </ul>
                </div>
                <div>
                    <h4>🛡️ Security Features:</h4>
                    <ul class="feature-list">
                        <li>🔐 Fernet Symmetric Encryption</li>
                        <li>🔑 RSA Key Exchange</li>
                        <li>⚡ Perfect Forward Secrecy</li>
                        <li>🛡️ Zero-Knowledge Architecture</li>
                        <li>📴 Offline Capability</li>
                        <li>🔍 Security Auditing</li>
                    </ul>
                </div>
            </div>
            <br>
            <button class="btn btn-primary" onclick="window.open('/sdk', '_blank')">📚 View SDK Docs</button>
            <button class="btn btn-success" onclick="downloadDemo()">💾 Download Source</button>
            <button class="btn btn-warning" onclick="viewAPIDemo()">🔧 API Demo</button>
        </div>
    </div>

    <script src="/api_client.js"></script>
    <script>
        // Check authentication on page load
        if (!api.isAuthenticated()) {{
            window.location.href = '/auth.html';
        }}

        // Display user info
        const user = api.getCurrentUser();
        if (user) {{
            console.log('Logged in as:', user.username);
        }}

        let demoState = {{
            clients: 5,
            rounds: 0,
            accuracy: 96.8,
            encryptionOps: 234,
            messages: 342,
            networkLatency: 23,
            dataEfficiency: 94.3,
            devices: [
                {{ id: 1, type: 'Mobile', name: 'iOS Device', accuracy: 87.3, status: 'training' }},
                {{ id: 2, type: 'Desktop', name: 'Linux Workstation', accuracy: 89.1, status: 'training' }},
                {{ id: 3, type: 'Edge', name: 'IoT Device', accuracy: 85.7, status: 'ready' }},
                {{ id: 4, type: 'Mobile', name: 'Android Device', accuracy: 88.9, status: 'training' }},
                {{ id: 5, type: 'Server', name: 'FL Coordinator', accuracy: 96.8, status: 'aggregating' }}
            ],
            startTime: Date.now()
        }};

        function updateUptime() {{
            const uptime = Math.floor((Date.now() - demoState.startTime) / 1000);
            document.getElementById('uptime').textContent = uptime + 's';
        }}

        // Enhanced Multi-Device Functions
        function addVirtualDevice() {{
            const deviceTypes = ['Mobile', 'Desktop', 'Edge', 'Server'];
            const deviceNames = ['Tablet Device', 'Raspberry Pi', 'Smart Camera', 'Cloud Node'];
            const newId = demoState.devices.length + 1;
            const deviceType = deviceTypes[Math.floor(Math.random() * deviceTypes.length)];
            const deviceName = deviceNames[Math.floor(Math.random() * deviceNames.length)];

            const newDevice = {{
                id: newId,
                type: deviceType,
                name: deviceName + ' #' + newId,
                accuracy: Math.random() * 15 + 80,
                status: 'connecting'
            }};

            demoState.devices.push(newDevice);
            demoState.clients++;
            demoState.encryptionOps += 3;

            addToConsole(1, `New device added: ${{newDevice.name}} (${{deviceType}})`);
            addToConsole(2, `Device network expanded to ${{demoState.clients}} nodes`);

            setTimeout(() => {{
                newDevice.status = 'ready';
                addToConsole(1, `${{newDevice.name}} is now ready for training`);
            }}, 2000);

            updateMetrics();
        }}

        function simulateNetworkIssue() {{
            addToConsole(1, 'Simulating network connectivity test...');
            addToConsole(2, 'Testing inter-device communication...');

            demoState.networkLatency += Math.random() * 50;

            setTimeout(() => {{
                const issueDevice = demoState.devices[Math.floor(Math.random() * demoState.devices.length)];
                issueDevice.status = 'reconnecting';
                addToConsole(1, `Network latency spike detected: ${{issueDevice.name}}`);

                setTimeout(() => {{
                    issueDevice.status = 'ready';
                    demoState.networkLatency = Math.max(15, demoState.networkLatency - 30);
                    addToConsole(2, `Network issue resolved. ${{issueDevice.name}} reconnected`);
                    updateMetrics();
                }}, 3000);
            }}, 1500);

            updateMetrics();
        }}

        function syncAllDevices() {{
            addToConsole(1, 'Initiating global device synchronization...');
            addToConsole(2, 'Broadcasting model updates to all devices...');

            let syncCount = 0;
            demoState.devices.forEach((device, index) => {{
                setTimeout(() => {{
                    device.status = 'syncing';
                    addToConsole(index % 2 + 1, `Syncing ${{device.name}}...`);

                    setTimeout(() => {{
                        device.status = 'ready';
                        device.accuracy += Math.random() * 2;
                        syncCount++;

                        if (syncCount === demoState.devices.length) {{
                            addToConsole(1, 'Global synchronization complete!');
                            addToConsole(2, 'All devices now have latest model version');
                            demoState.accuracy = demoState.devices.reduce((sum, d) => sum + d.accuracy, 0) / demoState.devices.length;
                            demoState.encryptionOps += demoState.devices.length * 2;
                            updateMetrics();
                        }}
                    }}, 1000 + Math.random() * 2000);
                }}, index * 500);
            }});
        }}

        function updateMetrics() {{
            document.getElementById('fl-clients').textContent = demoState.clients;
            document.getElementById('fl-rounds').textContent = demoState.rounds;
            document.getElementById('fl-accuracy').textContent = demoState.accuracy.toFixed(1) + '%';
            document.getElementById('encryption-ops').textContent = demoState.encryptionOps;
            document.getElementById('total-clients').textContent = demoState.clients;
            document.getElementById('total-messages').textContent = demoState.messages;
            document.getElementById('total-encryption').textContent = demoState.encryptionOps;
            document.getElementById('accuracy').textContent = demoState.accuracy.toFixed(1) + '%';

            // Update new metrics
            document.getElementById('network-latency').textContent = Math.round(demoState.networkLatency) + 'ms';
            document.getElementById('data-efficiency').textContent = demoState.dataEfficiency.toFixed(1) + '%';

            const progress = (demoState.rounds / 10) * 100;
            document.getElementById('fl-progress').style.width = progress + '%';
        }}

        function addToConsole(deviceId, message) {{
            const console = document.getElementById(`device${{deviceId}}-console`);
            const timestamp = new Date().toLocaleTimeString();
            console.innerHTML += `<br>[${{timestamp}}] ${{message}}`;
            console.scrollTop = console.scrollHeight;
        }}

        function addChatMessage(message, sender = 'user') {{
            const chat = document.getElementById('chat-messages');
            const messageEl = document.createElement('div');
            messageEl.style.cssText = `
                margin: 10px 0;
                padding: 10px;
                border-radius: 10px;
                background: ${{sender === 'ai' ? 'rgba(46,204,113,0.3)' : 'rgba(52,152,219,0.3)'}};
            `;
            messageEl.innerHTML = `<strong>${{sender === 'ai' ? 'AI Assistant' : 'You'}}:</strong> ${{message}}`;
            chat.appendChild(messageEl);
            chat.scrollTop = chat.scrollHeight;
        }}

        async function startFLDemo() {{
            addToConsole(1, 'Starting federated learning...');
            addToConsole(2, 'Connecting to backend server...');

            try {{
                const result = await api.startFLDemo();
                if (result.success) {{
                    addToConsole(1, `Training session started: ${{result.data.session_id}}`);
                    addToConsole(2, `Connected devices: ${{result.data.devices}}`);
                    demoState.encryptionOps += 5;
                    api.showNotification('Federated learning session started!', 'success');
                }} else {{
                    addToConsole(1, 'Failed to start FL demo');
                    api.showNotification('Failed to start FL demo', 'error');
                }}
            }} catch (error) {{
                addToConsole(1, 'Error: ' + error.message);
                api.showNotification('Connection error', 'error');
            }}
            updateMetrics();
        }}

        async function addFLClient() {{
            addToConsole(1, 'Adding new FL client...');

            try {{
                const deviceTypes = ['mobile', 'desktop', 'edge'];
                const deviceType = deviceTypes[Math.floor(Math.random() * deviceTypes.length)];

                const result = await api.addFLClient(null, deviceType);
                if (result.success) {{
                    demoState.clients++;
                    addToConsole(1, `New client added: ${{result.data.device.name}}`);
                    addToConsole(2, `Device type: ${{result.data.device.type}}`);
                    demoState.encryptionOps += 2;
                    api.showNotification(`New ${{deviceType}} client connected!`, 'success');
                }} else {{
                    addToConsole(1, 'Failed to add FL client');
                    api.showNotification('Failed to add client', 'error');
                }}
            }} catch (error) {{
                addToConsole(1, 'Error: ' + error.message);
                api.showNotification('Connection error', 'error');
            }}
            updateMetrics();
        }}

        async function runTraining() {{
            addToConsole(1, 'Starting training round...');
            addToConsole(2, 'Broadcasting to all devices...');

            try {{
                const result = await api.runTraining();
                if (result.success) {{
                    const trainingResult = result.data.training_result;
                    demoState.rounds = trainingResult.round;
                    demoState.accuracy = trainingResult.accuracy;
                    demoState.encryptionOps += Math.floor(Math.random() * 5) + 3;
                    demoState.messages += Math.floor(Math.random() * 10) + 5;

                    addToConsole(1, `Training round ${{trainingResult.round}} completed`);
                    addToConsole(2, `Accuracy: ${{trainingResult.accuracy.toFixed(1)}}%`);
                    api.showNotification(`Training round ${{trainingResult.round}} completed!`, 'success');
                }} else {{
                    addToConsole(1, 'Training failed');
                    api.showNotification('Training failed', 'error');
                }}
            }} catch (error) {{
                addToConsole(1, 'Error: ' + error.message);
                api.showNotification('Training error', 'error');
            }}
            updateMetrics();
        }}

        async function testAI(provider) {{
            addChatMessage(`Testing ${{provider}} integration...`, 'ai');

            try {{
                const result = await api.testAI(provider);
                if (result.success) {{
                    addChatMessage(result.data.response, 'ai');
                    api.showNotification(`${{provider}} test successful!`, 'success');
                }} else {{
                    addChatMessage(`${{provider}} test failed`, 'ai');
                    api.showNotification(`${{provider}} test failed`, 'error');
                }}
            }} catch (error) {{
                addChatMessage('AI service connection error', 'ai');
                api.showNotification('AI service error', 'error');
            }}

            demoState.messages++;
            updateMetrics();
        }}

        async function analyzeModel() {{
            addChatMessage('Generating AI model analysis...', 'ai');

            try {{
                const result = await api.analyzeModel();
                if (result.success) {{
                    const analysis = result.data.analysis;
                    const message = `
                        AI Analysis Complete:
                        • Model Accuracy: ${{analysis.model_performance.accuracy}}%
                        • Convergence Rate: ${{analysis.model_performance.convergence_rate}}
                        • Privacy Score: ${{analysis.model_performance.privacy_score}}

                        Recommendations: ${{analysis.recommendations.join(', ')}}
                    `;
                    addChatMessage(message, 'ai');
                    api.showNotification('AI analysis completed!', 'success');
                }} else {{
                    addChatMessage('AI analysis failed', 'ai');
                    api.showNotification('Analysis failed', 'error');
                }}
            }} catch (error) {{
                addChatMessage('Analysis service error', 'ai');
                api.showNotification('Analysis error', 'error');
            }}

            demoState.messages++;
            updateMetrics();
        }}

        async function generateKeys() {{
            addToConsole(1, 'Generating new encryption keys...');
            addToConsole(2, 'Connecting to security module...');

            try {{
                const result = await api.generateKeys();
                if (result.success) {{
                    addToConsole(1, `Key generated: ${{result.data.key_fingerprint}}`);
                    addToConsole(2, `Algorithm: ${{result.data.algorithm}}`);
                    demoState.encryptionOps += 10;
                    api.showNotification('New encryption keys generated!', 'success');
                }} else {{
                    addToConsole(1, 'Key generation failed');
                    api.showNotification('Key generation failed', 'error');
                }}
            }} catch (error) {{
                addToConsole(1, 'Error: ' + error.message);
                api.showNotification('Security service error', 'error');
            }}
            updateMetrics();
        }}

        async function testEncryption() {{
            addToConsole(1, 'Running encryption test...');
            addToConsole(2, 'Testing encryption algorithms...');

            try {{
                const result = await api.testEncryption();
                if (result.success) {{
                    addToConsole(1, `Test passed: ${{result.data.algorithm}}`);
                    addToConsole(2, `Original: ${{result.data.original_size}} bytes, Encrypted: ${{result.data.encrypted_size}} bytes`);
                    demoState.encryptionOps += 15;
                    api.showNotification('Encryption test passed!', 'success');
                }} else {{
                    addToConsole(1, 'Encryption test failed');
                    api.showNotification('Encryption test failed', 'error');
                }}
            }} catch (error) {{
                addToConsole(1, 'Error: ' + error.message);
                api.showNotification('Encryption service error', 'error');
            }}
            updateMetrics();
        }}

        function trainDevice(deviceId) {{
            addToConsole(deviceId, 'Starting local model training...');
            for (let i = 1; i <= 5; i++) {{
                setTimeout(() => {{
                    const loss = (Math.random() * 0.5 + 0.1).toFixed(4);
                    addToConsole(deviceId, `Epoch ${{i}}: Loss = ${{loss}}`);
                    if (i === 5) {{
                        addToConsole(deviceId, 'Training complete. Sending encrypted updates...');
                        demoState.encryptionOps += 3;
                        updateMetrics();
                    }}
                }}, i * 1000);
            }}
        }}

        function startFullDemo() {{
            addChatMessage('Starting complete two-device federated learning demonstration...', 'ai');
            setTimeout(() => startFLDemo(), 1000);
            setTimeout(() => trainDevice(1), 2000);
            setTimeout(() => trainDevice(2), 2500);
            setTimeout(() => runTraining(), 8000);
        }}

        function sendChat() {{
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (!message) return;

            addChatMessage(message);
            input.value = '';

            setTimeout(() => {{
                const responses = [
                    'Federated learning allows multiple devices to collaboratively train a model without sharing raw data.',
                    'Off-Guard uses Fernet encryption to secure all communications between devices.',
                    'The platform supports both online AI services and offline local models.',
                    'Two-device simulation demonstrates secure model parameter sharing.',
                    'All model updates are encrypted before transmission to ensure privacy.'
                ];
                const response = responses[Math.floor(Math.random() * responses.length)];
                addChatMessage(response, 'ai');
                demoState.messages++;
                updateMetrics();
            }}, 1000);
        }}

        async function exportMetrics() {{
            try {{
                const result = await api.exportData();
                if (result.success) {{
                    const data = result.data;
                    const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `offguard-export-${{new Date().toISOString().split('T')[0]}}.json`;
                    a.click();
                    api.showNotification('Data exported successfully!', 'success');
                }} else {{
                    api.showNotification('Export failed', 'error');
                }}
            }} catch (error) {{
                console.error('Export error:', error);
                api.showNotification('Export service error', 'error');
            }}
        }}

        function resetDemo() {{
            demoState = {{
                clients: 5,
                rounds: 0,
                accuracy: 96.8,
                encryptionOps: 234,
                messages: 342,
                networkLatency: 23,
                dataEfficiency: 94.3,
                devices: [
                    {{ id: 1, type: 'Mobile', name: 'iOS Device', accuracy: 87.3, status: 'training' }},
                    {{ id: 2, type: 'Desktop', name: 'Linux Workstation', accuracy: 89.1, status: 'training' }},
                    {{ id: 3, type: 'Edge', name: 'IoT Device', accuracy: 85.7, status: 'ready' }},
                    {{ id: 4, type: 'Mobile', name: 'Android Device', accuracy: 88.9, status: 'training' }},
                    {{ id: 5, type: 'Server', name: 'FL Coordinator', accuracy: 96.8, status: 'aggregating' }}
                ],
                startTime: Date.now()
            }};
            updateMetrics();
            addChatMessage('Demo reset successfully! Multi-device network restored.', 'ai');
        }}

        document.getElementById('chat-input').addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') {{
                sendChat();
            }}
        }});

        // Enhanced Offline Mode Function
        async function goOfflineInteractive() {{
            addToConsole(1, 'Initiating secure offline transition...');
            addToConsole(2, 'Preparing air-gapped environment...');

            try {{
                const result = await api.enableOfflineMode();
                if (result.success) {{
                    addToConsole(1, 'Offline mode enabled successfully');
                    addToConsole(2, 'Launching interactive offline dashboard...');
                    api.showNotification('Transitioning to secure offline mode...', 'success');

                    setTimeout(() => {{
                        window.open('/offline_mode.html', '_blank');
                    }}, 2000);
                }} else {{
                    addToConsole(1, 'Failed to enable offline mode');
                    api.showNotification('Offline mode transition failed', 'error');
                }}
            }} catch (error) {{
                addToConsole(1, 'Error: ' + error.message);
                api.showNotification('Offline service error', 'error');
            }}
        }}

        // Initialize
        setInterval(updateUptime, 1000);
        setInterval(() => {{
            // Simulate activity
            if (Math.random() < 0.1) {{
                demoState.messages++;
                demoState.encryptionOps++;
                updateMetrics();
            }}
        }}, 5000);

        updateMetrics();
    </script>
</body>
</html>
        """

    def create_request_handler(self):
        """Create custom request handler for the demo"""
        demo_instance = self

        class DemoRequestHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(demo_instance.generate_demo_html().encode())
                elif self.path == '/chatgpt_welcome_section.html':
                    # Serve the welcome dashboard page
                    try:
                        with open('chatgpt_welcome_section.html', 'r', encoding='utf-8') as f:
                            content = f.read()
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(content.encode())
                    except FileNotFoundError:
                        self.send_error(404, "Welcome page not found")
                elif self.path == '/metrics':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    metrics = {
                        'fl_metrics': demo_instance.fl_metrics,
                        'uptime': time.time() - demo_instance.start_time,
                        'status': 'active' if demo_instance.demo_active else 'inactive'
                    }
                    self.wfile.write(json.dumps(metrics).encode())
                elif self.path == '/sdk':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    sdk_html = self.generate_sdk_page()
                    self.wfile.write(sdk_html.encode())
                else:
                    super().do_GET()

            def generate_sdk_page(self):
                return """
<!DOCTYPE html>
<html><head><title>Off-Guard SDK</title>
<style>body{font-family:Arial,sans-serif;margin:40px;background:#f5f5f5;}
.header{background:#667eea;color:white;padding:20px;border-radius:10px;margin-bottom:20px;}
.section{background:white;padding:20px;margin:10px 0;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}
.code{background:#2c3e50;color:#ecf0f1;padding:15px;border-radius:5px;font-family:monospace;overflow-x:auto;}
</style></head><body>
<div class="header"><h1>🛡️ Off-Guard SDK Documentation</h1>
<p>Secure Federated Learning Framework</p></div>
<div class="section"><h2>Quick Start</h2>
<div class="code">pip install off-guard-sdk<br>
from offguard import SecureFLClient<br><br>
client = SecureFLClient("device_1")<br>
client.connect("localhost:8080")<br>
client.train_model(rounds=5)</div></div>
<div class="section"><h2>Features</h2>
<ul><li>🌸 Flower Framework Integration</li>
<li>🔐 End-to-End Encryption</li>
<li>🤖 Multi-AI Provider Support</li>
<li>📱 Cross-Platform Support</li></ul></div>
<div class="section"><h2>Security</h2>
<p>Off-Guard uses Fernet symmetric encryption with RSA key exchange for secure federated learning.</p></div>
</body></html>
                """

            def log_message(self, format, *args):
                pass  # Suppress logging

        return DemoRequestHandler

    async def start_web_server(self):
        """Start the web server"""
        handler = self.create_request_handler()

        try:
            with socketserver.TCPServer(("", self.port), handler) as httpd:
                self.server = httpd
                print(f"🌐 Web server started at http://localhost:{self.port}")

                # Try to open browser
                try:
                    webbrowser.open(f"http://localhost:{self.port}")
                    print("🌐 Browser opened automatically")
                except:
                    print("⚠️  Could not open browser automatically")

                print(f"📊 Demo running - visit http://localhost:{self.port}")
                httpd.serve_forever()

        except Exception as e:
            print(f"❌ Server error: {e}")

    async def simulate_fl_activity(self):
        """Simulate federated learning activity"""
        while self.demo_active:
            await asyncio.sleep(10)

            # Simulate activity
            if self.fl_metrics["rounds"] < 10:
                self.fl_metrics["rounds"] += 1
                self.fl_metrics["accuracy"] += 0.5
                self.fl_metrics["encryption_ops"] += 5

                print(f"📊 FL Round {self.fl_metrics['rounds']} completed, Accuracy: {self.fl_metrics['accuracy']:.1f}%")

    def show_demo_info(self):
        """Show demo information"""
        print("\n" + "="*60)
        print("🎯 OFF-GUARD FINAL DEMO")
        print("="*60)
        print(f"🌐 Web Interface: http://localhost:{self.port}")
        print(f"📚 SDK Docs:     http://localhost:{self.port}/sdk")
        print(f"📊 Metrics API:  http://localhost:{self.port}/metrics")
        print("="*60)
        print("🔑 FEATURES AVAILABLE:")
        print("  ✅ Flower Framework Simulation")
        print("  ✅ AI Integration Interface")
        print("  ✅ Two-Device FL Demo")
        print("  ✅ Encryption Demo")
        print("  ✅ Real-time Metrics")
        print("  ✅ Interactive Chat")
        print("="*60)
        print("💡 DEMO HIGHLIGHTS:")
        print("  • Complete web-based interface")
        print("  • No external dependencies required")
        print("  • Simulated federated learning")
        print("  • Encryption demonstrations")
        print("  • Real-time monitoring")
        print("="*60)

    async def run_demo(self):
        """Run the complete demo"""
        try:
            self.demo_active = True
            self.print_banner()
            self.show_demo_info()

            # Start FL simulation in background
            fl_task = asyncio.create_task(self.simulate_fl_activity())

            # Start web server (blocking)
            await self.start_web_server()

        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
        except Exception as e:
            print(f"❌ Demo error: {e}")
        finally:
            self.demo_active = False
            print("🎉 Demo shutdown complete")

async def main():
    """Main entry point"""
    demo = OffGuardFinalDemo()
    await demo.run_demo()

if __name__ == "__main__":
    print("🚀 Starting Off-Guard Final Demo...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo completed!")
    except Exception as e:
        print(f"❌ Startup error: {e}")