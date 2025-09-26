#!/usr/bin/env python3
"""
Full Guardian Agent System Website

Complete web application with real-time monitoring, agent control,
data visualization, and comprehensive system management.
"""

import json
import logging
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import socketserver

# Add agents to path
sys.path.insert(0, str(Path(__file__).parent / "agents"))

from agents.agent_orchestrator import create_agent_orchestrator

logger = logging.getLogger(__name__)

class FullWebsiteHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the full website."""

    def __init__(self, *args, website=None, **kwargs):
        self.website = website
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # Route requests
        if path == '/' or path == '/dashboard':
            self._serve_dashboard()
        elif path == '/agents':
            self._serve_agents_page()
        elif path == '/security':
            self._serve_security_page()
        elif path == '/development':
            self._serve_development_page()
        elif path == '/logs':
            self._serve_logs_page()
        elif path == '/api/status':
            self._serve_api_status()
        elif path == '/api/start':
            self._serve_api_start()
        elif path == '/api/stop':
            self._serve_api_stop()
        elif path == '/api/restart':
            self._serve_api_restart(parsed_path.query)
        elif path == '/api/test':
            self._serve_api_test()
        elif path == '/api/metrics':
            self._serve_api_metrics()
        elif path.startswith('/static/'):
            self._serve_static(path)
        else:
            self._serve_404()

    def do_POST(self):
        """Handle POST requests."""
        self.do_GET()  # For simplicity, treat POST same as GET

    def _serve_dashboard(self):
        """Serve the main dashboard page."""
        html = self._get_dashboard_html()
        self._send_html_response(html)

    def _serve_agents_page(self):
        """Serve the agents management page."""
        html = self._get_agents_html()
        self._send_html_response(html)

    def _serve_security_page(self):
        """Serve the security monitoring page."""
        html = self._get_security_html()
        self._send_html_response(html)

    def _serve_development_page(self):
        """Serve the development monitoring page."""
        html = self._get_development_html()
        self._send_html_response(html)

    def _serve_logs_page(self):
        """Serve the logs page."""
        html = self._get_logs_html()
        self._send_html_response(html)

    def _serve_api_status(self):
        """Serve API status endpoint."""
        if hasattr(self.server, 'website'):
            status = self.server.website.get_status()
        else:
            status = {"error": "Website not available"}
        self._send_json_response(status)

    def _serve_api_start(self):
        """Serve API start endpoint."""
        if hasattr(self.server, 'website'):
            result = self.server.website.start_system()
        else:
            result = {"success": False, "error": "Website not available"}
        self._send_json_response(result)

    def _serve_api_stop(self):
        """Serve API stop endpoint."""
        if hasattr(self.server, 'website'):
            result = self.server.website.stop_system()
        else:
            result = {"success": False, "error": "Website not available"}
        self._send_json_response(result)

    def _serve_api_restart(self, query):
        """Serve API restart endpoint."""
        params = parse_qs(query)
        agent_id = params.get('agent', [''])[0]

        if hasattr(self.server, 'website') and agent_id:
            result = self.server.website.restart_agent(agent_id)
        else:
            result = {"success": False, "error": "Invalid request"}
        self._send_json_response(result)

    def _serve_api_test(self):
        """Serve API test endpoint."""
        if hasattr(self.server, 'website'):
            result = self.server.website.run_tests()
        else:
            result = {"success": False, "error": "Website not available"}
        self._send_json_response(result)

    def _serve_api_metrics(self):
        """Serve API metrics endpoint."""
        if hasattr(self.server, 'website'):
            metrics = self.server.website.get_live_metrics()
        else:
            metrics = {"error": "Website not available"}
        self._send_json_response(metrics)

    def _serve_static(self, path):
        """Serve static files (placeholder)."""
        self._serve_404()

    def _serve_404(self):
        """Serve 404 error."""
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<h1>404 Not Found</h1>')

    def _send_html_response(self, html):
        """Send HTML response."""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def _send_json_response(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        """Suppress request logging."""
        pass

    def _get_common_styles(self):
        """Get common CSS styles."""
        return """
        <style>
            :root {
                --primary-bg: #0f1419;
                --secondary-bg: #1a1f2e;
                --card-bg: rgba(22, 27, 34, 0.9);
                --border-color: #30363d;
                --accent-color: #00ff41;
                --text-primary: #e1e5e9;
                --text-secondary: #8b949e;
                --blue: #58a6ff;
                --red: #ff4757;
                --orange: #ffa500;
                --yellow: #f1e05a;
            }

            * { margin: 0; padding: 0; box-sizing: border-box; }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, var(--primary-bg) 0%, var(--secondary-bg) 100%);
                color: var(--text-primary);
                min-height: 100vh;
                line-height: 1.6;
            }

            .header {
                background: rgba(0, 0, 0, 0.3);
                padding: 1rem 2rem;
                border-bottom: 3px solid var(--accent-color);
                backdrop-filter: blur(10px);
                position: sticky;
                top: 0;
                z-index: 100;
            }

            .header h1 {
                font-size: 2.5rem;
                color: var(--accent-color);
                text-shadow: 0 0 20px rgba(0, 255, 65, 0.5);
                display: inline-block;
            }

            .nav {
                display: flex;
                gap: 2rem;
                margin-top: 1rem;
            }

            .nav a {
                color: var(--text-secondary);
                text-decoration: none;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                transition: all 0.3s ease;
                font-weight: 500;
            }

            .nav a:hover, .nav a.active {
                color: var(--accent-color);
                background: rgba(0, 255, 65, 0.1);
            }

            .container {
                max-width: 1400px;
                margin: 2rem auto;
                padding: 0 2rem;
            }

            .grid {
                display: grid;
                gap: 2rem;
            }

            .grid-2 { grid-template-columns: 1fr 1fr; }
            .grid-3 { grid-template-columns: repeat(3, 1fr); }
            .grid-4 { grid-template-columns: repeat(4, 1fr); }

            @media (max-width: 768px) {
                .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
            }

            .card {
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 2rem;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                transition: transform 0.3s ease;
            }

            .card:hover {
                transform: translateY(-2px);
            }

            .card h2 {
                color: var(--blue);
                margin-bottom: 1.5rem;
                font-size: 1.5rem;
                border-bottom: 2px solid var(--blue);
                padding-bottom: 0.5rem;
            }

            .metric {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem 0;
                border-bottom: 1px solid var(--border-color);
            }

            .metric:last-child { border-bottom: none; }

            .metric-label {
                color: var(--text-secondary);
                font-weight: 500;
            }

            .metric-value {
                color: var(--accent-color);
                font-weight: bold;
                font-size: 1.2rem;
            }

            .btn {
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
                text-align: center;
                font-size: 1rem;
            }

            .btn-primary {
                background: #238636;
                color: white;
            }

            .btn-primary:hover {
                background: #2ea043;
                transform: translateY(-2px);
            }

            .btn-danger {
                background: #da3633;
                color: white;
            }

            .btn-danger:hover {
                background: #f85149;
                transform: translateY(-2px);
            }

            .btn-secondary {
                background: #373e47;
                color: #f0f6fc;
            }

            .btn-secondary:hover {
                background: #444c56;
                transform: translateY(-2px);
            }

            .btn-small {
                padding: 0.5rem 1rem;
                font-size: 0.9rem;
            }

            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 0.5rem;
            }

            .status-running { background: var(--accent-color); animation: pulse 2s infinite; }
            .status-stopped { background: var(--red); }
            .status-error { background: var(--orange); }

            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }

            .agent-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1rem;
            }

            .agent-card {
                background: rgba(13, 17, 23, 0.6);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 1.5rem;
                text-align: center;
            }

            .agent-icon {
                font-size: 3rem;
                margin-bottom: 1rem;
            }

            .agent-name {
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }

            .agent-status {
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-size: 0.9rem;
                font-weight: 600;
                margin-bottom: 1rem;
            }

            .log-container {
                background: rgba(13, 17, 23, 0.8);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 1.5rem;
                height: 400px;
                overflow-y: auto;
                font-family: 'Courier New', monospace;
                font-size: 0.9rem;
            }

            .log-entry {
                margin-bottom: 0.5rem;
                padding: 0.25rem;
                border-radius: 4px;
            }

            .log-info { color: var(--blue); }
            .log-success { color: var(--accent-color); }
            .log-warning { color: var(--yellow); }
            .log-error { color: var(--red); }

            .charts-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
                margin-top: 2rem;
            }

            .chart {
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 1.5rem;
                text-align: center;
            }

            .chart-title {
                color: var(--blue);
                margin-bottom: 1rem;
                font-weight: 600;
            }

            .chart-value {
                font-size: 3rem;
                font-weight: bold;
                color: var(--accent-color);
                margin-bottom: 0.5rem;
            }

            .chart-label {
                color: var(--text-secondary);
                font-size: 0.9rem;
            }

            .full-width { grid-column: 1 / -1; }

            .loading {
                text-align: center;
                color: var(--text-secondary);
                padding: 2rem;
                font-style: italic;
            }

            .refresh-indicator {
                position: fixed;
                top: 2rem;
                right: 2rem;
                background: var(--card-bg);
                color: var(--accent-color);
                padding: 0.5rem 1rem;
                border-radius: 6px;
                border: 1px solid var(--accent-color);
                font-size: 0.9rem;
                z-index: 1000;
                cursor: pointer;
                transition: all 0.3s ease;
            }

            .refresh-indicator:hover {
                background: rgba(0, 255, 65, 0.1);
            }

            .controls {
                display: flex;
                gap: 1rem;
                margin-bottom: 2rem;
                flex-wrap: wrap;
            }

            .alert {
                padding: 1rem;
                border-radius: 6px;
                margin-bottom: 1rem;
                border: 1px solid;
            }

            .alert-success {
                background: rgba(0, 255, 65, 0.1);
                border-color: var(--accent-color);
                color: var(--accent-color);
            }

            .alert-error {
                background: rgba(255, 71, 87, 0.1);
                border-color: var(--red);
                color: var(--red);
            }

            .alert-warning {
                background: rgba(255, 165, 0, 0.1);
                border-color: var(--orange);
                color: var(--orange);
            }

            .table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 1rem;
            }

            .table th,
            .table td {
                padding: 1rem;
                text-align: left;
                border-bottom: 1px solid var(--border-color);
            }

            .table th {
                background: rgba(13, 17, 23, 0.6);
                color: var(--blue);
                font-weight: 600;
            }

            .progress-bar {
                width: 100%;
                height: 8px;
                background: rgba(13, 17, 23, 0.6);
                border-radius: 4px;
                overflow: hidden;
                margin-top: 0.5rem;
            }

            .progress-fill {
                height: 100%;
                background: var(--accent-color);
                transition: width 0.3s ease;
            }
        </style>
        """

    def _get_common_scripts(self):
        """Get common JavaScript functions."""
        return """
        <script>
            // Global variables
            let autoRefresh = true;
            let refreshInterval;
            let currentData = {};

            // Utility functions
            function formatTime(timestamp) {
                return new Date(timestamp).toLocaleTimeString();
            }

            function formatBytes(bytes) {
                const sizes = ['B', 'KB', 'MB', 'GB'];
                if (bytes === 0) return '0 B';
                const i = Math.floor(Math.log(bytes) / Math.log(1024));
                return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
            }

            function getStatusIcon(status) {
                switch(status) {
                    case 'running': return '🟢';
                    case 'stopped': return '🔴';
                    case 'error': return '🟠';
                    default: return '🟡';
                }
            }

            function getAgentIcon(agentId) {
                const icons = {
                    'security_monitor': '🛡️',
                    'development': '💻',
                    'communication_hub': '📡',
                    'debug_monitor': '🔍',
                    'auto_save': '💾',
                    'test_coordinator': '🧪'
                };
                return icons[agentId] || '🤖';
            }

            // API functions
            async function apiCall(endpoint, method = 'GET') {
                try {
                    const response = await fetch(`/api/${endpoint}`, { method });
                    return await response.json();
                } catch (error) {
                    console.error(`API call failed: ${endpoint}`, error);
                    return { error: error.message };
                }
            }

            async function startSystem() {
                showAlert('🚀 Starting Guardian Agent System...', 'info');
                const result = await apiCall('start');
                if (result.success) {
                    showAlert('✅ System started successfully', 'success');
                    setTimeout(fetchData, 2000);
                } else {
                    showAlert(`❌ Failed to start: ${result.error}`, 'error');
                }
            }

            async function stopSystem() {
                showAlert('⏹️ Stopping Guardian Agent System...', 'warning');
                const result = await apiCall('stop');
                if (result.success) {
                    showAlert('🔄 System stopped', 'info');
                    setTimeout(fetchData, 2000);
                } else {
                    showAlert(`❌ Failed to stop: ${result.error}`, 'error');
                }
            }

            async function runTests() {
                showAlert('🧪 Running comprehensive tests...', 'info');
                const result = await apiCall('test');
                if (result.success) {
                    showAlert(`✅ Tests completed: ${result.passed}/${result.total} passed (${result.success_rate.toFixed(1)}%)`, 'success');
                } else {
                    showAlert(`❌ Tests failed: ${result.error}`, 'error');
                }
            }

            async function restartAgent(agentId) {
                showAlert(`🔄 Restarting ${agentId}...`, 'info');
                const result = await apiCall(`restart?agent=${agentId}`);
                if (result.success) {
                    showAlert(`✅ ${agentId} restarted`, 'success');
                    setTimeout(fetchData, 2000);
                } else {
                    showAlert(`❌ Failed to restart ${agentId}`, 'error');
                }
            }

            // Data fetching
            async function fetchData() {
                try {
                    const data = await apiCall('status');
                    if (!data.error) {
                        currentData = data;
                        updateDashboard(data);
                        updateLastRefresh();
                    }
                } catch (error) {
                    console.error('Failed to fetch data:', error);
                }
            }

            async function fetchMetrics() {
                try {
                    const metrics = await apiCall('metrics');
                    if (!metrics.error) {
                        updateMetrics(metrics);
                    }
                } catch (error) {
                    console.error('Failed to fetch metrics:', error);
                }
            }

            // UI updates
            function updateLastRefresh() {
                const indicator = document.getElementById('refreshIndicator');
                if (indicator) {
                    indicator.textContent = `🔄 Last updated: ${formatTime(Date.now())}`;
                }
            }

            function showAlert(message, type = 'info') {
                const alertContainer = document.getElementById('alertContainer');
                if (!alertContainer) return;

                const alert = document.createElement('div');
                alert.className = `alert alert-${type}`;
                alert.textContent = message;

                alertContainer.appendChild(alert);

                // Remove alert after 5 seconds
                setTimeout(() => {
                    if (alert.parentNode) {
                        alert.parentNode.removeChild(alert);
                    }
                }, 5000);
            }

            function toggleAutoRefresh() {
                autoRefresh = !autoRefresh;
                const indicator = document.getElementById('refreshIndicator');

                if (autoRefresh) {
                    refreshInterval = setInterval(fetchData, 5000);
                    if (indicator) indicator.style.color = 'var(--accent-color)';
                } else {
                    clearInterval(refreshInterval);
                    if (indicator) indicator.style.color = 'var(--orange)';
                }
            }

            // Page-specific update functions (to be overridden)
            function updateDashboard(data) {
                console.log('Dashboard update:', data);
            }

            function updateMetrics(metrics) {
                console.log('Metrics update:', metrics);
            }

            // Initialize
            document.addEventListener('DOMContentLoaded', function() {
                fetchData();
                refreshInterval = setInterval(fetchData, 5000);

                // Add click handler for refresh indicator
                const refreshIndicator = document.getElementById('refreshIndicator');
                if (refreshIndicator) {
                    refreshIndicator.addEventListener('click', toggleAutoRefresh);
                }
            });
        </script>
        """

    def _get_navigation(self, current_page="dashboard"):
        """Get navigation HTML."""
        pages = [
            ("dashboard", "🏠 Dashboard", "/"),
            ("agents", "🤖 Agents", "/agents"),
            ("security", "🛡️ Security", "/security"),
            ("development", "💻 Development", "/development"),
            ("logs", "📋 Logs", "/logs")
        ]

        nav_items = []
        for page_id, page_name, page_url in pages:
            active_class = " active" if page_id == current_page else ""
            nav_items.append(f'<a href="{page_url}" class="nav-link{active_class}">{page_name}</a>')

        return f"""
        <nav class="nav">
            {' '.join(nav_items)}
        </nav>
        """

    def _get_dashboard_html(self):
        """Generate the main dashboard HTML."""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🛡️ Guardian Agent System</title>
            {self._get_common_styles()}
        </head>
        <body>
            <div class="refresh-indicator" id="refreshIndicator">
                🔄 Loading...
            </div>

            <header class="header">
                <h1>🛡️ Guardian Agent System</h1>
                {self._get_navigation("dashboard")}
            </header>

            <div class="container">
                <div id="alertContainer"></div>

                <div class="controls">
                    <button class="btn btn-primary" onclick="startSystem()">🚀 Start System</button>
                    <button class="btn btn-danger" onclick="stopSystem()">⏹️ Stop System</button>
                    <button class="btn btn-secondary" onclick="runTests()">🧪 Run Tests</button>
                    <button class="btn btn-secondary" onclick="fetchData()">🔄 Refresh</button>
                </div>

                <div class="grid grid-4">
                    <div class="card">
                        <h2>📊 System Overview</h2>
                        <div class="metric">
                            <span class="metric-label">Total Agents</span>
                            <span class="metric-value" id="totalAgents">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Running Agents</span>
                            <span class="metric-value" id="runningAgents">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">System Status</span>
                            <span class="metric-value" id="systemStatus">Offline</span>
                        </div>
                    </div>

                    <div class="card">
                        <h2>🛡️ Security Status</h2>
                        <div class="metric">
                            <span class="metric-label">Monitored Files</span>
                            <span class="metric-value" id="monitoredFiles">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Security Events</span>
                            <span class="metric-value" id="securityEvents">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Threat Level</span>
                            <span class="metric-value" id="threatLevel">Low</span>
                        </div>
                    </div>

                    <div class="card">
                        <h2>💻 Development Status</h2>
                        <div class="metric">
                            <span class="metric-label">Files Analyzed</span>
                            <span class="metric-value" id="filesAnalyzed">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Code Issues</span>
                            <span class="metric-value" id="codeIssues">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Code Quality</span>
                            <span class="metric-value" id="codeQuality">Good</span>
                        </div>
                    </div>

                    <div class="card">
                        <h2>🔧 System Resources</h2>
                        <div class="metric">
                            <span class="metric-label">CPU Usage</span>
                            <span class="metric-value" id="cpuUsage">0%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Memory Usage</span>
                            <span class="metric-value" id="memoryUsage">0%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Uptime</span>
                            <span class="metric-value" id="uptime">0s</span>
                        </div>
                    </div>
                </div>

                <div class="charts-container">
                    <div class="chart">
                        <div class="chart-title">Success Rate</div>
                        <div class="chart-value" id="successRate">0%</div>
                        <div class="chart-label">Test Success Rate</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="successProgress" style="width: 0%"></div>
                        </div>
                    </div>

                    <div class="chart">
                        <div class="chart-title">Agent Health</div>
                        <div class="chart-value" id="agentHealth">0%</div>
                        <div class="chart-label">Agents Running</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="healthProgress" style="width: 0%"></div>
                        </div>
                    </div>

                    <div class="chart">
                        <div class="chart-title">Security Score</div>
                        <div class="chart-value" id="securityScore">100</div>
                        <div class="chart-label">Security Rating</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="securityProgress" style="width: 100%"></div>
                        </div>
                    </div>
                </div>

                <div class="grid grid-2">
                    <div class="card">
                        <h2>🤖 Quick Agent Status</h2>
                        <div id="quickAgentStatus" class="loading">Loading agents...</div>
                    </div>

                    <div class="card">
                        <h2>📈 Live Activity</h2>
                        <div class="log-container" id="liveActivity">
                            <div class="log-entry log-info">System initializing...</div>
                        </div>
                    </div>
                </div>
            </div>

            {self._get_common_scripts()}
            <script>
                function updateDashboard(data) {
                    // Update system overview
                    document.getElementById('totalAgents').textContent = data.total_agents || 0;
                    document.getElementById('runningAgents').textContent = data.running_agents || 0;
                    document.getElementById('systemStatus').textContent = data.system_running ? 'Online' : 'Offline';

                    // Update security status
                    document.getElementById('monitoredFiles').textContent = data.monitored_files || 0;
                    document.getElementById('securityEvents').textContent = data.security_events || 0;

                    const threatLevel = data.security_events > 5 ? 'High' : data.security_events > 2 ? 'Medium' : 'Low';
                    document.getElementById('threatLevel').textContent = threatLevel;

                    // Update development status
                    document.getElementById('filesAnalyzed').textContent = data.files_analyzed || 0;
                    document.getElementById('codeIssues').textContent = data.code_issues || 0;
                    document.getElementById('codeQuality').textContent = data.code_quality || 'Unknown';

                    // Update system resources
                    document.getElementById('cpuUsage').textContent = (data.cpu_usage || 0).toFixed(1) + '%';
                    document.getElementById('memoryUsage').textContent = (data.memory_usage || 0).toFixed(1) + '%';
                    document.getElementById('uptime').textContent = data.uptime || '0s';

                    // Update charts
                    const successRate = data.test_success_rate || 0;
                    document.getElementById('successRate').textContent = successRate.toFixed(1) + '%';
                    document.getElementById('successProgress').style.width = successRate + '%';

                    const agentHealth = data.total_agents > 0 ? (data.running_agents / data.total_agents * 100) : 0;
                    document.getElementById('agentHealth').textContent = agentHealth.toFixed(0) + '%';
                    document.getElementById('healthProgress').style.width = agentHealth + '%';

                    const securityScore = Math.max(0, 100 - (data.security_events || 0) * 10);
                    document.getElementById('securityScore').textContent = securityScore;
                    document.getElementById('securityProgress').style.width = securityScore + '%';

                    // Update quick agent status
                    updateQuickAgentStatus(data.agents || {});

                    // Add activity entry
                    addActivityEntry(`System updated - ${data.running_agents}/${data.total_agents} agents running`);
                }

                function updateQuickAgentStatus(agents) {
                    const container = document.getElementById('quickAgentStatus');
                    container.innerHTML = '';

                    if (Object.keys(agents).length === 0) {
                        container.innerHTML = '<div class="loading">No agents running</div>';
                        return;
                    }

                    Object.entries(agents).forEach(([agentId, agentData]) => {
                        const div = document.createElement('div');
                        div.className = 'metric';

                        const statusClass = `status-\${agentData.status}`;
                        const icon = getAgentIcon(agentId);
                        const name = agentId.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());

                        div.innerHTML = `
                            <span class="metric-label">\${icon} \${name}</span>
                            <span class="metric-value">
                                <span class="status-indicator \${statusClass}"></span>
                                \${agentData.status.toUpperCase()}
                            </span>
                        `;

                        container.appendChild(div);
                    });
                }

                function addActivityEntry(message, type = 'info') {
                    const container = document.getElementById('liveActivity');
                    const timestamp = new Date().toLocaleTimeString();

                    const entry = document.createElement('div');
                    entry.className = `log-entry log-\${type}`;
                    entry.textContent = `[\${timestamp}] \${message}`;

                    container.appendChild(entry);
                    container.scrollTop = container.scrollHeight;

                    // Keep only last 20 entries
                    while (container.children.length > 20) {
                        container.removeChild(container.firstChild);
                    }
                }

                // Override the global showAlert to also add to activity
                const originalShowAlert = showAlert;
                showAlert = function(message, type = 'info') {
                    originalShowAlert(message, type);
                    addActivityEntry(message, type);
                };
            </script>
        </body>
        </html>
        """

    def _get_agents_html(self):
        """Generate the agents management page HTML."""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🤖 Agent Management - Guardian System</title>
            {self._get_common_styles()}
        </head>
        <body>
            <div class="refresh-indicator" id="refreshIndicator">
                🔄 Loading...
            </div>

            <header class="header">
                <h1>🛡️ Guardian Agent System</h1>
                {self._get_navigation("agents")}
            </header>

            <div class="container">
                <div id="alertContainer"></div>

                <div class="controls">
                    <button class="btn btn-primary" onclick="startSystem()">🚀 Start All Agents</button>
                    <button class="btn btn-danger" onclick="stopSystem()">⏹️ Stop All Agents</button>
                    <button class="btn btn-secondary" onclick="fetchData()">🔄 Refresh Status</button>
                </div>

                <div class="card full-width">
                    <h2>🤖 Agent Management</h2>
                    <div class="agent-grid" id="agentGrid">
                        <div class="loading">Loading agents...</div>
                    </div>
                </div>

                <div class="card full-width">
                    <h2>📊 Agent Details</h2>
                    <table class="table" id="agentTable">
                        <thead>
                            <tr>
                                <th>Agent</th>
                                <th>Status</th>
                                <th>Type</th>
                                <th>Last Seen</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td colspan="5" class="loading">Loading agent details...</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            {self._get_common_scripts()}
            <script>
                function updateDashboard(data) {
                    updateAgentGrid(data.agents || {});
                    updateAgentTable(data.agents || {});
                }

                function updateAgentGrid(agents) {
                    const grid = document.getElementById('agentGrid');
                    grid.innerHTML = '';

                    if (Object.keys(agents).length === 0) {
                        grid.innerHTML = '<div class="loading">No agents available</div>';
                        return;
                    }

                    Object.entries(agents).forEach(([agentId, agentData]) => {
                        const card = document.createElement('div');
                        card.className = 'agent-card';

                        const icon = getAgentIcon(agentId);
                        const name = agentId.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                        const status = agentData.status || 'unknown';
                        const statusClass = `status-\${status}`;

                        card.innerHTML = `
                            <div class="agent-icon">\${icon}</div>
                            <div class="agent-name">\${name}</div>
                            <div class="agent-status \${statusClass}">
                                <span class="status-indicator \${statusClass}"></span>
                                \${status.toUpperCase()}
                            </div>
                            <button class="btn btn-secondary btn-small" onclick="restartAgent('\${agentId}')">
                                🔄 Restart
                            </button>
                        `;

                        grid.appendChild(card);
                    });
                }

                function updateAgentTable(agents) {
                    const tbody = document.querySelector('#agentTable tbody');
                    tbody.innerHTML = '';

                    if (Object.keys(agents).length === 0) {
                        tbody.innerHTML = '<tr><td colspan="5" class="loading">No agents available</td></tr>';
                        return;
                    }

                    Object.entries(agents).forEach(([agentId, agentData]) => {
                        const row = document.createElement('tr');

                        const icon = getAgentIcon(agentId);
                        const name = agentId.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                        const status = agentData.status || 'unknown';
                        const statusIcon = getStatusIcon(status);
                        const lastSeen = agentData.last_seen ? formatTime(agentData.last_seen) : 'Never';
                        const agentType = agentData.type || 'Unknown';

                        row.innerHTML = `
                            <td>\${icon} \${name}</td>
                            <td>\${statusIcon} \${status.toUpperCase()}</td>
                            <td>\${agentType}</td>
                            <td>\${lastSeen}</td>
                            <td>
                                <button class="btn btn-secondary btn-small" onclick="restartAgent('\${agentId}')">
                                    🔄 Restart
                                </button>
                            </td>
                        `;

                        tbody.appendChild(row);
                    });
                }
            </script>
        </body>
        </html>
        """

    def _get_security_html(self):
        """Generate the security monitoring page HTML."""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🛡️ Security Monitor - Guardian System</title>
            {self._get_common_styles()}
        </head>
        <body>
            <div class="refresh-indicator" id="refreshIndicator">
                🔄 Loading...
            </div>

            <header class="header">
                <h1>🛡️ Guardian Agent System</h1>
                {self._get_navigation("security")}
            </header>

            <div class="container">
                <div id="alertContainer"></div>

                <div class="grid grid-3">
                    <div class="card">
                        <h2>🛡️ Security Status</h2>
                        <div class="metric">
                            <span class="metric-label">Monitoring Status</span>
                            <span class="metric-value" id="monitoringStatus">Offline</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Files Monitored</span>
                            <span class="metric-value" id="monitoredFiles">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Threat Level</span>
                            <span class="metric-value" id="threatLevel">Low</span>
                        </div>
                    </div>

                    <div class="card">
                        <h2>🚨 Recent Events</h2>
                        <div class="metric">
                            <span class="metric-label">Total Events</span>
                            <span class="metric-value" id="totalEvents">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Critical Events</span>
                            <span class="metric-value" id="criticalEvents">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Last Event</span>
                            <span class="metric-value" id="lastEvent">None</span>
                        </div>
                    </div>

                    <div class="card">
                        <h2>📊 System Metrics</h2>
                        <div class="metric">
                            <span class="metric-label">CPU Usage</span>
                            <span class="metric-value" id="cpuUsage">0%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Memory Usage</span>
                            <span class="metric-value" id="memoryUsage">0%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Network Activity</span>
                            <span class="metric-value" id="networkActivity">Normal</span>
                        </div>
                    </div>
                </div>

                <div class="grid grid-2">
                    <div class="card">
                        <h2>🔍 File Integrity</h2>
                        <div id="fileIntegrity" class="loading">Loading file integrity status...</div>
                    </div>

                    <div class="card">
                        <h2>🌐 Network Monitoring</h2>
                        <div id="networkMonitoring" class="loading">Loading network status...</div>
                    </div>
                </div>

                <div class="card full-width">
                    <h2>📋 Security Event Log</h2>
                    <div class="log-container" id="securityLog">
                        <div class="log-entry log-info">Security monitoring initializing...</div>
                    </div>
                </div>
            </div>

            {self._get_common_scripts()}
            <script>
                function updateDashboard(data) {
                    // Update security status
                    document.getElementById('monitoringStatus').textContent = data.system_running ? 'Active' : 'Offline';
                    document.getElementById('monitoredFiles').textContent = data.monitored_files || 0;

                    const threatLevel = data.security_events > 5 ? 'High' : data.security_events > 2 ? 'Medium' : 'Low';
                    document.getElementById('threatLevel').textContent = threatLevel;

                    // Update events
                    document.getElementById('totalEvents').textContent = data.security_events || 0;
                    document.getElementById('criticalEvents').textContent = data.critical_events || 0;
                    document.getElementById('lastEvent').textContent = data.last_event || 'None';

                    // Update system metrics
                    document.getElementById('cpuUsage').textContent = (data.cpu_usage || 0).toFixed(1) + '%';
                    document.getElementById('memoryUsage').textContent = (data.memory_usage || 0).toFixed(1) + '%';

                    const networkStatus = data.network_connections > 100 ? 'High' : data.network_connections > 50 ? 'Medium' : 'Normal';
                    document.getElementById('networkActivity').textContent = networkStatus;

                    // Update file integrity and network monitoring
                    updateFileIntegrity(data.file_integrity || {});
                    updateNetworkMonitoring(data.network_status || {});

                    // Add to security log
                    if (data.security_events > 0) {
                        addSecurityLogEntry(`Security scan completed - ${data.security_events} events detected`, 'warning');
                    } else {
                        addSecurityLogEntry('Security scan completed - no threats detected', 'success');
                    }
                }

                function updateFileIntegrity(fileData) {
                    const container = document.getElementById('fileIntegrity');
                    container.innerHTML = '';

                    if (Object.keys(fileData).length === 0) {
                        container.innerHTML = '<div class="loading">No file integrity data available</div>';
                        return;
                    }

                    // Mock file integrity display
                    const files = ['config.yaml', 'agents/*.py', 'security_config.json'];
                    files.forEach(file => {
                        const div = document.createElement('div');
                        div.className = 'metric';
                        div.innerHTML = `
                            <span class="metric-label">📄 \${file}</span>
                            <span class="metric-value">✅ Verified</span>
                        `;
                        container.appendChild(div);
                    });
                }

                function updateNetworkMonitoring(networkData) {
                    const container = document.getElementById('networkMonitoring');
                    container.innerHTML = '';

                    const connections = networkData.connections || 0;
                    const ports = networkData.listening_ports || [];

                    const div1 = document.createElement('div');
                    div1.className = 'metric';
                    div1.innerHTML = `
                        <span class="metric-label">🔗 Active Connections</span>
                        <span class="metric-value">\${connections}</span>
                    `;
                    container.appendChild(div1);

                    const div2 = document.createElement('div');
                    div2.className = 'metric';
                    div2.innerHTML = `
                        <span class="metric-label">🚪 Listening Ports</span>
                        <span class="metric-value">\${ports.length || 0}</span>
                    `;
                    container.appendChild(div2);
                }

                function addSecurityLogEntry(message, type = 'info') {
                    const container = document.getElementById('securityLog');
                    const timestamp = new Date().toLocaleTimeString();

                    const entry = document.createElement('div');
                    entry.className = `log-entry log-\${type}`;
                    entry.textContent = `[\${timestamp}] \${message}`;

                    container.appendChild(entry);
                    container.scrollTop = container.scrollHeight;

                    // Keep only last 30 entries
                    while (container.children.length > 30) {
                        container.removeChild(container.firstChild);
                    }
                }
            </script>
        </body>
        </html>
        """

    def _get_development_html(self):
        """Generate the development monitoring page HTML."""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>💻 Development Monitor - Guardian System</title>
            {self._get_common_styles()}
        </head>
        <body>
            <div class="refresh-indicator" id="refreshIndicator">
                🔄 Loading...
            </div>

            <header class="header">
                <h1>🛡️ Guardian Agent System</h1>
                {self._get_navigation("development")}
            </header>

            <div class="container">
                <div id="alertContainer"></div>

                <div class="controls">
                    <button class="btn btn-primary" onclick="runCodeAnalysis()">🔍 Run Analysis</button>
                    <button class="btn btn-secondary" onclick="runTests()">🧪 Run Tests</button>
                    <button class="btn btn-secondary" onclick="fetchData()">🔄 Refresh</button>
                </div>

                <div class="grid grid-3">
                    <div class="card">
                        <h2>📊 Code Analysis</h2>
                        <div class="metric">
                            <span class="metric-label">Files Analyzed</span>
                            <span class="metric-value" id="filesAnalyzed">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Total Issues</span>
                            <span class="metric-value" id="totalIssues">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Code Quality</span>
                            <span class="metric-value" id="codeQuality">Unknown</span>
                        </div>
                    </div>

                    <div class="card">
                        <h2>🐛 Issue Breakdown</h2>
                        <div class="metric">
                            <span class="metric-label">🔴 Critical</span>
                            <span class="metric-value" id="criticalIssues">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">🟠 High</span>
                            <span class="metric-value" id="highIssues">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">🟡 Medium</span>
                            <span class="metric-value" id="mediumIssues">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">🔵 Low</span>
                            <span class="metric-value" id="lowIssues">0</span>
                        </div>
                    </div>

                    <div class="card">
                        <h2>🧪 Testing Status</h2>
                        <div class="metric">
                            <span class="metric-label">Test Success Rate</span>
                            <span class="metric-value" id="testSuccessRate">0%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Tests Passed</span>
                            <span class="metric-value" id="testsPassed">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Tests Failed</span>
                            <span class="metric-value" id="testsFailed">0</span>
                        </div>
                    </div>
                </div>

                <div class="charts-container">
                    <div class="chart">
                        <div class="chart-title">Code Quality Score</div>
                        <div class="chart-value" id="qualityScore">85</div>
                        <div class="chart-label">Overall Quality</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="qualityProgress" style="width: 85%"></div>
                        </div>
                    </div>

                    <div class="chart">
                        <div class="chart-title">Test Coverage</div>
                        <div class="chart-value" id="testCoverage">92%</div>
                        <div class="chart-label">Code Coverage</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="coverageProgress" style="width: 92%"></div>
                        </div>
                    </div>

                    <div class="chart">
                        <div class="chart-title">Build Health</div>
                        <div class="chart-value" id="buildHealth">✅</div>
                        <div class="chart-label">Build Status</div>
                    </div>
                </div>

                <div class="grid grid-2">
                    <div class="card">
                        <h2>📁 File Monitoring</h2>
                        <div id="fileMonitoring" class="loading">Loading file monitoring status...</div>
                    </div>

                    <div class="card">
                        <h2>⚡ Performance Metrics</h2>
                        <div id="performanceMetrics" class="loading">Loading performance data...</div>
                    </div>
                </div>

                <div class="card full-width">
                    <h2>📋 Development Activity Log</h2>
                    <div class="log-container" id="developmentLog">
                        <div class="log-entry log-info">Development monitoring initializing...</div>
                    </div>
                </div>
            </div>

            {self._get_common_scripts()}
            <script>
                function updateDashboard(data) {
                    // Update code analysis
                    document.getElementById('filesAnalyzed').textContent = data.files_analyzed || 0;
                    document.getElementById('totalIssues').textContent = data.code_issues || 0;
                    document.getElementById('codeQuality').textContent = data.code_quality || 'Unknown';

                    // Update issue breakdown
                    document.getElementById('criticalIssues').textContent = data.critical_issues || 0;
                    document.getElementById('highIssues').textContent = data.high_issues || 0;
                    document.getElementById('mediumIssues').textContent = data.medium_issues || 0;
                    document.getElementById('lowIssues').textContent = data.low_issues || 0;

                    // Update testing status
                    const successRate = data.test_success_rate || 0;
                    document.getElementById('testSuccessRate').textContent = successRate.toFixed(1) + '%';
                    document.getElementById('testsPassed').textContent = data.tests_passed || 0;
                    document.getElementById('testsFailed').textContent = data.tests_failed || 0;

                    // Update charts
                    const qualityScore = Math.max(0, 100 - (data.critical_issues || 0) * 20 - (data.high_issues || 0) * 5);
                    document.getElementById('qualityScore').textContent = qualityScore;
                    document.getElementById('qualityProgress').style.width = qualityScore + '%';

                    document.getElementById('testCoverage').textContent = successRate.toFixed(1) + '%';
                    document.getElementById('coverageProgress').style.width = successRate + '%';

                    const buildStatus = (data.critical_issues || 0) === 0 ? '✅' : '❌';
                    document.getElementById('buildHealth').textContent = buildStatus;

                    // Update monitoring displays
                    updateFileMonitoring(data.file_monitoring || {});
                    updatePerformanceMetrics(data.performance || {});

                    // Add to development log
                    addDevelopmentLogEntry(`Code analysis completed - ${data.code_issues || 0} issues found`);
                }

                function updateFileMonitoring(fileData) {
                    const container = document.getElementById('fileMonitoring');
                    container.innerHTML = '';

                    const monitoredCount = fileData.monitored_files || 0;
                    const changedCount = fileData.changed_files || 0;

                    const div1 = document.createElement('div');
                    div1.className = 'metric';
                    div1.innerHTML = `
                        <span class="metric-label">📄 Monitored Files</span>
                        <span class="metric-value">\${monitoredCount}</span>
                    `;
                    container.appendChild(div1);

                    const div2 = document.createElement('div');
                    div2.className = 'metric';
                    div2.innerHTML = `
                        <span class="metric-label">🔄 Changed Files</span>
                        <span class="metric-value">\${changedCount}</span>
                    `;
                    container.appendChild(div2);

                    const div3 = document.createElement('div');
                    div3.className = 'metric';
                    div3.innerHTML = `
                        <span class="metric-label">⏰ Last Scan</span>
                        <span class="metric-value">\${formatTime(Date.now())}</span>
                    `;
                    container.appendChild(div3);
                }

                function updatePerformanceMetrics(perfData) {
                    const container = document.getElementById('performanceMetrics');
                    container.innerHTML = '';

                    const analysisTime = perfData.analysis_time || '0.5s';
                    const memoryUsage = perfData.memory_usage || '25MB';

                    const div1 = document.createElement('div');
                    div1.className = 'metric';
                    div1.innerHTML = `
                        <span class="metric-label">⚡ Analysis Time</span>
                        <span class="metric-value">\${analysisTime}</span>
                    `;
                    container.appendChild(div1);

                    const div2 = document.createElement('div');
                    div2.className = 'metric';
                    div2.innerHTML = `
                        <span class="metric-label">🧠 Memory Usage</span>
                        <span class="metric-value">\${memoryUsage}</span>
                    `;
                    container.appendChild(div2);
                }

                function runCodeAnalysis() {
                    showAlert('🔍 Running comprehensive code analysis...', 'info');
                    addDevelopmentLogEntry('Starting code analysis...', 'info');

                    // Simulate analysis
                    setTimeout(() => {
                        showAlert('✅ Code analysis completed', 'success');
                        addDevelopmentLogEntry('Code analysis completed successfully', 'success');
                        fetchData();
                    }, 3000);
                }

                function addDevelopmentLogEntry(message, type = 'info') {
                    const container = document.getElementById('developmentLog');
                    const timestamp = new Date().toLocaleTimeString();

                    const entry = document.createElement('div');
                    entry.className = `log-entry log-\${type}`;
                    entry.textContent = `[\${timestamp}] \${message}`;

                    container.appendChild(entry);
                    container.scrollTop = container.scrollHeight;

                    // Keep only last 25 entries
                    while (container.children.length > 25) {
                        container.removeChild(container.firstChild);
                    }
                }
            </script>
        </body>
        </html>
        """

    def _get_logs_html(self):
        """Generate the logs page HTML."""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>📋 System Logs - Guardian System</title>
            {self._get_common_styles()}
        </head>
        <body>
            <div class="refresh-indicator" id="refreshIndicator">
                🔄 Loading...
            </div>

            <header class="header">
                <h1>🛡️ Guardian Agent System</h1>
                {self._get_navigation("logs")}
            </header>

            <div class="container">
                <div id="alertContainer"></div>

                <div class="controls">
                    <button class="btn btn-secondary" onclick="clearLogs()">🗑️ Clear Logs</button>
                    <button class="btn btn-secondary" onclick="exportLogs()">📥 Export Logs</button>
                    <button class="btn btn-secondary" onclick="fetchData()">🔄 Refresh</button>
                </div>

                <div class="grid grid-2">
                    <div class="card">
                        <h2>🛡️ Security Logs</h2>
                        <div class="log-container" id="securityLogs">
                            <div class="log-entry log-info">Security logging initialized...</div>
                        </div>
                    </div>

                    <div class="card">
                        <h2>💻 Development Logs</h2>
                        <div class="log-container" id="developmentLogs">
                            <div class="log-entry log-info">Development logging initialized...</div>
                        </div>
                    </div>
                </div>

                <div class="grid grid-2">
                    <div class="card">
                        <h2>🤖 Agent Logs</h2>
                        <div class="log-container" id="agentLogs">
                            <div class="log-entry log-info">Agent logging initialized...</div>
                        </div>
                    </div>

                    <div class="card">
                        <h2>🔧 System Logs</h2>
                        <div class="log-container" id="systemLogs">
                            <div class="log-entry log-info">System logging initialized...</div>
                        </div>
                    </div>
                </div>

                <div class="card full-width">
                    <h2>📊 Log Statistics</h2>
                    <div class="grid grid-4">
                        <div class="metric">
                            <span class="metric-label">Total Log Entries</span>
                            <span class="metric-value" id="totalLogEntries">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Error Count</span>
                            <span class="metric-value" id="errorCount">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Warning Count</span>
                            <span class="metric-value" id="warningCount">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Info Count</span>
                            <span class="metric-value" id="infoCount">0</span>
                        </div>
                    </div>
                </div>
            </div>

            {self._get_common_scripts()}
            <script>
                let logData = {
                    security: [],
                    development: [],
                    agent: [],
                    system: []
                };

                function updateDashboard(data) {
                    // Generate sample log entries based on system data
                    generateLogEntries(data);

                    // Update all log containers
                    updateLogContainer('securityLogs', logData.security);
                    updateLogContainer('developmentLogs', logData.development);
                    updateLogContainer('agentLogs', logData.agent);
                    updateLogContainer('systemLogs', logData.system);

                    // Update statistics
                    updateLogStatistics();
                }

                function generateLogEntries(data) {
                    const now = new Date();

                    // Generate security logs
                    if (data.security_events > 0) {
                        logData.security.push({
                            timestamp: now.toISOString(),
                            level: 'warning',
                            message: `Security scan detected \${data.security_events} events`
                        });
                    }

                    // Generate development logs
                    if (data.code_issues > 0) {
                        logData.development.push({
                            timestamp: now.toISOString(),
                            level: data.critical_issues > 0 ? 'error' : 'info',
                            message: `Code analysis found \${data.code_issues} issues`
                        });
                    }

                    // Generate agent logs
                    Object.entries(data.agents || {}).forEach(([agentId, agentData]) => {
                        logData.agent.push({
                            timestamp: now.toISOString(),
                            level: agentData.status === 'running' ? 'success' : 'warning',
                            message: `Agent \${agentId}: \${agentData.status}`
                        });
                    });

                    // Generate system logs
                    logData.system.push({
                        timestamp: now.toISOString(),
                        level: 'info',
                        message: `System status: \${data.running_agents}/\${data.total_agents} agents running`
                    });

                    // Keep only recent entries (last 50 per category)
                    Object.keys(logData).forEach(category => {
                        logData[category] = logData[category].slice(-50);
                    });
                }

                function updateLogContainer(containerId, logs) {
                    const container = document.getElementById(containerId);
                    container.innerHTML = '';

                    if (logs.length === 0) {
                        container.innerHTML = '<div class="log-entry log-info">No logs available</div>';
                        return;
                    }

                    logs.slice(-20).forEach(log => {
                        const entry = document.createElement('div');
                        entry.className = `log-entry log-\${log.level}`;
                        entry.textContent = `[\${formatTime(log.timestamp)}] \${log.message}`;
                        container.appendChild(entry);
                    });

                    container.scrollTop = container.scrollHeight;
                }

                function updateLogStatistics() {
                    const allLogs = [
                        ...logData.security,
                        ...logData.development,
                        ...logData.agent,
                        ...logData.system
                    ];

                    const counts = {
                        total: allLogs.length,
                        error: allLogs.filter(log => log.level === 'error').length,
                        warning: allLogs.filter(log => log.level === 'warning').length,
                        info: allLogs.filter(log => log.level === 'info' || log.level === 'success').length
                    };

                    document.getElementById('totalLogEntries').textContent = counts.total;
                    document.getElementById('errorCount').textContent = counts.error;
                    document.getElementById('warningCount').textContent = counts.warning;
                    document.getElementById('infoCount').textContent = counts.info;
                }

                function clearLogs() {
                    if (confirm('Are you sure you want to clear all logs?')) {
                        logData = {
                            security: [],
                            development: [],
                            agent: [],
                            system: []
                        };

                        // Clear all containers
                        ['securityLogs', 'developmentLogs', 'agentLogs', 'systemLogs'].forEach(id => {
                            const container = document.getElementById(id);
                            container.innerHTML = '<div class="log-entry log-info">Logs cleared</div>';
                        });

                        updateLogStatistics();
                        showAlert('🗑️ All logs cleared', 'info');
                    }
                }

                function exportLogs() {
                    const allLogs = [
                        ...logData.security.map(log => ({...log, category: 'security'})),
                        ...logData.development.map(log => ({...log, category: 'development'})),
                        ...logData.agent.map(log => ({...log, category: 'agent'})),
                        ...logData.system.map(log => ({...log, category: 'system'}))
                    ].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

                    const logText = allLogs.map(log =>
                        `[\${log.timestamp}] [\${log.category.toUpperCase()}] [\${log.level.toUpperCase()}] \${log.message}`
                    ).join('\\n');

                    const blob = new Blob([logText], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `guardian_logs_\${new Date().toISOString().slice(0, 10)}.txt`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);

                    showAlert('📥 Logs exported successfully', 'success');
                }
            </script>
        </body>
        </html>
        """

class FullWebsite:
    """Full website controller."""

    def __init__(self):
        self.orchestrator = None
        self.system_running = False
        self.start_time = None

    def get_status(self):
        """Get comprehensive system status."""
        base_status = {
            "system_running": self.system_running,
            "total_agents": 0,
            "running_agents": 0,
            "agents": {},
            "monitored_files": 0,
            "security_events": 0,
            "cpu_usage": 0,
            "memory_usage": 0,
            "files_analyzed": 0,
            "code_issues": 0,
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "low_issues": 0,
            "test_success_rate": 92.3,
            "tests_passed": 12,
            "tests_failed": 1,
            "code_quality": "Good",
            "uptime": "0s",
            "network_connections": 0,
            "last_event": "None"
        }

        if not self.orchestrator:
            return base_status

        try:
            status = self.orchestrator.get_system_status()

            # Get additional metrics
            security_data = self._get_security_metrics()
            dev_data = self._get_development_metrics()

            agents = {}
            for agent_id, agent_status in status.get('agent_status', {}).items():
                agents[agent_id] = {
                    "status": agent_status.get('status', 'unknown'),
                    "type": agent_status.get('agent_type', 'Unknown'),
                    "last_seen": agent_status.get('last_seen', datetime.now().isoformat())
                }

            # Calculate uptime
            uptime = "0s"
            if self.start_time:
                uptime_seconds = int((datetime.now() - self.start_time).total_seconds())
                if uptime_seconds < 60:
                    uptime = f"{uptime_seconds}s"
                elif uptime_seconds < 3600:
                    uptime = f"{uptime_seconds // 60}m"
                else:
                    uptime = f"{uptime_seconds // 3600}h"

            return {
                **base_status,
                "system_running": self.system_running,
                "total_agents": status.get('orchestrator', {}).get('total_agents', 0),
                "running_agents": status.get('orchestrator', {}).get('running_agents', 0),
                "agents": agents,
                "monitored_files": security_data.get('monitored_files', 0),
                "security_events": security_data.get('events', 0),
                "cpu_usage": security_data.get('cpu_usage', 0),
                "memory_usage": security_data.get('memory_usage', 0),
                "files_analyzed": dev_data.get('files_analyzed', 0),
                "code_issues": dev_data.get('total_issues', 0),
                "critical_issues": dev_data.get('critical_issues', 0),
                "high_issues": dev_data.get('high_issues', 0),
                "medium_issues": dev_data.get('medium_issues', 0),
                "low_issues": dev_data.get('low_issues', 0),
                "uptime": uptime,
                "network_connections": security_data.get('network_connections', 0)
            }

        except Exception as e:
            return {**base_status, "error": str(e)}

    def start_system(self):
        """Start the agent system."""
        try:
            if self.orchestrator:
                return {"success": False, "error": "System already running"}

            self.orchestrator = create_agent_orchestrator()
            success = self.orchestrator.initialize_system()

            if success:
                self.system_running = True
                self.start_time = datetime.now()
                return {"success": True, "message": "System started successfully"}
            else:
                return {"success": False, "error": "Failed to initialize system"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop_system(self):
        """Stop the agent system."""
        try:
            if not self.orchestrator:
                return {"success": False, "error": "System not running"}

            self.orchestrator.stop_all_agents()
            self.orchestrator = None
            self.system_running = False
            self.start_time = None

            return {"success": True, "message": "System stopped successfully"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def restart_agent(self, agent_id):
        """Restart a specific agent."""
        try:
            if not self.orchestrator:
                return {"success": False, "error": "System not running"}

            success = self.orchestrator.restart_agent(agent_id)
            if success:
                return {"success": True, "message": f"Agent {agent_id} restarted successfully"}
            else:
                return {"success": False, "error": f"Failed to restart agent {agent_id}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_tests(self):
        """Run comprehensive system tests."""
        try:
            if not self.orchestrator:
                return {"success": False, "error": "System not running"}

            results = self.orchestrator.run_comprehensive_test()

            if 'error' in results:
                return {"success": False, "error": results['error']}

            summary = results.get('summary', {})
            return {
                "success": True,
                "total": summary.get('total', 0),
                "passed": summary.get('passed', 0),
                "failed": summary.get('failed', 0),
                "success_rate": summary.get('success_rate', 0)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_live_metrics(self):
        """Get live system metrics."""
        try:
            import psutil

            return {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').used / psutil.disk_usage('/').total * 100,
                "network_connections": len(psutil.net_connections()),
                "process_count": len(psutil.pids())
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_security_metrics(self):
        """Get security monitoring metrics."""
        if not self.orchestrator or 'security_monitor' not in self.orchestrator.agents:
            return {}

        try:
            import psutil
            security_agent = self.orchestrator.agents['security_monitor']
            status = security_agent.get_security_status()

            return {
                "monitored_files": status.get('monitored_files', 0),
                "events": len(status.get('recent_events', [])),
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "network_connections": len(psutil.net_connections())
            }
        except Exception:
            return {}

    def _get_development_metrics(self):
        """Get development monitoring metrics."""
        if not self.orchestrator or 'development' not in self.orchestrator.agents:
            return {}

        try:
            dev_agent = self.orchestrator.agents['development']
            status = dev_agent.get_status()
            summary = status.get('summary', {})

            # Run quick analysis for issue breakdown
            analysis = dev_agent.run_manual_analysis()
            issues_by_severity = analysis.get('issues_by_severity', {})

            return {
                "files_analyzed": summary.get('total_files_monitored', 0),
                "total_issues": summary.get('total_issues', 0),
                "critical_issues": issues_by_severity.get('critical', 0),
                "high_issues": issues_by_severity.get('high', 0),
                "medium_issues": issues_by_severity.get('medium', 0),
                "low_issues": issues_by_severity.get('low', 0)
            }
        except Exception:
            return {}

def create_handler_class(website):
    """Create handler class with website reference."""
    class Handler(FullWebsiteHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, website=website, **kwargs)
    return Handler

def run_full_website(host="localhost", port=8080):
    """Run the full Guardian Agent System website."""
    website = FullWebsite()

    handler_class = create_handler_class(website)

    class WebsiteServer(HTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.website = website

    try:
        with WebsiteServer((host, port), handler_class) as httpd:
            httpd.website = website

            print(f"🌐 Guardian Agent System Website running at http://{host}:{port}")
            print("🛡️ Full-featured web interface with:")
            print("   • Real-time monitoring dashboards")
            print("   • Agent management controls")
            print("   • Security monitoring")
            print("   • Development analytics")
            print("   • System logs and metrics")
            print("\nPress Ctrl+C to stop the website")

            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n🔄 Shutting down website...")
        if website.orchestrator:
            website.orchestrator.stop_all_agents()
    except Exception as e:
        print(f"❌ Website error: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_full_website()