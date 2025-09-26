#!/usr/bin/env python3
"""
Simple Web Dashboard for UIOTA Guardian Agent System

Lightweight web interface using built-in Python HTTP server.
"""

import json
import logging
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import socketserver

# Add agents to path
sys.path.insert(0, str(Path(__file__).parent / "agents"))

from agents.agent_orchestrator import create_agent_orchestrator

logger = logging.getLogger(__name__)

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    def __init__(self, *args, dashboard=None, **kwargs):
        self.dashboard = dashboard
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/':
            self._serve_dashboard()
        elif parsed_path.path == '/api/status':
            self._serve_status()
        elif parsed_path.path == '/api/start':
            self._start_system()
        elif parsed_path.path == '/api/stop':
            self._stop_system()
        elif parsed_path.path == '/api/test':
            self._run_tests()
        else:
            self._serve_404()

    def do_POST(self):
        """Handle POST requests."""
        self.do_GET()  # For simplicity, treat POST same as GET

    def _serve_dashboard(self):
        """Serve the main dashboard HTML."""
        html = self._get_dashboard_html()

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_status(self):
        """Serve the status API."""
        if hasattr(self.server, 'dashboard'):
            status = self.server.dashboard.get_status()
        else:
            status = {"error": "Dashboard not available"}

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

    def _start_system(self):
        """Start the agent system."""
        if hasattr(self.server, 'dashboard'):
            result = self.server.dashboard.start_system()
        else:
            result = {"success": False, "error": "Dashboard not available"}

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def _stop_system(self):
        """Stop the agent system."""
        if hasattr(self.server, 'dashboard'):
            result = self.server.dashboard.stop_system()
        else:
            result = {"success": False, "error": "Dashboard not available"}

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def _run_tests(self):
        """Run system tests."""
        if hasattr(self.server, 'dashboard'):
            result = self.server.dashboard.run_tests()
        else:
            result = {"success": False, "error": "Dashboard not available"}

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def _serve_404(self):
        """Serve 404 error."""
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<h1>404 Not Found</h1>')

    def log_message(self, format, *args):
        """Suppress request logging."""
        pass

    def _get_dashboard_html(self):
        """Generate the dashboard HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ Guardian Agent Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
            color: #e1e5e9;
            min-height: 100vh;
        }

        .header {
            background: rgba(0, 0, 0, 0.3);
            padding: 2rem;
            text-align: center;
            border-bottom: 3px solid #00ff41;
        }

        .header h1 {
            font-size: 3rem;
            color: #00ff41;
            text-shadow: 0 0 20px rgba(0, 255, 65, 0.5);
            margin-bottom: 1rem;
        }

        .header .subtitle {
            color: #8b949e;
            font-size: 1.2rem;
        }

        .container {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 2rem;
        }

        .controls {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-bottom: 3rem;
        }

        .btn {
            padding: 1rem 2rem;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 150px;
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

        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 3rem;
        }

        .card {
            background: rgba(22, 27, 34, 0.9);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .card h2 {
            color: #58a6ff;
            margin-bottom: 1.5rem;
            font-size: 1.5rem;
            text-align: center;
            border-bottom: 2px solid #58a6ff;
            padding-bottom: 0.5rem;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0;
            border-bottom: 1px solid #21262d;
        }

        .metric:last-child {
            border-bottom: none;
        }

        .metric-label {
            color: #8b949e;
            font-weight: 500;
        }

        .metric-value {
            color: #00ff41;
            font-weight: bold;
            font-size: 1.2rem;
        }

        .agent-list {
            list-style: none;
        }

        .agent-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            margin-bottom: 0.5rem;
            background: rgba(13, 17, 23, 0.6);
            border-radius: 8px;
            border-left: 4px solid #21262d;
        }

        .agent-item.running {
            border-left-color: #00ff41;
        }

        .agent-item.stopped {
            border-left-color: #ff4757;
        }

        .agent-name {
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .agent-status {
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 600;
        }

        .status-running {
            background: rgba(0, 255, 65, 0.2);
            color: #00ff41;
        }

        .status-stopped {
            background: rgba(255, 71, 87, 0.2);
            color: #ff4757;
        }

        .log-container {
            background: rgba(13, 17, 23, 0.8);
            border: 1px solid #21262d;
            border-radius: 8px;
            padding: 1.5rem;
            height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }

        .log-entry {
            margin-bottom: 0.5rem;
            padding: 0.25rem;
        }

        .log-info { color: #58a6ff; }
        .log-success { color: #00ff41; }
        .log-warning { color: #f1e05a; }
        .log-error { color: #f85149; }

        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }

        .status-running .status-indicator {
            background: #00ff41;
            animation: pulse 2s infinite;
        }

        .status-stopped .status-indicator {
            background: #ff4757;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .refresh-indicator {
            position: fixed;
            top: 2rem;
            right: 2rem;
            background: rgba(22, 27, 34, 0.9);
            color: #00ff41;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            border: 1px solid #00ff41;
            font-size: 0.9rem;
        }

        .loading {
            text-align: center;
            color: #8b949e;
            padding: 2rem;
            font-style: italic;
        }

        .full-width {
            grid-column: 1 / -1;
        }
    </style>
</head>
<body>
    <div class="refresh-indicator" id="refreshIndicator">
        🔄 Auto-refresh: ON
    </div>

    <header class="header">
        <h1>🛡️ Guardian Agent Dashboard</h1>
        <p class="subtitle">Real-time monitoring and control of cybersecurity and development agents</p>
    </header>

    <div class="container">
        <div class="controls">
            <button class="btn btn-primary" onclick="startSystem()">🚀 Start System</button>
            <button class="btn btn-danger" onclick="stopSystem()">⏹️ Stop System</button>
            <button class="btn btn-secondary" onclick="runTests()">🧪 Run Tests</button>
            <button class="btn btn-secondary" onclick="refreshStatus()">🔄 Refresh</button>
        </div>

        <div class="status-grid">
            <!-- System Overview -->
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
                <div class="metric">
                    <span class="metric-label">Last Update</span>
                    <span class="metric-value" id="lastUpdate">Never</span>
                </div>
            </div>

            <!-- Security Monitoring -->
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
                    <span class="metric-label">CPU Usage</span>
                    <span class="metric-value" id="cpuUsage">0%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory Usage</span>
                    <span class="metric-value" id="memoryUsage">0%</span>
                </div>
            </div>

            <!-- Development Status -->
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
                    <span class="metric-label">Critical Issues</span>
                    <span class="metric-value" id="criticalIssues">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Test Status</span>
                    <span class="metric-value" id="testStatus">Not Run</span>
                </div>
            </div>

            <!-- Agent Status -->
            <div class="card">
                <h2>🤖 Agent Status</h2>
                <ul class="agent-list" id="agentList">
                    <li class="loading">Loading agent status...</li>
                </ul>
            </div>

            <!-- Activity Log -->
            <div class="card full-width">
                <h2>📋 Activity Log</h2>
                <div class="log-container" id="activityLog">
                    <div class="log-entry log-info">
                        [${new Date().toLocaleTimeString()}] 🔄 Dashboard initialized
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let autoRefresh = true;
        let refreshInterval;

        function addLogEntry(message, type = 'info') {
            const logContainer = document.getElementById('activityLog');
            const timestamp = new Date().toLocaleTimeString();

            const entry = document.createElement('div');
            entry.className = `log-entry log-${type}`;
            entry.textContent = `[${timestamp}] ${message}`;

            logContainer.appendChild(entry);
            logContainer.scrollTop = logContainer.scrollHeight;

            // Keep only last 50 entries
            while (logContainer.children.length > 50) {
                logContainer.removeChild(logContainer.firstChild);
            }
        }

        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                updateDashboard(data);
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
            } catch (error) {
                console.error('Failed to fetch status:', error);
                addLogEntry(`⚠️ Failed to fetch status: ${error.message}`, 'error');
            }
        }

        function updateDashboard(data) {
            // System overview
            document.getElementById('totalAgents').textContent = data.total_agents || 0;
            document.getElementById('runningAgents').textContent = data.running_agents || 0;
            document.getElementById('systemStatus').textContent = data.system_running ? 'Online' : 'Offline';

            // Security status
            document.getElementById('monitoredFiles').textContent = data.monitored_files || 0;
            document.getElementById('securityEvents').textContent = data.security_events || 0;
            document.getElementById('cpuUsage').textContent = (data.cpu_usage || 0).toFixed(1) + '%';
            document.getElementById('memoryUsage').textContent = (data.memory_usage || 0).toFixed(1) + '%';

            // Development status
            document.getElementById('filesAnalyzed').textContent = data.files_analyzed || 0;
            document.getElementById('codeIssues').textContent = data.code_issues || 0;
            document.getElementById('criticalIssues').textContent = data.critical_issues || 0;
            document.getElementById('testStatus').textContent = data.test_status || 'Not Run';

            // Update agent list
            updateAgentList(data.agents || {});
        }

        function updateAgentList(agents) {
            const agentList = document.getElementById('agentList');
            agentList.innerHTML = '';

            if (Object.keys(agents).length === 0) {
                const li = document.createElement('li');
                li.className = 'loading';
                li.textContent = 'No agents running';
                agentList.appendChild(li);
                return;
            }

            Object.entries(agents).forEach(([agentId, agentData]) => {
                const li = document.createElement('li');
                li.className = `agent-item ${agentData.status}`;

                const agentName = document.createElement('div');
                agentName.className = 'agent-name';
                agentName.innerHTML = `${getAgentIcon(agentId)} ${formatAgentName(agentId)}`;

                const agentStatus = document.createElement('div');
                agentStatus.className = `agent-status status-${agentData.status}`;
                agentStatus.innerHTML = `<span class="status-indicator"></span>${agentData.status.toUpperCase()}`;

                li.appendChild(agentName);
                li.appendChild(agentStatus);
                agentList.appendChild(li);
            });
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

        function formatAgentName(agentId) {
            return agentId.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
        }

        async function startSystem() {
            addLogEntry('🚀 Starting Guardian Agent System...', 'info');
            try {
                const response = await fetch('/api/start');
                const result = await response.json();
                if (result.success) {
                    addLogEntry('✅ Agent system started successfully', 'success');
                    setTimeout(fetchStatus, 2000); // Refresh after 2 seconds
                } else {
                    addLogEntry(`❌ Failed to start system: ${result.error}`, 'error');
                }
            } catch (error) {
                addLogEntry(`⚠️ Error starting system: ${error.message}`, 'error');
            }
        }

        async function stopSystem() {
            addLogEntry('⏹️ Stopping Guardian Agent System...', 'warning');
            try {
                const response = await fetch('/api/stop');
                const result = await response.json();
                if (result.success) {
                    addLogEntry('🔄 Agent system stopped', 'info');
                    setTimeout(fetchStatus, 2000); // Refresh after 2 seconds
                } else {
                    addLogEntry(`❌ Failed to stop system: ${result.error}`, 'error');
                }
            } catch (error) {
                addLogEntry(`⚠️ Error stopping system: ${error.message}`, 'error');
            }
        }

        async function runTests() {
            addLogEntry('🧪 Running comprehensive system tests...', 'info');
            try {
                const response = await fetch('/api/test');
                const result = await response.json();
                if (result.success) {
                    addLogEntry(`✅ Tests completed: ${result.passed}/${result.total} passed (${result.success_rate.toFixed(1)}%)`, 'success');
                } else {
                    addLogEntry(`❌ Test execution failed: ${result.error}`, 'error');
                }
            } catch (error) {
                addLogEntry(`⚠️ Error running tests: ${error.message}`, 'error');
            }
        }

        function refreshStatus() {
            addLogEntry('🔄 Refreshing status...', 'info');
            fetchStatus();
        }

        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            const indicator = document.getElementById('refreshIndicator');

            if (autoRefresh) {
                indicator.textContent = '🔄 Auto-refresh: ON';
                refreshInterval = setInterval(fetchStatus, 5000);
            } else {
                indicator.textContent = '⏸️ Auto-refresh: OFF';
                clearInterval(refreshInterval);
            }
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            addLogEntry('🔄 Initializing Guardian Agent Dashboard...', 'info');
            fetchStatus();

            // Start auto-refresh
            refreshInterval = setInterval(fetchStatus, 5000);

            // Click refresh indicator to toggle auto-refresh
            document.getElementById('refreshIndicator').addEventListener('click', toggleAutoRefresh);

            addLogEntry('✅ Dashboard ready - click Start System to begin', 'success');
        });
    </script>
</body>
</html>
        """

class SimpleDashboard:
    """Simple dashboard controller."""

    def __init__(self):
        self.orchestrator = None
        self.system_running = False

    def get_status(self):
        """Get system status."""
        if not self.orchestrator:
            return {
                "system_running": False,
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
                "test_status": "Not Run"
            }

        try:
            status = self.orchestrator.get_system_status()

            # Get additional metrics
            security_data = self._get_security_metrics()
            dev_data = self._get_development_metrics()

            agents = {}
            for agent_id, agent_status in status.get('agent_status', {}).items():
                agents[agent_id] = {"status": agent_status.get('status', 'unknown')}

            return {
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
                "test_status": "Ready"
            }

        except Exception as e:
            return {"error": str(e)}

    def start_system(self):
        """Start the agent system."""
        try:
            if self.orchestrator:
                return {"success": False, "error": "System already running"}

            self.orchestrator = create_agent_orchestrator()
            success = self.orchestrator.initialize_system()

            if success:
                self.system_running = True
                return {"success": True, "message": "System started"}
            else:
                return {"success": False, "error": "Failed to initialize"}

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

            return {"success": True, "message": "System stopped"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_tests(self):
        """Run system tests."""
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
                "memory_usage": psutil.virtual_memory().percent
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

            return {
                "files_analyzed": summary.get('total_files_monitored', 0),
                "total_issues": summary.get('total_issues', 0),
                "critical_issues": summary.get('critical_issues', 0)
            }
        except Exception:
            return {}

def create_handler_class(dashboard):
    """Create handler class with dashboard reference."""
    class Handler(DashboardHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, dashboard=dashboard, **kwargs)
    return Handler

def run_dashboard(host="localhost", port=8080):
    """Run the simple web dashboard."""
    dashboard = SimpleDashboard()

    handler_class = create_handler_class(dashboard)

    class DashboardServer(HTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dashboard = dashboard

    try:
        with DashboardServer((host, port), handler_class) as httpd:
            httpd.dashboard = dashboard

            print(f"🌐 Guardian Agent Dashboard running at http://{host}:{port}")
            print("🛡️ Real-time monitoring and control interface")
            print("📊 Click 'Start System' to initialize all agents")
            print("\nPress Ctrl+C to stop the dashboard")

            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n🔄 Shutting down dashboard...")
        if dashboard.orchestrator:
            dashboard.orchestrator.stop_all_agents()
    except Exception as e:
        print(f"❌ Dashboard error: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dashboard()