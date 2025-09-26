#!/usr/bin/env python3
"""
Simple Demo Launcher - No External Dependencies
Federated learning demonstration with built-in HTTP server
"""

import json
import logging
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

class FLDemoHandler(BaseHTTPRequestHandler):
    """HTTP handler for the FL demo."""

    def __init__(self, *args, demo_launcher=None, **kwargs):
        self.demo_launcher = demo_launcher
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/':
            self._serve_dashboard()
        elif parsed_path.path == '/enhanced_widgets.js':
            self._serve_widgets()
        elif parsed_path.path == '/api/status':
            self._serve_status()
        elif parsed_path.path.startswith('/api/'):
            self._handle_api(parsed_path)
        else:
            self._serve_404()

    def do_POST(self):
        """Handle POST requests."""
        self.do_GET()  # Simple implementation

    def _serve_dashboard(self):
        """Serve the main dashboard."""
        try:
            dashboard_file = Path(__file__).parent / "advanced_fl_dashboard.html"
            if dashboard_file.exists():
                with open(dashboard_file, 'r') as f:
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(content.encode())
            else:
                self._serve_basic_dashboard()
        except Exception as e:
            logger.error(f"Error serving dashboard: {e}")
            self._serve_error(str(e))

    def _serve_widgets(self):
        """Serve the enhanced widgets JavaScript."""
        try:
            widgets_file = Path(__file__).parent / "enhanced_widgets.js"
            if widgets_file.exists():
                with open(widgets_file, 'r') as f:
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-type', 'application/javascript')
                self.end_headers()
                self.wfile.write(content.encode())
            else:
                self._serve_404()
        except Exception as e:
            logger.error(f"Error serving widgets: {e}")
            self._serve_error(str(e))

    def _serve_basic_dashboard(self):
        """Serve a basic dashboard if advanced version not available."""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 FL Demo Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px;
               background: #0f0f0f; color: white; }
        .header { text-align: center; margin-bottom: 30px; }
        .status { background: #1a1a1a; padding: 20px; border-radius: 10px;
                 margin-bottom: 20px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                  gap: 20px; }
        .metric { background: #262626; padding: 15px; border-radius: 8px; text-align: center; }
        .value { font-size: 24px; color: #10a37f; margin-bottom: 5px; }
        .label { color: #999; }
        .terminal { background: #000; border-radius: 8px; padding: 20px;
                   font-family: monospace; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Federated Learning Demo</h1>
        <p>Advanced simulation environment for distributed machine learning</p>
    </div>

    <div class="status">
        <h2>📊 System Status</h2>
        <p id="status-text">Demo running successfully</p>
    </div>

    <div class="metrics">
        <div class="metric">
            <div class="value" id="clients">12</div>
            <div class="label">Active Clients</div>
        </div>
        <div class="metric">
            <div class="value" id="rounds">247</div>
            <div class="label">Training Rounds</div>
        </div>
        <div class="metric">
            <div class="value" id="accuracy">94.7%</div>
            <div class="label">Global Accuracy</div>
        </div>
        <div class="metric">
            <div class="value" id="loss">0.087</div>
            <div class="label">Training Loss</div>
        </div>
    </div>

    <div class="terminal">
        <h3>📟 System Terminal</h3>
        <div id="terminal-output">
            <div>[System] FL Demo initialized successfully</div>
            <div>[Status] All systems operational</div>
            <div>[Info] Ready for federated learning simulation</div>
        </div>
    </div>

    <script>
        function updateMetrics() {
            // Simulate real-time updates
            const rounds = parseInt(document.getElementById('rounds').textContent) + 1;
            document.getElementById('rounds').textContent = rounds;

            const accuracy = 94.7 + Math.sin(rounds / 10) * 2;
            document.getElementById('accuracy').textContent = accuracy.toFixed(1) + '%';

            const loss = 0.087 + Math.sin(rounds / 15) * 0.01;
            document.getElementById('loss').textContent = loss.toFixed(3);

            // Add terminal message occasionally
            if (rounds % 5 === 0) {
                const terminal = document.getElementById('terminal-output');
                const newLine = document.createElement('div');
                newLine.textContent = `[${new Date().toLocaleTimeString()}] Round ${rounds}: Accuracy ${accuracy.toFixed(1)}%`;
                terminal.appendChild(newLine);

                // Keep only last 10 lines
                while (terminal.children.length > 10) {
                    terminal.removeChild(terminal.firstChild);
                }
            }
        }

        // Update every 3 seconds
        setInterval(updateMetrics, 3000);

        console.log('🚀 FL Demo Dashboard initialized');
    </script>
</body>
</html>
        """

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_status(self):
        """Serve status API."""
        if hasattr(self.server, 'demo_launcher'):
            status = self.server.demo_launcher.get_status()
        else:
            status = {
                "demo_running": True,
                "simulation_data": {
                    "current_round": 247,
                    "global_accuracy": 94.7,
                    "active_clients": 12,
                    "training_loss": 0.087
                },
                "timestamp": datetime.now().isoformat()
            }

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

    def _handle_api(self, parsed_path):
        """Handle API endpoints."""
        path = parsed_path.path

        if path == '/api/start-demo':
            result = {"success": True, "message": "Demo started"}
        elif path == '/api/stop-demo':
            result = {"success": True, "message": "Demo stopped"}
        elif path == '/api/terminal-command':
            result = {"success": True, "response": ["Command executed"]}
        else:
            result = {"success": False, "error": "Unknown API endpoint"}

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
        html = "<h1>404 Not Found</h1><p>The requested resource was not found.</p>"
        self.wfile.write(html.encode())

    def _serve_error(self, error_msg):
        """Serve error page."""
        self.send_response(500)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        html = f"<h1>Error</h1><p>{error_msg}</p>"
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        """Suppress request logging."""
        pass

class SimpleDemoLauncher:
    """Simple demo launcher using built-in HTTP server."""

    def __init__(self, host="localhost", port=8888):
        self.host = host
        self.port = port
        self.demo_running = False
        self.simulation_data = {
            "current_round": 0,
            "global_accuracy": 85.0,
            "active_clients": 0,
            "training_loss": 0.245
        }

    def get_status(self):
        """Get demo status."""
        return {
            "demo_running": self.demo_running,
            "simulation_data": self.simulation_data,
            "timestamp": datetime.now().isoformat()
        }

    def start_demo(self):
        """Start the demo simulation."""
        self.demo_running = True
        self.simulation_data["active_clients"] = 8
        self.simulation_data["current_round"] = 1

    def stop_demo(self):
        """Stop the demo simulation."""
        self.demo_running = False

    def run(self):
        """Run the demo launcher."""
        print("🚀 Simple FL Demo Launcher")
        print("=" * 40)
        print(f"🌐 Dashboard: http://{self.host}:{self.port}")
        print("🤖 Federated learning simulation")
        print("💻 Built-in HTTP server")
        print("=" * 40)
        print("Press Ctrl+C to stop")
        print()

        # Create handler class with demo reference
        def create_handler(*args, **kwargs):
            return FLDemoHandler(*args, demo_launcher=self, **kwargs)

        # Create HTTP server
        class DemoServer(HTTPServer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.demo_launcher = self

        try:
            with DemoServer((self.host, self.port), create_handler) as httpd:
                # Auto-open browser
                def open_browser():
                    time.sleep(2)
                    try:
                        webbrowser.open(f"http://{self.host}:{self.port}")
                    except:
                        pass

                browser_thread = threading.Thread(target=open_browser, daemon=True)
                browser_thread.start()

                # Start demo
                self.start_demo()

                print(f"✅ Server running on http://{self.host}:{self.port}")
                print("🎯 Demo started automatically")

                httpd.serve_forever()

        except KeyboardInterrupt:
            print("\n🔄 Shutting down demo...")
            self.stop_demo()
            print("✅ Shutdown complete")
        except Exception as e:
            print(f"❌ Server error: {e}")

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple FL Demo Launcher")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to")
    args = parser.parse_args()

    launcher = SimpleDemoLauncher(args.host, args.port)
    launcher.run()

if __name__ == "__main__":
    main()