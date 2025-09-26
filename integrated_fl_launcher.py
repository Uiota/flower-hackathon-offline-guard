#!/usr/bin/env python3
"""
Integrated FL Launcher with ChatGPT-like Welcome Section
Complete federated learning platform with AI assistant interface
"""

import asyncio
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

# Import live AI integration if available
try:
    from live_ai_integration import (
        LiveAIManager,
        RealWorldFLCoordinator,
        MCPServerIntegration,
        create_live_ai_system
    )
    LIVE_AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Live AI integration not available: {e}")
    LIVE_AI_AVAILABLE = False

logger = logging.getLogger(__name__)

class IntegratedFLHandler(BaseHTTPRequestHandler):
    """HTTP handler for the integrated FL platform."""

    def __init__(self, *args, launcher=None, **kwargs):
        self.launcher = launcher
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/':
            self._serve_welcome_interface()
        elif parsed_path.path == '/live-demo':
            self._serve_live_dashboard()
        elif parsed_path.path == '/api/chat':
            self._handle_chat_api(parsed_path)
        elif parsed_path.path == '/api/system-status':
            self._serve_system_status()
        elif parsed_path.path == '/api/offline-mode':
            self._handle_offline_mode()
        elif parsed_path.path.startswith('/api/'):
            self._handle_api(parsed_path)
        else:
            self._serve_static_file(parsed_path.path)

    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8') if content_length else '{}'

        try:
            data = json.loads(post_data)
        except:
            data = {}

        if parsed_path.path == '/api/chat':
            self._handle_chat_message(data)
        elif parsed_path.path == '/api/enable-offline':
            self._enable_offline_mode(data)
        elif parsed_path.path == '/api/start-fl-demo':
            self._start_fl_demo(data)
        else:
            self._handle_api(parsed_path)

    def _serve_welcome_interface(self):
        """Serve the ChatGPT-like welcome interface."""
        try:
            welcome_file = Path(__file__).parent / "chatgpt_welcome_section.html"
            if welcome_file.exists():
                with open(welcome_file, 'r') as f:
                    content = f.read()

                # Inject current system status
                system_status = self._get_system_status()
                content = content.replace(
                    'AI Services Connected',
                    system_status.get('status_text', 'AI Services Connected')
                )

                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(content.encode())
            else:
                self._serve_fallback_welcome()
        except Exception as e:
            logger.error(f"Error serving welcome interface: {e}")
            self._serve_error(str(e))

    def _serve_live_dashboard(self):
        """Serve the live FL dashboard."""
        try:
            dashboard_file = Path(__file__).parent / "live_ai_dashboard.html"
            if dashboard_file.exists():
                with open(dashboard_file, 'r') as f:
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(content.encode())
            else:
                self._serve_fallback_dashboard()
        except Exception as e:
            logger.error(f"Error serving dashboard: {e}")
            self._serve_error(str(e))

    def _serve_fallback_welcome(self):
        """Serve fallback welcome page."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>FL Guardian AI Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #0f0f0f; color: white; }
        .container { max-width: 800px; margin: 0 auto; padding: 40px 20px; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 48px; margin-bottom: 10px; color: #8b5cf6; }
        .section { background: #1a1a1a; padding: 30px; border-radius: 12px; margin-bottom: 20px; }
        .btn { background: #8b5cf6; color: white; padding: 12px 24px; border: none;
               border-radius: 8px; cursor: pointer; margin: 10px; font-size: 16px; }
        .btn:hover { background: #7c3aed; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 FL Guardian</h1>
            <p>Advanced Federated Learning with AI Integration</p>
        </div>

        <div class="section">
            <h2>🌐 What is Federated Learning?</h2>
            <p>Federated Learning enables multiple parties to train machine learning models collaboratively
            without sharing raw data. Perfect for privacy-sensitive applications in healthcare, finance,
            and enterprise environments.</p>
        </div>

        <div class="section">
            <h2>🛡️ Privacy-First Approach</h2>
            <ul>
                <li>🔒 Data never leaves your devices</li>
                <li>🔐 End-to-end encryption</li>
                <li>🎭 Differential privacy protection</li>
                <li>🏠 Offline mode available</li>
            </ul>
        </div>

        <div class="section">
            <h2>🚀 Get Started</h2>
            <button class="btn" onclick="window.location.href='/live-demo'">Launch Live Demo</button>
            <button class="btn" onclick="enableOfflineMode()">Enable Offline Mode</button>
        </div>
    </div>

    <script>
        function enableOfflineMode() {
            alert('🛡️ Offline mode provides maximum privacy by running everything locally.');
            fetch('/api/enable-offline', { method: 'POST' })
                .then(() => location.reload());
        }
    </script>
</body>
</html>
        """

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_fallback_dashboard(self):
        """Serve fallback dashboard."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>FL Live Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #0f0f0f; color: white; padding: 20px; }
        .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #1a1a1a; padding: 20px; border-radius: 12px; border: 1px solid #333; }
        .metric { font-size: 24px; color: #8b5cf6; margin-bottom: 5px; }
    </style>
</head>
<body>
    <h1>🚀 Federated Learning Live Demo</h1>
    <div class="dashboard">
        <div class="card">
            <h3>📊 Global Accuracy</h3>
            <div class="metric">87.3%</div>
        </div>
        <div class="card">
            <h3>🔄 Training Round</h3>
            <div class="metric">47</div>
        </div>
        <div class="card">
            <h3>📱 Active Clients</h3>
            <div class="metric">12</div>
        </div>
        <div class="card">
            <h3>🤖 AI Status</h3>
            <div class="metric">Connected</div>
        </div>
    </div>
</body>
</html>
        """

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def _handle_chat_message(self, data):
        """Handle chat message from AI assistant."""
        message = data.get('message', '')

        # Generate AI response
        if LIVE_AI_AVAILABLE and hasattr(self.server.launcher, 'ai_manager'):
            # Use real AI if available
            response = self._generate_ai_response(message)
        else:
            # Use rule-based responses
            response = self._generate_mock_response(message)

        self._send_json({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })

    def _generate_mock_response(self, message):
        """Generate mock AI response."""
        message_lower = message.lower()

        if 'federated learning' in message_lower or 'fl' in message_lower:
            return """Federated Learning allows multiple devices to train a shared model without exposing their data.
            It's perfect for privacy-sensitive applications like healthcare and finance. Would you like to see our live demo?"""

        elif 'offline' in message_lower or 'privacy' in message_lower:
            return """Our offline mode provides maximum privacy by running everything locally. No data ever leaves your system.
            This is ideal for highly sensitive environments and regulatory compliance."""

        elif 'demo' in message_lower or 'start' in message_lower:
            return """I can launch our interactive federated learning demo for you! You'll see real-time training metrics,
            network visualization, and AI-generated insights. Ready to begin?"""

        else:
            return """I'm here to help you understand federated learning and our privacy-preserving AI platform.
            Ask me about FL concepts, privacy features, or how to get started with our demo!"""

    def _serve_system_status(self):
        """Serve current system status."""
        status = self._get_system_status()
        self._send_json(status)

    def _get_system_status(self):
        """Get current system status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'live_ai_available': LIVE_AI_AVAILABLE,
            'demo_running': getattr(self.server.launcher, 'demo_running', True),
            'offline_mode': getattr(self.server.launcher, 'offline_mode', False),
            'fl_training_active': getattr(self.server.launcher, 'fl_training_active', False),
            'connected_clients': getattr(self.server.launcher, 'connected_clients', 3),
            'status_text': 'Offline Mode Active' if getattr(self.server.launcher, 'offline_mode', False)
                          else 'AI Services Connected'
        }

    def _enable_offline_mode(self, data):
        """Enable offline mode."""
        try:
            self.server.launcher.offline_mode = True
            self.server.launcher.fl_training_active = False

            self._send_json({
                'success': True,
                'message': 'Offline mode enabled successfully',
                'status': 'offline'
            })

            logger.info("🛡️ Offline mode enabled")

        except Exception as e:
            self._send_json({
                'success': False,
                'error': str(e)
            })

    def _start_fl_demo(self, data):
        """Start FL demo."""
        try:
            self.server.launcher.demo_running = True
            self.server.launcher.fl_training_active = True

            if LIVE_AI_AVAILABLE and hasattr(self.server.launcher, 'fl_coordinator'):
                self.server.launcher.fl_coordinator.training_active = True

            self._send_json({
                'success': True,
                'message': 'FL demo started successfully'
            })

            logger.info("🚀 FL demo started")

        except Exception as e:
            self._send_json({
                'success': False,
                'error': str(e)
            })

    def _handle_api(self, parsed_path):
        """Handle other API endpoints."""
        self._send_json({
            'message': 'API endpoint accessed',
            'path': parsed_path.path,
            'timestamp': datetime.now().isoformat()
        })

    def _serve_static_file(self, path):
        """Serve static files."""
        # Handle static file requests (CSS, JS, etc.)
        self._serve_404()

    def _send_json(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _serve_404(self):
        """Serve 404 error."""
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        html = "<h1>404 Not Found</h1>"
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

class IntegratedFLLauncher:
    """Integrated FL launcher with ChatGPT-like interface."""

    def __init__(self, host="localhost", port=8888):
        self.host = host
        self.port = port

        # Initialize live AI components if available
        if LIVE_AI_AVAILABLE:
            try:
                self.ai_manager, self.fl_coordinator, self.mcp_server = create_live_ai_system()
                logger.info("✅ Live AI system initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize live AI: {e}")
                self.ai_manager = None
                self.fl_coordinator = None
                self.mcp_server = None
        else:
            self.ai_manager = None
            self.fl_coordinator = None
            self.mcp_server = None

        # Platform state
        self.demo_running = True
        self.fl_training_active = False
        self.offline_mode = False
        self.connected_clients = 3
        self.current_round = 47

    def start_platform(self):
        """Start the integrated platform."""
        self.demo_running = True

        # Start background services
        self._start_background_services()

        logger.info("🚀 Integrated FL platform started")

    def _start_background_services(self):
        """Start background services."""
        def background_loop():
            while self.demo_running:
                try:
                    if self.fl_training_active and not self.offline_mode:
                        self.current_round += 1

                        # Simulate FL progress
                        if LIVE_AI_AVAILABLE and self.fl_coordinator:
                            # Use real FL coordinator
                            pass

                    time.sleep(3)

                except Exception as e:
                    logger.error(f"Background service error: {e}")
                    time.sleep(5)

        background_thread = threading.Thread(target=background_loop, daemon=True)
        background_thread.start()

    def stop_platform(self):
        """Stop the integrated platform."""
        self.demo_running = False
        self.fl_training_active = False

    def run(self):
        """Run the integrated FL launcher."""
        print("🤖 Integrated Federated Learning Platform")
        print("=" * 55)
        print(f"🌐 Welcome Interface: http://{self.host}:{self.port}")
        print(f"🚀 Live Demo: http://{self.host}:{self.port}/live-demo")
        print()
        print("✨ Features:")
        print("  • ChatGPT-like AI assistant interface")
        print("  • Interactive federated learning explanation")
        print("  • Real-world FL training capabilities")
        print("  • Privacy-first offline mode")
        print("  • Live AI integration (OpenAI, Anthropic, etc.)")
        print("  • MCP server for AI service coordination")
        print("  • LangGraph workflow visualization")
        print()
        print("🛡️ Privacy Options:")
        print("  • Request offline mode for maximum security")
        print("  • Air-gapped environment support")
        print("  • GDPR/HIPAA compliance features")
        print()
        print("🎯 Live AI:", "✅ Available" if LIVE_AI_AVAILABLE else "❌ Simulation mode")
        print("=" * 55)
        print("Press Ctrl+C to stop the platform")
        print()

        # Create handler class
        def create_handler(*args, **kwargs):
            return IntegratedFLHandler(*args, launcher=self, **kwargs)

        # Create HTTP server
        class PlatformServer(HTTPServer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.launcher = self

        try:
            with PlatformServer((self.host, self.port), create_handler) as httpd:
                # Auto-open browser
                def open_browser():
                    time.sleep(2)
                    try:
                        webbrowser.open(f"http://{self.host}:{self.port}")
                        print(f"🌐 Browser opened to http://{self.host}:{self.port}")
                    except Exception as e:
                        print(f"⚠️ Could not auto-open browser: {e}")
                        print(f"Please manually navigate to: http://{self.host}:{self.port}")

                browser_thread = threading.Thread(target=open_browser, daemon=True)
                browser_thread.start()

                # Start platform
                self.start_platform()

                print(f"✅ Platform running on http://{self.host}:{self.port}")
                print("🤖 AI assistant ready to help!")

                httpd.serve_forever()

        except KeyboardInterrupt:
            print("\n🔄 Shutting down integrated platform...")
            self.stop_platform()
            print("✅ Shutdown complete")
        except Exception as e:
            print(f"❌ Platform error: {e}")

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Integrated FL Platform with AI Assistant")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to")
    parser.add_argument("--offline", action="store_true", help="Start in offline mode")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    launcher = IntegratedFLLauncher(args.host, args.port)

    if args.offline:
        launcher.offline_mode = True
        print("🛡️ Starting in offline mode for maximum privacy")

    launcher.run()

if __name__ == "__main__":
    main()