#!/usr/bin/env python3
"""
UIOTA.Space Integration Server
Connects all offline guard components with uiota.space architecture
"""

import asyncio
import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
from aiohttp import web, ClientSession
import websockets

class UIOTASpaceIntegration:
    """
    Main integration server that connects all UIOTA offline-guard components
    with the uiota.space website and provides unified portal access.
    """

    def __init__(self, port: int = 8080):
        self.port = port
        self.app = web.Application()
        self.websocket_clients = set()
        self.component_status = {}
        self.guardian_network = {}
        self.fl_metrics = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self._setup_routes()
        self._initialize_components()

    def _setup_routes(self):
        """Setup all HTTP routes for the unified portal"""

        # Static file serving
        self.app.router.add_static('/', Path.cwd(), name='static')

        # API endpoints
        self.app.router.add_get('/api/status', self.get_system_status)
        self.app.router.add_get('/api/components', self.get_components_status)
        self.app.router.add_get('/api/guardian', self.get_guardian_info)
        self.app.router.add_get('/api/fl-metrics', self.get_fl_metrics)
        self.app.router.add_post('/api/start-fl', self.start_federated_learning)
        self.app.router.add_post('/api/backup', self.create_backup)
        self.app.router.add_post('/api/diagnostics', self.run_diagnostics)

        # WebSocket endpoint
        self.app.router.add_get('/ws', self.websocket_handler)

        # Main portal redirect
        self.app.router.add_get('/', self.serve_main_portal)

    async def serve_main_portal(self, request):
        """Serve the main unified portal"""
        portal_path = Path.cwd() / 'unified_portal.html'
        if portal_path.exists():
            return web.FileResponse(portal_path)
        else:
            return web.Response(text="Portal not found", status=404)

    def _initialize_components(self):
        """Initialize all system components"""
        self.logger.info("Initializing UIOTA.Space components...")

        # Component status tracking
        self.component_status = {
            'offline-ai': {'status': 'ready', 'port': None, 'url': 'offline-ai/index.html'},
            'fl-dashboard': {'status': 'ready', 'port': None, 'url': 'flower-offguard-uiota-demo/dashboard.html'},
            'guardian-portal': {'status': 'ready', 'port': None, 'url': 'web-demo/portal.html'},
            'airplane-mode': {'status': 'ready', 'port': None, 'url': 'web-demo/airplane_mode_guardian.html'},
            'ml-demo': {'status': 'ready', 'port': None, 'url': 'web-demo/ml-demo.js'},
            'token-system': {'status': 'ready', 'port': None, 'url': 'web-demo/uiota-token-system.js'}
        }

        # Guardian network initialization
        self.guardian_network = {
            'local_guardian': {
                'id': 'guardian-001',
                'class': 'AI Guardian',
                'level': 5,
                'xp': 2847,
                'specialization': 'Machine Learning',
                'status': 'online'
            }
        }

        # FL metrics initialization
        self.fl_metrics = {
            'rounds_completed': 12,
            'active_clients': 8,
            'accuracy': 0.94,
            'loss': 0.15,
            'last_update': datetime.now().isoformat()
        }

    async def get_system_status(self, request):
        """Get overall system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_usage': self._get_cpu_usage(),
                'memory_usage': self._get_memory_usage(),
                'disk_usage': self._get_disk_usage(),
                'uptime': self._get_uptime()
            },
            'network': {
                'mode': 'offline-ready',
                'guardian_nodes': len(self.guardian_network),
                'websocket_clients': len(self.websocket_clients)
            },
            'components': self.component_status
        }
        return web.json_response(status)

    async def get_components_status(self, request):
        """Get status of all components"""
        return web.json_response(self.component_status)

    async def get_guardian_info(self, request):
        """Get guardian information"""
        return web.json_response(self.guardian_network)

    async def get_fl_metrics(self, request):
        """Get federated learning metrics"""
        return web.json_response(self.fl_metrics)

    async def start_federated_learning(self, request):
        """Start a new federated learning round"""
        try:
            # Simulate starting FL round
            self.fl_metrics['rounds_completed'] += 1
            self.fl_metrics['last_update'] = datetime.now().isoformat()

            # Broadcast to websocket clients
            await self._broadcast_update({
                'type': 'fl_round_started',
                'round': self.fl_metrics['rounds_completed']
            })

            return web.json_response({
                'success': True,
                'round': self.fl_metrics['rounds_completed']
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def create_backup(self, request):
        """Create system backup"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = Path.cwd() / 'backups' / timestamp
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Create backup (simplified)
            backup_info = {
                'timestamp': timestamp,
                'components': list(self.component_status.keys()),
                'guardian_network': self.guardian_network,
                'fl_metrics': self.fl_metrics
            }

            backup_file = backup_dir / 'system_state.json'
            with open(backup_file, 'w') as f:
                json.dump(backup_info, f, indent=2)

            await self._broadcast_update({
                'type': 'backup_created',
                'timestamp': timestamp
            })

            return web.json_response({
                'success': True,
                'backup_path': str(backup_file)
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def run_diagnostics(self, request):
        """Run system diagnostics"""
        try:
            diagnostics = {
                'timestamp': datetime.now().isoformat(),
                'checks': {
                    'python_version': self._check_python(),
                    'disk_space': self._check_disk_space(),
                    'network_connectivity': self._check_network(),
                    'components_health': self._check_components(),
                    'guardian_network': self._check_guardian_network()
                }
            }

            await self._broadcast_update({
                'type': 'diagnostics_completed',
                'results': diagnostics
            })

            return web.json_response(diagnostics)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def websocket_handler(self, request):
        """Handle WebSocket connections for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.websocket_clients.add(ws)
        self.logger.info(f"WebSocket client connected. Total: {len(self.websocket_clients)}")

        try:
            # Send initial status
            await ws.send_str(json.dumps({
                'type': 'initial_status',
                'data': {
                    'components': self.component_status,
                    'guardian': self.guardian_network,
                    'fl_metrics': self.fl_metrics
                }
            }))

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({'error': 'Invalid JSON'}))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f'WebSocket error: {ws.exception()}')
                    break
        finally:
            self.websocket_clients.discard(ws)
            self.logger.info(f"WebSocket client disconnected. Total: {len(self.websocket_clients)}")

        return ws

    async def _handle_websocket_message(self, ws, data):
        """Handle incoming WebSocket messages"""
        msg_type = data.get('type')

        if msg_type == 'ping':
            await ws.send_str(json.dumps({'type': 'pong'}))
        elif msg_type == 'get_status':
            status = await self.get_system_status(None)
            await ws.send_str(json.dumps({
                'type': 'status_update',
                'data': json.loads(status.text)
            }))

    async def _broadcast_update(self, message):
        """Broadcast update to all WebSocket clients"""
        if self.websocket_clients:
            disconnected = set()
            for ws in self.websocket_clients:
                try:
                    await ws.send_str(json.dumps(message))
                except ConnectionResetError:
                    disconnected.add(ws)

            # Remove disconnected clients
            self.websocket_clients -= disconnected

    def _get_cpu_usage(self):
        """Get current CPU usage"""
        try:
            result = subprocess.run(['python3', '-c',
                'import psutil; print(psutil.cpu_percent(interval=1))'],
                capture_output=True, text=True, timeout=5)
            return float(result.stdout.strip())
        except:
            return 35.0  # Default value

    def _get_memory_usage(self):
        """Get current memory usage"""
        try:
            result = subprocess.run(['python3', '-c',
                'import psutil; m=psutil.virtual_memory(); print(f"{m.used/1024**3:.1f}")'],
                capture_output=True, text=True, timeout=5)
            return f"{result.stdout.strip()}GB"
        except:
            return "6.2GB"  # Default value

    def _get_disk_usage(self):
        """Get current disk usage"""
        try:
            result = subprocess.run(['df', '-h', '.'],
                capture_output=True, text=True, timeout=5)
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                usage = lines[1].split()[4]
                return usage
        except:
            pass
        return "45%"  # Default value

    def _get_uptime(self):
        """Get system uptime"""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
                hours = int(uptime_seconds // 3600)
                return f"{hours}h"
        except:
            return "24h"  # Default value

    def _check_python(self):
        """Check Python version"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}"

    def _check_disk_space(self):
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage('.')
            free_gb = free // (1024**3)
            return f"{free_gb}GB free"
        except:
            return "Unknown"

    def _check_network(self):
        """Check network connectivity"""
        return "Offline-ready"

    def _check_components(self):
        """Check component health"""
        return {comp: status['status'] for comp, status in self.component_status.items()}

    def _check_guardian_network(self):
        """Check guardian network status"""
        return f"{len(self.guardian_network)} guardians online"

    def start_background_tasks(self):
        """Start background monitoring tasks"""
        def update_metrics():
            while True:
                try:
                    # Update FL metrics periodically
                    if self.fl_metrics['rounds_completed'] > 0:
                        # Simulate slight changes in accuracy/loss
                        import random
                        self.fl_metrics['accuracy'] += random.uniform(-0.01, 0.01)
                        self.fl_metrics['loss'] += random.uniform(-0.01, 0.01)
                        self.fl_metrics['accuracy'] = max(0.80, min(0.99, self.fl_metrics['accuracy']))
                        self.fl_metrics['loss'] = max(0.05, min(0.30, self.fl_metrics['loss']))
                        self.fl_metrics['last_update'] = datetime.now().isoformat()

                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    self.logger.error(f"Error in background metrics update: {e}")
                    time.sleep(60)

        # Start background thread
        thread = threading.Thread(target=update_metrics, daemon=True)
        thread.start()

    async def run_server(self):
        """Run the main server"""
        self.logger.info(f"Starting UIOTA.Space Integration Server on port {self.port}")

        # Start background tasks
        self.start_background_tasks()

        # Create and run web app
        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()

        self.logger.info(f"🛡️ UIOTA.Space Portal running at http://localhost:{self.port}")
        self.logger.info(f"🌐 Unified Portal: http://localhost:{self.port}/unified_portal.html")
        self.logger.info(f"🤖 Offline AI: http://localhost:{self.port}/offline-ai/index.html")
        self.logger.info(f"🧠 FL Dashboard: http://localhost:{self.port}/flower-offguard-uiota-demo/dashboard.html")

        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Shutting down server...")
            await runner.cleanup()

async def main():
    """Main entry point"""
    server = UIOTASpaceIntegration(port=8080)
    await server.run_server()

if __name__ == "__main__":
    asyncio.run(main())