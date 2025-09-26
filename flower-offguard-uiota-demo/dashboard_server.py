#!/usr/bin/env python3
"""
State-of-the-art Federated Learning Dashboard Server
Real-time monitoring with WebSocket updates and modern UI
"""

import asyncio
import json
import time
import threading
import random
import math
import os
from pathlib import Path
from typing import Dict, List, Any
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
import urllib.parse as urlparse
import webbrowser

# Set environment
os.environ["OFFLINE_MODE"] = "1"

class FLDashboardServer:
    """Advanced FL Dashboard with real-time monitoring."""

    def __init__(self, port=8081):
        self.port = port
        self.clients = {}
        self.training_data = {
            "global_metrics": {
                "round": 0,
                "accuracy": 0.0,
                "loss": 0.0,
                "total_clients": 0,
                "active_clients": 0
            },
            "client_metrics": {},
            "training_history": [],
            "client_status": {},
            "system_metrics": {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "network_io": 0.0
            }
        }
        self.fl_thread = None
        self.running = False

    def start_fl_training(self):
        """Start federated learning training simulation."""
        self.running = True
        self.fl_thread = threading.Thread(target=self._run_fl_simulation, daemon=True)
        self.fl_thread.start()

    def _run_fl_simulation(self):
        """Run realistic federated learning simulation."""
        num_clients = 8
        num_rounds = 10

        # Initialize clients
        for i in range(num_clients):
            self.training_data["client_metrics"][f"client_{i}"] = {
                "id": f"client_{i}",
                "status": "initializing",
                "accuracy": 0.0,
                "loss": 2.5,
                "samples": random.randint(800, 1200),
                "epochs_completed": 0,
                "last_update": time.time()
            }

        self.training_data["global_metrics"]["total_clients"] = num_clients

        for round_num in range(1, num_rounds + 1):
            if not self.running:
                break

            print(f"🔄 Dashboard: Starting Round {round_num}")

            # Update global round
            self.training_data["global_metrics"]["round"] = round_num

            # Client training phase
            active_clients = random.randint(5, num_clients)
            self.training_data["global_metrics"]["active_clients"] = active_clients

            active_client_ids = random.sample(list(self.training_data["client_metrics"].keys()), active_clients)

            # Update client statuses
            for client_id in self.training_data["client_metrics"]:
                if client_id in active_client_ids:
                    self.training_data["client_metrics"][client_id]["status"] = "training"
                else:
                    self.training_data["client_metrics"][client_id]["status"] = "offline"

            # Simulate training epochs for each active client
            for epoch in range(3):
                for client_id in active_client_ids:
                    if not self.running:
                        break

                    client = self.training_data["client_metrics"][client_id]

                    # Simulate training progress
                    current_loss = client["loss"]
                    improvement = random.uniform(0.05, 0.15)
                    new_loss = max(0.1, current_loss - improvement)

                    # Calculate accuracy from loss (inverse relationship)
                    new_accuracy = min(0.99, 1.0 - (new_loss / 2.5))

                    client["loss"] = new_loss
                    client["accuracy"] = new_accuracy
                    client["epochs_completed"] = epoch + 1
                    client["last_update"] = time.time()
                    client["status"] = f"epoch_{epoch+1}/3"

                    time.sleep(0.3)  # Realistic training time

            # Server aggregation phase
            print(f"📊 Dashboard: Aggregating Round {round_num}")

            # Calculate global metrics
            total_accuracy = 0.0
            total_loss = 0.0

            for client_id in active_client_ids:
                client = self.training_data["client_metrics"][client_id]
                total_accuracy += client["accuracy"]
                total_loss += client["loss"]
                client["status"] = "aggregating"

            global_accuracy = total_accuracy / len(active_client_ids)
            global_loss = total_loss / len(active_client_ids)

            self.training_data["global_metrics"]["accuracy"] = global_accuracy
            self.training_data["global_metrics"]["loss"] = global_loss

            # Add to history
            self.training_data["training_history"].append({
                "round": round_num,
                "accuracy": global_accuracy,
                "loss": global_loss,
                "active_clients": active_clients,
                "timestamp": time.time()
            })

            # Update system metrics
            self.training_data["system_metrics"]["cpu_usage"] = random.uniform(30, 80)
            self.training_data["system_metrics"]["memory_usage"] = random.uniform(40, 70)
            self.training_data["system_metrics"]["network_io"] = random.uniform(10, 50)

            # Set clients to completed
            for client_id in active_client_ids:
                self.training_data["client_metrics"][client_id]["status"] = "completed"

            print(f"✅ Dashboard: Round {round_num} complete - Accuracy: {global_accuracy:.2%}")
            time.sleep(2)  # Pause between rounds

        # Training complete
        for client_id in self.training_data["client_metrics"]:
            self.training_data["client_metrics"][client_id]["status"] = "training_complete"

        print("🏁 Dashboard: Federated Learning Complete")

class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for the dashboard."""

    def __init__(self, *args, dashboard_server=None, **kwargs):
        self.dashboard_server = dashboard_server
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            self.path = '/dashboard.html'
        elif self.path == '/api/metrics':
            self._serve_metrics()
            return
        elif self.path == '/api/start':
            self._start_training()
            return
        elif self.path == '/api/stop':
            self._stop_training()
            return

        super().do_GET()

    def _serve_metrics(self):
        """Serve current training metrics as JSON."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        response = json.dumps(self.dashboard_server.training_data, indent=2)
        self.wfile.write(response.encode())

    def _start_training(self):
        """Start FL training."""
        if not self.dashboard_server.running:
            self.dashboard_server.start_fl_training()

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(b'{"status": "started"}')

    def _stop_training(self):
        """Stop FL training."""
        self.dashboard_server.running = False

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(b'{"status": "stopped"}')

def run_dashboard_server():
    """Run the dashboard server."""
    dashboard = FLDashboardServer()

    class CustomHandler(DashboardHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, dashboard_server=dashboard, **kwargs)

    print("🚀 Starting Federated Learning Dashboard Server")
    print(f"🌐 Dashboard will be available at: http://localhost:{dashboard.port}")
    print("📊 Features: Real-time monitoring, live charts, client status")

    try:
        with HTTPServer(("", dashboard.port), CustomHandler) as httpd:
            print(f"✅ Server started on port {dashboard.port}")
            print("🎯 Open http://localhost:8080 in your browser")

            # Auto-open browser
            try:
                webbrowser.open(f'http://localhost:{dashboard.port}')
            except:
                pass

            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n⏹️  Dashboard server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    run_dashboard_server()