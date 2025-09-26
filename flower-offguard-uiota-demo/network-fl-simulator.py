#!/usr/bin/env python3
"""
Real-time Network Traffic FL Simulator with Frequency Tracking
Complete functional federated learning with network simulation and sensor monitoring
"""

import sys
import os
import time
import threading
import socket
import json
import random
import math
import queue
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
import subprocess
from datetime import datetime
import csv

# Add project paths
sys.path.insert(0, 'src')

# Import our working modules
from src.guard import GuardConfig, new_keypair, sign_blob, verify_blob

class NetworkTrafficSimulator:
    """Simulates realistic network traffic patterns for FL communications."""

    def __init__(self):
        self.traffic_log = []
        self.frequency_tracker = {}
        self.bandwidth_usage = []
        self.packet_loss_rate = 0.05  # 5% packet loss
        self.latency_base = 50  # 50ms base latency
        self.is_monitoring = True

    def simulate_packet_transmission(self, source, destination, payload_size, packet_type):
        """Simulate packet transmission with realistic network conditions."""
        # Calculate transmission time based on bandwidth (simulated 1 Mbps)
        transmission_time = payload_size / (1024 * 128)  # 1 Mbps = 128 KB/s

        # Add network latency
        latency = self.latency_base + random.uniform(0, 30)  # Variable latency

        # Simulate packet loss
        packet_lost = random.random() < self.packet_loss_rate

        # Track frequency
        freq_key = f"{source}->{destination}"
        if freq_key not in self.frequency_tracker:
            self.frequency_tracker[freq_key] = []

        self.frequency_tracker[freq_key].append(time.time())

        # Log traffic
        traffic_entry = {
            'timestamp': time.time(),
            'source': source,
            'destination': destination,
            'packet_type': packet_type,
            'payload_size': payload_size,
            'transmission_time': transmission_time,
            'latency': latency,
            'packet_lost': packet_lost,
            'bandwidth_mbps': (payload_size * 8) / (transmission_time * 1000000) if transmission_time > 0 else 0
        }

        self.traffic_log.append(traffic_entry)

        # Simulate actual transmission delay
        time.sleep(transmission_time + latency/1000)

        return not packet_lost

    def get_frequency_stats(self, window_seconds=60):
        """Calculate transmission frequency over time window."""
        current_time = time.time()
        stats = {}

        for connection, timestamps in self.frequency_tracker.items():
            recent_transmissions = [t for t in timestamps if current_time - t < window_seconds]
            frequency_hz = len(recent_transmissions) / window_seconds
            stats[connection] = {
                'frequency_hz': frequency_hz,
                'total_packets': len(recent_transmissions),
                'window_seconds': window_seconds
            }

        return stats

    def get_bandwidth_utilization(self, window_seconds=60):
        """Calculate bandwidth utilization over time window."""
        current_time = time.time()
        recent_traffic = [t for t in self.traffic_log if current_time - t['timestamp'] < window_seconds]

        total_bytes = sum(t['payload_size'] for t in recent_traffic)
        avg_bandwidth_mbps = (total_bytes * 8) / (window_seconds * 1000000) if window_seconds > 0 else 0

        return {
            'avg_bandwidth_mbps': avg_bandwidth_mbps,
            'total_packets': len(recent_traffic),
            'total_bytes': total_bytes,
            'packet_loss_rate': sum(1 for t in recent_traffic if t['packet_lost']) / len(recent_traffic) if recent_traffic else 0
        }

class NetworkSensor:
    """Real-time network monitoring sensor."""

    def __init__(self, traffic_sim):
        self.traffic_sim = traffic_sim
        self.sensor_data = []
        self.monitoring = True

    def start_monitoring(self):
        """Start continuous network monitoring."""
        def monitor_loop():
            while self.monitoring:
                # Collect real-time metrics
                freq_stats = self.traffic_sim.get_frequency_stats(10)  # 10-second window
                bandwidth_stats = self.traffic_sim.get_bandwidth_utilization(10)

                sensor_reading = {
                    'timestamp': time.time(),
                    'frequency_stats': freq_stats,
                    'bandwidth_stats': bandwidth_stats,
                    'active_connections': len(freq_stats),
                    'total_traffic_entries': len(self.traffic_sim.traffic_log)
                }

                self.sensor_data.append(sensor_reading)

                # Keep only last 100 readings
                if len(self.sensor_data) > 100:
                    self.sensor_data.pop(0)

                time.sleep(1)  # Sample every second

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def get_latest_reading(self):
        """Get the most recent sensor reading."""
        return self.sensor_data[-1] if self.sensor_data else None

class WorkingFLModel:
    """Functional ML model using pure Python."""

    def __init__(self, input_size=20, hidden_size=50, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights randomly
        self.w1 = [[random.gauss(0, 0.1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [random.gauss(0, 0.1) for _ in range(hidden_size)]
        self.w2 = [[random.gauss(0, 0.1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [random.gauss(0, 0.1) for _ in range(output_size)]

        self.loss_history = []
        self.accuracy_history = []

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-max(min(x, 500), -500)))

    def softmax(self, x):
        max_x = max(x)
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp = sum(exp_x)
        return [xi / sum_exp for xi in exp_x]

    def forward(self, x):
        # Hidden layer
        hidden = []
        for j in range(self.hidden_size):
            activation = sum(x[i] * self.w1[i][j] for i in range(len(x))) + self.b1[j]
            hidden.append(self.sigmoid(activation))

        # Output layer
        output = []
        for k in range(self.output_size):
            activation = sum(hidden[j] * self.w2[j][k] for j in range(self.hidden_size)) + self.b2[k]
            output.append(activation)

        return self.softmax(output), hidden

    def train_step(self, X, y, learning_rate=0.01):
        total_loss = 0
        correct = 0

        for i in range(len(X)):
            output, hidden = self.forward(X[i])

            # Calculate loss
            target = [0] * self.output_size
            target[y[i]] = 1
            loss = -sum(target[j] * math.log(max(output[j], 1e-15)) for j in range(self.output_size))
            total_loss += loss

            # Check accuracy
            predicted = output.index(max(output))
            if predicted == y[i]:
                correct += 1

            # Backpropagation (simplified)
            output_grad = [output[j] - target[j] for j in range(self.output_size)]

            # Update output weights
            for j in range(self.hidden_size):
                for k in range(self.output_size):
                    self.w2[j][k] -= learning_rate * output_grad[k] * hidden[j]
            for k in range(self.output_size):
                self.b2[k] -= learning_rate * output_grad[k]

            # Update hidden weights
            hidden_grad = []
            for j in range(self.hidden_size):
                grad = sum(output_grad[k] * self.w2[j][k] for k in range(self.output_size))
                grad *= hidden[j] * (1 - hidden[j])
                hidden_grad.append(grad)

            for i_idx in range(len(X[i])):
                for j in range(self.hidden_size):
                    self.w1[i_idx][j] -= learning_rate * hidden_grad[j] * X[i][i_idx]
            for j in range(self.hidden_size):
                self.b1[j] -= learning_rate * hidden_grad[j]

        avg_loss = total_loss / len(X)
        accuracy = correct / len(X)

        self.loss_history.append(avg_loss)
        self.accuracy_history.append(accuracy)

        return avg_loss, accuracy

    def get_weights(self):
        return {
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2
        }

    def set_weights(self, weights):
        self.w1 = weights['w1']
        self.b1 = weights['b1']
        self.w2 = weights['w2']
        self.b2 = weights['b2']

class NetworkedFLClient:
    """FL Client with network traffic simulation."""

    def __init__(self, client_id, data, labels, traffic_sim):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self.model = WorkingFLModel()
        self.traffic_sim = traffic_sim

        # Create mesh directory safely
        mesh_dir = Path(f"./mesh_data/client_{client_id}")
        mesh_dir.mkdir(parents=True, exist_ok=True)

        # Security
        self.private_key, self.public_key = new_keypair()

        print(f"✅ Client {client_id} initialized with {len(data)} samples")

    def send_update_to_server(self, update_payload):
        """Send model update to server with network simulation."""
        payload_json = json.dumps(update_payload, default=str)
        payload_size = len(payload_json.encode())

        # Simulate network transmission
        success = self.traffic_sim.simulate_packet_transmission(
            f"client_{self.client_id}",
            "fl_server",
            payload_size,
            "model_update"
        )

        if success:
            print(f"📤 Client {self.client_id} -> Server: {payload_size} bytes transmitted")
            return True
        else:
            print(f"❌ Client {self.client_id} -> Server: Packet lost")
            return False

    def receive_global_model(self, global_weights):
        """Receive global model with network simulation."""
        payload_json = json.dumps(global_weights, default=str)
        payload_size = len(payload_json.encode())

        # Simulate network transmission
        success = self.traffic_sim.simulate_packet_transmission(
            "fl_server",
            f"client_{self.client_id}",
            payload_size,
            "global_model"
        )

        if success:
            print(f"📥 Server -> Client {self.client_id}: {payload_size} bytes received")
            return global_weights
        else:
            print(f"❌ Server -> Client {self.client_id}: Packet lost, using cached model")
            return self.model.get_weights()  # Use cached model

    def train_round(self, global_weights=None):
        """Perform local training with network communications."""
        if global_weights:
            received_weights = self.receive_global_model(global_weights)
            self.model.set_weights(received_weights)

        print(f"🔄 Client {self.client_id} starting local training...")

        # Train for epochs
        for epoch in range(2):
            loss, acc = self.model.train_step(self.data, self.labels)

        weights = self.model.get_weights()

        # Sign the weights
        weights_data = json.dumps(weights, default=str).encode()
        signature = sign_blob(self.private_key, weights_data)

        update_payload = {
            'weights': weights,
            'num_samples': len(self.data),
            'loss': loss,
            'accuracy': acc,
            'signature': signature.hex(),
            'client_id': self.client_id,
            'timestamp': time.time()
        }

        # Send update with network simulation
        transmission_success = self.send_update_to_server(update_payload)

        if transmission_success:
            return update_payload
        else:
            # Retry once on failure
            print(f"🔄 Client {self.client_id} retrying transmission...")
            time.sleep(0.1)
            if self.send_update_to_server(update_payload):
                return update_payload
            else:
                print(f"❌ Client {self.client_id} failed to send update")
                return None

class NetworkedFLServer:
    """FL Server with network traffic simulation."""

    def __init__(self, num_clients, traffic_sim):
        self.num_clients = num_clients
        self.global_model = WorkingFLModel()
        self.round_number = 0
        self.client_updates = []
        self.training_history = []
        self.traffic_sim = traffic_sim

        # Security
        self.private_key, self.public_key = new_keypair()

        print(f"🖥️  FL Server initialized for {num_clients} clients")

    def broadcast_global_model(self, clients):
        """Broadcast global model to all clients with network simulation."""
        global_weights = self.global_model.get_weights()
        payload_json = json.dumps(global_weights, default=str)
        payload_size = len(payload_json.encode())

        print(f"📡 Broadcasting global model ({payload_size} bytes) to {len(clients)} clients")

        # Simulate broadcast to each client
        for client in clients:
            self.traffic_sim.simulate_packet_transmission(
                "fl_server",
                f"client_{client.client_id}",
                payload_size,
                "global_model_broadcast"
            )

    def aggregate_weights(self, client_updates):
        """Federated averaging with network-aware aggregation."""
        if not client_updates:
            return self.global_model.get_weights()

        print(f"🔄 Aggregating {len(client_updates)} client updates...")

        # Initialize aggregated weights
        first_weights = client_updates[0]['weights']
        aggregated = {
            'w1': [[0.0 for _ in range(len(first_weights['w1'][0]))] for _ in range(len(first_weights['w1']))],
            'b1': [0.0 for _ in range(len(first_weights['b1']))],
            'w2': [[0.0 for _ in range(len(first_weights['w2'][0]))] for _ in range(len(first_weights['w2']))],
            'b2': [0.0 for _ in range(len(first_weights['b2']))]
        }

        total_samples = sum(update['num_samples'] for update in client_updates)

        # Weighted averaging
        for update in client_updates:
            weight_factor = update['num_samples'] / total_samples
            weights = update['weights']

            # Aggregate all weights
            for i in range(len(weights['w1'])):
                for j in range(len(weights['w1'][i])):
                    aggregated['w1'][i][j] += weights['w1'][i][j] * weight_factor

            for i in range(len(weights['b1'])):
                aggregated['b1'][i] += weights['b1'][i] * weight_factor

            for i in range(len(weights['w2'])):
                for j in range(len(weights['w2'][i])):
                    aggregated['w2'][i][j] += weights['w2'][i][j] * weight_factor

            for i in range(len(weights['b2'])):
                aggregated['b2'][i] += weights['b2'][i] * weight_factor

        return aggregated

    def run_federated_round(self, clients):
        """Run one federated round with full network simulation."""
        self.round_number += 1
        print(f"\n🔄 Starting Federated Round {self.round_number}")
        print("=" * 60)

        # Broadcast global model
        self.broadcast_global_model(clients)

        # Collect client updates
        client_updates = []
        global_weights = self.global_model.get_weights()

        for client in clients:
            update = client.train_round(global_weights)
            if update:  # Only include successful updates
                client_updates.append(update)

        if not client_updates:
            print("❌ No client updates received this round")
            return None

        # Aggregate weights
        new_global_weights = self.aggregate_weights(client_updates)
        self.global_model.set_weights(new_global_weights)

        # Calculate statistics
        avg_loss = sum(update['loss'] for update in client_updates) / len(client_updates)
        avg_accuracy = sum(update['accuracy'] for update in client_updates) / len(client_updates)

        # Get network statistics
        freq_stats = self.traffic_sim.get_frequency_stats(30)
        bandwidth_stats = self.traffic_sim.get_bandwidth_utilization(30)

        round_stats = {
            'round': self.round_number,
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'num_clients': len(client_updates),
            'total_samples': sum(update['num_samples'] for update in client_updates),
            'network_frequency': freq_stats,
            'network_bandwidth': bandwidth_stats,
            'timestamp': time.time()
        }

        self.training_history.append(round_stats)

        print(f"📊 Round {self.round_number} Results:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Average Accuracy: {avg_accuracy:.4f}")
        print(f"   Participating Clients: {len(client_updates)}")
        print(f"   Network Bandwidth: {bandwidth_stats['avg_bandwidth_mbps']:.2f} Mbps")
        print(f"   Packet Loss Rate: {bandwidth_stats['packet_loss_rate']:.1%}")

        return round_stats

def generate_non_iid_data(total_samples=600, num_clients=3, num_classes=4):
    """Generate non-IID data distribution."""
    print("📊 Generating non-IID data distribution...")

    client_data = []
    client_labels = []

    for client_id in range(num_clients):
        samples_per_client = total_samples // num_clients

        # Different class distributions for each client
        class_probs = [0.25, 0.25, 0.25, 0.25]  # Default
        if client_id == 0:
            class_probs = [0.7, 0.1, 0.1, 0.1]
        elif client_id == 1:
            class_probs = [0.1, 0.7, 0.1, 0.1]
        elif client_id == 2:
            class_probs = [0.1, 0.1, 0.7, 0.1]

        data = []
        labels = []

        for _ in range(samples_per_client):
            class_choice = random.choices(range(num_classes), weights=class_probs)[0]

            # Generate features with class bias
            sample = [random.gauss(0, 1) for _ in range(20)]

            if class_choice == 0:
                sample[:5] = [x + 2 for x in sample[:5]]
            elif class_choice == 1:
                sample[5:10] = [x + 2 for x in sample[5:10]]
            elif class_choice == 2:
                sample[10:15] = [x + 2 for x in sample[10:15]]
            else:
                sample[15:20] = [x + 2 for x in sample[15:20]]

            data.append(sample)
            labels.append(class_choice)

        client_data.append(data)
        client_labels.append(labels)

        class_dist = [labels.count(i) for i in range(num_classes)]
        print(f"  Client {client_id}: {samples_per_client} samples, classes: {class_dist}")

    return client_data, client_labels

class DashboardHandler(SimpleHTTPRequestHandler):
    """Enhanced dashboard with network monitoring."""

    def __init__(self, *args, fl_server=None, sensor=None, **kwargs):
        self.fl_server = fl_server
        self.sensor = sensor
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            sensor_data = self.sensor.get_latest_reading() if self.sensor else {}

            status = {
                'server_status': 'running',
                'current_round': getattr(self.fl_server, 'round_number', 0),
                'training_history': getattr(self.fl_server, 'training_history', []),
                'network_sensor': sensor_data,
                'timestamp': time.time()
            }

            self.wfile.write(json.dumps(status, default=str).encode())
            return

        elif self.path == '/':
            self.path = '/dashboard.html'

        return super().do_GET()

def create_enhanced_dashboard():
    """Create dashboard with network frequency monitoring."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>FL Network Traffic Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 1.8em; font-weight: bold; color: #3498db; }
        .metric-label { color: #7f8c8d; font-size: 0.85em; }
        .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        .chart-container { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .network-section { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        canvas { width: 100%; height: 250px; }
        .frequency-display { font-family: monospace; background: #ecf0f1; padding: 10px; border-radius: 3px; margin: 10px 0; }
        .status-indicator { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
        .status-active { background: #27ae60; animation: pulse 2s infinite; }
        .status-idle { background: #f39c12; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 FL Network Traffic Dashboard</h1>
            <p>Real-time federated learning with network frequency monitoring</p>
        </div>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value" id="current-round">0</div>
                <div class="metric-label">Current Round</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="avg-accuracy">0.000</div>
                <div class="metric-label">Average Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="avg-loss">0.000</div>
                <div class="metric-label">Average Loss</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="bandwidth">0.0</div>
                <div class="metric-label">Bandwidth (Mbps)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="packet-loss">0.0%</div>
                <div class="metric-label">Packet Loss</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="total-packets">0</div>
                <div class="metric-label">Total Packets</div>
            </div>
        </div>

        <div class="charts">
            <div class="chart-container">
                <h3>Training Progress</h3>
                <canvas id="training-chart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Network Bandwidth</h3>
                <canvas id="network-chart"></canvas>
            </div>
        </div>

        <div class="network-section">
            <h3>📡 Real-time Network Frequency Monitoring</h3>
            <div id="frequency-monitor">
                <p>Initializing network sensor...</p>
            </div>

            <h4>Connection Status</h4>
            <div id="connection-status">
                <p><span class="status-indicator status-active"></span>FL Server: Active</p>
                <p><span class="status-indicator status-active"></span>Network Sensor: Monitoring</p>
                <p><span class="status-indicator status-active"></span>Traffic Simulation: Running</p>
            </div>
        </div>
    </div>

    <script>
        let trainingHistory = [];
        let networkHistory = [];

        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update basic metrics
                    document.getElementById('current-round').textContent = data.current_round;

                    if (data.training_history.length > 0) {
                        const latest = data.training_history[data.training_history.length - 1];
                        document.getElementById('avg-accuracy').textContent = latest.avg_accuracy.toFixed(3);
                        document.getElementById('avg-loss').textContent = latest.avg_loss.toFixed(3);

                        trainingHistory = data.training_history;
                        updateTrainingChart();
                    }

                    // Update network metrics
                    if (data.network_sensor && data.network_sensor.bandwidth_stats) {
                        const netStats = data.network_sensor.bandwidth_stats;
                        document.getElementById('bandwidth').textContent = netStats.avg_bandwidth_mbps.toFixed(2);
                        document.getElementById('packet-loss').textContent = (netStats.packet_loss_rate * 100).toFixed(1) + '%';
                        document.getElementById('total-packets').textContent = netStats.total_packets;

                        networkHistory.push({
                            timestamp: data.timestamp,
                            bandwidth: netStats.avg_bandwidth_mbps,
                            packet_loss: netStats.packet_loss_rate
                        });

                        if (networkHistory.length > 50) networkHistory.shift();
                        updateNetworkChart();
                    }

                    // Update frequency monitor
                    if (data.network_sensor && data.network_sensor.frequency_stats) {
                        updateFrequencyMonitor(data.network_sensor.frequency_stats);
                    }
                })
                .catch(error => console.error('Error:', error));
        }

        function updateFrequencyMonitor(freqStats) {
            const monitor = document.getElementById('frequency-monitor');
            let html = '<h4>📊 Connection Frequencies</h4>';

            Object.keys(freqStats).forEach(connection => {
                const stats = freqStats[connection];
                const freq = stats.frequency_hz.toFixed(2);
                const packets = stats.total_packets;

                html += `
                    <div class="frequency-display">
                        <strong>${connection}</strong><br>
                        Frequency: ${freq} Hz | Packets: ${packets} | Status:
                        <span class="status-indicator ${freq > 0.1 ? 'status-active' : 'status-idle'}"></span>
                        ${freq > 0.1 ? 'Active' : 'Idle'}
                    </div>
                `;
            });

            if (Object.keys(freqStats).length === 0) {
                html += '<p>No active connections detected</p>';
            }

            monitor.innerHTML = html;
        }

        function updateTrainingChart() {
            const canvas = document.getElementById('training-chart');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (trainingHistory.length === 0) return;

            const width = canvas.width;
            const height = canvas.height;
            const margin = 30;

            // Draw axes
            ctx.strokeStyle = '#ddd';
            ctx.beginPath();
            ctx.moveTo(margin, margin);
            ctx.lineTo(margin, height - margin);
            ctx.lineTo(width - margin, height - margin);
            ctx.stroke();

            // Draw accuracy line
            ctx.strokeStyle = '#27ae60';
            ctx.lineWidth = 2;
            ctx.beginPath();

            for (let i = 0; i < trainingHistory.length; i++) {
                const x = margin + (i / Math.max(1, trainingHistory.length - 1)) * (width - 2 * margin);
                const y = height - margin - (trainingHistory[i].avg_accuracy * (height - 2 * margin));

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Labels
            ctx.fillStyle = '#2c3e50';
            ctx.font = '12px Arial';
            ctx.fillText('Accuracy', margin, 20);
        }

        function updateNetworkChart() {
            const canvas = document.getElementById('network-chart');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (networkHistory.length === 0) return;

            const width = canvas.width;
            const height = canvas.height;
            const margin = 30;

            // Draw axes
            ctx.strokeStyle = '#ddd';
            ctx.beginPath();
            ctx.moveTo(margin, margin);
            ctx.lineTo(margin, height - margin);
            ctx.lineTo(width - margin, height - margin);
            ctx.stroke();

            // Draw bandwidth line
            ctx.strokeStyle = '#3498db';
            ctx.lineWidth = 2;
            ctx.beginPath();

            const maxBandwidth = Math.max(...networkHistory.map(h => h.bandwidth), 0.1);

            for (let i = 0; i < networkHistory.length; i++) {
                const x = margin + (i / Math.max(1, networkHistory.length - 1)) * (width - 2 * margin);
                const y = height - margin - ((networkHistory[i].bandwidth / maxBandwidth) * (height - 2 * margin));

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            ctx.fillStyle = '#2c3e50';
            ctx.font = '12px Arial';
            ctx.fillText('Bandwidth (Mbps)', margin, 20);
        }

        // Update every 1 second for real-time monitoring
        setInterval(updateDashboard, 1000);
        updateDashboard();

        // Setup canvas
        window.addEventListener('load', function() {
            ['training-chart', 'network-chart'].forEach(id => {
                const canvas = document.getElementById(id);
                canvas.width = canvas.offsetWidth;
                canvas.height = 250;
            });
        });
    </script>
</body>
</html>
"""
    return html

def run_complete_fl_network_simulation():
    """Run complete FL simulation with network traffic and frequency monitoring."""
    print("🚀 Starting Complete FL Network Simulation")
    print("🌐 With Real-time Traffic and Frequency Monitoring")
    print("=" * 70)

    # Set environment
    os.environ["OFFLINE_MODE"] = "1"

    # Configuration
    num_clients = 3
    num_rounds = 8
    total_samples = 600

    print(f"📋 Configuration:")
    print(f"   Clients: {num_clients}")
    print(f"   Rounds: {num_rounds}")
    print(f"   Total Samples: {total_samples}")
    print()

    # Initialize network simulation
    traffic_sim = NetworkTrafficSimulator()
    sensor = NetworkSensor(traffic_sim)
    sensor.start_monitoring()

    print("✅ Network traffic simulator initialized")
    print("✅ Network sensor monitoring started")

    # Generate data
    client_data, client_labels = generate_non_iid_data(total_samples, num_clients)

    # Initialize server
    server = NetworkedFLServer(num_clients, traffic_sim)

    # Initialize clients
    clients = []
    for i in range(num_clients):
        client = NetworkedFLClient(i, client_data[i], client_labels[i], traffic_sim)
        clients.append(client)

    # Set up dashboard
    print("🌐 Setting up enhanced dashboard with network monitoring...")
    dashboard_html = create_enhanced_dashboard()

    os.makedirs("dashboard_static", exist_ok=True)
    with open("dashboard_static/dashboard.html", "w") as f:
        f.write(dashboard_html)

    # Start dashboard server
    def run_dashboard():
        os.chdir("dashboard_static")
        handler = lambda *args, **kwargs: DashboardHandler(*args, fl_server=server, sensor=sensor, **kwargs)
        httpd = ThreadingHTTPServer(("localhost", 8081), handler)
        print("✅ Enhanced dashboard available at: http://localhost:8081")
        httpd.serve_forever()

    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()

    print("✅ All systems operational!")
    print("\n🌐 Dashboard: http://localhost:8081")
    print("📡 Real-time network monitoring active")
    print("📊 Frequency tracking enabled")
    print("🔄 Starting federated learning with network simulation...\n")

    # Run federated learning
    try:
        for round_num in range(num_rounds):
            round_stats = server.run_federated_round(clients)

            if round_stats:
                # Print network statistics
                print("\n📡 Network Statistics:")
                for connection, stats in round_stats['network_frequency'].items():
                    print(f"   {connection}: {stats['frequency_hz']:.2f} Hz ({stats['total_packets']} packets)")

                bandwidth_stats = round_stats['network_bandwidth']
                print(f"   Total Bandwidth: {bandwidth_stats['avg_bandwidth_mbps']:.2f} Mbps")
                print(f"   Packet Loss: {bandwidth_stats['packet_loss_rate']:.1%}")

            time.sleep(3)  # Pause to observe real-time updates

        print("\n🎉 Federated Learning with Network Simulation Complete!")
        print("=" * 70)

        if server.training_history:
            final_stats = server.training_history[-1]
            print("📊 Final Results:")
            print(f"   Final Accuracy: {final_stats['avg_accuracy']:.4f}")
            print(f"   Final Loss: {final_stats['avg_loss']:.4f}")
            print(f"   Total Rounds: {final_stats['round']}")
            print(f"   Network Efficiency: {final_stats['network_bandwidth']['avg_bandwidth_mbps']:.2f} Mbps avg")

        print("\n🌐 Dashboard continues running at: http://localhost:8081")
        print("📡 Network sensor continues monitoring")
        print("Press Ctrl+C to stop all services")

        # Keep services running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down all services...")
            sensor.monitoring = False

    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    pass

if __name__ == "__main__":
    run_complete_fl_network_simulation()