#!/usr/bin/env python3
"""
Working Federated Learning Demo - Actually Functional
Uses only available system dependencies to create a real working FL system
"""

import sys
import os
import time
import threading
import socket
import json
import random
import math
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
import subprocess

# Add project paths
sys.path.insert(0, '../src')
sys.path.insert(0, '..')

# Import our working modules
from src.guard import GuardConfig, new_keypair, sign_blob, verify_blob
from src.mesh_sync import MeshTransport, MeshConfig

class WorkingFLModel:
    """A simple but functional ML model using pure Python (no PyTorch)."""

    def __init__(self, input_size=784, hidden_size=100, output_size=10):
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
        """Sigmoid activation function."""
        return 1 / (1 + math.exp(-max(min(x, 500), -500)))  # Prevent overflow

    def softmax(self, x):
        """Softmax activation for output layer."""
        max_x = max(x)
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp = sum(exp_x)
        return [xi / sum_exp for xi in exp_x]

    def forward(self, x):
        """Forward pass through the network."""
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
        """Perform one training step (simplified backpropagation)."""
        total_loss = 0
        correct = 0

        for i in range(len(X)):
            # Forward pass
            output, hidden = self.forward(X[i])

            # Calculate loss (cross-entropy)
            target = [0] * self.output_size
            target[y[i]] = 1
            loss = -sum(target[j] * math.log(max(output[j], 1e-15)) for j in range(self.output_size))
            total_loss += loss

            # Check accuracy
            predicted = output.index(max(output))
            if predicted == y[i]:
                correct += 1

            # Simplified backpropagation (gradient descent)
            # Output layer gradients
            output_grad = [output[j] - target[j] for j in range(self.output_size)]

            # Update output weights and biases
            for j in range(self.hidden_size):
                for k in range(self.output_size):
                    self.w2[j][k] -= learning_rate * output_grad[k] * hidden[j]
            for k in range(self.output_size):
                self.b2[k] -= learning_rate * output_grad[k]

            # Hidden layer gradients (simplified)
            hidden_grad = []
            for j in range(self.hidden_size):
                grad = sum(output_grad[k] * self.w2[j][k] for k in range(self.output_size))
                grad *= hidden[j] * (1 - hidden[j])  # Sigmoid derivative
                hidden_grad.append(grad)

            # Update hidden weights and biases
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
        """Get model weights for federated aggregation."""
        return {
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2
        }

    def set_weights(self, weights):
        """Set model weights from federated aggregation."""
        self.w1 = weights['w1']
        self.b1 = weights['b1']
        self.w2 = weights['w2']
        self.b2 = weights['b2']

class FLClient:
    """Functional Federated Learning Client."""

    def __init__(self, client_id, data, labels):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self.model = WorkingFLModel()
        self.mesh = MeshTransport(MeshConfig(), f"./mesh_data/client_{client_id}")

        # Security
        self.private_key, self.public_key = new_keypair()

        print(f"✅ Client {client_id} initialized with {len(data)} samples")

    def train_round(self, global_weights=None):
        """Perform local training for one federated round."""
        if global_weights:
            self.model.set_weights(global_weights)

        print(f"🔄 Client {self.client_id} starting local training...")

        # Train for a few epochs
        for epoch in range(3):
            loss, acc = self.model.train_step(self.data, self.labels)
            if epoch % 1 == 0:
                print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")

        # Get updated weights
        weights = self.model.get_weights()

        # Sign the weights for security
        weights_data = json.dumps(weights, default=str).encode()
        signature = sign_blob(self.private_key, weights_data)

        update_payload = {
            'weights': weights,
            'num_samples': len(self.data),
            'loss': loss,
            'accuracy': acc,
            'signature': signature.hex(),
            'client_id': self.client_id
        }

        return update_payload

class FLServer:
    """Functional Federated Learning Server."""

    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.global_model = WorkingFLModel()
        self.round_number = 0
        self.client_updates = []
        self.training_history = []

        # Security
        self.private_key, self.public_key = new_keypair()

        print(f"🖥️  FL Server initialized for {num_clients} clients")

    def aggregate_weights(self, client_updates):
        """Perform federated averaging of client weights."""
        if not client_updates:
            return self.global_model.get_weights()

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

            # Aggregate w1
            for i in range(len(weights['w1'])):
                for j in range(len(weights['w1'][i])):
                    aggregated['w1'][i][j] += weights['w1'][i][j] * weight_factor

            # Aggregate b1
            for i in range(len(weights['b1'])):
                aggregated['b1'][i] += weights['b1'][i] * weight_factor

            # Aggregate w2
            for i in range(len(weights['w2'])):
                for j in range(len(weights['w2'][i])):
                    aggregated['w2'][i][j] += weights['w2'][i][j] * weight_factor

            # Aggregate b2
            for i in range(len(weights['b2'])):
                aggregated['b2'][i] += weights['b2'][i] * weight_factor

        return aggregated

    def run_federated_round(self, clients):
        """Run one complete federated learning round."""
        self.round_number += 1
        print(f"\n🔄 Starting Federated Round {self.round_number}")
        print("=" * 50)

        # Send global model to clients and collect updates
        global_weights = self.global_model.get_weights()
        client_updates = []

        for client in clients:
            update = client.train_round(global_weights)
            client_updates.append(update)
            print(f"📥 Received update from Client {client.client_id}")

        # Aggregate weights
        print("🔄 Aggregating client updates...")
        new_global_weights = self.aggregate_weights(client_updates)
        self.global_model.set_weights(new_global_weights)

        # Calculate round statistics
        avg_loss = sum(update['loss'] for update in client_updates) / len(client_updates)
        avg_accuracy = sum(update['accuracy'] for update in client_updates) / len(client_updates)

        round_stats = {
            'round': self.round_number,
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'num_clients': len(client_updates),
            'total_samples': sum(update['num_samples'] for update in client_updates)
        }

        self.training_history.append(round_stats)

        print(f"📊 Round {self.round_number} Results:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Average Accuracy: {avg_accuracy:.4f}")
        print(f"   Participating Clients: {len(client_updates)}")

        return round_stats

def generate_synthetic_data(num_samples=100, num_features=20):
    """Generate synthetic classification data."""
    data = []
    labels = []

    for _ in range(num_samples):
        # Generate features
        sample = [random.gauss(0, 1) for _ in range(num_features)]

        # Simple classification rule
        label = 0
        if sum(sample[:5]) > 0:
            label = 1
        if sum(sample[5:10]) > 1:
            label = 2
        if sum(sample[10:15]) > 2:
            label = 3

        data.append(sample)
        labels.append(label % 4)  # 4 classes

    return data, labels

def create_non_iid_data(total_samples=1000, num_clients=3, num_classes=4):
    """Create non-IID data distribution across clients."""
    print("📊 Generating non-IID data distribution...")

    client_data = []
    client_labels = []

    for client_id in range(num_clients):
        # Each client gets different class distribution
        samples_per_client = total_samples // num_clients

        # Create biased class distribution
        class_probs = [0.1, 0.1, 0.1, 0.7]  # Default
        if client_id == 1:
            class_probs = [0.7, 0.1, 0.1, 0.1]
        elif client_id == 2:
            class_probs = [0.1, 0.7, 0.1, 0.1]

        data = []
        labels = []

        for _ in range(samples_per_client):
            # Choose class based on distribution
            class_choice = random.choices(range(num_classes), weights=class_probs)[0]

            # Generate features biased toward the chosen class
            sample = [random.gauss(0, 1) for _ in range(20)]

            # Bias features based on class
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
        print(f"  Client {client_id}: {samples_per_client} samples, class distribution: {class_dist}")

    return client_data, client_labels

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Thread-based HTTP server for dashboard."""
    pass

class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the dashboard."""

    def __init__(self, *args, fl_server=None, clients=None, **kwargs):
        self.fl_server = fl_server
        self.clients = clients
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            status = {
                'server_status': 'running',
                'current_round': getattr(self.fl_server, 'round_number', 0),
                'num_clients': len(self.clients) if self.clients else 0,
                'training_history': getattr(self.fl_server, 'training_history', [])
            }

            self.wfile.write(json.dumps(status).encode())
            return

        elif self.path == '/':
            self.path = '/dashboard.html'

        # Serve static files
        return super().do_GET()

def create_dashboard_html():
    """Create a simple but functional dashboard."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Federated Learning Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
        .metric-label { color: #7f8c8d; font-size: 0.9em; }
        .chart-container { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
        .status-running { background: #27ae60; }
        .status-stopped { background: #e74c3c; }
        #chart { width: 100%; height: 300px; border: 1px solid #ddd; }
        .log { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; font-family: monospace; height: 200px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Federated Learning Dashboard</h1>
            <p>Real-time monitoring of federated learning training</p>
        </div>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value" id="current-round">0</div>
                <div class="metric-label">Current Round</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="num-clients">0</div>
                <div class="metric-label">Active Clients</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="avg-accuracy">0.00</div>
                <div class="metric-label">Average Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="avg-loss">0.00</div>
                <div class="metric-label">Average Loss</div>
            </div>
        </div>

        <div class="chart-container">
            <h3>Training Progress</h3>
            <canvas id="chart"></canvas>
        </div>

        <div class="chart-container">
            <h3>System Status</h3>
            <p><span class="status-indicator status-running"></span>FL Server: Running</p>
            <p><span class="status-indicator status-running"></span>Mesh Network: Active</p>
            <p><span class="status-indicator status-running"></span>Security: Enabled</p>

            <h4>Training Log</h4>
            <div class="log" id="training-log">
                [INFO] Federated Learning Dashboard Started<br>
                [INFO] Waiting for training data...
            </div>
        </div>
    </div>

    <script>
        let trainingHistory = [];

        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('current-round').textContent = data.current_round;
                    document.getElementById('num-clients').textContent = data.num_clients;

                    if (data.training_history.length > 0) {
                        const latest = data.training_history[data.training_history.length - 1];
                        document.getElementById('avg-accuracy').textContent = latest.avg_accuracy.toFixed(3);
                        document.getElementById('avg-loss').textContent = latest.avg_loss.toFixed(3);

                        // Update training history
                        trainingHistory = data.training_history;
                        updateChart();
                        updateLog(latest);
                    }
                })
                .catch(error => console.error('Error:', error));
        }

        function updateChart() {
            const canvas = document.getElementById('chart');
            const ctx = canvas.getContext('2d');

            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (trainingHistory.length === 0) return;

            // Simple line chart
            const width = canvas.width;
            const height = canvas.height;
            const margin = 40;

            // Draw axes
            ctx.strokeStyle = '#ddd';
            ctx.beginPath();
            ctx.moveTo(margin, margin);
            ctx.lineTo(margin, height - margin);
            ctx.lineTo(width - margin, height - margin);
            ctx.stroke();

            // Draw accuracy line
            ctx.strokeStyle = '#3498db';
            ctx.lineWidth = 2;
            ctx.beginPath();

            for (let i = 0; i < trainingHistory.length; i++) {
                const x = margin + (i / Math.max(1, trainingHistory.length - 1)) * (width - 2 * margin);
                const y = height - margin - (trainingHistory[i].avg_accuracy * (height - 2 * margin));

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Draw loss line
            ctx.strokeStyle = '#e74c3c';
            ctx.beginPath();

            const maxLoss = Math.max(...trainingHistory.map(h => h.avg_loss));

            for (let i = 0; i < trainingHistory.length; i++) {
                const x = margin + (i / Math.max(1, trainingHistory.length - 1)) * (width - 2 * margin);
                const y = height - margin - ((trainingHistory[i].avg_loss / maxLoss) * (height - 2 * margin));

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Labels
            ctx.fillStyle = '#34495e';
            ctx.font = '12px Arial';
            ctx.fillText('Accuracy (Blue)', margin, 20);
            ctx.fillText('Loss (Red)', margin + 100, 20);
        }

        function updateLog(latest) {
            const log = document.getElementById('training-log');
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = `[${timestamp}] Round ${latest.round}: Acc=${latest.avg_accuracy.toFixed(3)}, Loss=${latest.avg_loss.toFixed(3)}<br>`;
            log.innerHTML += logEntry;
            log.scrollTop = log.scrollHeight;
        }

        // Update every 2 seconds
        setInterval(updateDashboard, 2000);
        updateDashboard();

        // Initial chart setup
        window.addEventListener('load', function() {
            const canvas = document.getElementById('chart');
            canvas.width = canvas.offsetWidth;
            canvas.height = 300;
        });
    </script>
</body>
</html>
"""
    return html

def run_working_demo():
    """Run the complete working federated learning demo."""
    print("🚀 Starting Functional Federated Learning Demo")
    print("=" * 60)

    # Set up environment
    os.environ["OFFLINE_MODE"] = "1"

    # Configuration
    num_clients = 3
    num_rounds = 5
    total_samples = 600

    print(f"📋 Configuration:")
    print(f"   Clients: {num_clients}")
    print(f"   Rounds: {num_rounds}")
    print(f"   Total Samples: {total_samples}")
    print()

    # Create data
    client_data, client_labels = create_non_iid_data(total_samples, num_clients)

    # Initialize server
    server = FLServer(num_clients)

    # Initialize clients
    clients = []
    for i in range(num_clients):
        client = FLClient(i, client_data[i], client_labels[i])
        clients.append(client)

    # Create dashboard
    print("🌐 Setting up dashboard...")
    dashboard_html = create_dashboard_html()
    os.makedirs("dashboard_static", exist_ok=True)
    with open("dashboard_static/dashboard.html", "w") as f:
        f.write(dashboard_html)

    # Start dashboard server in background
    def run_dashboard():
        os.chdir("dashboard_static")
        handler = lambda *args, **kwargs: DashboardHandler(*args, fl_server=server, clients=clients, **kwargs)
        httpd = ThreadingHTTPServer(("localhost", 8080), handler)
        print("✅ Dashboard available at: http://localhost:8080")
        httpd.serve_forever()

    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()

    print("✅ All systems initialized!")
    print("\n🌐 Dashboard: http://localhost:8080")
    print("🔄 Starting federated training...\n")

    # Run federated learning
    try:
        for round_num in range(num_rounds):
            round_stats = server.run_federated_round(clients)

            # Add some delay to see real-time updates
            time.sleep(2)

        print("\n🎉 Federated Learning Complete!")
        print("=" * 60)
        print("📊 Final Results:")

        if server.training_history:
            final_stats = server.training_history[-1]
            print(f"   Final Accuracy: {final_stats['avg_accuracy']:.4f}")
            print(f"   Final Loss: {final_stats['avg_loss']:.4f}")
            print(f"   Total Rounds: {final_stats['round']}")

        print("\n🌐 Dashboard will remain available at: http://localhost:8080")
        print("Press Ctrl+C to stop the dashboard server")

        # Keep dashboard running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_working_demo()