"""
Flask-based Federated Learning Server with WebSocket support
"""

import json
import time
import threading
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit, disconnect
import flwr as fl
from flwr.common import NDArrays, Scalar
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import SimpleClientManager

import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared.models import get_model, count_parameters
from shared.datasets import get_client_data
from shared.utils import (
    Config, MetricsCollector, setup_logging, Timer,
    generate_keypair, save_json, load_json
)

logger = logging.getLogger(__name__)


class CustomFedAvgStrategy(FedAvg):
    """Custom Federated Averaging strategy with metrics collection."""

    def __init__(self, metrics_collector: MetricsCollector, **kwargs):
        super().__init__(**kwargs)
        self.metrics_collector = metrics_collector
        self.round_num = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes] | Tuple[fl.server.client_proxy.ClientProxy, fl.server.client_proxy.EvaluateRes]]
    ) -> Tuple[Optional[NDArrays], Dict[str, Scalar]]:
        """Aggregate fit results and collect metrics."""

        if not results:
            return None, {}

        # Collect metrics from clients
        total_loss = 0.0
        total_samples = 0
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            samples = fit_res.num_examples
            loss = fit_res.metrics.get("loss", 0.0)

            total_loss += loss * samples
            total_samples += samples

            # Store client metrics
            self.metrics_collector.add_metric("client_loss", loss, server_round, client_id)
            self.metrics_collector.add_metric("client_samples", samples, server_round, client_id)

        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        self.metrics_collector.add_metric("server_loss", avg_loss, server_round)

        # Log round info
        logger.info(f"Round {server_round}: {len(results)} clients, avg_loss={avg_loss:.4f}")

        # Perform aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Add server metrics
        aggregated_metrics["server_loss"] = avg_loss
        aggregated_metrics["num_clients"] = len(results)
        aggregated_metrics["total_samples"] = total_samples

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes] | Tuple[fl.server.client_proxy.ClientProxy, fl.server.client_proxy.EvaluateRes]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results and collect metrics."""

        if not results:
            return None, {}

        # Collect evaluation metrics
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        for client_proxy, eval_res in results:
            client_id = client_proxy.cid
            samples = eval_res.num_examples
            loss = eval_res.loss
            accuracy = eval_res.metrics.get("accuracy", 0.0)

            total_loss += loss * samples
            total_accuracy += accuracy * samples
            total_samples += samples

            # Store client evaluation metrics
            self.metrics_collector.add_metric("client_eval_loss", loss, server_round, client_id)
            self.metrics_collector.add_metric("client_eval_accuracy", accuracy, server_round, client_id)

        # Calculate aggregated metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0

        self.metrics_collector.add_metric("server_eval_loss", avg_loss, server_round)
        self.metrics_collector.add_metric("server_eval_accuracy", avg_accuracy, server_round)

        logger.info(f"Round {server_round} evaluation: loss={avg_loss:.4f}, accuracy={avg_accuracy:.4f}")

        return avg_loss, {
            "accuracy": avg_accuracy,
            "num_clients": len(results),
            "total_samples": total_samples
        }


class FederatedLearningServer:
    """Main federated learning server."""

    def __init__(self, config: Config):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.is_running = False
        self.server_thread = None
        self.fl_server = None
        self.model = None

        # Initialize model
        self._init_model()

        # Generate server keypair
        self.private_key, self.public_key = generate_keypair()

        # Save server keys
        keys_dir = Path("artifacts")
        keys_dir.mkdir(exist_ok=True)

        with open(keys_dir / "server_private_key.pem", "wb") as f:
            f.write(self.private_key)

        with open(keys_dir / "server_public_key.pem", "wb") as f:
            f.write(self.public_key)

    def _init_model(self):
        """Initialize the global model."""
        model_name = self.config.get("model", "cnn")
        dataset = self.config.get("dataset", "mnist")
        self.model = get_model(model_name, dataset)

        logger.info(f"Initialized {model_name} model for {dataset}")
        logger.info(f"Model parameters: {count_parameters(self.model):,}")

    def get_model_parameters(self) -> NDArrays:
        """Get current model parameters."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def start_training(self) -> bool:
        """Start federated training."""
        if self.is_running:
            logger.warning("Training is already running")
            return False

        try:
            self.is_running = True

            # Create strategy
            strategy = CustomFedAvgStrategy(
                metrics_collector=self.metrics_collector,
                fraction_fit=self.config.get("fraction_fit", 1.0),
                fraction_evaluate=self.config.get("fraction_evaluate", 1.0),
                min_fit_clients=self.config.get("min_fit_clients", 2),
                min_evaluate_clients=self.config.get("min_evaluate_clients", 2),
                min_available_clients=self.config.get("min_available_clients", 2),
                initial_parameters=fl.common.ndarrays_to_parameters(self.get_model_parameters())
            )

            # Create server configuration
            server_config = fl.server.ServerConfig(
                num_rounds=self.config.get("num_rounds", 10)
            )

            # Start server in separate thread
            self.server_thread = threading.Thread(
                target=self._run_server,
                args=(strategy, server_config),
                daemon=True
            )
            self.server_thread.start()

            logger.info("Federated learning server started")
            return True

        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            self.is_running = False
            return False

    def _run_server(self, strategy, config):
        """Run the federated learning server."""
        try:
            # Start flower server
            fl.server.start_server(
                server_address=f"0.0.0.0:{self.config.get('port', 8080)}",
                config=config,
                strategy=strategy
            )
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.is_running = False

    def stop_training(self) -> bool:
        """Stop federated training."""
        if not self.is_running:
            logger.warning("Training is not running")
            return False

        try:
            self.is_running = False
            if self.server_thread and self.server_thread.is_alive():
                # Note: Flower server doesn't have a clean shutdown method
                # In production, you'd need to implement proper shutdown
                logger.info("Stopping federated learning server")
            return True
        except Exception as e:
            logger.error(f"Failed to stop training: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current server status."""
        return {
            "is_running": self.is_running,
            "model_parameters": count_parameters(self.model),
            "metrics": self.metrics_collector.to_dict(),
            "config": self.config.to_dict()
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return self.metrics_collector.to_dict()


# Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'federated-learning-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global server instance
fl_server = None


def init_server(config_path: str = None):
    """Initialize the federated learning server."""
    global fl_server

    # Load configuration
    if config_path and Path(config_path).exists():
        config = Config.from_file(config_path)
    else:
        # Default configuration
        config = Config({
            "model": "cnn",
            "dataset": "mnist",
            "num_rounds": 10,
            "port": 8080,
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
            "min_fit_clients": 2,
            "min_evaluate_clients": 2,
            "min_available_clients": 2
        })

    fl_server = FederatedLearningServer(config)
    logger.info("Federated learning server initialized")


# Web Routes
@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')


# REST API Routes
@app.route('/api/status')
def get_status():
    """Get server status."""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500

    return jsonify(fl_server.get_status())


@app.route('/api/start', methods=['POST'])
def start_training():
    """Start federated training."""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500

    success = fl_server.start_training()
    if success:
        # Broadcast status update
        socketio.emit('training_started', fl_server.get_status())
        return jsonify({"status": "started"})
    else:
        return jsonify({"error": "Failed to start training"}), 500


@app.route('/api/stop', methods=['POST'])
def stop_training():
    """Stop federated training."""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500

    success = fl_server.stop_training()
    if success:
        # Broadcast status update
        socketio.emit('training_stopped', fl_server.get_status())
        return jsonify({"status": "stopped"})
    else:
        return jsonify({"error": "Failed to stop training"}), 500


@app.route('/api/metrics')
def get_metrics():
    """Get training metrics."""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500

    return jsonify(fl_server.get_metrics())


@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
    """Get or update configuration."""
    if fl_server is None:
        return jsonify({"error": "Server not initialized"}), 500

    if request.method == 'GET':
        return jsonify(fl_server.config.to_dict())

    elif request.method == 'POST':
        try:
            new_config = request.get_json()
            fl_server.config.update(new_config)
            return jsonify({"status": "updated", "config": fl_server.config.to_dict()})
        except Exception as e:
            return jsonify({"error": str(e)}), 400


# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    if fl_server:
        emit('status_update', fl_server.get_status())


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('request_status')
def handle_status_request():
    """Handle status request."""
    if fl_server:
        emit('status_update', fl_server.get_status())


@socketio.on('request_metrics')
def handle_metrics_request():
    """Handle metrics request."""
    if fl_server:
        emit('metrics_update', fl_server.get_metrics())


# Background task to broadcast metrics
def background_metrics_broadcaster():
    """Broadcast metrics periodically."""
    while True:
        time.sleep(5)  # Broadcast every 5 seconds
        if fl_server and fl_server.is_running:
            socketio.emit('metrics_update', fl_server.get_metrics())
            socketio.emit('status_update', fl_server.get_status())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="FL Server Web Application")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--fl-port", type=int, default=8080, help="FL server port")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Initialize server
    init_server(args.config)

    # Update FL port if specified
    if fl_server:
        fl_server.config.set("port", args.fl_port)

    # Start background metrics broadcaster
    metrics_thread = threading.Thread(target=background_metrics_broadcaster, daemon=True)
    metrics_thread.start()

    # Run web application
    logger.info(f"Starting web application on {args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=False)