"""
Functional Federated Learning Client with Real ML Training
"""

import argparse
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import asyncio
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import flwr as fl
from flwr.common import NDArrays, Scalar

# Add parent directory to path for shared modules
sys.path.append(str(Path(__file__).parent.parent))

from shared.models import get_model, count_parameters
from shared.datasets import get_client_data
from shared.utils import (
    Config, MetricsCollector, setup_logging, Timer,
    generate_keypair, sign_data, save_json, get_device
)

logger = logging.getLogger(__name__)


class FunctionalFLClient(fl.client.NumPyClient):
    """Functional Federated Learning Client with actual ML training."""

    def __init__(self, client_id: str, config: Config):
        self.client_id = client_id
        self.config = config
        self.device = get_device()
        self.metrics_collector = MetricsCollector()

        # Security components
        self.private_key, self.public_key = generate_keypair()

        # ML components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None

        # Training state
        self.current_round = 0
        self.total_samples = 0

        # Initialize components
        self._init_model()
        self._init_data()
        self._init_training()

        logger.info(f"Client {self.client_id} initialized with {count_parameters(self.model):,} parameters")

    def _init_model(self):
        """Initialize the ML model."""
        model_name = self.config.get("model", "cnn")
        dataset = self.config.get("dataset", "mnist")

        self.model = get_model(model_name, dataset)
        self.model.to(self.device)

        logger.info(f"Client {self.client_id}: Initialized {model_name} model for {dataset}")

    def _init_data(self):
        """Initialize client's data partition."""
        dataset = self.config.get("dataset", "mnist")
        num_clients = self.config.get("num_clients", 10)
        batch_size = self.config.get("batch_size", 32)
        alpha = self.config.get("alpha", 0.5)
        data_path = self.config.get("data_path", "./data")

        # Convert client_id to integer for data partitioning
        client_idx = int(self.client_id) if self.client_id.isdigit() else hash(self.client_id) % num_clients

        self.train_loader, self.test_loader = get_client_data(
            dataset=dataset,
            client_id=client_idx,
            num_clients=num_clients,
            batch_size=batch_size,
            alpha=alpha,
            data_path=data_path
        )

        # Calculate total samples for this client
        self.total_samples = len(self.train_loader.dataset)

        logger.info(f"Client {self.client_id}: Data loaded - {self.total_samples} training samples")

    def _init_training(self):
        """Initialize training components."""
        lr = self.config.get("learning_rate", 0.01)
        momentum = self.config.get("momentum", 0.9)
        weight_decay = self.config.get("weight_decay", 0.0001)

        # Initialize optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()

        logger.info(f"Client {self.client_id}: Training components initialized (lr={lr})")

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return current model parameters."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from server."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on local data."""
        self.current_round = int(config.get("server_round", 0))

        logger.info(f"Client {self.client_id}: Starting training round {self.current_round}")

        # Set parameters from server
        self.set_parameters(parameters)

        # Train the model
        train_loss, train_samples = self._train_model()

        # Get updated parameters
        updated_parameters = self.get_parameters({})

        # Prepare metrics
        metrics = {
            "loss": train_loss,
            "samples": train_samples,
            "client_id": self.client_id,
            "round": self.current_round
        }

        # Store metrics
        self.metrics_collector.add_metric("train_loss", train_loss, self.current_round, self.client_id)
        self.metrics_collector.add_metric("train_samples", train_samples, self.current_round, self.client_id)

        logger.info(f"Client {self.client_id}: Round {self.current_round} complete - Loss: {train_loss:.4f}")

        return updated_parameters, train_samples, metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on local test data."""
        # Set parameters from server
        self.set_parameters(parameters)

        # Evaluate the model
        test_loss, test_accuracy, test_samples = self._evaluate_model()

        # Prepare metrics
        metrics = {
            "accuracy": test_accuracy,
            "client_id": self.client_id,
            "round": self.current_round
        }

        # Store metrics
        self.metrics_collector.add_metric("test_loss", test_loss, self.current_round, self.client_id)
        self.metrics_collector.add_metric("test_accuracy", test_accuracy, self.current_round, self.client_id)

        logger.info(f"Client {self.client_id}: Evaluation - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

        return test_loss, test_samples, metrics

    def _train_model(self) -> Tuple[float, int]:
        """Perform actual model training."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        num_epochs = self.config.get("local_epochs", 1)

        with Timer(f"Client {self.client_id} training") as timer:
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                epoch_samples = 0

                for batch_idx, (data, target) in enumerate(self.train_loader):
                    # Move data to device
                    data, target = data.to(self.device), target.to(self.device)

                    # Zero gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    output = self.model(data)
                    loss = self.criterion(output, target)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    # Accumulate metrics
                    batch_loss = loss.item() * len(data)
                    epoch_loss += batch_loss
                    epoch_samples += len(data)

                    # Log progress for long training
                    if batch_idx % 50 == 0:
                        logger.debug(f"Client {self.client_id}: Epoch {epoch+1}/{num_epochs}, "
                                   f"Batch {batch_idx}/{len(self.train_loader)}, "
                                   f"Loss: {loss.item():.4f}")

                total_loss += epoch_loss
                total_samples += epoch_samples

                logger.debug(f"Client {self.client_id}: Epoch {epoch+1} complete - "
                           f"Loss: {epoch_loss/epoch_samples:.4f}")

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        logger.info(f"Client {self.client_id}: Training completed in {timer.elapsed:.2f}s")

        return avg_loss, total_samples

    def _evaluate_model(self) -> Tuple[float, float, int]:
        """Perform actual model evaluation."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with Timer(f"Client {self.client_id} evaluation") as timer:
            with torch.no_grad():
                for data, target in self.test_loader:
                    # Move data to device
                    data, target = data.to(self.device), target.to(self.device)

                    # Forward pass
                    output = self.model(data)
                    loss = self.criterion(output, target)

                    # Accumulate metrics
                    total_loss += loss.item() * len(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total_samples += len(data)

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        accuracy = correct / total_samples if total_samples > 0 else 0.0

        logger.info(f"Client {self.client_id}: Evaluation completed in {timer.elapsed:.2f}s")

        return avg_loss, accuracy, total_samples

    def get_client_info(self) -> Dict:
        """Get client information and status."""
        return {
            "client_id": self.client_id,
            "device": str(self.device),
            "model_parameters": count_parameters(self.model),
            "training_samples": self.total_samples,
            "current_round": self.current_round,
            "config": self.config.to_dict(),
            "metrics": self.metrics_collector.to_dict()
        }


class ClientManager:
    """Manage the FL client lifecycle."""

    def __init__(self, client_id: str, config: Config):
        self.client_id = client_id
        self.config = config
        self.client = None
        self.is_running = False
        self.client_thread = None

    def start_client(self, server_address: str) -> bool:
        """Start the FL client."""
        if self.is_running:
            logger.warning(f"Client {self.client_id} is already running")
            return False

        try:
            # Initialize the FL client
            self.client = FunctionalFLClient(self.client_id, self.config)

            # Start client in separate thread
            self.client_thread = threading.Thread(
                target=self._run_client,
                args=(server_address,),
                daemon=True
            )

            self.is_running = True
            self.client_thread.start()

            logger.info(f"Client {self.client_id} started, connecting to {server_address}")
            return True

        except Exception as e:
            logger.error(f"Failed to start client {self.client_id}: {e}")
            self.is_running = False
            return False

    def _run_client(self, server_address: str):
        """Run the FL client."""
        try:
            # Connect to the FL server
            fl.client.start_numpy_client(
                server_address=server_address,
                client=self.client
            )
        except Exception as e:
            logger.error(f"Client {self.client_id} error: {e}")
        finally:
            self.is_running = False
            logger.info(f"Client {self.client_id} stopped")

    def stop_client(self) -> bool:
        """Stop the FL client."""
        if not self.is_running:
            logger.warning(f"Client {self.client_id} is not running")
            return False

        try:
            self.is_running = False
            # Note: Flower client doesn't have a clean shutdown method
            # In production, you'd need to implement proper shutdown
            logger.info(f"Stopping client {self.client_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop client {self.client_id}: {e}")
            return False

    def get_status(self) -> Dict:
        """Get client status."""
        status = {
            "client_id": self.client_id,
            "is_running": self.is_running,
            "config": self.config.to_dict()
        }

        if self.client:
            status.update(self.client.get_client_info())

        return status


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Functional FL Client")

    # Client identification
    parser.add_argument("--client-id", required=True, help="Unique client identifier")
    parser.add_argument("--server-address", default="localhost:8080", help="FL server address")

    # Dataset and model configuration
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "synthetic"], default="mnist", help="Dataset")
    parser.add_argument("--model", choices=["cnn", "linear"], default="cnn", help="Model type")

    # Training configuration
    parser.add_argument("--local-epochs", type=int, default=1, help="Local training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay")

    # Data configuration
    parser.add_argument("--num-clients", type=int, default=10, help="Total number of clients")
    parser.add_argument("--alpha", type=float, default=0.5, help="Non-IID parameter")
    parser.add_argument("--data-path", default="./data", help="Data storage path")

    # System configuration
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--config-file", help="Configuration file path")

    return parser.parse_args()


def main():
    """Main function to run the FL client."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)

    # Create configuration
    config = Config()

    # Load configuration from file if provided
    if args.config_file and Path(args.config_file).exists():
        file_config = Config.from_file(args.config_file)
        config.update(file_config.to_dict())

    # Update with command line arguments
    config.update({
        "dataset": args.dataset,
        "model": args.model,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "num_clients": args.num_clients,
        "alpha": args.alpha,
        "data_path": args.data_path
    })

    # Create and start client manager
    client_manager = ClientManager(args.client_id, config)

    logger.info(f"Starting FL client {args.client_id}")
    logger.info(f"Configuration: {config.to_dict()}")

    success = client_manager.start_client(args.server_address)

    if success:
        try:
            # Keep the main thread alive
            while client_manager.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")

        client_manager.stop_client()
    else:
        logger.error("Failed to start client")
        return 1

    logger.info("Client shutdown complete")
    return 0


if __name__ == "__main__":
    exit(main())