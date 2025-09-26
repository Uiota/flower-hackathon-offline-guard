"""
Federated Learning Client with Differential Privacy and Off-Guard Security
"""

import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Differential Privacy imports
try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

from . import datasets
from . import guard
from . import mesh_sync
from . import models
from . import utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedClient(fl.client.NumPyClient):
    """Federated Learning client with DP and security features."""

    def __init__(self, cid: str, args):
        self.cid = cid
        self.args = args
        self.server_public_key = None
        self.client_keypair = None

        # Initialize security
        self._init_security()

        # Initialize model and data
        self._init_model()
        self._init_data()

        # Initialize differential privacy if enabled
        self.privacy_engine = None
        if args.dp == "on" and OPACUS_AVAILABLE:
            self._init_privacy()

    def _init_security(self):
        """Initialize security components."""
        logger.info(f"Client {self.cid}: Initializing security...")

        # Run preflight checks
        guard.preflight_check()

        # Load server public key
        server_pubkey_path = Path("artifacts/server_public_key.pkl")
        if server_pubkey_path.exists():
            with open(server_pubkey_path, 'rb') as f:
                self.server_public_key = pickle.load(f)
        else:
            logger.warning("Server public key not found - signature verification disabled")

        # Generate client keypair
        self.client_keypair = guard.new_keypair()
        logger.info(f"Client {self.cid}: Security initialized")

    def _init_model(self):
        """Initialize model."""
        if self.args.dataset == "mnist":
            self.model = models.SmallCNN(num_classes=10)
        elif self.args.dataset == "cifar10":
            self.model = models.CIFAR10CNN(num_classes=10)
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)

    def _init_data(self):
        """Initialize dataset."""
        logger.info(f"Client {self.cid}: Loading {self.args.dataset} dataset...")

        if self.args.dataset == "mnist":
            self.train_loader, self.test_loader = datasets.get_mnist_client_data(
                client_id=int(self.cid) if self.cid.isdigit() else 0,
                num_clients=10,
                batch_size=self.args.batch_size,
                alpha=0.5  # Non-IID parameter
            )
        elif self.args.dataset == "cifar10":
            self.train_loader, self.test_loader = datasets.get_cifar10_client_data(
                client_id=int(self.cid) if self.cid.isdigit() else 0,
                num_clients=10,
                batch_size=self.args.batch_size,
                alpha=0.5
            )

        logger.info(f"Client {self.cid}: Data loaded - {len(self.train_loader)} training batches")

    def _init_privacy(self):
        """Initialize differential privacy."""
        if not OPACUS_AVAILABLE:
            logger.warning("Opacus not available - DP disabled")
            return

        logger.info(f"Client {self.cid}: Initializing Differential Privacy...")

        self.privacy_engine = PrivacyEngine()

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            epochs=self.args.epochs,
            target_epsilon=self.args.dp_epsilon,
            target_delta=self.args.dp_delta,
            max_grad_norm=self.args.dp_max_grad_norm,
        )

        logger.info(f"Client {self.cid}: DP initialized (ε={self.args.dp_epsilon}, δ={self.args.dp_delta})")

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> fl.common.NDArrays:
        """Return model parameters."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: fl.common.NDArrays) -> None:
        """Set model parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Tuple[fl.common.NDArrays, int, Dict[str, fl.common.Scalar]]:
        """Train the model on local data."""
        logger.info(f"Client {self.cid}: Starting training round")

        # Set parameters from server
        self.set_parameters(parameters)

        # Train model
        self.model.train()
        train_loss = 0.0
        train_samples = 0

        for epoch in range(self.args.epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * len(data)
                train_samples += len(data)

        avg_loss = train_loss / train_samples

        # Get updated parameters
        updated_parameters = self.get_parameters({})

        # Sign parameters if security enabled
        if self.client_keypair:
            # Serialize parameters for signing
            params_bytes = utils.serialize_parameters(updated_parameters)
            signature = guard.sign_blob(self.client_keypair[0], params_bytes)  # private key

            # Store signature in config for strategy to use
            config["signature"] = utils.bytes_to_base64(signature)
            config["client_public_key"] = utils.serialize_public_key(self.client_keypair[1])

        # Report privacy spent if DP enabled
        metrics = {"loss": avg_loss}
        if self.privacy_engine:
            epsilon = self.privacy_engine.get_epsilon(self.args.dp_delta)
            metrics["privacy_epsilon"] = epsilon
            logger.info(f"Client {self.cid}: Privacy spent ε = {epsilon:.3f}")

        logger.info(f"Client {self.cid}: Training complete - Loss: {avg_loss:.4f}")

        return updated_parameters, train_samples, metrics

    def evaluate(self, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        """Evaluate model on local test data."""
        self.set_parameters(parameters)

        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += self.criterion(output, target).item() * len(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(data)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = test_loss / total if total > 0 else float('inf')

        logger.info(f"Client {self.cid}: Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return avg_loss, total, {"accuracy": accuracy}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Federated Learning Client")

    # Client identification
    parser.add_argument("--cid", type=str, required=True, help="Client ID")
    parser.add_argument("--server-address", default="localhost:8080", help="Server address")

    # Training settings
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist", help="Dataset")
    parser.add_argument("--epochs", type=int, default=1, help="Local training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")

    # Differential Privacy settings
    parser.add_argument("--dp", choices=["on", "off"], default="off", help="Enable DP")
    parser.add_argument("--dp-epsilon", type=float, default=1.0, help="DP epsilon")
    parser.add_argument("--dp-delta", type=float, default=1e-5, help="DP delta")
    parser.add_argument("--dp-max-grad-norm", type=float, default=1.0, help="DP max grad norm")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Create and start client
    client = FederatedClient(args.cid, args)
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )