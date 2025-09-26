"""
Federated Learning Server with Off-Guard Security and UIOTA Mesh Integration
"""

import argparse
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import flwr as fl
import torch
import numpy as np

from . import guard
from . import mesh_sync
from . import models
from . import utils
from .strategy_custom import OffGuardFedAvg, OffGuardFedProx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedServer:
    """Main federated learning server with security and mesh integration."""

    def __init__(self, args):
        self.args = args
        self.server_keypair = None
        self.artifacts_dir = Path("artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)

        # Initialize security
        self._init_security()

        # Initialize global model
        self._init_model()

    def _init_security(self):
        """Initialize cryptographic keys and security checks."""
        logger.info("Initializing Off-Guard security...")

        # Run preflight security checks
        guard.preflight_check()

        # Generate or load server keypair
        keypair_path = self.artifacts_dir / "server_keypair.pkl"
        if keypair_path.exists():
            logger.info("Loading existing server keypair...")
            with open(keypair_path, 'rb') as f:
                self.server_keypair = pickle.load(f)
        else:
            logger.info("Generating new server keypair...")
            self.server_keypair = guard.new_keypair()
            with open(keypair_path, 'wb') as f:
                pickle.dump(self.server_keypair, f)

        # Save public key for clients
        pub_key_path = self.artifacts_dir / "server_public_key.pkl"
        with open(pub_key_path, 'wb') as f:
            pickle.dump(self.server_keypair[1], f)  # public key

        logger.info("Server security initialized successfully")

    def _init_model(self):
        """Initialize the global model."""
        logger.info(f"Initializing global model for {self.args.dataset}...")

        if self.args.dataset == "mnist":
            self.global_model = models.SmallCNN(num_classes=10)
        elif self.args.dataset == "cifar10":
            self.global_model = models.CIFAR10CNN(num_classes=10)
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        # Set deterministic seed
        torch.manual_seed(42)
        np.random.seed(42)

        logger.info("Global model initialized")

    def run(self):
        """Start the federated learning server."""
        logger.info("Starting Federated Learning Server...")
        logger.info(f"Dataset: {self.args.dataset}")
        logger.info(f"Strategy: {self.args.strategy}")
        logger.info(f"Rounds: {self.args.rounds}")
        logger.info(f"Clients per round: {self.args.clients_per_round}")

        # Choose aggregation strategy
        if self.args.strategy == "fedavg":
            strategy = OffGuardFedAvg(
                server_keypair=self.server_keypair,
                initial_parameters=fl.common.ndarrays_to_parameters(
                    [val.cpu().numpy() for val in self.global_model.state_dict().values()]
                ),
                min_fit_clients=max(2, self.args.clients_per_round // 2),
                min_evaluate_clients=max(2, self.args.clients_per_round // 2),
                min_available_clients=self.args.clients_per_round,
                evaluate_fn=self._get_evaluate_fn(),
            )
        elif self.args.strategy == "fedprox":
            strategy = OffGuardFedProx(
                server_keypair=self.server_keypair,
                initial_parameters=fl.common.ndarrays_to_parameters(
                    [val.cpu().numpy() for val in self.global_model.state_dict().values()]
                ),
                min_fit_clients=max(2, self.args.clients_per_round // 2),
                min_evaluate_clients=max(2, self.args.clients_per_round // 2),
                min_available_clients=self.args.clients_per_round,
                evaluate_fn=self._get_evaluate_fn(),
                proximal_mu=0.1,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.args.strategy}")

        # Configure Flower server
        config = fl.server.ServerConfig(num_rounds=self.args.rounds)

        # Start server
        server_address = self.args.server_address.split(':')
        host, port = server_address[0], int(server_address[1])

        logger.info(f"Server starting on {host}:{port}")
        fl.server.start_server(
            server_address=f"{host}:{port}",
            config=config,
            strategy=strategy,
        )

    def _get_evaluate_fn(self):
        """Return evaluation function for server-side evaluation."""
        def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
            # This would implement server-side evaluation
            # For now, we rely on client-side evaluation
            return None
        return evaluate


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Federated Learning Server")

    # Core FL settings
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--clients-per-round", type=int, default=10, help="Clients per round")
    parser.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg", help="FL strategy")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist", help="Dataset")
    parser.add_argument("--server-address", default="localhost:8080", help="Server address")

    # Privacy settings
    parser.add_argument("--dp", choices=["on", "off"], default="off", help="Differential Privacy")

    # Network simulation
    parser.add_argument("--latency-ms", type=int, default=50, help="Base latency (ms)")
    parser.add_argument("--jitter-ms", type=int, default=25, help="Latency jitter (ms)")
    parser.add_argument("--dropout-pct", type=float, default=0.1, help="Client dropout probability")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    server = FederatedServer(args)
    server.run()