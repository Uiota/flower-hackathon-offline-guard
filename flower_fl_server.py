#!/usr/bin/env python3
"""
Flower Federated Learning Server with Off-Guard Integration
Secure federated learning with encrypted offline communication
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, EvaluateRes
import torch
import torch.nn as nn
from cryptography.fernet import Fernet
import json
import os
from pathlib import Path

class SecureOffGuardStrategy(FedAvg):
    """Custom Flower strategy with Off-Guard encryption"""

    def __init__(self, encryption_key: bytes = None, **kwargs):
        super().__init__(**kwargs)
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.client_metrics = {}

    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.client.Client, FitRes]], failures: List[Tuple[fl.client.Client, FitRes]]) -> Tuple[Optional[Parameters], Dict[str, any]]:
        """Aggregate model updates with encryption"""
        logging.info(f"Round {server_round}: Aggregating {len(results)} client updates")

        # Decrypt client updates
        decrypted_results = []
        for client, fit_res in results:
            try:
                # Decrypt parameters if they're encrypted
                if hasattr(fit_res, 'encrypted') and fit_res.encrypted:
                    decrypted_params = self._decrypt_parameters(fit_res.parameters)
                    fit_res.parameters = decrypted_params
                decrypted_results.append((client, fit_res))
            except Exception as e:
                logging.error(f"Failed to decrypt client update: {e}")

        # Standard aggregation
        aggregated_params, metrics = super().aggregate_fit(server_round, decrypted_results, failures)

        # Encrypt aggregated model for distribution
        if aggregated_params:
            encrypted_params = self._encrypt_parameters(aggregated_params)

        return encrypted_params, metrics

    def _encrypt_parameters(self, parameters: Parameters) -> Parameters:
        """Encrypt model parameters"""
        try:
            # Convert parameters to bytes and encrypt
            param_bytes = fl.common.parameters_to_ndarrays(parameters)
            serialized = json.dumps([arr.tolist() for arr in param_bytes])
            encrypted_data = self.cipher.encrypt(serialized.encode())

            # Create new Parameters object with encrypted data
            encrypted_params = fl.common.ndarrays_to_parameters([encrypted_data])
            return encrypted_params
        except Exception as e:
            logging.error(f"Encryption failed: {e}")
            return parameters

    def _decrypt_parameters(self, parameters: Parameters) -> Parameters:
        """Decrypt model parameters"""
        try:
            param_arrays = fl.common.parameters_to_ndarrays(parameters)
            encrypted_data = param_arrays[0]

            decrypted_data = self.cipher.decrypt(encrypted_data)
            param_lists = json.loads(decrypted_data.decode())

            import numpy as np
            param_arrays = [np.array(arr) for arr in param_lists]
            return fl.common.ndarrays_to_parameters(param_arrays)
        except Exception as e:
            logging.error(f"Decryption failed: {e}")
            return parameters

class OffGuardFLServer:
    """Off-Guard Federated Learning Server"""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.encryption_key = Fernet.generate_key()
        self.strategy = SecureOffGuardStrategy(encryption_key=self.encryption_key)
        self.server_metrics = {
            "rounds_completed": 0,
            "clients_connected": 0,
            "model_accuracy": 0.0,
            "encryption_status": "active"
        }

    async def start_server(self, num_rounds: int = 5):
        """Start the federated learning server"""
        logging.info(f"Starting Off-Guard FL Server on {self.host}:{self.port}")
        logging.info(f"Encryption key: {self.encryption_key.decode()}")

        # Save encryption key for clients
        key_file = Path("fl_encryption_key.txt")
        with open(key_file, "wb") as f:
            f.write(self.encryption_key)

        # Configure server
        config = fl.server.ServerConfig(num_rounds=num_rounds)

        try:
            # Start Flower server
            fl.server.start_server(
                server_address=f"{self.host}:{self.port}",
                config=config,
                strategy=self.strategy,
            )
        except Exception as e:
            logging.error(f"Server startup failed: {e}")

    def get_server_status(self) -> Dict:
        """Get current server status"""
        return {
            "status": "running",
            "host": self.host,
            "port": self.port,
            "encryption_enabled": True,
            "metrics": self.server_metrics,
            "strategy": "SecureOffGuardStrategy"
        }

def main():
    """Main server entry point"""
    logging.basicConfig(level=logging.INFO)

    server = OffGuardFLServer(host="0.0.0.0", port=8080)

    print("🌸 Off-Guard Flower FL Server Starting...")
    print(f"🔐 Encryption: Enabled")
    print(f"🌐 Address: {server.host}:{server.port}")

    try:
        asyncio.run(server.start_server(num_rounds=10))
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()