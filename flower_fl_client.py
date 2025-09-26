#!/usr/bin/env python3
"""
Flower Federated Learning Client with Off-Guard Integration
Secure federated learning client with encrypted offline communication
"""

import logging
import asyncio
from typing import Dict, List, Tuple
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cryptography.fernet import Fernet
import json
import os
from pathlib import Path

class SimpleModel(nn.Module):
    """Simple neural network for demonstration"""

    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class OffGuardFLClient(fl.client.NumPyClient):
    """Secure Off-Guard Federated Learning Client"""

    def __init__(self, client_id: str, model: nn.Module, encryption_key: bytes = None):
        self.client_id = client_id
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Encryption setup
        self.encryption_key = encryption_key
        if self.encryption_key:
            self.cipher = Fernet(self.encryption_key)

        # Generate synthetic data for demo
        self.train_loader, self.test_loader = self._create_demo_data()

        # Training setup
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.metrics = {
            "client_id": client_id,
            "rounds_participated": 0,
            "local_accuracy": 0.0,
            "data_samples": 1000
        }

    def _create_demo_data(self) -> Tuple[DataLoader, DataLoader]:
        """Create synthetic data for demonstration"""
        # Generate random data for demo
        train_data = torch.randn(1000, 784)
        train_labels = torch.randint(0, 10, (1000,))
        test_data = torch.randn(200, 784)
        test_labels = torch.randint(0, 10, (200,))

        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get model parameters"""
        params = [param.cpu().detach().numpy() for param in self.model.parameters()]

        # Encrypt parameters if encryption is enabled
        if self.encryption_key:
            params = self._encrypt_parameters(params)

        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters"""
        try:
            # Decrypt parameters if encryption is enabled
            if self.encryption_key:
                parameters = self._decrypt_parameters(parameters)

            # Set parameters
            params_dict = zip(self.model.parameters(), parameters)
            for param, new_param in params_dict:
                param.data.copy_(torch.tensor(new_param))

        except Exception as e:
            logging.error(f"Failed to set parameters: {e}")

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model"""
        self.set_parameters(parameters)

        # Local training
        self.model.train()
        for epoch in range(2):  # Quick training for demo
            for batch_data, batch_labels in self.train_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

        # Update metrics
        self.metrics["rounds_participated"] += 1
        self.metrics["local_accuracy"] = self._evaluate_model()

        num_examples = len(self.train_loader.dataset)
        return self.get_parameters({}), num_examples, self.metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model"""
        self.set_parameters(parameters)
        loss, accuracy = self._evaluate_model()

        num_examples = len(self.test_loader.dataset)
        return float(loss), num_examples, {"accuracy": accuracy}

    def _evaluate_model(self) -> Tuple[float, float]:
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data, batch_labels in self.test_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(self.test_loader)
        return avg_loss, accuracy

    def _encrypt_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Encrypt model parameters"""
        try:
            serialized = json.dumps([param.tolist() for param in parameters])
            encrypted_data = self.cipher.encrypt(serialized.encode())
            return [encrypted_data]
        except Exception as e:
            logging.error(f"Parameter encryption failed: {e}")
            return parameters

    def _decrypt_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Decrypt model parameters"""
        try:
            if len(parameters) == 1 and isinstance(parameters[0], bytes):
                decrypted_data = self.cipher.decrypt(parameters[0])
                param_lists = json.loads(decrypted_data.decode())
                return [np.array(param) for param in param_lists]
            return parameters
        except Exception as e:
            logging.error(f"Parameter decryption failed: {e}")
            return parameters

def create_client(client_id: str, server_address: str = "localhost:8080") -> None:
    """Create and start a federated learning client"""
    logging.basicConfig(level=logging.INFO)

    # Load encryption key
    key_file = Path("fl_encryption_key.txt")
    encryption_key = None
    if key_file.exists():
        with open(key_file, "rb") as f:
            encryption_key = f.read()

    # Create model and client
    model = SimpleModel()
    client = OffGuardFLClient(client_id=client_id, model=model, encryption_key=encryption_key)

    print(f"🌸 Off-Guard FL Client {client_id} Starting...")
    print(f"🔐 Encryption: {'Enabled' if encryption_key else 'Disabled'}")
    print(f"🌐 Server: {server_address}")

    try:
        # Start client
        fl.client.start_numpy_client(server_address=server_address, client=client)
    except Exception as e:
        print(f"❌ Client error: {e}")

def main():
    """Main client entry point"""
    import sys

    client_id = sys.argv[1] if len(sys.argv) > 1 else "client_1"
    server_address = sys.argv[2] if len(sys.argv) > 2 else "localhost:8080"

    create_client(client_id, server_address)

if __name__ == "__main__":
    main()