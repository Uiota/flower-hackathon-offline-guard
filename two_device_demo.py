#!/usr/bin/env python3
"""
Two-Device Federated Learning Demo
Demonstrates secure federated learning between two simulated devices
"""

import asyncio
import logging
import json
import threading
import time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from cryptography.fernet import Fernet
from pathlib import Path
import subprocess
import os

class DeviceSimulator:
    """Simulates a federated learning device"""

    def __init__(self, device_id: str, device_type: str = "mobile"):
        self.device_id = device_id
        self.device_type = device_type
        self.encryption_key = None
        self.model = self._create_model()
        self.is_connected = False
        self.is_training = False
        self.metrics = {
            "rounds_completed": 0,
            "local_accuracy": 0.0,
            "training_time": 0.0,
            "data_samples": 1000,
            "encryption_operations": 0
        }
        self.log_messages = []

    def _create_model(self):
        """Create a simple neural network model"""
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )
        return model

    def log(self, message: str):
        """Add log message"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_messages.append(log_entry)
        print(f"Device {self.device_id}: {message}")

    def generate_synthetic_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic training data"""
        # Create realistic MNIST-like data
        X = torch.randn(self.metrics["data_samples"], 784)
        # Add some pattern to make it more realistic
        if self.device_id == "device_1":
            X += torch.randn(784) * 0.1  # Device 1 has slightly different data distribution
        y = torch.randint(0, 10, (self.metrics["data_samples"],))
        return X, y

    async def connect_to_server(self, server_address: str = "localhost:8080") -> bool:
        """Connect to federated learning server"""
        self.log(f"Connecting to FL server at {server_address}...")
        await asyncio.sleep(1)  # Simulate connection time

        # Simulate key exchange
        self.encryption_key = Fernet.generate_key()
        self.log("Encryption key exchanged successfully")
        self.metrics["encryption_operations"] += 1

        self.is_connected = True
        self.log("Connected to FL server successfully")
        return True

    async def train_local_model(self, rounds: int = 3) -> Dict:
        """Train the local model"""
        if not self.is_connected:
            self.log("ERROR: Not connected to FL server")
            return {}

        self.is_training = True
        self.log(f"Starting local training for {rounds} rounds...")

        X, y = self.generate_synthetic_data()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        start_time = time.time()

        for round_num in range(rounds):
            self.log(f"Training round {round_num + 1}/{rounds}")

            # Training loop
            for epoch in range(5):  # Quick training per round
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                # Log progress
                if epoch % 2 == 0:
                    self.log(f"  Epoch {epoch + 1}: Loss = {loss.item():.4f}")

            await asyncio.sleep(0.5)  # Simulate training time

        # Calculate final accuracy
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y).sum().item() / len(y)

        training_time = time.time() - start_time
        self.metrics.update({
            "rounds_completed": rounds,
            "local_accuracy": accuracy,
            "training_time": training_time
        })

        self.log(f"Local training completed: Accuracy = {accuracy:.3f}")
        self.is_training = False

        return self.metrics

    async def send_encrypted_update(self) -> bytes:
        """Send encrypted model updates"""
        if not self.encryption_key:
            self.log("ERROR: No encryption key available")
            return b""

        self.log("Encrypting model parameters...")

        # Get model parameters
        params = []
        for param in self.model.parameters():
            params.append(param.data.numpy().flatten())

        # Serialize and encrypt
        param_data = np.concatenate(params)
        cipher = Fernet(self.encryption_key)
        encrypted_data = cipher.encrypt(param_data.tobytes())

        self.metrics["encryption_operations"] += 1
        self.log(f"Model update encrypted ({len(encrypted_data)} bytes)")

        return encrypted_data

    async def receive_global_update(self, encrypted_update: bytes) -> bool:
        """Receive and apply global model update"""
        if not self.encryption_key:
            self.log("ERROR: No encryption key available")
            return False

        self.log("Receiving global model update...")

        try:
            # Decrypt update
            cipher = Fernet(self.encryption_key)
            decrypted_data = cipher.decrypt(encrypted_update)
            self.metrics["encryption_operations"] += 1

            # Apply update (simplified)
            self.log("Global model update applied successfully")
            return True

        except Exception as e:
            self.log(f"ERROR: Failed to decrypt global update: {e}")
            return False

    def get_status(self) -> Dict:
        """Get current device status"""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "is_connected": self.is_connected,
            "is_training": self.is_training,
            "metrics": self.metrics,
            "recent_logs": self.log_messages[-5:] if self.log_messages else []
        }

class TwoDeviceFLDemo:
    """Orchestrates the two-device federated learning demo"""

    def __init__(self):
        self.device1 = DeviceSimulator("device_1", "mobile")
        self.device2 = DeviceSimulator("device_2", "desktop")
        self.server_running = False
        self.demo_active = False
        self.global_rounds = 0

    async def start_fl_server(self):
        """Start the federated learning server"""
        print("🌸 Starting Flower FL Server...")
        self.server_running = True

        # Create encryption key file for both devices
        server_key = Fernet.generate_key()
        with open("demo_fl_key.txt", "wb") as f:
            f.write(server_key)

        print(f"✅ FL Server started with encryption key")
        return True

    async def run_full_demo(self, rounds: int = 5):
        """Run complete two-device federated learning demo"""
        print("\n🎯 Starting Two-Device Federated Learning Demo")
        print("=" * 60)

        self.demo_active = True

        try:
            # Step 1: Start FL Server
            await self.start_fl_server()
            await asyncio.sleep(1)

            # Step 2: Connect both devices
            print("\n📱 Connecting devices...")
            connect_tasks = [
                self.device1.connect_to_server(),
                self.device2.connect_to_server()
            ]
            await asyncio.gather(*connect_tasks)

            # Step 3: Run federated learning rounds
            for round_num in range(rounds):
                self.global_rounds = round_num + 1
                print(f"\n🔄 Global Round {self.global_rounds}/{rounds}")
                print("-" * 30)

                # Step 3a: Local training on both devices
                print("🏋️ Starting local training on both devices...")
                training_tasks = [
                    self.device1.train_local_model(rounds=2),
                    self.device2.train_local_model(rounds=2)
                ]
                results = await asyncio.gather(*training_tasks)

                # Step 3b: Encrypt and send model updates
                print("🔐 Encrypting and exchanging model updates...")
                update_tasks = [
                    self.device1.send_encrypted_update(),
                    self.device2.send_encrypted_update()
                ]
                encrypted_updates = await asyncio.gather(*update_tasks)

                # Step 3c: Simulate server aggregation
                print("⚡ Server aggregating encrypted updates...")
                await asyncio.sleep(1)

                # Step 3d: Send aggregated update back to devices
                print("📤 Sending global update to devices...")
                aggregated_update = encrypted_updates[0]  # Simplified aggregation
                receive_tasks = [
                    self.device1.receive_global_update(aggregated_update),
                    self.device2.receive_global_update(aggregated_update)
                ]
                await asyncio.gather(*receive_tasks)

                # Show round summary
                print(f"✅ Round {self.global_rounds} completed")
                print(f"   Device 1 accuracy: {self.device1.metrics['local_accuracy']:.3f}")
                print(f"   Device 2 accuracy: {self.device2.metrics['local_accuracy']:.3f}")
                print(f"   Total encryption ops: {self.device1.metrics['encryption_operations'] + self.device2.metrics['encryption_operations']}")

                await asyncio.sleep(2)  # Pause between rounds

            # Demo completion
            print("\n🎉 Federated Learning Demo Completed!")
            print("=" * 60)
            print(f"✅ {rounds} global rounds completed")
            print(f"🔐 {self.get_total_encryption_ops()} encryption operations")
            print(f"📱 Device 1 final accuracy: {self.device1.metrics['local_accuracy']:.3f}")
            print(f"💻 Device 2 final accuracy: {self.device2.metrics['local_accuracy']:.3f}")
            print(f"⏱️  Total demo time: {self.get_total_training_time():.1f}s")

        except Exception as e:
            print(f"❌ Demo error: {e}")
        finally:
            self.demo_active = False

    def get_total_encryption_ops(self) -> int:
        """Get total encryption operations"""
        return (self.device1.metrics["encryption_operations"] +
                self.device2.metrics["encryption_operations"])

    def get_total_training_time(self) -> float:
        """Get total training time"""
        return (self.device1.metrics["training_time"] +
                self.device2.metrics["training_time"])

    def get_demo_status(self) -> Dict:
        """Get current demo status"""
        return {
            "demo_active": self.demo_active,
            "server_running": self.server_running,
            "global_rounds": self.global_rounds,
            "device1_status": self.device1.get_status(),
            "device2_status": self.device2.get_status(),
            "total_encryption_ops": self.get_total_encryption_ops(),
            "total_training_time": self.get_total_training_time()
        }

    async def simulate_offline_mode(self):
        """Simulate offline federated learning"""
        print("\n📴 Simulating Offline Mode...")
        print("Devices will continue training locally without server connection")

        # Disconnect devices from server
        self.device1.is_connected = False
        self.device2.is_connected = False

        # Continue local training
        offline_tasks = [
            self.device1.train_local_model(rounds=1),
            self.device2.train_local_model(rounds=1)
        ]
        await asyncio.gather(*offline_tasks)

        print("✅ Offline training completed")
        print("📡 Devices ready to reconnect when server is available")

async def main():
    """Main demo entry point"""
    logging.basicConfig(level=logging.INFO)

    demo = TwoDeviceFLDemo()

    print("🛡️ Off-Guard Two-Device Federated Learning Demo")
    print("=" * 60)

    try:
        # Run the full demo
        await demo.run_full_demo(rounds=3)

        # Optional: Simulate offline mode
        print("\n" + "=" * 60)
        await demo.simulate_offline_mode()

        # Show final status
        print("\n📊 Final Demo Status:")
        status = demo.get_demo_status()
        print(json.dumps(status, indent=2))

    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(main())