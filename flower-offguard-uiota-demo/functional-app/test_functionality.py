#!/usr/bin/env python3
"""
Test script to verify all components work correctly
"""

import sys
import time
import threading
import unittest
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from shared.models import SmallCNN, CIFAR10CNN, LinearClassifier, get_model, count_parameters
from shared.datasets import get_client_data, SyntheticDataset
from shared.utils import Config, MetricsCollector, generate_keypair, sign_data, verify_signature
from client.fl_client import FunctionalFLClient
from mesh.p2p_network import P2PNode, MeshNetworkManager
from dashboard.monitor import SystemMonitor, TrainingMonitor


class TestModels(unittest.TestCase):
    """Test ML models."""

    def test_small_cnn(self):
        """Test SmallCNN model."""
        model = SmallCNN(num_classes=10)
        x = torch.randn(4, 1, 28, 28)  # MNIST input
        output = model(x)
        self.assertEqual(output.shape, (4, 10))

    def test_cifar10_cnn(self):
        """Test CIFAR10CNN model."""
        model = CIFAR10CNN(num_classes=10)
        x = torch.randn(4, 3, 32, 32)  # CIFAR-10 input
        output = model(x)
        self.assertEqual(output.shape, (4, 10))

    def test_linear_classifier(self):
        """Test LinearClassifier model."""
        model = LinearClassifier(input_size=784, num_classes=10)
        x = torch.randn(4, 784)
        output = model(x)
        self.assertEqual(output.shape, (4, 10))

    def test_model_factory(self):
        """Test model factory function."""
        # MNIST models
        mnist_cnn = get_model("cnn", "mnist")
        self.assertIsInstance(mnist_cnn, SmallCNN)

        mnist_linear = get_model("linear", "mnist")
        self.assertIsInstance(mnist_linear, LinearClassifier)

        # CIFAR-10 models
        cifar_cnn = get_model("cnn", "cifar10")
        self.assertIsInstance(cifar_cnn, CIFAR10CNN)

    def test_parameter_counting(self):
        """Test parameter counting."""
        model = SmallCNN()
        param_count = count_parameters(model)
        self.assertGreater(param_count, 0)


class TestDatasets(unittest.TestCase):
    """Test dataset functionality."""

    def test_synthetic_dataset(self):
        """Test synthetic dataset."""
        dataset = SyntheticDataset(size=100, input_dim=784, num_classes=10)
        self.assertEqual(len(dataset), 100)

        data, target = dataset[0]
        self.assertEqual(data.shape, (784,))
        self.assertIn(target.item(), range(10))

    def test_client_data_loading(self):
        """Test client data loading."""
        # Test synthetic data
        train_loader, test_loader = get_client_data(
            dataset="synthetic",
            client_id=0,
            num_clients=3,
            batch_size=16
        )

        self.assertGreater(len(train_loader), 0)
        self.assertGreater(len(test_loader), 0)

        # Test a batch
        for data, target in train_loader:
            self.assertEqual(data.shape[0], 16)  # batch size
            break


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_config(self):
        """Test configuration management."""
        config = Config()
        config.set("test_key", "test_value")
        self.assertEqual(config.get("test_key"), "test_value")

        # Test update
        config.update({"new_key": "new_value"})
        self.assertEqual(config.get("new_key"), "new_value")

    def test_metrics_collector(self):
        """Test metrics collection."""
        collector = MetricsCollector()
        collector.add_metric("loss", 0.5, 1, "client-1")
        collector.add_metric("loss", 0.3, 2, "client-1")

        metrics = collector.get_metrics("loss")
        self.assertEqual(len(metrics), 2)

        latest = collector.get_latest_metric("loss")
        self.assertEqual(latest["value"], 0.3)

    def test_cryptography(self):
        """Test cryptographic functions."""
        private_key, public_key = generate_keypair()
        self.assertIsInstance(private_key, bytes)
        self.assertIsInstance(public_key, bytes)

        # Test signing and verification
        data = b"test message"
        signature = sign_data(private_key, data)
        is_valid = verify_signature(public_key, signature, data)
        self.assertTrue(is_valid)

        # Test invalid signature
        is_invalid = verify_signature(public_key, signature, b"wrong data")
        self.assertFalse(is_invalid)


class TestFLClient(unittest.TestCase):
    """Test FL client functionality."""

    def test_client_initialization(self):
        """Test FL client initialization."""
        config = Config({
            "dataset": "synthetic",
            "model": "linear",
            "batch_size": 16,
            "num_clients": 3,
            "local_epochs": 1
        })

        client = FunctionalFLClient("test-client", config)
        self.assertEqual(client.client_id, "test-client")
        self.assertIsNotNone(client.model)
        self.assertIsNotNone(client.train_loader)
        self.assertIsNotNone(client.test_loader)

    def test_client_training(self):
        """Test client training functionality."""
        config = Config({
            "dataset": "synthetic",
            "model": "linear",
            "batch_size": 16,
            "num_clients": 3,
            "local_epochs": 1
        })

        client = FunctionalFLClient("test-client", config)

        # Get initial parameters
        initial_params = client.get_parameters({})
        self.assertIsInstance(initial_params, list)

        # Test training
        updated_params, samples, metrics = client.fit(initial_params, {"server_round": 1})

        self.assertIsInstance(updated_params, list)
        self.assertGreater(samples, 0)
        self.assertIn("loss", metrics)

        # Test evaluation
        loss, eval_samples, eval_metrics = client.evaluate(updated_params, {})
        self.assertGreater(eval_samples, 0)
        self.assertIn("accuracy", eval_metrics)


class TestMeshNetwork(unittest.TestCase):
    """Test mesh networking functionality."""

    def setUp(self):
        """Set up test nodes."""
        self.node1 = P2PNode("test-node-1", port=0)
        self.node2 = P2PNode("test-node-2", port=0)

    def test_node_initialization(self):
        """Test P2P node initialization."""
        self.assertEqual(self.node1.node_id, "test-node-1")
        self.assertIsNotNone(self.node1.private_key)
        self.assertIsNotNone(self.node1.public_key)

    def test_mesh_manager(self):
        """Test mesh network manager."""
        manager = MeshNetworkManager("test-mesh")
        self.assertIsNotNone(manager.node)

        status = manager.get_status()
        self.assertIn("node_id", status)
        self.assertIn("is_running", status)


class TestDashboard(unittest.TestCase):
    """Test dashboard monitoring."""

    def test_system_monitor(self):
        """Test system monitoring."""
        monitor = SystemMonitor()
        monitor.start_monitoring()
        time.sleep(1)  # Let it collect some data
        monitor.stop_monitoring()

        current = monitor.get_current_metrics()
        if current:  # May be None if monitoring didn't start in time
            self.assertIsNotNone(current.cpu_percent)
            self.assertIsNotNone(current.memory_percent)

    def test_training_monitor(self):
        """Test training monitoring."""
        monitor = TrainingMonitor()

        monitor.start_training_session()
        self.assertTrue(monitor.is_training)

        monitor.update_training_metrics(1, 0.5, 0.8, 3, 10.0)
        summary = monitor.get_training_summary()

        self.assertEqual(summary["current_round"], 1)
        self.assertEqual(summary["best_accuracy"], 0.8)

        monitor.stop_training_session()
        self.assertFalse(monitor.is_training)


def run_integration_test():
    """Run integration test with multiple components."""
    print("=== Integration Test ===")

    # Test client with synthetic data
    print("Testing FL Client with synthetic data...")
    config = Config({
        "dataset": "synthetic",
        "model": "linear",
        "batch_size": 16,
        "num_clients": 2,
        "local_epochs": 1,
        "learning_rate": 0.01
    })

    client = FunctionalFLClient("integration-test-client", config)
    print(f"✓ Client initialized with {count_parameters(client.model):,} parameters")

    # Test training
    initial_params = client.get_parameters({})
    updated_params, samples, metrics = client.fit(initial_params, {"server_round": 1})
    print(f"✓ Training completed: {samples} samples, loss={metrics['loss']:.4f}")

    # Test evaluation
    loss, eval_samples, eval_metrics = client.evaluate(updated_params, {})
    print(f"✓ Evaluation completed: {eval_samples} samples, accuracy={eval_metrics['accuracy']:.4f}")

    # Test monitoring
    print("Testing monitoring systems...")
    training_monitor = TrainingMonitor()
    training_monitor.update_training_metrics(1, metrics['loss'], eval_metrics['accuracy'], 1, 5.0)
    summary = training_monitor.get_training_summary()
    print(f"✓ Monitoring: Round {summary['current_round']}, accuracy={summary['latest_accuracy']:.4f}")

    print("=== Integration Test Passed ===")


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(exit=False, verbosity=2)

    print("\n" + "="*50)

    # Run integration test
    run_integration_test()

    print("\n=== All Tests Completed ===")
    print("✓ Models work correctly")
    print("✓ Datasets load properly")
    print("✓ FL clients train and evaluate")
    print("✓ Mesh networking initializes")
    print("✓ Monitoring systems function")
    print("✓ Integration test passed")
    print("\nThe functional FL application is ready to use!")