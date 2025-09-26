#!/usr/bin/env python3
"""
SDK Packager Agent - Packages the Flower Off-Guard UIOTA demo as a simple SDK.

This agent handles:
- Creating pip-installable package structure
- Generating proper setup.py with dependencies
- Creating example usage scripts
- Building comprehensive documentation
- Creating SDK distribution files (wheel, tarball)
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='📦 [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class SDKPackagerAgent:
    """Main SDK packager automation agent."""

    def __init__(self, project_root: Path, output_dir: Path = None):
        self.project_root = Path(project_root).resolve()
        self.demo_dir = self.project_root / "flower-offguard-uiota-demo"
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "sdk-dist"
        self.build_dir = self.output_dir / "build"

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.build_dir.mkdir(parents=True, exist_ok=True)

        # SDK configuration
        self.config = {
            "package_name": "flower-offguard-uiota",
            "version": "1.0.0",
            "author": "UIOTA Team",
            "author_email": "dev@uiota.org",
            "description": "Flower Off-Guard UIOTA Federated Learning SDK",
            "url": "https://github.com/uiota/offline-guard",
            "license": "MIT",
            "python_requires": ">=3.8",
            "classifiers": [
                "Development Status :: 4 - Beta",
                "Intended Audience :: Developers",
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10",
                "Programming Language :: Python :: 3.11",
                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Topic :: System :: Distributed Computing",
                "Topic :: Security :: Cryptography",
            ],
            "keywords": "federated-learning flower ai security uiota mesh-networking",
            "console_scripts": [
                "flower-offguard-server=flower_offguard_uiota.server:main",
                "flower-offguard-client=flower_offguard_uiota.client:main",
                "flower-offguard-demo=flower_offguard_uiota.demo:main",
            ]
        }

    def validate_source(self) -> bool:
        """Validate source demo directory."""
        logger.info("Validating source demo directory...")

        if not self.demo_dir.exists():
            logger.error(f"Demo directory not found: {self.demo_dir}")
            return False

        src_dir = self.demo_dir / "src"
        if not src_dir.exists():
            logger.error(f"Source directory not found: {src_dir}")
            return False

        requirements_file = self.demo_dir / "requirements.txt"
        if not requirements_file.exists():
            logger.error(f"Requirements file not found: {requirements_file}")
            return False

        # Check for essential source files
        essential_files = ["server.py", "models.py", "utils.py", "guard.py"]
        for file_name in essential_files:
            file_path = src_dir / file_name
            if not file_path.exists():
                logger.error(f"Essential file missing: {file_path}")
                return False

        logger.info("✅ Source validation passed")
        return True

    def create_sdk_structure(self) -> Path:
        """Create proper Python package structure for SDK."""
        logger.info("Creating SDK package structure...")

        package_dir = self.build_dir / self.config["package_name"].replace("-", "_")

        # Remove existing package directory
        if package_dir.exists():
            shutil.rmtree(package_dir)

        package_dir.mkdir(parents=True)

        # Create package structure
        (package_dir / "__init__.py").write_text(self._generate_init_file())

        # Copy and transform source files
        self._copy_source_files(package_dir)

        # Create client wrapper if it doesn't exist
        if not (package_dir / "client.py").exists():
            (package_dir / "client.py").write_text(self._generate_client_wrapper())

        # Create demo wrapper
        (package_dir / "demo.py").write_text(self._generate_demo_wrapper())

        # Create examples directory
        examples_dir = package_dir / "examples"
        examples_dir.mkdir()
        self._create_example_scripts(examples_dir)

        logger.info(f"✅ SDK structure created: {package_dir}")
        return package_dir

    def _generate_init_file(self) -> str:
        """Generate __init__.py for the package."""
        return f'''"""
Flower Off-Guard UIOTA Federated Learning SDK

A comprehensive federated learning framework with:
- Flower AI integration
- Off-Guard security features
- UIOTA mesh networking
- Differential privacy support
- CPU-optimized performance
"""

__version__ = "{self.config["version"]}"
__author__ = "{self.config["author"]}"
__license__ = "{self.config["license"]}"

# Core imports
from .server import FederatedServer
from .models import SmallCNN, CIFAR10CNN
from .guard import (
    preflight_check,
    new_keypair,
    verify_model_integrity,
    apply_differential_privacy
)
from .utils import (
    set_random_seeds,
    get_device,
    calculate_model_size,
    format_metrics
)

# Strategy imports
from .strategy_custom import OffGuardFedAvg, OffGuardFedProx

# Dataset utilities
from .datasets import (
    create_mnist_partitions,
    create_cifar10_partitions,
    load_dataset
)

# Mesh networking
from .mesh_sync import MeshCoordinator, P2PNode

__all__ = [
    # Core classes
    "FederatedServer",
    "SmallCNN",
    "CIFAR10CNN",

    # Security functions
    "preflight_check",
    "new_keypair",
    "verify_model_integrity",
    "apply_differential_privacy",

    # Utilities
    "set_random_seeds",
    "get_device",
    "calculate_model_size",
    "format_metrics",

    # Strategies
    "OffGuardFedAvg",
    "OffGuardFedProx",

    # Dataset utilities
    "create_mnist_partitions",
    "create_cifar10_partitions",
    "load_dataset",

    # Mesh networking
    "MeshCoordinator",
    "P2PNode",

    # Metadata
    "__version__",
    "__author__",
    "__license__"
]

# Quick start helper
def quick_start():
    """Print quick start instructions."""
    print(f"""
🌸 Flower Off-Guard UIOTA SDK v{__version__}

📖 Quick Start:

1. Start a federated server:
   >>> from flower_offguard_uiota import FederatedServer
   >>> server = FederatedServer({{'dataset': 'mnist', 'rounds': 5}})
   >>> server.run()

2. Or use the CLI:
   $ flower-offguard-server --dataset mnist --rounds 5

3. Run demo:
   $ flower-offguard-demo

📚 Documentation: https://github.com/uiota/offline-guard
🛟 Support: dev@uiota.org
""")
'''

    def _copy_source_files(self, package_dir: Path) -> None:
        """Copy and adapt source files for SDK."""
        src_dir = self.demo_dir / "src"

        # Files to copy directly
        direct_copy_files = [
            "models.py", "utils.py", "guard.py",
            "mesh_sync.py", "datasets.py", "strategy_custom.py"
        ]

        for file_name in direct_copy_files:
            src_file = src_dir / file_name
            if src_file.exists():
                dest_file = package_dir / file_name

                # Read, modify imports, and write
                content = src_file.read_text(encoding='utf-8')

                # Convert relative imports to absolute
                content = content.replace("from .", f"from flower_offguard_uiota.")
                content = content.replace("from . import", f"from flower_offguard_uiota import")

                dest_file.write_text(content)

        # Special handling for server.py
        server_file = src_dir / "server.py"
        if server_file.exists():
            content = server_file.read_text(encoding='utf-8')

            # Convert imports
            content = content.replace("from . import", "from flower_offguard_uiota import")
            content = content.replace("from .", "from flower_offguard_uiota.")

            # Wrap main functionality in a class/function for SDK use
            content = self._wrap_server_for_sdk(content)

            (package_dir / "server.py").write_text(content)

    def _wrap_server_for_sdk(self, server_content: str) -> str:
        """Wrap server.py for SDK compatibility."""
        # Add SDK-specific imports and modifications
        sdk_wrapper = '''#!/usr/bin/env python3
"""
Flower Off-Guard UIOTA Server - SDK Version

Enhanced for programmatic use as an SDK component.
"""

# Add SDK-friendly interface
class ServerConfig:
    """Configuration class for easier SDK integration."""

    def __init__(self, **kwargs):
        # Default configuration
        self.rounds = kwargs.get('rounds', 5)
        self.clients_per_round = kwargs.get('clients_per_round', 10)
        self.strategy = kwargs.get('strategy', 'fedavg')
        self.dataset = kwargs.get('dataset', 'mnist')
        self.server_address = kwargs.get('server_address', 'localhost:8080')
        self.dp = kwargs.get('dp', 'off')
        self.latency_ms = kwargs.get('latency_ms', 50)
        self.jitter_ms = kwargs.get('jitter_ms', 25)
        self.dropout_pct = kwargs.get('dropout_pct', 0.1)

def create_server(config_dict=None, **kwargs):
    """Create a FederatedServer instance with configuration.

    Args:
        config_dict: Dictionary with server configuration
        **kwargs: Additional configuration parameters

    Returns:
        FederatedServer instance ready to run
    """
    if config_dict:
        kwargs.update(config_dict)

    config = ServerConfig(**kwargs)
    return FederatedServer(config)

''' + server_content

        return sdk_wrapper

    def _generate_client_wrapper(self) -> str:
        """Generate client wrapper if client.py doesn't exist."""
        return '''#!/usr/bin/env python3
"""
Flower Off-Guard UIOTA Client - SDK Version

A federated learning client with security and mesh networking.
"""

import argparse
import logging
from typing import Dict, Any

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import models
from . import datasets
from . import guard
from . import utils

logger = logging.getLogger(__name__)

class OffGuardClient(fl.client.NumPyClient):
    """Federated learning client with Off-Guard security."""

    def __init__(self, client_id: str, dataset: str = "mnist", batch_size: int = 32):
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size

        # Initialize model
        if dataset == "mnist":
            self.model = models.SmallCNN(num_classes=10)
        elif dataset == "cifar10":
            self.model = models.CIFAR10CNN(num_classes=10)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Load data
        self.train_loader, self.test_loader = self._load_data()

        # Security initialization
        guard.preflight_check()
        self.keypair = guard.new_keypair()

        logger.info(f"Client {client_id} initialized for {dataset}")

    def _load_data(self):
        """Load and partition data for this client."""
        if self.dataset == "mnist":
            return datasets.create_mnist_partitions(
                num_clients=10,  # This would typically be configured
                client_id=int(self.client_id),
                batch_size=self.batch_size
            )
        elif self.dataset == "cifar10":
            return datasets.create_cifar10_partitions(
                num_clients=10,
                client_id=int(self.client_id),
                batch_size=self.batch_size
            )

    def get_parameters(self, config):
        """Return current model parameters."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Update model parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model on local data."""
        self.set_parameters(parameters)

        # Training configuration
        epochs = config.get("local_epochs", 1)
        lr = config.get("learning_rate", 0.01)

        # Train model
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Return updated parameters
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on local data."""
        self.set_parameters(parameters)

        criterion = nn.CrossEntropyLoss()
        self.model.eval()

        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        return float(test_loss), len(self.test_loader.dataset), {"accuracy": accuracy}


def create_client(client_id: str, server_address: str = "localhost:8080", **kwargs):
    """Create and start a federated learning client.

    Args:
        client_id: Unique identifier for this client
        server_address: Server address to connect to
        **kwargs: Additional client configuration

    Returns:
        Started client instance
    """
    client = OffGuardClient(client_id, **kwargs)

    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )

    return client


def main():
    """Main entry point for client CLI."""
    parser = argparse.ArgumentParser(description="Flower Off-Guard Client")
    parser.add_argument("--client-id", default="0", help="Client ID")
    parser.add_argument("--server-address", default="localhost:8080", help="Server address")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist", help="Dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    create_client(
        client_id=args.client_id,
        server_address=args.server_address,
        dataset=args.dataset,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
'''

    def _generate_demo_wrapper(self) -> str:
        """Generate demo wrapper script."""
        return '''#!/usr/bin/env python3
"""
Flower Off-Guard UIOTA Demo Wrapper

Interactive demo launcher with multiple scenarios.
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class DemoLauncher:
    """Interactive demo launcher."""

    def __init__(self):
        self.scenarios = {
            "quickstart": {
                "name": "Quick Start Demo",
                "description": "Basic federated learning with 2 clients",
                "server_args": ["--rounds", "3", "--clients-per-round", "2"],
                "num_clients": 2
            },
            "secure": {
                "name": "Security Demo",
                "description": "Federated learning with differential privacy",
                "server_args": ["--rounds", "5", "--dp", "on"],
                "num_clients": 3
            },
            "mesh": {
                "name": "Mesh Networking Demo",
                "description": "P2P mesh networking with multiple nodes",
                "server_args": ["--rounds", "5", "--clients-per-round", "5"],
                "num_clients": 5
            },
            "performance": {
                "name": "Performance Demo",
                "description": "Larger scale demo with network simulation",
                "server_args": ["--rounds", "10", "--clients-per-round", "8", "--latency-ms", "100"],
                "num_clients": 8
            }
        }

    def show_menu(self):
        """Display demo scenario menu."""
        print("\\n🌸 Flower Off-Guard UIOTA Demo Launcher")
        print("=" * 50)
        print("Choose a demo scenario:\\n")

        for key, scenario in self.scenarios.items():
            print(f"{key:12} - {scenario['name']}")
            print(f"{'':14} {scenario['description']}\\n")

        print("custom      - Custom configuration")
        print("quit        - Exit demo launcher\\n")

    def run_scenario(self, scenario_key: str):
        """Run a specific demo scenario."""
        if scenario_key not in self.scenarios:
            print(f"❌ Unknown scenario: {scenario_key}")
            return False

        scenario = self.scenarios[scenario_key]
        print(f"\\n🚀 Starting {scenario['name']}...")
        print(f"📝 {scenario['description']}\\n")

        try:
            # Start server in background
            print("🖥️  Starting federated server...")
            server_cmd = ["flower-offguard-server"] + scenario["server_args"]

            server_process = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for server to start
            time.sleep(3)

            # Start clients
            client_processes = []
            for i in range(scenario["num_clients"]):
                print(f"👤 Starting client {i+1}/{scenario['num_clients']}...")

                client_cmd = [
                    "flower-offguard-client",
                    "--client-id", str(i),
                    "--server-address", "localhost:8080"
                ]

                client_proc = subprocess.Popen(
                    client_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                client_processes.append(client_proc)
                time.sleep(1)

            print("\\n✅ Demo scenario started!")
            print("Press Ctrl+C to stop the demo\\n")

            # Wait for completion or interruption
            try:
                server_process.wait()
                print("🏁 Demo completed successfully!")
            except KeyboardInterrupt:
                print("\\n⏹️  Stopping demo...")

                # Terminate all processes
                server_process.terminate()
                for client_proc in client_processes:
                    client_proc.terminate()

                print("✅ Demo stopped")

        except FileNotFoundError:
            print("❌ Demo commands not found. Make sure the SDK is properly installed.")
            return False
        except Exception as e:
            print(f"❌ Error running demo: {e}")
            return False

        return True

    def run_custom_demo(self):
        """Run custom demo with user input."""
        print("\\n🔧 Custom Demo Configuration")
        print("-" * 30)

        try:
            rounds = int(input("Number of rounds (default: 5): ") or "5")
            clients = int(input("Number of clients (default: 3): ") or "3")
            dataset = input("Dataset [mnist/cifar10] (default: mnist): ") or "mnist"

            if dataset not in ["mnist", "cifar10"]:
                print("❌ Invalid dataset. Using mnist.")
                dataset = "mnist"

            dp = input("Enable differential privacy? [y/N]: ").lower().startswith('y')

            print(f"\\n🚀 Starting custom demo...")
            print(f"   Rounds: {rounds}")
            print(f"   Clients: {clients}")
            print(f"   Dataset: {dataset}")
            print(f"   Differential Privacy: {'Yes' if dp else 'No'}\\n")

            # Build server command
            server_args = [
                "--rounds", str(rounds),
                "--clients-per-round", str(clients),
                "--dataset", dataset
            ]
            if dp:
                server_args.extend(["--dp", "on"])

            # Create custom scenario
            custom_scenario = {
                "name": "Custom Demo",
                "description": f"{rounds} rounds, {clients} clients, {dataset}",
                "server_args": server_args,
                "num_clients": clients
            }

            self.scenarios["custom"] = custom_scenario
            return self.run_scenario("custom")

        except ValueError:
            print("❌ Invalid input. Please enter numbers for rounds and clients.")
            return False
        except KeyboardInterrupt:
            print("\\n❌ Demo setup cancelled.")
            return False

    def interactive_mode(self):
        """Run in interactive mode."""
        while True:
            self.show_menu()

            try:
                choice = input("Select scenario: ").strip().lower()

                if choice == "quit" or choice == "q":
                    print("👋 Goodbye!")
                    break
                elif choice == "custom":
                    self.run_custom_demo()
                elif choice in self.scenarios:
                    self.run_scenario(choice)
                else:
                    print(f"❌ Invalid choice: {choice}")

                input("\\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\\n👋 Goodbye!")
                break
            except EOFError:
                print("\\n👋 Goodbye!")
                break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Flower Off-Guard UIOTA Demo")
    parser.add_argument("--scenario", choices=["quickstart", "secure", "mesh", "performance"],
                       help="Run specific scenario")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    launcher = DemoLauncher()

    if args.scenario:
        success = launcher.run_scenario(args.scenario)
        sys.exit(0 if success else 1)
    else:
        launcher.interactive_mode()


if __name__ == "__main__":
    main()
'''

    def _create_example_scripts(self, examples_dir: Path) -> None:
        """Create example usage scripts."""

        # Basic server example
        (examples_dir / "basic_server.py").write_text('''#!/usr/bin/env python3
"""
Basic server example for Flower Off-Guard UIOTA SDK.
"""

from flower_offguard_uiota import create_server

def main():
    """Run a basic federated learning server."""
    # Create server with simple configuration
    server = create_server(
        dataset="mnist",
        rounds=5,
        clients_per_round=3,
        strategy="fedavg"
    )

    print("🌸 Starting basic federated learning server...")
    server.run()

if __name__ == "__main__":
    main()
''')

        # Security example
        (examples_dir / "secure_server.py").write_text('''#!/usr/bin/env python3
"""
Security-enhanced server example with differential privacy.
"""

from flower_offguard_uiota import create_server

def main():
    """Run a secure federated learning server."""
    server = create_server(
        dataset="cifar10",
        rounds=10,
        clients_per_round=5,
        strategy="fedprox",
        dp="on"  # Enable differential privacy
    )

    print("🛡️ Starting secure federated learning server...")
    print("🔒 Differential privacy enabled")
    server.run()

if __name__ == "__main__":
    main()
''')

        # Client example
        (examples_dir / "basic_client.py").write_text('''#!/usr/bin/env python3
"""
Basic client example for Flower Off-Guard UIOTA SDK.
"""

from flower_offguard_uiota.client import create_client

def main():
    """Run a basic federated learning client."""
    print("👤 Starting federated learning client...")

    client = create_client(
        client_id="example_client",
        server_address="localhost:8080",
        dataset="mnist",
        batch_size=32
    )

if __name__ == "__main__":
    main()
''')

        # Custom model example
        (examples_dir / "custom_model.py").write_text('''#!/usr/bin/env python3
"""
Example of using custom models with the SDK.
"""

import torch
import torch.nn as nn
from flower_offguard_uiota import models, utils

class CustomCNN(nn.Module):
    """Custom CNN model for demonstration."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def main():
    """Demonstrate custom model usage."""
    # Create custom model
    model = CustomCNN(num_classes=10)

    # Use SDK utilities
    device = utils.get_device()
    model = model.to(device)

    print(f"📊 Model size: {utils.calculate_model_size(model):.2f} MB")
    print(f"🖥️  Device: {device}")

    # Set random seeds for reproducibility
    utils.set_random_seeds(42)

    print("✅ Custom model ready for federated learning!")

if __name__ == "__main__":
    main()
''')

        # Mesh networking example
        (examples_dir / "mesh_demo.py").write_text('''#!/usr/bin/env python3
"""
UIOTA mesh networking demonstration.
"""

from flower_offguard_uiota.mesh_sync import MeshCoordinator, P2PNode

def main():
    """Demonstrate mesh networking capabilities."""
    print("🌐 UIOTA Mesh Networking Demo")

    # Create mesh coordinator
    coordinator = MeshCoordinator(
        node_id="demo_node",
        port=9000
    )

    print("🔗 Starting mesh coordinator...")
    # In a real scenario, you would start the coordinator
    # coordinator.start()

    print("✅ Mesh networking demo setup complete!")
    print("📝 Note: This is a demonstration of the API structure.")

if __name__ == "__main__":
    main()
''')

    def generate_setup_py(self, package_dir: Path) -> str:
        """Generate setup.py for the SDK package."""

        # Read requirements from demo
        requirements = []
        requirements_file = self.demo_dir / "requirements.txt"
        if requirements_file.exists():
            requirements = [
                line.strip() for line in requirements_file.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]

        classifiers_str = ",\\n        ".join(f'"{c}"' for c in self.config["classifiers"])
        console_scripts_str = ",\\n            ".join(f'"{s}"' for s in self.config["console_scripts"])
        requirements_str = ",\\n        ".join(f'"{r}"' for r in requirements)

        return f'''#!/usr/bin/env python3
"""
Setup script for {self.config["package_name"]} SDK.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="{self.config["package_name"]}",
    version="{self.config["version"]}",
    description="{self.config["description"]}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="{self.config["author"]}",
    author_email="{self.config["author_email"]}",
    url="{self.config["url"]}",
    license="{self.config["license"]}",

    packages=find_packages(),
    python_requires="{self.config["python_requires"]}",

    install_requires=[
        {requirements_str}
    ],

    extras_require={{
        "dev": [
            "pytest>=7.4.3",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "twine>=4.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
        ],
        "examples": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
        ],
    }},

    classifiers=[
        {classifiers_str}
    ],

    keywords="{self.config["keywords"]}",

    entry_points={{
        "console_scripts": [
            {console_scripts_str}
        ],
    }},

    include_package_data=True,
    package_data={{
        "{self.config["package_name"].replace("-", "_")}": [
            "examples/*.py",
            "*.yaml",
            "*.json",
            "*.txt",
            "*.md"
        ],
    }},

    project_urls={{
        "Documentation": "{self.config["url"]}/docs",
        "Source": "{self.config["url"]}",
        "Tracker": "{self.config["url"]}/issues",
    }},
)
'''

    def generate_sdk_readme(self) -> str:
        """Generate README.md for SDK package."""
        return f'''# {self.config["package_name"].title()} SDK

{self.config["description"]}

## 🌟 Features

- 🌸 **Flower AI Integration** - Built on the robust Flower federated learning framework
- 🛡️ **Off-Guard Security** - Comprehensive security with cryptographic protection
- 🌐 **UIOTA Mesh Networking** - Peer-to-peer mesh networking for resilient FL
- 🔒 **Differential Privacy** - Configurable privacy protection
- 🚀 **CPU Optimized** - Designed for CPU-only environments
- 📦 **Easy Integration** - Simple SDK for quick deployment

## 🚀 Quick Start

### Installation

```bash
pip install {self.config["package_name"]}
```

### Basic Usage

#### Start a Federated Server

```python
from flower_offguard_uiota import create_server

# Create and start server
server = create_server(
    dataset="mnist",
    rounds=5,
    clients_per_round=3
)
server.run()
```

#### Or use the CLI

```bash
flower-offguard-server --dataset mnist --rounds 5
```

#### Start a Client

```python
from flower_offguard_uiota.client import create_client

create_client(
    client_id="client_1",
    server_address="localhost:8080",
    dataset="mnist"
)
```

#### Or use the CLI

```bash
flower-offguard-client --client-id 1 --server-address localhost:8080
```

### Interactive Demo

```bash
flower-offguard-demo --interactive
```

## 📖 Documentation

### Core Components

#### Server

```python
from flower_offguard_uiota import FederatedServer

server = FederatedServer({{
    'dataset': 'cifar10',
    'rounds': 10,
    'clients_per_round': 5,
    'strategy': 'fedprox',
    'dp': 'on'  # Enable differential privacy
}})
```

#### Models

```python
from flower_offguard_uiota import SmallCNN, CIFAR10CNN

# For MNIST
model = SmallCNN(num_classes=10)

# For CIFAR-10
model = CIFAR10CNN(num_classes=10)
```

#### Security

```python
from flower_offguard_uiota import guard

# Security checks
guard.preflight_check()

# Generate cryptographic keys
keypair = guard.new_keypair()

# Apply differential privacy
guard.apply_differential_privacy(model, noise_multiplier=1.0)
```

#### Utilities

```python
from flower_offguard_uiota import utils

# Set reproducible seeds
utils.set_random_seeds(42)

# Get appropriate device
device = utils.get_device()

# Calculate model size
size_mb = utils.calculate_model_size(model)
```

### Configuration

The SDK supports various configuration options:

```python
config = {{
    # Core FL settings
    'dataset': 'mnist',  # mnist, cifar10
    'rounds': 5,
    'clients_per_round': 10,
    'strategy': 'fedavg',  # fedavg, fedprox

    # Security settings
    'dp': 'off',  # on, off

    # Network settings
    'server_address': 'localhost:8080',
    'latency_ms': 50,
    'jitter_ms': 25,
    'dropout_pct': 0.1,
}}
```

## 🔧 Examples

The SDK includes comprehensive examples in the `examples/` directory:

- `basic_server.py` - Simple federated server
- `secure_server.py` - Server with differential privacy
- `basic_client.py` - Basic federated client
- `custom_model.py` - Using custom neural network models
- `mesh_demo.py` - UIOTA mesh networking demonstration

## 🛠️ Development

### Installation for Development

```bash
git clone {self.config["url"]}.git
cd offline-guard
pip install -e ".[dev,docs,examples]"
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
black flower_offguard_uiota/
flake8 flower_offguard_uiota/
mypy flower_offguard_uiota/
```

## 📋 Requirements

- Python {self.config["python_requires"]}
- CPU-only (no GPU required)
- 2GB RAM minimum
- Network connection for initial setup

## 📄 License

This project is licensed under the {self.config["license"]} License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md]({self.config["url"]}/blob/main/CONTRIBUTING.md) for guidelines.

## 📞 Support

- 📚 Documentation: {self.config["url"]}/docs
- 🐛 Issues: {self.config["url"]}/issues
- 📧 Email: {self.config["author_email"]}

## 🙏 Acknowledgments

Built with:
- [Flower](https://flower.dev/) - Federated Learning Framework
- [PyTorch](https://pytorch.org/) - Deep Learning Library
- [Opacus](https://opacus.ai/) - Differential Privacy

---

Made with ❤️ by the {self.config["author"]}
'''

    def build_distributions(self, package_root: Path) -> Dict:
        """Build wheel and source distributions."""
        logger.info("Building distribution packages...")

        results = {
            "success": False,
            "wheel_file": None,
            "tarball_file": None,
            "errors": []
        }

        try:
            # Create setup.py in package root
            setup_py_content = self.generate_setup_py(package_root)
            setup_py_path = package_root / "setup.py"
            setup_py_path.write_text(setup_py_content)

            # Create README.md
            readme_content = self.generate_sdk_readme()
            (package_root / "README.md").write_text(readme_content)

            # Build distributions using subprocess
            build_cmd = [
                sys.executable, "setup.py",
                "sdist", "bdist_wheel",
                "--dist-dir", str(self.output_dir)
            ]

            result = subprocess.run(
                build_cmd,
                cwd=package_root,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                results["errors"].append(f"Build failed: {result.stderr}")
                return results

            # Find generated files
            for file_path in self.output_dir.iterdir():
                if file_path.suffix == ".whl":
                    results["wheel_file"] = file_path
                elif file_path.name.endswith(".tar.gz"):
                    results["tarball_file"] = file_path

            results["success"] = True
            logger.info("✅ Distribution packages built successfully")

        except subprocess.TimeoutExpired:
            results["errors"].append("Build process timed out")
        except Exception as e:
            results["errors"].append(f"Build error: {e}")

        return results

    def validate_package(self, distributions: Dict) -> bool:
        """Validate built package."""
        logger.info("Validating built packages...")

        if not distributions.get("success"):
            logger.error("Cannot validate - build failed")
            return False

        # Check wheel file
        wheel_file = distributions.get("wheel_file")
        if not wheel_file or not wheel_file.exists():
            logger.error("Wheel file not found")
            return False

        # Check tarball
        tarball_file = distributions.get("tarball_file")
        if not tarball_file or not tarball_file.exists():
            logger.error("Source tarball not found")
            return False

        # Basic file size checks
        if wheel_file.stat().st_size < 1000:  # Less than 1KB suggests error
            logger.error("Wheel file suspiciously small")
            return False

        if tarball_file.stat().st_size < 1000:
            logger.error("Tarball file suspiciously small")
            return False

        logger.info("✅ Package validation passed")
        return True

    def package_sdk(self, skip_validation: bool = False) -> Dict:
        """Main SDK packaging process."""
        logger.info("🚀 Starting SDK packaging process...")

        results = {
            "success": False,
            "package_dir": None,
            "distributions": {},
            "errors": []
        }

        try:
            # Validate source
            if not self.validate_source():
                results["errors"].append("Source validation failed")
                return results

            # Create SDK structure
            package_dir = self.create_sdk_structure()
            results["package_dir"] = package_dir

            # Build distributions
            distributions = self.build_distributions(package_dir.parent)
            results["distributions"] = distributions

            if not distributions["success"]:
                results["errors"].extend(distributions["errors"])
                return results

            # Validate package
            if not skip_validation and not self.validate_package(distributions):
                results["errors"].append("Package validation failed")
                return results

            results["success"] = True
            logger.info("🎉 SDK packaging completed successfully!")

        except Exception as e:
            logger.error(f"SDK packaging failed: {e}")
            results["errors"].append(str(e))

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Flower Off-Guard UIOTA SDK Packager")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output-dir", help="Output directory for SDK distribution")
    parser.add_argument("--skip-validation", action="store_true", help="Skip package validation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize packager
    packager = SDKPackagerAgent(
        project_root=args.project_root,
        output_dir=args.output_dir
    )

    # Package SDK
    results = packager.package_sdk(skip_validation=args.skip_validation)

    # Print results
    if results["success"]:
        logger.info("📦 SDK Packaging Summary:")
        logger.info(f"   Package directory: {results['package_dir']}")

        distributions = results["distributions"]
        if distributions.get("wheel_file"):
            logger.info(f"   Wheel: {distributions['wheel_file']}")
        if distributions.get("tarball_file"):
            logger.info(f"   Source: {distributions['tarball_file']}")

        logger.info("\\n🚀 Installation commands:")
        if distributions.get("wheel_file"):
            logger.info(f"   pip install {distributions['wheel_file']}")
        if distributions.get("tarball_file"):
            logger.info(f"   pip install {distributions['tarball_file']}")

        return 0
    else:
        logger.error("❌ SDK packaging failed!")
        for error in results["errors"]:
            logger.error(f"   - {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())