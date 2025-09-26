#!/usr/bin/env python3
"""
Launcher script for FL Clients with mesh networking
"""

import argparse
import logging
import sys
import time
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from shared.utils import setup_logging, Config
from client.fl_client import ClientManager
from mesh.p2p_network import MeshNetworkManager

def main():
    """Main function to run an FL client with mesh networking."""
    parser = argparse.ArgumentParser(description="FL Client with Mesh Networking")

    # Client identification
    parser.add_argument("--client-id", required=True, help="Unique client identifier")
    parser.add_argument("--server-address", default="localhost:8080", help="FL server address")

    # Dataset and model configuration
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "synthetic"],
                       default="mnist", help="Dataset")
    parser.add_argument("--model", choices=["cnn", "linear"],
                       default="cnn", help="Model type")

    # Training configuration
    parser.add_argument("--local-epochs", type=int, default=1, help="Local training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")

    # Data configuration
    parser.add_argument("--num-clients", type=int, default=10, help="Total number of clients")
    parser.add_argument("--alpha", type=float, default=0.5, help="Non-IID parameter")
    parser.add_argument("--data-path", default="./data", help="Data storage path")

    # Mesh networking
    parser.add_argument("--enable-mesh", action="store_true", help="Enable mesh networking")
    parser.add_argument("--mesh-port", type=int, default=8081, help="Mesh network port")
    parser.add_argument("--mesh-peers", nargs="*", help="Initial mesh peers")

    # System configuration
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--config-file", help="Configuration file path")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

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
        "num_clients": args.num_clients,
        "alpha": args.alpha,
        "data_path": args.data_path
    })

    # Save configuration for reference
    config_path = project_root / "config" / f"client_{args.client_id}_config.json"
    config_path.parent.mkdir(exist_ok=True)
    config.save_to_file(str(config_path))

    logger.info(f"Starting FL Client {args.client_id}")
    logger.info(f"Configuration: {config.to_dict()}")

    # Initialize mesh networking if enabled
    mesh_manager = None
    if args.enable_mesh:
        try:
            mesh_manager = MeshNetworkManager(f"mesh-{args.client_id}")
            mesh_manager.start(args.mesh_port, args.mesh_peers)
            logger.info(f"Mesh networking enabled on port {args.mesh_port}")
        except Exception as e:
            logger.error(f"Failed to start mesh networking: {e}")
            mesh_manager = None

    # Create and start client manager
    client_manager = ClientManager(args.client_id, config)

    success = client_manager.start_client(args.server_address)

    if success:
        try:
            logger.info(f"FL Client {args.client_id} is running")

            # Keep the main thread alive and monitor status
            while client_manager.is_running:
                # Print status periodically
                if hasattr(client_manager, 'client') and client_manager.client:
                    status = client_manager.get_status()
                    logger.debug(f"Client status: Round {status.get('current_round', 0)}")

                # Print mesh status if enabled
                if mesh_manager:
                    mesh_status = mesh_manager.get_status()
                    logger.debug(f"Mesh peers: {mesh_status.get('peer_count', 0)}")

                time.sleep(30)  # Status update every 30 seconds

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            # Cleanup
            client_manager.stop_client()
            if mesh_manager:
                mesh_manager.stop()
    else:
        logger.error("Failed to start client")
        return 1

    logger.info("Client shutdown complete")
    return 0

if __name__ == "__main__":
    exit(main())