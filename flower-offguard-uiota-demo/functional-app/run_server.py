#!/usr/bin/env python3
"""
Launcher script for the FL Server with integrated dashboard
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from shared.utils import setup_logging, Config
from app.server import init_server, app, socketio

def main():
    """Main function to run the FL server."""
    parser = argparse.ArgumentParser(description="Federated Learning Server")

    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--web-port", type=int, default=5000, help="Web server port")
    parser.add_argument("--fl-port", type=int, default=8080, help="FL server port")

    # FL configuration
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "synthetic"],
                       default="mnist", help="Dataset")
    parser.add_argument("--model", choices=["cnn", "linear"],
                       default="cnn", help="Model type")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients")

    # System configuration
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--log-file", help="Log file path")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    # Create configuration
    if args.config and Path(args.config).exists():
        config = Config.from_file(args.config)
    else:
        config = Config()

    # Update configuration with command line arguments
    config.update({
        "dataset": args.dataset,
        "model": args.model,
        "num_rounds": args.num_rounds,
        "port": args.fl_port,
        "min_fit_clients": args.min_clients,
        "min_evaluate_clients": args.min_clients,
        "min_available_clients": args.min_clients,
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0
    })

    # Save configuration for reference
    config_path = project_root / "config" / "server_config.json"
    config_path.parent.mkdir(exist_ok=True)
    config.save_to_file(str(config_path))

    logger.info(f"Starting FL Server with configuration:")
    logger.info(f"  Dataset: {config.get('dataset')}")
    logger.info(f"  Model: {config.get('model')}")
    logger.info(f"  FL Rounds: {config.get('num_rounds')}")
    logger.info(f"  FL Port: {config.get('port')}")
    logger.info(f"  Web Port: {args.web_port}")
    logger.info(f"  Min Clients: {config.get('min_fit_clients')}")

    # Initialize FL server
    init_server(str(config_path))

    # Run web application
    logger.info(f"Starting web dashboard on http://{args.host}:{args.web_port}")
    socketio.run(app, host=args.host, port=args.web_port, debug=False)

if __name__ == "__main__":
    main()