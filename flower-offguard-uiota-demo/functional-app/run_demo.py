#!/usr/bin/env python3
"""
Complete demo launcher for the federated learning system
"""

import argparse
import logging
import subprocess
import sys
import time
import signal
import threading
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from shared.utils import setup_logging, Config

class FLDemo:
    """Orchestrate the complete FL demo."""

    def __init__(self, config: Config):
        self.config = config
        self.processes = []
        self.is_running = False
        self.logger = logging.getLogger(__name__)

    def start_server(self):
        """Start the FL server."""
        cmd = [
            sys.executable, "run_server.py",
            "--web-port", str(self.config.get("web_port", 5000)),
            "--fl-port", str(self.config.get("fl_port", 8080)),
            "--dataset", self.config.get("dataset", "mnist"),
            "--model", self.config.get("model", "cnn"),
            "--num-rounds", str(self.config.get("num_rounds", 10)),
            "--min-clients", str(self.config.get("min_clients", 2)),
            "--log-level", self.config.get("log_level", "INFO")
        ]

        self.logger.info("Starting FL Server...")
        process = subprocess.Popen(cmd, cwd=project_root)
        self.processes.append(("server", process))
        return process

    def start_client(self, client_id: str, delay: int = 0):
        """Start an FL client."""
        if delay > 0:
            time.sleep(delay)

        cmd = [
            sys.executable, "run_client.py",
            "--client-id", client_id,
            "--server-address", f"localhost:{self.config.get('fl_port', 8080)}",
            "--dataset", self.config.get("dataset", "mnist"),
            "--model", self.config.get("model", "cnn"),
            "--local-epochs", str(self.config.get("local_epochs", 2)),
            "--batch-size", str(self.config.get("batch_size", 32)),
            "--num-clients", str(self.config.get("num_clients", 10)),
            "--log-level", self.config.get("log_level", "INFO")
        ]

        if self.config.get("enable_mesh", False):
            cmd.extend(["--enable-mesh", "--mesh-port", str(8081 + int(client_id.split('-')[-1]))])

        self.logger.info(f"Starting FL Client {client_id}...")
        process = subprocess.Popen(cmd, cwd=project_root)
        self.processes.append((f"client-{client_id}", process))
        return process

    def start_demo(self):
        """Start the complete demo."""
        if self.is_running:
            self.logger.warning("Demo is already running")
            return

        self.is_running = True
        self.logger.info("Starting Federated Learning Demo")

        try:
            # Start server
            self.start_server()
            time.sleep(5)  # Give server time to start

            # Start clients with staggered timing
            num_clients = self.config.get("num_clients", 3)
            client_threads = []

            for i in range(num_clients):
                client_id = f"client-{i+1}"
                delay = i * 2  # Stagger client starts by 2 seconds

                thread = threading.Thread(
                    target=self.start_client,
                    args=(client_id, delay),
                    daemon=True
                )
                thread.start()
                client_threads.append(thread)

            # Wait for all clients to start
            for thread in client_threads:
                thread.join()

            self.logger.info("All components started")
            self.logger.info(f"Dashboard available at: http://localhost:{self.config.get('web_port', 5000)}")
            self.logger.info("Press Ctrl+C to stop the demo")

            # Monitor processes
            self._monitor_processes()

        except Exception as e:
            self.logger.error(f"Error starting demo: {e}")
            self.stop_demo()

    def stop_demo(self):
        """Stop the demo and cleanup."""
        if not self.is_running:
            return

        self.is_running = False
        self.logger.info("Stopping Federated Learning Demo...")

        # Terminate all processes
        for name, process in self.processes:
            try:
                self.logger.info(f"Stopping {name}...")
                process.terminate()

                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Force killing {name}...")
                    process.kill()
                    process.wait()

            except Exception as e:
                self.logger.error(f"Error stopping {name}: {e}")

        self.processes.clear()
        self.logger.info("Demo stopped")

    def _monitor_processes(self):
        """Monitor running processes."""
        while self.is_running:
            try:
                # Check if any process has died unexpectedly
                for i, (name, process) in enumerate(self.processes):
                    if process.poll() is not None:
                        self.logger.warning(f"Process {name} has stopped unexpectedly")
                        # Could implement restart logic here

                time.sleep(5)

            except Exception as e:
                self.logger.error(f"Error monitoring processes: {e}")
                break

    def get_status(self):
        """Get status of all processes."""
        status = {
            "is_running": self.is_running,
            "processes": []
        }

        for name, process in self.processes:
            status["processes"].append({
                "name": name,
                "pid": process.pid,
                "running": process.poll() is None
            })

        return status


def signal_handler(demo, signum, frame):
    """Handle shutdown signals."""
    print("\nReceived shutdown signal...")
    demo.stop_demo()
    sys.exit(0)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="FL Demo Launcher")

    # Demo configuration
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "synthetic"],
                       default="mnist", help="Dataset")
    parser.add_argument("--model", choices=["cnn", "linear"],
                       default="cnn", help="Model type")
    parser.add_argument("--num-rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--local-epochs", type=int, default=2, help="Local epochs per client")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # Server ports
    parser.add_argument("--web-port", type=int, default=5000, help="Web dashboard port")
    parser.add_argument("--fl-port", type=int, default=8080, help="FL server port")

    # Features
    parser.add_argument("--enable-mesh", action="store_true", help="Enable mesh networking")

    # System
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

    # Update with command line arguments
    config.update({
        "dataset": args.dataset,
        "model": args.model,
        "num_rounds": args.num_rounds,
        "num_clients": args.num_clients,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "web_port": args.web_port,
        "fl_port": args.fl_port,
        "min_clients": min(2, args.num_clients),
        "enable_mesh": args.enable_mesh,
        "log_level": args.log_level
    })

    # Save demo configuration
    demo_config_path = project_root / "config" / "demo_config.json"
    demo_config_path.parent.mkdir(exist_ok=True)
    config.save_to_file(str(demo_config_path))

    logger.info("=== Federated Learning Demo ===")
    logger.info(f"Configuration: {json.dumps(config.to_dict(), indent=2)}")

    # Create and start demo
    demo = FLDemo(config)

    # Setup signal handlers
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(demo, s, f))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(demo, s, f))

    try:
        demo.start_demo()

        # Keep main thread alive
        while demo.is_running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
    finally:
        demo.stop_demo()

    logger.info("Demo completed")
    return 0


if __name__ == "__main__":
    exit(main())