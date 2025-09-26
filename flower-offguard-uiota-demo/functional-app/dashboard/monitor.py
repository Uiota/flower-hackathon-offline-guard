"""
Real-time Monitoring Dashboard for Federated Learning
"""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil
import sys

sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import MetricsCollector, Config, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_sent: int
    network_recv: int
    timestamp: float


@dataclass
class TrainingMetrics:
    """Training progress metrics."""
    round_number: int
    loss: float
    accuracy: float
    clients_participated: int
    training_time: float
    timestamp: float


@dataclass
class ClientStatus:
    """Individual client status."""
    client_id: str
    status: str  # "connected", "training", "idle", "disconnected"
    last_seen: float
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    samples: Optional[int] = None


class SystemMonitor:
    """Monitor system resources."""

    def __init__(self):
        self.is_running = False
        self.metrics_history: List[SystemMetrics] = []
        self.max_history = 1000  # Keep last 1000 data points

    def start_monitoring(self):
        """Start system monitoring."""
        if self.is_running:
            return

        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring."""
        self.is_running = False
        logger.info("System monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        last_network = psutil.net_io_counters()

        while self.is_running:
            try:
                # Get current network stats
                current_network = psutil.net_io_counters()

                # Calculate network throughput (bytes/second)
                network_sent = current_network.bytes_sent - last_network.bytes_sent
                network_recv = current_network.bytes_recv - last_network.bytes_recv

                # Create metrics
                metrics = SystemMetrics(
                    cpu_percent=psutil.cpu_percent(interval=1),
                    memory_percent=psutil.virtual_memory().percent,
                    disk_usage=psutil.disk_usage('/').percent,
                    network_sent=network_sent,
                    network_recv=network_recv,
                    timestamp=time.time()
                )

                # Store metrics
                self.metrics_history.append(metrics)

                # Limit history size
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)

                # Update last network stats
                last_network = current_network

                time.sleep(5)  # Monitor every 5 seconds

            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(5)

    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_history(self, seconds: int = 300) -> List[SystemMetrics]:
        """Get metrics history for the last N seconds."""
        cutoff_time = time.time() - seconds
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def get_average_metrics(self, seconds: int = 60) -> Optional[SystemMetrics]:
        """Get average metrics for the last N seconds."""
        recent_metrics = self.get_metrics_history(seconds)
        if not recent_metrics:
            return None

        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_usage for m in recent_metrics) / len(recent_metrics)
        total_sent = sum(m.network_sent for m in recent_metrics)
        total_recv = sum(m.network_recv for m in recent_metrics)

        return SystemMetrics(
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            disk_usage=avg_disk,
            network_sent=total_sent,
            network_recv=total_recv,
            timestamp=time.time()
        )


class TrainingMonitor:
    """Monitor federated learning training progress."""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.client_status: Dict[str, ClientStatus] = {}
        self.training_history: List[TrainingMetrics] = []
        self.is_training = False
        self.current_round = 0

    def update_training_metrics(self, round_num: int, loss: float, accuracy: float,
                              clients_participated: int, training_time: float):
        """Update training metrics for a round."""
        metrics = TrainingMetrics(
            round_number=round_num,
            loss=loss,
            accuracy=accuracy,
            clients_participated=clients_participated,
            training_time=training_time,
            timestamp=time.time()
        )

        self.training_history.append(metrics)
        self.current_round = round_num

        # Update metrics collector
        self.metrics_collector.add_metric("round_loss", loss, round_num)
        self.metrics_collector.add_metric("round_accuracy", accuracy, round_num)
        self.metrics_collector.add_metric("clients_participated", clients_participated, round_num)

        logger.info(f"Training metrics updated for round {round_num}")

    def update_client_status(self, client_id: str, status: str, loss: float = None,
                           accuracy: float = None, samples: int = None):
        """Update individual client status."""
        self.client_status[client_id] = ClientStatus(
            client_id=client_id,
            status=status,
            last_seen=time.time(),
            loss=loss,
            accuracy=accuracy,
            samples=samples
        )

        logger.debug(f"Client {client_id} status updated: {status}")

    def start_training_session(self):
        """Mark the start of a training session."""
        self.is_training = True
        self.current_round = 0
        self.training_history.clear()
        logger.info("Training session started")

    def stop_training_session(self):
        """Mark the end of a training session."""
        self.is_training = False
        logger.info("Training session stopped")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training session summary."""
        if not self.training_history:
            return {
                "is_training": self.is_training,
                "current_round": self.current_round,
                "total_rounds": 0,
                "best_accuracy": 0.0,
                "latest_loss": 0.0,
                "active_clients": 0
            }

        latest_metrics = self.training_history[-1]
        best_accuracy = max(m.accuracy for m in self.training_history)
        active_clients = len([c for c in self.client_status.values()
                            if c.status in ["connected", "training"]])

        return {
            "is_training": self.is_training,
            "current_round": self.current_round,
            "total_rounds": len(self.training_history),
            "best_accuracy": best_accuracy,
            "latest_loss": latest_metrics.loss,
            "latest_accuracy": latest_metrics.accuracy,
            "active_clients": active_clients,
            "last_updated": time.time()
        }

    def get_client_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all client statuses."""
        current_time = time.time()
        client_summaries = []

        for client in self.client_status.values():
            # Mark clients as disconnected if not seen recently
            time_since_seen = current_time - client.last_seen
            status = "disconnected" if time_since_seen > 120 else client.status

            client_summaries.append({
                "client_id": client.client_id,
                "status": status,
                "last_seen": client.last_seen,
                "time_since_seen": time_since_seen,
                "loss": client.loss,
                "accuracy": client.accuracy,
                "samples": client.samples
            })

        return sorted(client_summaries, key=lambda x: x["last_seen"], reverse=True)

    def cleanup_old_clients(self, timeout_seconds: int = 300):
        """Remove clients that haven't been seen recently."""
        current_time = time.time()
        old_clients = [
            client_id for client_id, client in self.client_status.items()
            if current_time - client.last_seen > timeout_seconds
        ]

        for client_id in old_clients:
            del self.client_status[client_id]
            logger.debug(f"Removed old client: {client_id}")


class NetworkTopologyMonitor:
    """Monitor network topology and P2P connections."""

    def __init__(self):
        self.peer_connections: Dict[str, Dict[str, Any]] = {}
        self.connection_history: List[Dict[str, Any]] = []

    def update_peer_status(self, peer_id: str, status: str, address: str = "",
                          capabilities: List[str] = None):
        """Update peer connection status."""
        self.peer_connections[peer_id] = {
            "peer_id": peer_id,
            "status": status,
            "address": address,
            "capabilities": capabilities or [],
            "last_seen": time.time()
        }

        # Record connection event
        self.connection_history.append({
            "peer_id": peer_id,
            "event": "status_update",
            "status": status,
            "timestamp": time.time()
        })

        logger.debug(f"Peer {peer_id} status updated: {status}")

    def get_network_topology(self) -> Dict[str, Any]:
        """Get current network topology."""
        active_peers = len([p for p in self.peer_connections.values()
                          if p["status"] == "connected"])

        return {
            "total_peers": len(self.peer_connections),
            "active_peers": active_peers,
            "peers": list(self.peer_connections.values()),
            "last_updated": time.time()
        }

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        recent_events = [e for e in self.connection_history
                        if time.time() - e["timestamp"] < 3600]  # Last hour

        connects = len([e for e in recent_events if e["status"] == "connected"])
        disconnects = len([e for e in recent_events if e["status"] == "disconnected"])

        return {
            "recent_connects": connects,
            "recent_disconnects": disconnects,
            "total_events": len(recent_events),
            "connection_stability": connects / max(connects + disconnects, 1)
        }


class DashboardMonitor:
    """Main dashboard monitoring system."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.system_monitor = SystemMonitor()
        self.training_monitor = TrainingMonitor()
        self.network_monitor = NetworkTopologyMonitor()

        self.is_running = False
        self.update_interval = self.config.get("update_interval", 5)
        self.cleanup_interval = self.config.get("cleanup_interval", 60)

        # Background task for periodic cleanup
        self.cleanup_thread = None

    def start(self):
        """Start all monitoring components."""
        if self.is_running:
            logger.warning("Dashboard monitor is already running")
            return

        self.is_running = True

        # Start system monitoring
        self.system_monitor.start_monitoring()

        # Start cleanup task
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

        logger.info("Dashboard monitoring started")

    def stop(self):
        """Stop all monitoring components."""
        if not self.is_running:
            return

        self.is_running = False

        # Stop system monitoring
        self.system_monitor.stop_monitoring()

        logger.info("Dashboard monitoring stopped")

    def _cleanup_loop(self):
        """Periodic cleanup of old data."""
        while self.is_running:
            try:
                # Cleanup old clients
                self.training_monitor.cleanup_old_clients()

                # Cleanup old connection history
                cutoff_time = time.time() - 86400  # Keep 24 hours
                self.network_monitor.connection_history = [
                    event for event in self.network_monitor.connection_history
                    if event["timestamp"] >= cutoff_time
                ]

                time.sleep(self.cleanup_interval)

            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                time.sleep(self.cleanup_interval)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "system": {
                "current": asdict(self.system_monitor.get_current_metrics()) if self.system_monitor.get_current_metrics() else None,
                "average": asdict(self.system_monitor.get_average_metrics()) if self.system_monitor.get_average_metrics() else None,
                "history": [asdict(m) for m in self.system_monitor.get_metrics_history()]
            },
            "training": {
                "summary": self.training_monitor.get_training_summary(),
                "clients": self.training_monitor.get_client_summary(),
                "history": [asdict(m) for m in self.training_monitor.training_history[-100:]]  # Last 100 rounds
            },
            "network": {
                "topology": self.network_monitor.get_network_topology(),
                "stats": self.network_monitor.get_connection_stats()
            },
            "timestamp": time.time()
        }

    def update_from_fl_server(self, server_status: Dict[str, Any]):
        """Update monitoring data from FL server status."""
        if server_status.get("is_running"):
            if not self.training_monitor.is_training:
                self.training_monitor.start_training_session()
        else:
            if self.training_monitor.is_training:
                self.training_monitor.stop_training_session()

        # Update metrics if available
        metrics = server_status.get("metrics", {})
        if metrics:
            # Process server metrics
            for metric_name, metric_values in metrics.items():
                if isinstance(metric_values, list) and metric_values:
                    latest_metric = metric_values[-1]
                    if "server_loss" in metric_name:
                        round_num = latest_metric.get("round", 0)
                        loss_value = latest_metric.get("value", 0.0)
                        # We need accuracy too, but it might come from a different metric
                        # For now, we'll update when we have both

    def update_from_mesh_network(self, network_status: Dict[str, Any]):
        """Update monitoring data from mesh network status."""
        peers = network_status.get("peers", [])
        for peer in peers:
            self.network_monitor.update_peer_status(
                peer_id=peer.get("peer_id", ""),
                status=peer.get("status", "unknown"),
                address=f"{peer.get('address', '')}:{peer.get('port', '')}",
                capabilities=peer.get("capabilities", [])
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FL Dashboard Monitor")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--update-interval", type=int, default=5, help="Update interval")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Create and start monitor
    config = Config({
        "update_interval": args.update_interval
    })

    monitor = DashboardMonitor(config)
    monitor.start()

    try:
        while True:
            # Print dashboard data periodically
            data = monitor.get_dashboard_data()
            print(f"Dashboard data: {json.dumps(data, indent=2)}")
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard monitor")
        monitor.stop()