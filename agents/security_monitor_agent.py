#!/usr/bin/env python3
"""
Security Monitor Agent for UIOTA Offline Guard

Real-time security monitoring, threat detection, and defensive response
for the Guardian ecosystem. Monitors system integrity, network traffic,
and agent interactions for potential security threats.
"""

import asyncio
import json
import logging
import os
import psutil
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import socket
import subprocess

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Represents a security event or threat."""
    timestamp: str
    event_type: str
    severity: str  # critical, high, medium, low
    source: str
    description: str
    details: Dict[str, Any]
    action_taken: Optional[str] = None
    resolved: bool = False

@dataclass
class ThreatSignature:
    """Signature for threat detection."""
    name: str
    pattern: str
    signature_type: str  # file, network, process, api
    severity: str
    description: str
    detection_logic: Optional[str] = None

class SecurityMonitorAgent:
    """
    Continuous security monitoring agent for the Guardian ecosystem.
    Provides real-time threat detection, system integrity monitoring,
    and automated defensive responses.
    """

    def __init__(self, config_path: Path = None):
        """
        Initialize the security monitor agent.

        Args:
            config_path: Path to security configuration file
        """
        self.config_path = config_path or Path.home() / ".uiota" / "security_config.json"
        self.alerts_dir = Path.home() / ".uiota" / "security_alerts"
        self.quarantine_dir = Path.home() / ".uiota" / "quarantine"

        # Monitoring state
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._event_queue = deque(maxlen=1000)
        self._security_events: List[SecurityEvent] = []
        self._threat_signatures: List[ThreatSignature] = []

        # Metrics tracking
        self._system_baselines: Dict[str, float] = {}
        self._network_connections: Set[Tuple[str, int]] = set()
        self._file_hashes: Dict[str, str] = {}
        self._process_whitelist: Set[str] = set()

        # Rate limiting for alerts
        self._alert_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

        self._init_directories()
        self._load_config()
        self._load_threat_signatures()
        self._establish_baselines()

        logger.info("SecurityMonitorAgent initialized")

    def _init_directories(self) -> None:
        """Create necessary directories for security operations."""
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        # Create logs directory
        logs_dir = Path.home() / ".uiota" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> None:
        """Load security configuration."""
        default_config = {
            "monitoring": {
                "check_interval": 5,
                "file_monitoring": True,
                "network_monitoring": True,
                "process_monitoring": True,
                "performance_monitoring": True
            },
            "thresholds": {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "disk_usage": 90.0,
                "network_connections": 50,
                "failed_logins": 5
            },
            "response": {
                "auto_quarantine": True,
                "alert_rate_limit": 60,
                "emergency_shutdown": False
            },
            "whitelist": {
                "processes": ["python", "podman", "jupyter"],
                "ports": [8080, 8888, 22],
                "directories": [str(Path.home() / ".uiota")]
            }
        }

        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    self.config = {**default_config, **loaded_config}
            except Exception as e:
                logger.warning(f"Failed to load config, using defaults: {e}")
                self.config = default_config
        else:
            self.config = default_config
            self._save_config()

    def _save_config(self) -> None:
        """Save current configuration."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def _load_threat_signatures(self) -> None:
        """Load threat detection signatures."""
        # Basic threat signatures for demonstration
        self._threat_signatures = [
            ThreatSignature(
                name="suspicious_network_scan",
                pattern="multiple_connections_same_host",
                signature_type="network",
                severity="medium",
                description="Potential network scanning activity detected"
            ),
            ThreatSignature(
                name="unauthorized_file_access",
                pattern="access_outside_whitelist",
                signature_type="file",
                severity="high",
                description="File access outside authorized directories"
            ),
            ThreatSignature(
                name="resource_exhaustion",
                pattern="high_resource_usage",
                signature_type="process",
                severity="medium",
                description="Abnormally high resource consumption"
            ),
            ThreatSignature(
                name="privilege_escalation_attempt",
                pattern="sudo_without_auth",
                signature_type="process",
                severity="critical",
                description="Potential privilege escalation attempt"
            )
        ]

    def _establish_baselines(self) -> None:
        """Establish security baselines for system monitoring."""
        try:
            # CPU and memory baselines
            cpu_samples = []
            memory_samples = []

            for _ in range(10):
                cpu_samples.append(psutil.cpu_percent())
                memory_samples.append(psutil.virtual_memory().percent)
                time.sleep(0.5)

            self._system_baselines = {
                "cpu_baseline": sum(cpu_samples) / len(cpu_samples),
                "memory_baseline": sum(memory_samples) / len(memory_samples),
                "established_at": time.time()
            }

            # Network baseline
            connections = psutil.net_connections()
            self._network_connections = {
                (conn.laddr.ip if conn.laddr else "", conn.laddr.port if conn.laddr else 0)
                for conn in connections if conn.status == 'LISTEN'
            }

            # File integrity baselines for critical files
            self._establish_file_baselines()

            logger.info(f"Security baselines established: {self._system_baselines}")

        except Exception as e:
            logger.error(f"Failed to establish baselines: {e}")

    def _establish_file_baselines(self) -> None:
        """Establish file integrity baselines."""
        critical_files = [
            Path.home() / ".uiota" / "security_config.json",
            Path.cwd() / ".guardian" / "config.yaml",
            Path.cwd() / "agents" / "*.py"
        ]

        for file_pattern in critical_files:
            if '*' in str(file_pattern):
                # Handle glob patterns
                parent = file_pattern.parent
                pattern = file_pattern.name
                if parent.exists():
                    for file_path in parent.glob(pattern):
                        if file_path.is_file():
                            self._file_hashes[str(file_path)] = self._calculate_file_hash(file_path)
            else:
                if file_pattern.exists() and file_pattern.is_file():
                    self._file_hashes[str(file_pattern)] = self._calculate_file_hash(file_pattern)

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash file {file_path}: {e}")
            return ""

    def _check_file_integrity(self) -> List[SecurityEvent]:
        """Check file integrity against baselines."""
        events = []

        for file_path, baseline_hash in self._file_hashes.items():
            if not Path(file_path).exists():
                events.append(SecurityEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="file_deletion",
                    severity="high",
                    source="file_monitor",
                    description=f"Critical file deleted: {file_path}",
                    details={"file_path": file_path, "baseline_hash": baseline_hash}
                ))
                continue

            current_hash = self._calculate_file_hash(Path(file_path))
            if current_hash and current_hash != baseline_hash:
                events.append(SecurityEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="file_modification",
                    severity="medium",
                    source="file_monitor",
                    description=f"Critical file modified: {file_path}",
                    details={
                        "file_path": file_path,
                        "baseline_hash": baseline_hash,
                        "current_hash": current_hash
                    }
                ))
                # Update baseline after detecting change
                self._file_hashes[file_path] = current_hash

        return events

    def _check_system_resources(self) -> List[SecurityEvent]:
        """Monitor system resource usage for anomalies."""
        events = []

        try:
            # CPU monitoring
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > self.config["thresholds"]["cpu_usage"]:
                events.append(SecurityEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="high_cpu_usage",
                    severity="medium",
                    source="resource_monitor",
                    description=f"High CPU usage detected: {cpu_percent}%",
                    details={"cpu_percent": cpu_percent, "threshold": self.config["thresholds"]["cpu_usage"]}
                ))

            # Memory monitoring
            memory = psutil.virtual_memory()
            if memory.percent > self.config["thresholds"]["memory_usage"]:
                events.append(SecurityEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="high_memory_usage",
                    severity="medium",
                    source="resource_monitor",
                    description=f"High memory usage detected: {memory.percent}%",
                    details={"memory_percent": memory.percent, "threshold": self.config["thresholds"]["memory_usage"]}
                ))

            # Disk monitoring
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > self.config["thresholds"]["disk_usage"]:
                events.append(SecurityEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="high_disk_usage",
                    severity="high",
                    source="resource_monitor",
                    description=f"High disk usage detected: {disk_percent:.1f}%",
                    details={"disk_percent": disk_percent, "threshold": self.config["thresholds"]["disk_usage"]}
                ))

        except Exception as e:
            logger.error(f"Resource monitoring error: {e}")

        return events

    def _check_network_activity(self) -> List[SecurityEvent]:
        """Monitor network connections for suspicious activity."""
        events = []

        try:
            current_connections = set()
            new_connections = set()

            connections = psutil.net_connections()
            for conn in connections:
                if conn.laddr:
                    addr_tuple = (conn.laddr.ip, conn.laddr.port)
                    current_connections.add(addr_tuple)

                    if addr_tuple not in self._network_connections:
                        new_connections.add(addr_tuple)

            # Check for too many new connections
            if len(new_connections) > 5:
                events.append(SecurityEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="suspicious_network_activity",
                    severity="medium",
                    source="network_monitor",
                    description=f"Multiple new network connections detected: {len(new_connections)}",
                    details={"new_connections": list(new_connections)}
                ))

            # Update baseline with legitimate connections
            self._network_connections.update(current_connections)

        except Exception as e:
            logger.error(f"Network monitoring error: {e}")

        return events

    def _check_process_activity(self) -> List[SecurityEvent]:
        """Monitor running processes for suspicious activity."""
        events = []

        try:
            suspicious_processes = []
            high_resource_processes = []

            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    proc_name = proc_info['name']

                    # Check against whitelist
                    if not any(allowed in proc_name.lower() for allowed in self.config["whitelist"]["processes"]):
                        # Check if process is consuming unusual resources
                        if (proc_info['cpu_percent'] > 20.0 or proc_info['memory_percent'] > 10.0):
                            high_resource_processes.append({
                                'pid': proc_info['pid'],
                                'name': proc_name,
                                'cpu': proc_info['cpu_percent'],
                                'memory': proc_info['memory_percent']
                            })

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if high_resource_processes:
                events.append(SecurityEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="suspicious_process_activity",
                    severity="medium",
                    source="process_monitor",
                    description=f"Processes with high resource usage detected: {len(high_resource_processes)}",
                    details={"processes": high_resource_processes}
                ))

        except Exception as e:
            logger.error(f"Process monitoring error: {e}")

        return events

    def _handle_security_event(self, event: SecurityEvent) -> None:
        """Handle a detected security event."""
        # Rate limiting
        event_key = f"{event.event_type}_{event.source}"
        now = time.time()

        # Clean old timestamps
        cutoff = now - self.config["response"]["alert_rate_limit"]
        while self._alert_timestamps[event_key] and self._alert_timestamps[event_key][0] < cutoff:
            self._alert_timestamps[event_key].popleft()

        # Check rate limit
        if len(self._alert_timestamps[event_key]) >= 5:
            logger.debug(f"Rate limiting event: {event_key}")
            return

        self._alert_timestamps[event_key].append(now)

        # Log the event
        logger.warning(f"Security Event [{event.severity}]: {event.description}")

        # Save event to disk
        self._save_security_event(event)

        # Add to internal tracking
        self._security_events.append(event)

        # Take automatic response actions
        self._respond_to_event(event)

    def _save_security_event(self, event: SecurityEvent) -> None:
        """Save security event to persistent storage."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            alerts_file = self.alerts_dir / f"security_events_{timestamp}.json"

            # Load existing events
            events = []
            if alerts_file.exists():
                with open(alerts_file, 'r') as f:
                    events = json.load(f)

            # Add new event
            events.append(asdict(event))

            # Save back to file
            with open(alerts_file, 'w') as f:
                json.dump(events, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save security event: {e}")

    def _respond_to_event(self, event: SecurityEvent) -> None:
        """Take automated response to security events."""
        try:
            if event.severity == "critical":
                logger.critical(f"CRITICAL SECURITY EVENT: {event.description}")

                if self.config["response"]["emergency_shutdown"]:
                    self._emergency_shutdown()

            elif event.severity == "high":
                if event.event_type == "file_deletion" and self.config["response"]["auto_quarantine"]:
                    self._quarantine_threat(event)

        except Exception as e:
            logger.error(f"Failed to respond to security event: {e}")

    def _quarantine_threat(self, event: SecurityEvent) -> None:
        """Quarantine detected threats."""
        try:
            quarantine_file = self.quarantine_dir / f"threat_{int(time.time())}.json"
            with open(quarantine_file, 'w') as f:
                json.dump(asdict(event), f, indent=2)

            logger.info(f"Threat quarantined: {quarantine_file}")

        except Exception as e:
            logger.error(f"Failed to quarantine threat: {e}")

    def _emergency_shutdown(self) -> None:
        """Emergency shutdown of critical systems."""
        logger.critical("INITIATING EMERGENCY SECURITY SHUTDOWN")

        try:
            # Stop containers
            subprocess.run(["podman", "stop", "--all"], timeout=30)

            # Stop the monitoring agent itself
            self.stop()

        except Exception as e:
            logger.error(f"Emergency shutdown failed: {e}")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Security monitoring started")

        while self._running:
            try:
                events = []

                # Run all monitoring checks
                if self.config["monitoring"]["file_monitoring"]:
                    events.extend(self._check_file_integrity())

                if self.config["monitoring"]["performance_monitoring"]:
                    events.extend(self._check_system_resources())

                if self.config["monitoring"]["network_monitoring"]:
                    events.extend(self._check_network_activity())

                if self.config["monitoring"]["process_monitoring"]:
                    events.extend(self._check_process_activity())

                # Handle detected events
                for event in events:
                    self._handle_security_event(event)

                # Sleep until next check
                time.sleep(self.config["monitoring"]["check_interval"])

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)  # Brief pause before retrying

        logger.info("Security monitoring stopped")

    def start(self) -> None:
        """Start the security monitoring agent."""
        if self._running:
            logger.warning("Security monitor is already running")
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info("SecurityMonitorAgent started")

    def stop(self) -> None:
        """Stop the security monitoring agent."""
        if not self._running:
            return

        self._running = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)

        logger.info("SecurityMonitorAgent stopped")

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status and recent events."""
        recent_events = [
            asdict(event) for event in self._security_events[-10:]
        ]

        return {
            "status": "monitoring" if self._running else "stopped",
            "baselines_established": bool(self._system_baselines),
            "monitored_files": len(self._file_hashes),
            "recent_events": recent_events,
            "config": self.config
        }

    def add_threat_signature(self, signature: ThreatSignature) -> None:
        """Add a new threat detection signature."""
        self._threat_signatures.append(signature)
        logger.info(f"Added threat signature: {signature.name}")

    def whitelist_process(self, process_name: str) -> None:
        """Add a process to the whitelist."""
        self.config["whitelist"]["processes"].append(process_name)
        self._save_config()
        logger.info(f"Whitelisted process: {process_name}")

def create_security_monitor() -> SecurityMonitorAgent:
    """Factory function to create a configured SecurityMonitorAgent."""
    return SecurityMonitorAgent()

if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    monitor = create_security_monitor()

    try:
        monitor.start()

        # Run for demo period
        time.sleep(30)

        status = monitor.get_security_status()
        print(f"Security Status: {json.dumps(status, indent=2)}")

    except KeyboardInterrupt:
        print("Demo interrupted")
    finally:
        monitor.stop()