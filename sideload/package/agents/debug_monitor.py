#!/usr/bin/env python3
"""
Debug Monitor for UIOTA Offline Guard

Real-time debugging and monitoring tools for Guardian agent interactions,
system diagnostics, and performance analysis.
"""

import asyncio
import json
import logging
import psutil
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import traceback
import sys
import gc
import weakref

from communication_hub import CommunicationHub
from auto_save_agent import AutoSaveAgent
from security_monitor_agent import SecurityMonitorAgent
from development_agent import DevelopmentAgent
from test_coordinator import TestCoordinator

logger = logging.getLogger(__name__)

@dataclass
class DebugEvent:
    """Represents a debug event or diagnostic finding."""
    timestamp: str
    event_type: str  # error, warning, info, performance, interaction
    source: str
    message: str
    details: Dict[str, Any]
    severity: str = "info"  # critical, high, medium, low, info
    stack_trace: Optional[str] = None

@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    timestamp: str
    metric_name: str
    value: float
    unit: str
    source: str
    metadata: Dict[str, Any] = None

@dataclass
class InteractionTrace:
    """Trace of interaction between agents."""
    trace_id: str
    start_time: str
    end_time: Optional[str]
    source_agent: str
    target_agent: str
    interaction_type: str  # message, task, api_call
    data: Dict[str, Any]
    status: str = "active"  # active, completed, failed, timeout

class DebugMonitor:
    """
    Comprehensive debugging and monitoring system for Guardian agents.
    Provides real-time diagnostics, performance tracking, and interaction analysis.
    """

    def __init__(self, communication_hub: CommunicationHub = None):
        """
        Initialize the debug monitor.

        Args:
            communication_hub: Reference to the communication hub for monitoring
        """
        self.debug_dir = Path.home() / ".uiota" / "debug"
        self.logs_dir = self.debug_dir / "logs"
        self.traces_dir = self.debug_dir / "traces"
        self.reports_dir = self.debug_dir / "reports"

        # External references
        self.communication_hub = communication_hub
        self.monitored_agents: Dict[str, weakref.ReferenceType] = {}

        # Debug data collection
        self.debug_events: deque = deque(maxlen=5000)
        self.performance_metrics: deque = deque(maxlen=10000)
        self.interaction_traces: Dict[str, InteractionTrace] = {}

        # Real-time monitoring
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._performance_thread: Optional[threading.Thread] = None

        # Event handlers and hooks
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.performance_thresholds: Dict[str, float] = {}

        # Debugging state
        self.debug_sessions: Dict[str, Dict[str, Any]] = {}
        self.breakpoints: Set[str] = set()
        self.watch_variables: Dict[str, Any] = {}

        # Configuration
        self.config = {
            "monitoring": {
                "collect_performance": True,
                "trace_interactions": True,
                "log_all_events": True,
                "monitor_memory": True,
                "monitor_cpu": True
            },
            "thresholds": {
                "memory_usage_mb": 500,
                "cpu_usage_percent": 80,
                "response_time_ms": 1000,
                "error_rate_percent": 5
            },
            "debugging": {
                "auto_breakpoint_errors": True,
                "capture_stack_traces": True,
                "deep_inspection": False
            },
            "reporting": {
                "generate_hourly_reports": True,
                "save_interaction_maps": True,
                "performance_dashboards": True
            }
        }

        self._init_directories()
        self._setup_performance_thresholds()
        self._setup_event_hooks()

        logger.info("DebugMonitor initialized")

    def _init_directories(self) -> None:
        """Create necessary directories for debug operations."""
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _setup_performance_thresholds(self) -> None:
        """Set up performance monitoring thresholds."""
        self.performance_thresholds = {
            "memory_usage": self.config["thresholds"]["memory_usage_mb"],
            "cpu_usage": self.config["thresholds"]["cpu_usage_percent"],
            "response_time": self.config["thresholds"]["response_time_ms"],
            "error_rate": self.config["thresholds"]["error_rate_percent"]
        }

    def _setup_event_hooks(self) -> None:
        """Set up event hooks for automatic monitoring."""
        if self.communication_hub:
            # Hook into communication hub events
            self.communication_hub.add_event_handler("agent_registered", self._on_agent_registered)
            self.communication_hub.add_event_handler("agent_unregistered", self._on_agent_unregistered)

    def register_agent_for_debugging(self, agent_id: str, agent_instance: Any) -> None:
        """
        Register an agent for detailed debugging monitoring.

        Args:
            agent_id: Agent identifier
            agent_instance: Reference to the agent instance
        """
        self.monitored_agents[agent_id] = weakref.ref(agent_instance)

        # Start debug session
        self.start_debug_session(agent_id, {
            "start_time": datetime.now().isoformat(),
            "agent_type": type(agent_instance).__name__,
            "monitoring_enabled": True
        })

        self._log_debug_event(
            event_type="info",
            source="debug_monitor",
            message=f"Agent registered for debugging: {agent_id}",
            details={"agent_type": type(agent_instance).__name__}
        )

    def _on_agent_registered(self, event_data: Dict[str, Any]) -> None:
        """Handle agent registration events."""
        agent_id = event_data.get("agent_id")
        self._log_debug_event(
            event_type="interaction",
            source="communication_hub",
            message=f"Agent registered: {agent_id}",
            details=event_data
        )

    def _on_agent_unregistered(self, event_data: Dict[str, Any]) -> None:
        """Handle agent unregistration events."""
        agent_id = event_data.get("agent_id")
        self._log_debug_event(
            event_type="interaction",
            source="communication_hub",
            message=f"Agent unregistered: {agent_id}",
            details=event_data
        )

        # End debug session if exists
        if agent_id in self.debug_sessions:
            self.end_debug_session(agent_id)

    def start_debug_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """Start a new debugging session."""
        self.debug_sessions[session_id] = {
            **session_data,
            "events": [],
            "performance_data": [],
            "interactions": []
        }

        self._log_debug_event(
            event_type="info",
            source="debug_monitor",
            message=f"Debug session started: {session_id}",
            details=session_data
        )

    def end_debug_session(self, session_id: str) -> Dict[str, Any]:
        """End a debugging session and return summary."""
        if session_id not in self.debug_sessions:
            return {"error": f"Session not found: {session_id}"}

        session = self.debug_sessions[session_id]
        session["end_time"] = datetime.now().isoformat()

        # Generate session report
        report = self._generate_session_report(session_id, session)

        # Save session data
        self._save_debug_session(session_id, session, report)

        # Clean up
        del self.debug_sessions[session_id]

        self._log_debug_event(
            event_type="info",
            source="debug_monitor",
            message=f"Debug session ended: {session_id}",
            details={"report_summary": report.get("summary", {})}
        )

        return report

    def _log_debug_event(self, event_type: str, source: str, message: str,
                        details: Dict[str, Any] = None, severity: str = "info") -> None:
        """Log a debug event."""
        if details is None:
            details = {}

        # Capture stack trace for errors
        stack_trace = None
        if (event_type == "error" and
            self.config["debugging"]["capture_stack_traces"]):
            stack_trace = traceback.format_exc()

        event = DebugEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            source=source,
            message=message,
            details=details,
            severity=severity,
            stack_trace=stack_trace
        )

        self.debug_events.append(event)

        # Add to active debug sessions
        for session_id, session in self.debug_sessions.items():
            if source in session_id or session.get("monitor_all", False):
                session["events"].append(asdict(event))

        # Log to file
        self._save_debug_event(event)

        # Trigger event handlers
        self._trigger_debug_handlers(event)

        # Auto-breakpoint on errors if configured
        if (event_type == "error" and
            self.config["debugging"]["auto_breakpoint_errors"]):
            self.add_breakpoint(f"error_{source}_{int(time.time())}")

    def _record_performance_metric(self, metric_name: str, value: float, unit: str,
                                  source: str, metadata: Dict[str, Any] = None) -> None:
        """Record a performance metric."""
        if not self.config["monitoring"]["collect_performance"]:
            return

        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            metric_name=metric_name,
            value=value,
            unit=unit,
            source=source,
            metadata=metadata or {}
        )

        self.performance_metrics.append(metric)

        # Check thresholds
        if metric_name in self.performance_thresholds:
            threshold = self.performance_thresholds[metric_name]
            if value > threshold:
                self._log_debug_event(
                    event_type="warning",
                    source="performance_monitor",
                    message=f"Performance threshold exceeded: {metric_name}",
                    details={
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "source": source
                    },
                    severity="medium"
                )

        # Add to active debug sessions
        for session_id, session in self.debug_sessions.items():
            if source in session_id or session.get("monitor_all", False):
                session["performance_data"].append(asdict(metric))

    def trace_interaction(self, source_agent: str, target_agent: str,
                         interaction_type: str, data: Dict[str, Any]) -> str:
        """Start tracing an interaction between agents."""
        if not self.config["monitoring"]["trace_interactions"]:
            return ""

        trace_id = f"{source_agent}_{target_agent}_{int(time.time() * 1000)}"

        trace = InteractionTrace(
            trace_id=trace_id,
            start_time=datetime.now().isoformat(),
            end_time=None,
            source_agent=source_agent,
            target_agent=target_agent,
            interaction_type=interaction_type,
            data=data,
            status="active"
        )

        self.interaction_traces[trace_id] = trace

        self._log_debug_event(
            event_type="interaction",
            source="trace_monitor",
            message=f"Interaction trace started: {source_agent} -> {target_agent}",
            details={
                "trace_id": trace_id,
                "interaction_type": interaction_type,
                "data_keys": list(data.keys())
            }
        )

        return trace_id

    def end_interaction_trace(self, trace_id: str, status: str = "completed",
                             result: Dict[str, Any] = None) -> None:
        """End an interaction trace."""
        if trace_id not in self.interaction_traces:
            return

        trace = self.interaction_traces[trace_id]
        trace.end_time = datetime.now().isoformat()
        trace.status = status

        if result:
            trace.data.update({"result": result})

        # Calculate duration
        start_time = datetime.fromisoformat(trace.start_time)
        end_time = datetime.fromisoformat(trace.end_time)
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Record performance metric
        self._record_performance_metric(
            metric_name="interaction_duration",
            value=duration_ms,
            unit="ms",
            source="trace_monitor",
            metadata={
                "trace_id": trace_id,
                "interaction_type": trace.interaction_type,
                "status": status
            }
        )

        self._log_debug_event(
            event_type="interaction",
            source="trace_monitor",
            message=f"Interaction trace ended: {trace.source_agent} -> {trace.target_agent}",
            details={
                "trace_id": trace_id,
                "duration_ms": duration_ms,
                "status": status
            }
        )

        # Add to debug sessions
        for session_id, session in self.debug_sessions.items():
            session["interactions"].append(asdict(trace))

        # Save trace
        self._save_interaction_trace(trace)

    def add_breakpoint(self, breakpoint_id: str) -> None:
        """Add a debugging breakpoint."""
        self.breakpoints.add(breakpoint_id)

        self._log_debug_event(
            event_type="info",
            source="debug_monitor",
            message=f"Breakpoint added: {breakpoint_id}",
            details={"breakpoint_id": breakpoint_id}
        )

    def remove_breakpoint(self, breakpoint_id: str) -> None:
        """Remove a debugging breakpoint."""
        self.breakpoints.discard(breakpoint_id)

        self._log_debug_event(
            event_type="info",
            source="debug_monitor",
            message=f"Breakpoint removed: {breakpoint_id}",
            details={"breakpoint_id": breakpoint_id}
        )

    def add_watch_variable(self, var_name: str, var_value: Any) -> None:
        """Add a variable to the watch list."""
        self.watch_variables[var_name] = var_value

        self._log_debug_event(
            event_type="info",
            source="debug_monitor",
            message=f"Variable added to watch: {var_name}",
            details={"variable": var_name, "type": type(var_value).__name__}
        )

    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics."""
        try:
            # System resource usage
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')

            # Python process info
            process = psutil.Process()
            process_memory = process.memory_info()

            # Agent status
            agent_status = {}
            if self.communication_hub:
                agent_status = self.communication_hub.get_agent_status()

            # Recent debug events by type
            event_counts = defaultdict(int)
            recent_events = list(self.debug_events)[-100:]
            for event in recent_events:
                event_counts[event.event_type] += 1

            # Performance metrics summary
            recent_metrics = list(self.performance_metrics)[-100:]
            metric_summary = {}
            for metric in recent_metrics:
                if metric.metric_name not in metric_summary:
                    metric_summary[metric.metric_name] = {
                        "count": 0, "total": 0, "min": float('inf'), "max": 0
                    }
                summary = metric_summary[metric.metric_name]
                summary["count"] += 1
                summary["total"] += metric.value
                summary["min"] = min(summary["min"], metric.value)
                summary["max"] = max(summary["max"], metric.value)

            for name, summary in metric_summary.items():
                summary["average"] = summary["total"] / summary["count"]

            return {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "memory_percent": memory.percent,
                    "memory_available_mb": memory.available / (1024 * 1024),
                    "cpu_percent": cpu_percent,
                    "disk_percent": (disk.used / disk.total) * 100,
                    "disk_free_gb": disk.free / (1024 * 1024 * 1024)
                },
                "process": {
                    "memory_rss_mb": process_memory.rss / (1024 * 1024),
                    "memory_vms_mb": process_memory.vms / (1024 * 1024),
                    "cpu_percent": process.cpu_percent(),
                    "num_threads": process.num_threads()
                },
                "agents": agent_status,
                "debug_events": {
                    "total_events": len(self.debug_events),
                    "event_counts": dict(event_counts)
                },
                "performance": {
                    "total_metrics": len(self.performance_metrics),
                    "metric_summary": metric_summary
                },
                "debugging": {
                    "active_sessions": len(self.debug_sessions),
                    "breakpoints": len(self.breakpoints),
                    "watch_variables": len(self.watch_variables),
                    "interaction_traces": len(self.interaction_traces)
                }
            }

        except Exception as e:
            logger.error(f"Failed to get system diagnostics: {e}")
            return {"error": str(e)}

    def _system_monitoring_loop(self) -> None:
        """Main system monitoring loop."""
        logger.info("System monitoring started")

        while self._running:
            try:
                if self.config["monitoring"]["monitor_memory"]:
                    memory = psutil.virtual_memory()
                    self._record_performance_metric(
                        "memory_usage", memory.percent, "percent", "system"
                    )

                if self.config["monitoring"]["monitor_cpu"]:
                    cpu_percent = psutil.cpu_percent()
                    self._record_performance_metric(
                        "cpu_usage", cpu_percent, "percent", "system"
                    )

                # Monitor Python process
                process = psutil.Process()
                process_memory = process.memory_info().rss / (1024 * 1024)  # MB
                self._record_performance_metric(
                    "process_memory", process_memory, "MB", "python_process"
                )

                # Garbage collection stats
                gc_stats = gc.get_stats()
                if gc_stats:
                    self._record_performance_metric(
                        "gc_collections", sum(stat['collections'] for stat in gc_stats),
                        "count", "garbage_collector"
                    )

                time.sleep(10)  # Monitor every 10 seconds

            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(5)

        logger.info("System monitoring stopped")

    def _trigger_debug_handlers(self, event: DebugEvent) -> None:
        """Trigger registered debug event handlers."""
        for handler in self.event_handlers[event.event_type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Debug event handler error: {e}")

    def _save_debug_event(self, event: DebugEvent) -> None:
        """Save debug event to persistent storage."""
        if not self.config["monitoring"]["log_all_events"]:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = self.logs_dir / f"debug_events_{timestamp}.jsonl"

            with open(log_file, 'a') as f:
                f.write(json.dumps(asdict(event)) + '\n')

        except Exception as e:
            logger.error(f"Failed to save debug event: {e}")

    def _save_interaction_trace(self, trace: InteractionTrace) -> None:
        """Save interaction trace to persistent storage."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            trace_file = self.traces_dir / f"interactions_{timestamp}.jsonl"

            with open(trace_file, 'a') as f:
                f.write(json.dumps(asdict(trace)) + '\n')

        except Exception as e:
            logger.error(f"Failed to save interaction trace: {e}")

    def _save_debug_session(self, session_id: str, session: Dict[str, Any],
                           report: Dict[str, Any]) -> None:
        """Save debug session data."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_file = self.reports_dir / f"debug_session_{session_id}_{timestamp}.json"

            session_data = {
                "session_id": session_id,
                "session": session,
                "report": report
            }

            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save debug session: {e}")

    def _generate_session_report(self, session_id: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive debug session report."""
        events = session.get("events", [])
        performance_data = session.get("performance_data", [])
        interactions = session.get("interactions", [])

        # Event analysis
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        for event in events:
            event_counts[event["event_type"]] += 1
            severity_counts[event["severity"]] += 1

        # Performance analysis
        perf_summary = defaultdict(list)
        for metric in performance_data:
            perf_summary[metric["metric_name"]].append(metric["value"])

        perf_stats = {}
        for metric_name, values in perf_summary.items():
            if values:
                perf_stats[metric_name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "average": sum(values) / len(values)
                }

        # Interaction analysis
        interaction_stats = {
            "total_interactions": len(interactions),
            "completed": len([i for i in interactions if i["status"] == "completed"]),
            "failed": len([i for i in interactions if i["status"] == "failed"]),
            "active": len([i for i in interactions if i["status"] == "active"])
        }

        return {
            "session_id": session_id,
            "duration": self._calculate_session_duration(session),
            "summary": {
                "total_events": len(events),
                "event_distribution": dict(event_counts),
                "severity_distribution": dict(severity_counts),
                "performance_metrics": len(performance_data),
                "performance_stats": perf_stats,
                "interaction_stats": interaction_stats
            },
            "recommendations": self._generate_recommendations(events, performance_data, interactions)
        }

    def _calculate_session_duration(self, session: Dict[str, Any]) -> float:
        """Calculate session duration in seconds."""
        try:
            start_time = datetime.fromisoformat(session["start_time"])
            end_time = datetime.fromisoformat(session.get("end_time", datetime.now().isoformat()))
            return (end_time - start_time).total_seconds()
        except:
            return 0.0

    def _generate_recommendations(self, events: List[Dict], performance_data: List[Dict],
                                 interactions: List[Dict]) -> List[str]:
        """Generate debugging and optimization recommendations."""
        recommendations = []

        # Error analysis
        errors = [e for e in events if e["event_type"] == "error"]
        if len(errors) > 5:
            recommendations.append(f"High error rate detected ({len(errors)} errors). Review error handling.")

        # Performance analysis
        if performance_data:
            high_memory = [p for p in performance_data if p["metric_name"] == "memory_usage" and p["value"] > 80]
            if high_memory:
                recommendations.append("High memory usage detected. Consider memory optimization.")

            high_cpu = [p for p in performance_data if p["metric_name"] == "cpu_usage" and p["value"] > 90]
            if high_cpu:
                recommendations.append("High CPU usage detected. Review computational efficiency.")

        # Interaction analysis
        failed_interactions = [i for i in interactions if i["status"] == "failed"]
        if len(failed_interactions) > len(interactions) * 0.1:  # >10% failure rate
            recommendations.append("High interaction failure rate. Review agent communication.")

        return recommendations

    def start(self) -> None:
        """Start the debug monitor."""
        if self._running:
            logger.warning("Debug monitor is already running")
            return

        self._running = True

        # Start monitoring threads
        self._monitor_thread = threading.Thread(target=self._system_monitoring_loop, daemon=True)
        self._monitor_thread.start()

        logger.info("DebugMonitor started")

    def stop(self) -> None:
        """Stop the debug monitor."""
        if not self._running:
            return

        self._running = False

        # Stop monitoring threads
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)

        # End all active debug sessions
        for session_id in list(self.debug_sessions.keys()):
            self.end_debug_session(session_id)

        logger.info("DebugMonitor stopped")

def create_debug_monitor(communication_hub: CommunicationHub = None) -> DebugMonitor:
    """Factory function to create a configured DebugMonitor."""
    return DebugMonitor(communication_hub)

if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    monitor = create_debug_monitor()

    try:
        monitor.start()

        # Demo debug session
        monitor.start_debug_session("demo_session", {"purpose": "testing"})

        # Simulate some events
        monitor._log_debug_event("info", "demo", "Demo event 1")
        monitor._log_debug_event("warning", "demo", "Demo warning")
        monitor._log_debug_event("error", "demo", "Demo error", severity="high")

        # Record some metrics
        monitor._record_performance_metric("demo_metric", 42.5, "units", "demo")

        # Trace an interaction
        trace_id = monitor.trace_interaction("agent_a", "agent_b", "message", {"data": "test"})
        time.sleep(1)
        monitor.end_interaction_trace(trace_id, "completed", {"result": "success"})

        time.sleep(5)

        # Get diagnostics
        diagnostics = monitor.get_system_diagnostics()
        print(f"System Diagnostics: {json.dumps(diagnostics, indent=2)}")

        # End session
        report = monitor.end_debug_session("demo_session")
        print(f"Session Report: {json.dumps(report, indent=2)}")

    except KeyboardInterrupt:
        print("Demo interrupted")
    finally:
        monitor.stop()