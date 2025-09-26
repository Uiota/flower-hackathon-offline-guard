#!/usr/bin/env python3
"""
Agent Orchestrator for UIOTA Offline Guard

Central orchestration system that coordinates all Guardian agents,
manages their lifecycle, and ensures proper collaboration.
"""

import asyncio
import json
import logging
import signal
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict

from auto_save_agent import AutoSaveAgent, create_auto_save_agent
from smooth_setup_agent import SmoothSetupAgent, create_smooth_setup_agent
from security_monitor_agent import SecurityMonitorAgent, create_security_monitor
from development_agent import DevelopmentAgent, create_development_agent
from test_coordinator import TestCoordinator, create_test_coordinator
from communication_hub import CommunicationHub, create_communication_hub
from debug_monitor import DebugMonitor, create_debug_monitor

logger = logging.getLogger(__name__)

@dataclass
class AgentStatus:
    """Status information for an agent."""
    agent_id: str
    agent_type: str
    status: str  # starting, running, stopping, stopped, error
    start_time: Optional[str] = None
    last_heartbeat: Optional[str] = None
    error_message: Optional[str] = None

class AgentOrchestrator:
    """
    Central orchestrator for all Guardian agents in the ecosystem.
    Manages agent lifecycle, coordination, and system-wide operations.
    """

    def __init__(self, project_root: Path = None):
        """
        Initialize the agent orchestrator.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.orchestrator_dir = Path.home() / ".uiota" / "orchestrator"
        self.state_file = self.orchestrator_dir / "orchestrator_state.json"

        # Agent instances
        self.agents: Dict[str, Any] = {}
        self.agent_status: Dict[str, AgentStatus] = {}

        # Core system components
        self.communication_hub: Optional[CommunicationHub] = None
        self.debug_monitor: Optional[DebugMonitor] = None

        # System state
        self._running = False
        self._shutdown_requested = False
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Configuration
        self.config = {
            "orchestration": {
                "auto_start_agents": True,
                "auto_restart_failed": True,
                "heartbeat_interval": 30,
                "graceful_shutdown_timeout": 60
            },
            "agents": {
                "auto_save": {"enabled": True, "priority": 1},
                "communication_hub": {"enabled": True, "priority": 0},
                "debug_monitor": {"enabled": True, "priority": 0},
                "security_monitor": {"enabled": True, "priority": 2},
                "development": {"enabled": True, "priority": 3},
                "smooth_setup": {"enabled": False, "priority": 4},  # On-demand
                "test_coordinator": {"enabled": False, "priority": 5}  # On-demand
            },
            "coordination": {
                "enable_inter_agent_communication": True,
                "enable_debugging": True,
                "enable_security_monitoring": True
            }
        }

        self._init_directories()
        self._setup_signal_handlers()

        logger.info("AgentOrchestrator initialized")

    def _init_directories(self) -> None:
        """Create necessary directories for orchestrator operations."""
        self.orchestrator_dir.mkdir(parents=True, exist_ok=True)

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_requested = True
            self.stop_all_agents()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def initialize_system(self) -> bool:
        """
        Initialize the complete Guardian agent system.

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing Guardian agent system...")

            # 1. Start core infrastructure agents first
            if not self._start_core_agents():
                logger.error("Failed to start core agents")
                return False

            # 2. Start application agents
            if not self._start_application_agents():
                logger.error("Failed to start application agents")
                return False

            # 3. Establish inter-agent communication
            if not self._setup_agent_communication():
                logger.error("Failed to setup agent communication")
                return False

            # 4. Start system monitoring
            if not self._start_system_monitoring():
                logger.error("Failed to start system monitoring")
                return False

            self._running = True

            # 5. Start heartbeat monitoring
            self._start_heartbeat_monitoring()

            logger.info("Guardian agent system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.stop_all_agents()
            return False

    def _start_core_agents(self) -> bool:
        """Start core infrastructure agents."""
        try:
            # Communication Hub (highest priority)
            if self.config["agents"]["communication_hub"]["enabled"]:
                self.communication_hub = create_communication_hub()
                self.communication_hub.start()
                self.agents["communication_hub"] = self.communication_hub
                self._update_agent_status("communication_hub", "running")
                logger.info("Communication hub started")

            # Debug Monitor
            if self.config["agents"]["debug_monitor"]["enabled"]:
                self.debug_monitor = create_debug_monitor(self.communication_hub)
                self.debug_monitor.start()
                self.agents["debug_monitor"] = self.debug_monitor
                self._update_agent_status("debug_monitor", "running")
                logger.info("Debug monitor started")

            return True

        except Exception as e:
            logger.error(f"Failed to start core agents: {e}")
            return False

    def _start_application_agents(self) -> bool:
        """Start application-specific agents."""
        try:
            # Auto Save Agent
            if self.config["agents"]["auto_save"]["enabled"]:
                auto_save = create_auto_save_agent()
                auto_save.start()
                self.agents["auto_save"] = auto_save
                self._update_agent_status("auto_save", "running")
                logger.info("Auto save agent started")

            # Security Monitor
            if self.config["agents"]["security_monitor"]["enabled"]:
                security_monitor = create_security_monitor()
                security_monitor.start()
                self.agents["security_monitor"] = security_monitor
                self._update_agent_status("security_monitor", "running")
                logger.info("Security monitor started")

            # Development Agent
            if self.config["agents"]["development"]["enabled"]:
                development = create_development_agent(self.project_root)
                development.start()
                self.agents["development"] = development
                self._update_agent_status("development", "running")
                logger.info("Development agent started")

            return True

        except Exception as e:
            logger.error(f"Failed to start application agents: {e}")
            return False

    def _setup_agent_communication(self) -> bool:
        """Set up communication between agents."""
        if not self.communication_hub or not self.config["coordination"]["enable_inter_agent_communication"]:
            return True

        try:
            # Register all agents with the communication hub
            for agent_id, agent in self.agents.items():
                if agent_id == "communication_hub":
                    continue

                capabilities = self._get_agent_capabilities(agent_id, agent)

                self.communication_hub.register_agent(
                    agent_id=agent_id,
                    agent_type=type(agent).__name__,
                    capabilities=capabilities,
                    agent_instance=agent,
                    message_handler=getattr(agent, 'handle_message', None),
                    task_handler=getattr(agent, 'handle_task', None)
                )

            # Register agents with debug monitor
            if self.debug_monitor:
                for agent_id, agent in self.agents.items():
                    if agent_id not in ["communication_hub", "debug_monitor"]:
                        self.debug_monitor.register_agent_for_debugging(agent_id, agent)

            logger.info("Agent communication established")
            return True

        except Exception as e:
            logger.error(f"Failed to setup agent communication: {e}")
            return False

    def _get_agent_capabilities(self, agent_id: str, agent: Any) -> List[str]:
        """Get capabilities for an agent."""
        capabilities_map = {
            "auto_save": ["save_state", "backup", "restore"],
            "security_monitor": ["threat_detection", "file_monitoring", "security_analysis"],
            "development": ["code_analysis", "testing", "quality_check"],
            "smooth_setup": ["system_setup", "dependency_management"],
            "test_coordinator": ["test_execution", "test_coordination"],
            "debug_monitor": ["debugging", "performance_monitoring", "diagnostics"]
        }

        return capabilities_map.get(agent_id, [])

    def _start_system_monitoring(self) -> bool:
        """Start system-wide monitoring."""
        try:
            if self.debug_monitor and self.config["coordination"]["enable_debugging"]:
                # Start monitoring system resources
                self.debug_monitor.start_debug_session("system_monitoring", {
                    "purpose": "System-wide monitoring",
                    "monitor_all": True
                })

            return True

        except Exception as e:
            logger.error(f"Failed to start system monitoring: {e}")
            return False

    def _start_heartbeat_monitoring(self) -> None:
        """Start heartbeat monitoring for all agents."""
        def heartbeat_loop():
            while self._running and not self._shutdown_requested:
                try:
                    self._check_agent_health()
                    time.sleep(self.config["orchestration"]["heartbeat_interval"])
                except Exception as e:
                    logger.error(f"Heartbeat monitoring error: {e}")
                    time.sleep(5)

        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _check_agent_health(self) -> None:
        """Check the health of all agents."""
        for agent_id, agent in self.agents.items():
            try:
                # Check if agent is still responsive
                if hasattr(agent, 'get_status'):
                    status = agent.get_status()
                    self._update_agent_heartbeat(agent_id)
                elif hasattr(agent, 'get_security_status'):
                    status = agent.get_security_status()
                    self._update_agent_heartbeat(agent_id)
                elif hasattr(agent, '_running'):
                    if agent._running:
                        self._update_agent_heartbeat(agent_id)
                    else:
                        self._update_agent_status(agent_id, "stopped")
                else:
                    # Basic check - agent exists
                    self._update_agent_heartbeat(agent_id)

            except Exception as e:
                logger.warning(f"Agent {agent_id} health check failed: {e}")
                self._update_agent_status(agent_id, "error", str(e))

                # Auto-restart if configured
                if self.config["orchestration"]["auto_restart_failed"]:
                    self.restart_agent(agent_id)

    def _update_agent_status(self, agent_id: str, status: str, error_message: str = None) -> None:
        """Update the status of an agent."""
        if agent_id not in self.agent_status:
            self.agent_status[agent_id] = AgentStatus(
                agent_id=agent_id,
                agent_type=type(self.agents.get(agent_id, None)).__name__ if self.agents.get(agent_id) else "Unknown",
                status=status,
                start_time=datetime.now().isoformat() if status == "running" else None
            )
        else:
            self.agent_status[agent_id].status = status
            self.agent_status[agent_id].error_message = error_message

        if self.debug_monitor:
            self.debug_monitor._log_debug_event(
                event_type="info",
                source="orchestrator",
                message=f"Agent {agent_id} status changed to {status}",
                details={"agent_id": agent_id, "status": status, "error": error_message}
            )

    def _update_agent_heartbeat(self, agent_id: str) -> None:
        """Update the heartbeat timestamp for an agent."""
        if agent_id in self.agent_status:
            self.agent_status[agent_id].last_heartbeat = datetime.now().isoformat()

    def start_agent_on_demand(self, agent_type: str) -> bool:
        """
        Start an agent on demand.

        Args:
            agent_type: Type of agent to start

        Returns:
            True if agent started successfully
        """
        if agent_type in self.agents:
            logger.warning(f"Agent {agent_type} is already running")
            return True

        try:
            if agent_type == "smooth_setup":
                agent = create_smooth_setup_agent(self.project_root)
                self.agents[agent_type] = agent
                self._update_agent_status(agent_type, "running")

            elif agent_type == "test_coordinator":
                agent = create_test_coordinator(self.project_root)
                self.agents[agent_type] = agent
                self._update_agent_status(agent_type, "running")

            else:
                logger.error(f"Unknown agent type: {agent_type}")
                return False

            # Register with communication hub
            if self.communication_hub:
                capabilities = self._get_agent_capabilities(agent_type, agent)
                self.communication_hub.register_agent(
                    agent_id=agent_type,
                    agent_type=type(agent).__name__,
                    capabilities=capabilities,
                    agent_instance=agent
                )

            # Register with debug monitor
            if self.debug_monitor:
                self.debug_monitor.register_agent_for_debugging(agent_type, agent)

            logger.info(f"On-demand agent started: {agent_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to start on-demand agent {agent_type}: {e}")
            return False

    def restart_agent(self, agent_id: str) -> bool:
        """
        Restart a specific agent.

        Args:
            agent_id: ID of the agent to restart

        Returns:
            True if restart successful
        """
        try:
            logger.info(f"Restarting agent: {agent_id}")

            # Stop the agent
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if hasattr(agent, 'stop'):
                    agent.stop()
                del self.agents[agent_id]

            # Unregister from communication hub
            if self.communication_hub:
                self.communication_hub.unregister_agent(agent_id)

            self._update_agent_status(agent_id, "stopping")

            # Wait a moment
            time.sleep(2)

            # Restart based on agent type
            restart_success = False
            if agent_id == "auto_save":
                new_agent = create_auto_save_agent()
                new_agent.start()
                restart_success = True
            elif agent_id == "security_monitor":
                new_agent = create_security_monitor()
                new_agent.start()
                restart_success = True
            elif agent_id == "development":
                new_agent = create_development_agent(self.project_root)
                new_agent.start()
                restart_success = True
            elif agent_id == "smooth_setup":
                new_agent = create_smooth_setup_agent(self.project_root)
                restart_success = True
            elif agent_id == "test_coordinator":
                new_agent = create_test_coordinator(self.project_root)
                restart_success = True

            if restart_success:
                self.agents[agent_id] = new_agent
                self._update_agent_status(agent_id, "running")

                # Re-register with communication hub
                if self.communication_hub:
                    capabilities = self._get_agent_capabilities(agent_id, new_agent)
                    self.communication_hub.register_agent(
                        agent_id=agent_id,
                        agent_type=type(new_agent).__name__,
                        capabilities=capabilities,
                        agent_instance=new_agent
                    )

                logger.info(f"Agent restarted successfully: {agent_id}")
                return True
            else:
                logger.error(f"Failed to restart agent: {agent_id}")
                self._update_agent_status(agent_id, "error", "Restart failed")
                return False

        except Exception as e:
            logger.error(f"Error restarting agent {agent_id}: {e}")
            self._update_agent_status(agent_id, "error", str(e))
            return False

    def stop_agent(self, agent_id: str) -> bool:
        """
        Stop a specific agent.

        Args:
            agent_id: ID of the agent to stop

        Returns:
            True if stop successful
        """
        try:
            if agent_id not in self.agents:
                logger.warning(f"Agent not found: {agent_id}")
                return False

            logger.info(f"Stopping agent: {agent_id}")

            agent = self.agents[agent_id]
            self._update_agent_status(agent_id, "stopping")

            if hasattr(agent, 'stop'):
                agent.stop()

            # Unregister from communication hub
            if self.communication_hub:
                self.communication_hub.unregister_agent(agent_id)

            del self.agents[agent_id]
            self._update_agent_status(agent_id, "stopped")

            logger.info(f"Agent stopped: {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Error stopping agent {agent_id}: {e}")
            self._update_agent_status(agent_id, "error", str(e))
            return False

    def stop_all_agents(self) -> None:
        """Stop all running agents in proper order."""
        logger.info("Stopping all agents...")

        # Stop in reverse priority order
        agent_priorities = [
            (self.config["agents"][agent_id]["priority"], agent_id)
            for agent_id in self.agents.keys()
            if agent_id in self.config["agents"]
        ]
        agent_priorities.sort(reverse=True)

        for _, agent_id in agent_priorities:
            self.stop_agent(agent_id)

        # Stop core agents last
        if "debug_monitor" in self.agents:
            self.stop_agent("debug_monitor")

        if "communication_hub" in self.agents:
            self.stop_agent("communication_hub")

        self._running = False

        # Stop heartbeat monitoring
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)

        logger.info("All agents stopped")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        agent_statuses = {
            agent_id: asdict(status)
            for agent_id, status in self.agent_status.items()
        }

        # Get additional status from agents
        detailed_status = {}
        for agent_id, agent in self.agents.items():
            try:
                if hasattr(agent, 'get_status'):
                    detailed_status[agent_id] = agent.get_status()
                elif hasattr(agent, 'get_security_status'):
                    detailed_status[agent_id] = agent.get_security_status()
                elif hasattr(agent, 'get_communication_stats'):
                    detailed_status[agent_id] = agent.get_communication_stats()
            except Exception as e:
                detailed_status[agent_id] = {"error": str(e)}

        return {
            "orchestrator": {
                "running": self._running,
                "shutdown_requested": self._shutdown_requested,
                "total_agents": len(self.agents),
                "running_agents": len([s for s in self.agent_status.values() if s.status == "running"])
            },
            "agent_status": agent_statuses,
            "detailed_status": detailed_status,
            "communication_hub": {
                "active": self.communication_hub is not None,
                "stats": self.communication_hub.get_communication_stats() if self.communication_hub else None
            },
            "debug_monitor": {
                "active": self.debug_monitor is not None,
                "diagnostics": self.debug_monitor.get_system_diagnostics() if self.debug_monitor else None
            }
        }

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive system test."""
        if "test_coordinator" not in self.agents:
            if not self.start_agent_on_demand("test_coordinator"):
                return {"error": "Failed to start test coordinator"}

        test_coordinator = self.agents["test_coordinator"]
        return test_coordinator.run_tests()

    def run_system_setup(self) -> Dict[str, Any]:
        """Run system setup using smooth setup agent."""
        if "smooth_setup" not in self.agents:
            if not self.start_agent_on_demand("smooth_setup"):
                return {"error": "Failed to start smooth setup agent"}

        smooth_setup = self.agents["smooth_setup"]

        # Run setup asynchronously
        async def run_setup():
            return await smooth_setup.run_setup()

        # Create new event loop for this operation
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_setup())
            loop.close()
            return {"setup_completed": result}
        except Exception as e:
            return {"error": f"Setup failed: {e}"}

    def save_system_state(self) -> bool:
        """Save the current system state."""
        try:
            state_data = {
                "timestamp": datetime.now().isoformat(),
                "running": self._running,
                "agent_status": {
                    agent_id: asdict(status)
                    for agent_id, status in self.agent_status.items()
                },
                "config": self.config
            }

            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            # Trigger auto-save for all agents
            if "auto_save" in self.agents:
                self.agents["auto_save"].force_save()

            logger.info("System state saved")
            return True

        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
            return False

def create_agent_orchestrator(project_root: Path = None) -> AgentOrchestrator:
    """Factory function to create a configured AgentOrchestrator."""
    return AgentOrchestrator(project_root)

if __name__ == "__main__":
    # Main orchestrator execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    orchestrator = create_agent_orchestrator()

    try:
        # Initialize the complete system
        if orchestrator.initialize_system():
            logger.info("Guardian Agent System is now operational")

            # Run indefinitely
            while not orchestrator._shutdown_requested:
                time.sleep(10)

                # Periodic system health check
                status = orchestrator.get_system_status()
                running_agents = status["orchestrator"]["running_agents"]
                total_agents = status["orchestrator"]["total_agents"]

                if running_agents < total_agents:
                    logger.warning(f"Some agents are not running: {running_agents}/{total_agents}")

        else:
            logger.error("Failed to initialize Guardian Agent System")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        # Save final state
        orchestrator.save_system_state()
        orchestrator.stop_all_agents()
        logger.info("Guardian Agent System shutdown complete")