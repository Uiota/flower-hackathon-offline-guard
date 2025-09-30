#!/usr/bin/env python3
"""
Base Agent System for Offline AI OS
Foundation classes for all agent types with lifecycle management
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle states"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentMessage:
    """Message format for inter-agent communication"""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: str
    payload: Dict[str, Any]
    priority: MessagePriority
    timestamp: str
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'priority': self.priority.value
        }


@dataclass
class AgentCapabilities:
    """Defines agent capabilities and skills"""
    detection: List[str] = field(default_factory=list)
    analysis: List[str] = field(default_factory=list)
    response: List[str] = field(default_factory=list)
    learning: List[str] = field(default_factory=list)
    communication: List[str] = field(default_factory=list)

    def has_capability(self, category: str, capability: str) -> bool:
        """Check if agent has specific capability"""
        return capability in getattr(self, category, [])

    def all_capabilities(self) -> List[str]:
        """Get all capabilities as flat list"""
        return (self.detection + self.analysis + self.response +
                self.learning + self.communication)


@dataclass
class AgentMetrics:
    """Agent performance and health metrics"""
    messages_received: int = 0
    messages_sent: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    errors_count: int = 0
    avg_task_duration_ms: float = 0.0
    uptime_seconds: float = 0.0
    last_activity: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class BaseAgent(ABC):
    """
    Abstract base agent with core functionality
    All specialized agents inherit from this
    """

    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        # Identity
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.agent_type = self.__class__.__name__

        # Configuration
        self.config = config or {}

        # State management
        self.state = AgentState.INITIALIZING
        self.start_time = time.time()

        # Capabilities
        self.capabilities = AgentCapabilities()

        # Metrics
        self.metrics = AgentMetrics()

        # Message handling
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_handlers: Dict[str, Callable] = {}

        # Task management
        self.current_task: Optional[asyncio.Task] = None
        self.background_tasks: List[asyncio.Task] = []

        # Lifecycle callbacks
        self.on_start_callbacks: List[Callable] = []
        self.on_stop_callbacks: List[Callable] = []

        logger.info(f"Agent {self.agent_id} created (type: {self.agent_type})")

    # ==================== LIFECYCLE METHODS ====================

    async def initialize(self):
        """Initialize agent and transition to IDLE state"""
        logger.info(f"Initializing agent {self.agent_id}")

        try:
            # Register default message handlers
            self._register_default_handlers()

            # Agent-specific initialization
            await self._on_initialize()

            # Start message processor
            self.background_tasks.append(
                asyncio.create_task(self._process_messages())
            )

            # Start health monitor
            self.background_tasks.append(
                asyncio.create_task(self._health_monitor())
            )

            # Transition to IDLE
            self.state = AgentState.IDLE

            # Call start callbacks
            for callback in self.on_start_callbacks:
                await callback(self)

            logger.info(f"Agent {self.agent_id} initialized successfully")

        except Exception as e:
            logger.error(f"Agent {self.agent_id} initialization failed: {e}")
            self.state = AgentState.ERROR
            raise

    @abstractmethod
    async def _on_initialize(self):
        """Agent-specific initialization logic"""
        pass

    async def start(self):
        """Start agent operations"""
        if self.state != AgentState.IDLE:
            logger.warning(f"Agent {self.agent_id} not in IDLE state, cannot start")
            return

        self.state = AgentState.ACTIVE
        logger.info(f"Agent {self.agent_id} started")

    def pause(self):
        """Pause agent operations"""
        if self.state == AgentState.ACTIVE:
            self.state = AgentState.PAUSED
            logger.info(f"Agent {self.agent_id} paused")

    def resume(self):
        """Resume agent operations"""
        if self.state == AgentState.PAUSED:
            self.state = AgentState.ACTIVE
            logger.info(f"Agent {self.agent_id} resumed")

    async def terminate(self):
        """Gracefully terminate agent"""
        logger.info(f"Terminating agent {self.agent_id}")
        self.state = AgentState.TERMINATING

        # Call stop callbacks
        for callback in self.on_stop_callbacks:
            await callback(self)

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        # Agent-specific cleanup
        await self._on_terminate()

        self.state = AgentState.TERMINATED
        logger.info(f"Agent {self.agent_id} terminated")

    @abstractmethod
    async def _on_terminate(self):
        """Agent-specific cleanup logic"""
        pass

    # ==================== MESSAGE HANDLING ====================

    async def send_message(self,
                          recipient_id: str,
                          message_type: str,
                          payload: Dict[str, Any],
                          priority: MessagePriority = MessagePriority.NORMAL):
        """Send message to another agent"""
        message = AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            priority=priority,
            timestamp=datetime.now().isoformat()
        )

        # In real implementation, this would go through message bus
        # For now, simulate sending
        self.metrics.messages_sent += 1
        self.metrics.last_activity = datetime.now().isoformat()

        logger.debug(f"Agent {self.agent_id} sent message to {recipient_id}: {message_type}")

        return message

    async def receive_message(self, message: AgentMessage):
        """Receive message from another agent"""
        await self.message_queue.put(message)
        self.metrics.messages_received += 1
        self.metrics.last_activity = datetime.now().isoformat()

    async def _process_messages(self):
        """Background task to process incoming messages"""
        while self.state != AgentState.TERMINATED:
            try:
                # Get message from queue with timeout
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Only process if active
                if self.state != AgentState.ACTIVE:
                    # Re-queue if not active
                    await self.message_queue.put(message)
                    await asyncio.sleep(0.1)
                    continue

                # Find handler for message type
                handler = self.message_handlers.get(message.message_type)

                if handler:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Error handling message in {self.agent_id}: {e}")
                        self.metrics.errors_count += 1
                else:
                    logger.warning(f"No handler for message type: {message.message_type}")

            except Exception as e:
                logger.error(f"Error in message processor for {self.agent_id}: {e}")
                await asyncio.sleep(1)

    def register_message_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message type"""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type} in {self.agent_id}")

    def _register_default_handlers(self):
        """Register default message handlers"""
        self.register_message_handler("ping", self._handle_ping)
        self.register_message_handler("status_request", self._handle_status_request)
        self.register_message_handler("shutdown", self._handle_shutdown)

    async def _handle_ping(self, message: AgentMessage):
        """Handle ping message"""
        await self.send_message(
            message.sender_id,
            "pong",
            {"agent_id": self.agent_id, "state": self.state.value},
            priority=message.priority
        )

    async def _handle_status_request(self, message: AgentMessage):
        """Handle status request"""
        status = self.get_status()
        await self.send_message(
            message.sender_id,
            "status_response",
            status,
            priority=message.priority
        )

    async def _handle_shutdown(self, message: AgentMessage):
        """Handle shutdown command"""
        await self.terminate()

    # ==================== TASK EXECUTION ====================

    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task (to be overridden by specialized agents)"""
        start_time = time.time()

        try:
            result = await self._execute_task_impl(task_data)

            # Update metrics
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.tasks_completed += 1
            self._update_avg_duration(duration_ms)

            return {
                "success": True,
                "result": result,
                "duration_ms": duration_ms
            }

        except Exception as e:
            logger.error(f"Task execution failed in {self.agent_id}: {e}")
            self.metrics.tasks_failed += 1
            self.metrics.errors_count += 1

            return {
                "success": False,
                "error": str(e),
                "duration_ms": (time.time() - start_time) * 1000
            }

    @abstractmethod
    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Any:
        """Agent-specific task execution logic"""
        pass

    # ==================== HEALTH & MONITORING ====================

    async def _health_monitor(self):
        """Background task to monitor agent health"""
        while self.state != AgentState.TERMINATED:
            try:
                # Update uptime
                self.metrics.uptime_seconds = time.time() - self.start_time

                # Check health
                is_healthy = await self._check_health()

                if not is_healthy and self.state == AgentState.ACTIVE:
                    logger.warning(f"Agent {self.agent_id} health check failed")
                    self.state = AgentState.ERROR

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in health monitor for {self.agent_id}: {e}")
                await asyncio.sleep(30)

    async def _check_health(self) -> bool:
        """Check agent health (can be overridden)"""
        # Basic health checks
        if self.metrics.errors_count > 100:
            return False

        if self.message_queue.qsize() > 1000:
            return False

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self.state.value,
            "capabilities": self.capabilities.all_capabilities(),
            "metrics": self.metrics.to_dict(),
            "config": self.config,
            "queue_size": self.message_queue.qsize()
        }

    def _update_avg_duration(self, new_duration_ms: float):
        """Update average task duration"""
        completed = self.metrics.tasks_completed
        if completed == 1:
            self.metrics.avg_task_duration_ms = new_duration_ms
        else:
            current_avg = self.metrics.avg_task_duration_ms
            self.metrics.avg_task_duration_ms = (
                (current_avg * (completed - 1) + new_duration_ms) / completed
            )

    # ==================== CAPABILITY MANAGEMENT ====================

    def add_capability(self, category: str, capability: str):
        """Add capability to agent"""
        cap_list = getattr(self.capabilities, category, None)
        if cap_list is not None and capability not in cap_list:
            cap_list.append(capability)
            logger.info(f"Added capability {capability} to {self.agent_id}")

    def remove_capability(self, category: str, capability: str):
        """Remove capability from agent"""
        cap_list = getattr(self.capabilities, category, None)
        if cap_list is not None and capability in cap_list:
            cap_list.remove(capability)
            logger.info(f"Removed capability {capability} from {self.agent_id}")

    # ==================== UTILITY METHODS ====================

    def __repr__(self):
        return f"<{self.agent_type}(id={self.agent_id}, state={self.state.value})>"

    def __str__(self):
        return f"{self.agent_type}[{self.agent_id}]"


# ==================== EXAMPLE SPECIALIZED AGENT ====================

class ThreatDetectorAgent(BaseAgent):
    """Example specialized agent for threat detection"""

    async def _on_initialize(self):
        """Initialize threat detector"""
        # Add detection capabilities
        self.add_capability("detection", "signature_matching")
        self.add_capability("detection", "anomaly_detection")
        self.add_capability("analysis", "threat_classification")

        # Register custom handlers
        self.register_message_handler("analyze_threat", self._handle_analyze_threat)

        # Load detection models (placeholder)
        self.detection_model = None  # Load actual model here

        logger.info(f"ThreatDetectorAgent {self.agent_id} initialized")

    async def _on_terminate(self):
        """Cleanup threat detector"""
        # Unload models, close connections, etc.
        logger.info(f"ThreatDetectorAgent {self.agent_id} cleaned up")

    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Any:
        """Execute threat detection task"""
        # Simulate threat detection
        await asyncio.sleep(0.1)  # Simulate processing

        return {
            "threats_detected": 2,
            "threat_types": ["malware", "intrusion_attempt"],
            "confidence": 0.95
        }

    async def _handle_analyze_threat(self, message: AgentMessage):
        """Handle threat analysis request"""
        threat_data = message.payload

        # Perform analysis
        result = await self.execute_task(threat_data)

        # Send response
        await self.send_message(
            message.sender_id,
            "threat_analysis_result",
            result,
            priority=MessagePriority.HIGH
        )


# ==================== DEMO ====================

async def demo_base_agent_system():
    """Demonstrate base agent system"""
    print("=" * 70)
    print("Base Agent System Demo")
    print("=" * 70)

    # Create threat detector agent
    detector = ThreatDetectorAgent(config={"threshold": 0.85})

    # Initialize
    await detector.initialize()
    print(f"\n✓ Created and initialized: {detector}")

    # Start agent
    await detector.start()
    print(f"✓ Agent started: {detector.state.value}")

    # Show capabilities
    print(f"\n✓ Capabilities: {detector.capabilities.all_capabilities()}")

    # Execute task
    print("\n→ Executing threat detection task...")
    result = await detector.execute_task({
        "source_ip": "192.168.1.100",
        "suspicious_pattern": "buffer_overflow_attempt"
    })
    print(f"✓ Task completed: {result}")

    # Send and receive messages
    print("\n→ Testing message handling...")
    ping_msg = AgentMessage(
        message_id="test_msg_001",
        sender_id="test_sender",
        recipient_id=detector.agent_id,
        message_type="ping",
        payload={},
        priority=MessagePriority.NORMAL,
        timestamp=datetime.now().isoformat()
    )
    await detector.receive_message(ping_msg)
    await asyncio.sleep(0.5)  # Allow processing
    print("✓ Ping message sent and processed")

    # Get status
    print("\n→ Agent Status:")
    status = detector.get_status()
    print(f"  State: {status['state']}")
    print(f"  Tasks Completed: {status['metrics']['tasks_completed']}")
    print(f"  Messages Received: {status['metrics']['messages_received']}")
    print(f"  Messages Sent: {status['metrics']['messages_sent']}")
    print(f"  Uptime: {status['metrics']['uptime_seconds']:.1f}s")

    # Pause and resume
    print("\n→ Testing pause/resume...")
    detector.pause()
    print(f"✓ Agent paused: {detector.state.value}")
    detector.resume()
    print(f"✓ Agent resumed: {detector.state.value}")

    # Terminate
    print("\n→ Terminating agent...")
    await detector.terminate()
    print(f"✓ Agent terminated: {detector.state.value}")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_base_agent_system())