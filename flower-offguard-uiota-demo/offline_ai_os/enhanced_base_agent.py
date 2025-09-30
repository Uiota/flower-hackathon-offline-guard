#!/usr/bin/env python3
"""
Offline AI Operating System - Base Agent Implementation
Core agent system with factory pattern for cybersecurity operations
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from queue import PriorityQueue
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ==================== ENUMS & DATA CLASSES ====================

class AgentState(Enum):
    """Agent lifecycle states"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"

class AgentType(Enum):
    """Specialized agent types"""
    COORDINATOR = "coordinator"
    THREAT_DETECTOR = "threat_detector"
    MALWARE_ANALYZER = "malware_analyzer"
    NETWORK_MONITOR = "network_monitor"
    INCIDENT_RESPONDER = "incident_responder"
    LOG_ANALYZER = "log_analyzer"
    VULNERABILITY_SCANNER = "vulnerability_scanner"
    LEARNING_AGENT = "learning_agent"

class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class AgentCapability:
    """Defines an agent capability"""
    name: str
    description: str
    version: str
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentMessage:
    """Inter-agent communication message"""
    id: str
    sender_id: str
    receiver_id: str
    message_type: str
    payload: Dict[str, Any]
    priority: MessagePriority
    timestamp: datetime
    encrypted: bool = False

    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority.value < other.priority.value

@dataclass
class ThreatEvent:
    """Security threat event"""
    event_id: str
    severity: str
    source_ip: str
    destination_ip: str
    event_type: str
    description: str
    timestamp: datetime
    raw_data: Dict[str, Any]

# ==================== BASE AGENT CLASS ====================

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    Implements core functionality: lifecycle, communication, monitoring.
    """

    def __init__(self, agent_id: str, agent_type: AgentType, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.state = AgentState.INITIALIZING
        self.capabilities: List[AgentCapability] = []
        self.message_queue = PriorityQueue()
        self.logger = logging.getLogger(f"{agent_type.value}_{agent_id[:8]}")
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "tasks_completed": 0,
            "errors": 0,
            "start_time": datetime.now()
        }

    async def initialize(self):
        """Initialize agent and load capabilities"""
        self.logger.info(f"Initializing agent {self.agent_id}")
        await self.load_capabilities()
        self.state = AgentState.IDLE
        self.logger.info(f"Agent {self.agent_id} initialized successfully")

    @abstractmethod
    async def load_capabilities(self):
        """Load agent-specific capabilities"""
        pass

    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message and optionally return response"""
        pass

    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute assigned task"""
        pass

    async def send_message(self, receiver_id: str, message_type: str,
                          payload: Dict[str, Any], priority: MessagePriority = MessagePriority.MEDIUM):
        """Send message to another agent"""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            priority=priority,
            timestamp=datetime.now()
        )
        self.metrics["messages_sent"] += 1
        self.logger.debug(f"Sending message {message.id} to {receiver_id}")
        # In real implementation, this would go through message broker
        return message

    async def receive_message(self, message: AgentMessage):
        """Receive and queue message"""
        self.message_queue.put(message)
        self.metrics["messages_received"] += 1
        self.logger.debug(f"Received message {message.id} from {message.sender_id}")

    async def run(self):
        """Main agent loop"""
        await self.initialize()
        self.logger.info(f"Agent {self.agent_id} starting main loop")

        while self.state != AgentState.TERMINATED:
            try:
                if self.state == AgentState.PAUSED:
                    await asyncio.sleep(1)
                    continue

                self.state = AgentState.ACTIVE

                # Process queued messages
                if not self.message_queue.empty():
                    message = self.message_queue.get()
                    response = await self.process_message(message)
                    if response:
                        await self.send_message(
                            message.sender_id,
                            "response",
                            response.payload,
                            response.priority
                        )

                # Perform agent-specific work
                await self.work_cycle()

                self.state = AgentState.IDLE
                await asyncio.sleep(0.1)  # Prevent CPU spinning

            except Exception as e:
                self.logger.error(f"Error in agent loop: {str(e)}")
                self.metrics["errors"] += 1
                self.state = AgentState.ERROR
                await asyncio.sleep(5)  # Backoff on error

    async def work_cycle(self):
        """Agent-specific work cycle - override in subclasses"""
        pass

    def pause(self):
        """Pause agent execution"""
        self.state = AgentState.PAUSED
        self.logger.info(f"Agent {self.agent_id} paused")

    def resume(self):
        """Resume agent execution"""
        if self.state == AgentState.PAUSED:
            self.state = AgentState.IDLE
            self.logger.info(f"Agent {self.agent_id} resumed")

    def terminate(self):
        """Terminate agent"""
        self.state = AgentState.TERMINATED
        self.logger.info(f"Agent {self.agent_id} terminated")

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        uptime = (datetime.now() - self.metrics["start_time"]).total_seconds()
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "state": self.state.value,
            "uptime_seconds": uptime,
            **self.metrics
        }

# ==================== SPECIALIZED AGENTS ====================

class ThreatDetectorAgent(BaseAgent):
    """Agent specialized in detecting security threats"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.THREAT_DETECTOR, config)
        self.threat_patterns = []
        self.detection_threshold = config.get("threshold", 0.75)

    async def load_capabilities(self):
        """Load threat detection capabilities"""
        self.capabilities = [
            AgentCapability(
                name="anomaly_detection",
                description="Detect behavioral anomalies in network traffic",
                version="1.0.0",
                parameters={"sensitivity": self.detection_threshold}
            ),
            AgentCapability(
                name="signature_matching",
                description="Match known attack signatures",
                version="1.0.0"
            )
        ]
        # Load threat patterns (in real system, from local DB)
        self.threat_patterns = self.config.get("patterns", [])

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming security events"""
        if message.message_type == "analyze_traffic":
            traffic_data = message.payload.get("traffic_data")
            threat = await self.analyze_traffic(traffic_data)

            if threat:
                return AgentMessage(
                    id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type="threat_detected",
                    payload={"threat": threat},
                    priority=MessagePriority.HIGH,
                    timestamp=datetime.now()
                )
        return None

    async def analyze_traffic(self, traffic_data: Dict[str, Any]) -> Optional[ThreatEvent]:
        """Analyze network traffic for threats"""
        # Simplified threat detection logic
        anomaly_score = self.calculate_anomaly_score(traffic_data)

        if anomaly_score > self.detection_threshold:
            threat = ThreatEvent(
                event_id=str(uuid.uuid4()),
                severity="high" if anomaly_score > 0.9 else "medium",
                source_ip=traffic_data.get("source_ip", "unknown"),
                destination_ip=traffic_data.get("dest_ip", "unknown"),
                event_type="anomaly_detected",
                description=f"Anomalous traffic detected (score: {anomaly_score:.2f})",
                timestamp=datetime.now(),
                raw_data=traffic_data
            )
            self.logger.warning(f"Threat detected: {threat.event_id}")
            return threat
        return None

    def calculate_anomaly_score(self, traffic_data: Dict[str, Any]) -> float:
        """Calculate anomaly score for traffic"""
        # Simplified scoring - in real system, use ML model
        score = 0.0

        # Check packet size
        if traffic_data.get("packet_size", 0) > 1500:
            score += 0.3

        # Check connection rate
        if traffic_data.get("connections_per_second", 0) > 100:
            score += 0.4

        # Check for suspicious ports
        suspicious_ports = [4444, 31337, 12345]
        if traffic_data.get("dest_port") in suspicious_ports:
            score += 0.5

        return min(score, 1.0)

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute threat detection task"""
        task_type = task.get("type")

        if task_type == "scan_network":
            # Simulate network scanning
            results = {
                "threats_detected": 0,
                "scanned_hosts": task.get("host_count", 0),
                "timestamp": datetime.now().isoformat()
            }
            self.metrics["tasks_completed"] += 1
            return results

        return {"status": "unknown_task_type"}

    async def work_cycle(self):
        """Continuous threat monitoring"""
        # In real system, this would monitor live traffic
        await asyncio.sleep(1)


class IncidentResponderAgent(BaseAgent):
    """Agent specialized in responding to security incidents"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.INCIDENT_RESPONDER, config)
        self.response_playbooks = {}

    async def load_capabilities(self):
        """Load incident response capabilities"""
        self.capabilities = [
            AgentCapability(
                name="auto_isolation",
                description="Automatically isolate compromised hosts",
                version="1.0.0"
            ),
            AgentCapability(
                name="threat_mitigation",
                description="Execute threat mitigation procedures",
                version="1.0.0"
            )
        ]

        # Load response playbooks
        self.response_playbooks = {
            "malware_detected": self.respond_to_malware,
            "ddos_attack": self.respond_to_ddos,
            "data_exfiltration": self.respond_to_exfiltration
        }

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incident alerts"""
        if message.message_type == "threat_detected":
            threat = message.payload.get("threat")
            response = await self.respond_to_threat(threat)

            return AgentMessage(
                id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="response_executed",
                payload={"response": response},
                priority=MessagePriority.HIGH,
                timestamp=datetime.now()
            )
        return None

    async def respond_to_threat(self, threat: Dict[str, Any]) -> Dict[str, Any]:
        """Execute response to detected threat"""
        event_type = threat.get("event_type")
        playbook = self.response_playbooks.get(event_type, self.default_response)

        self.logger.info(f"Executing response for {event_type}")
        response_result = await playbook(threat)

        return {
            "threat_id": threat.get("event_id"),
            "response_type": event_type,
            "actions_taken": response_result,
            "timestamp": datetime.now().isoformat()
        }

    async def respond_to_malware(self, threat: Dict[str, Any]) -> List[str]:
        """Respond to malware detection"""
        actions = []

        # Isolate infected host
        source_ip = threat.get("source_ip")
        actions.append(f"Isolated host: {source_ip}")

        # Scan for spread
        actions.append("Initiated network-wide malware scan")

        # Alert admins
        actions.append("Alert sent to security team")

        return actions

    async def respond_to_ddos(self, threat: Dict[str, Any]) -> List[str]:
        """Respond to DDoS attack"""
        actions = []
        actions.append("Activated rate limiting")
        actions.append("Blocked malicious source IPs")
        actions.append("Scaled up resources")
        return actions

    async def respond_to_exfiltration(self, threat: Dict[str, Any]) -> List[str]:
        """Respond to data exfiltration attempt"""
        actions = []
        actions.append("Blocked outbound connections")
        actions.append("Preserved forensic evidence")
        actions.append("Initiated investigation workflow")
        return actions

    async def default_response(self, threat: Dict[str, Any]) -> List[str]:
        """Default response for unknown threats"""
        return ["Logged incident", "Alerted security team"]

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute incident response task"""
        self.metrics["tasks_completed"] += 1
        return {"status": "completed", "task_id": task.get("id")}


class CoordinatorAgent(BaseAgent):
    """Agent that coordinates other agents"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.COORDINATOR, config)
        self.managed_agents: Dict[str, BaseAgent] = {}

    async def load_capabilities(self):
        """Load coordination capabilities"""
        self.capabilities = [
            AgentCapability(
                name="agent_orchestration",
                description="Coordinate multiple specialized agents",
                version="1.0.0"
            ),
            AgentCapability(
                name="task_distribution",
                description="Distribute tasks to appropriate agents",
                version="1.0.0"
            )
        ]

    async def register_agent(self, agent: BaseAgent):
        """Register an agent for coordination"""
        self.managed_agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent {agent.agent_id} ({agent.agent_type.value})")

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process coordination requests"""
        if message.message_type == "request_analysis":
            # Delegate to appropriate agent
            target_agent = self.find_suitable_agent(message.payload)
            if target_agent:
                await target_agent.receive_message(message)
        return None

    def find_suitable_agent(self, task_requirements: Dict[str, Any]) -> Optional[BaseAgent]:
        """Find agent suitable for task"""
        required_type = task_requirements.get("agent_type")

        for agent in self.managed_agents.values():
            if agent.agent_type.value == required_type and agent.state == AgentState.IDLE:
                return agent
        return None

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate task execution across agents"""
        self.metrics["tasks_completed"] += 1
        return {"status": "coordinated"}

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all managed agents"""
        return {
            "coordinator_id": self.agent_id,
            "total_agents": len(self.managed_agents),
            "agents": [
                {
                    "id": agent.agent_id,
                    "type": agent.agent_type.value,
                    "state": agent.state.value,
                    "metrics": agent.get_metrics()
                }
                for agent in self.managed_agents.values()
            ]
        }

# ==================== AGENT FACTORY ====================

class AgentFactory:
    """
    Factory for creating and managing agents.
    Implements factory pattern with agent pooling and lifecycle management.
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_pool: Dict[AgentType, List[BaseAgent]] = {
            agent_type: [] for agent_type in AgentType
        }
        self.logger = logging.getLogger("AgentFactory")

    def create_agent(self, agent_type: AgentType, config: Optional[Dict[str, Any]] = None) -> BaseAgent:
        """Create new agent instance"""
        if config is None:
            config = {}

        agent_id = self.generate_agent_id(agent_type)

        # Agent type mapping
        agent_classes = {
            AgentType.THREAT_DETECTOR: ThreatDetectorAgent,
            AgentType.INCIDENT_RESPONDER: IncidentResponderAgent,
            AgentType.COORDINATOR: CoordinatorAgent,
        }

        agent_class = agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent = agent_class(agent_id, config)
        self.agents[agent_id] = agent
        self.agent_pool[agent_type].append(agent)

        self.logger.info(f"Created agent {agent_id} of type {agent_type.value}")
        return agent

    def generate_agent_id(self, agent_type: AgentType) -> str:
        """Generate unique agent ID"""
        timestamp = datetime.now().isoformat()
        data = f"{agent_type.value}_{timestamp}_{uuid.uuid4()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Retrieve agent by ID"""
        return self.agents.get(agent_id)

    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all agents of specific type"""
        return self.agent_pool.get(agent_type, [])

    def terminate_agent(self, agent_id: str):
        """Terminate and remove agent"""
        agent = self.agents.get(agent_id)
        if agent:
            agent.terminate()
            self.agent_pool[agent.agent_type].remove(agent)
            del self.agents[agent_id]
            self.logger.info(f"Terminated agent {agent_id}")

    def get_factory_status(self) -> Dict[str, Any]:
        """Get factory statistics"""
        return {
            "total_agents": len(self.agents),
            "agents_by_type": {
                agent_type.value: len(agents)
                for agent_type, agents in self.agent_pool.items()
            },
            "active_agents": sum(
                1 for agent in self.agents.values()
                if agent.state == AgentState.ACTIVE
            )
        }

# ==================== DEMO USAGE ====================

async def demo_system():
    """Demonstrate the agent system"""
    print("=" * 60)
    print("Offline AI OS - Agent System Demo")
    print("=" * 60)

    # Create factory
    factory = AgentFactory()

    # Create coordinator
    coordinator = factory.create_agent(AgentType.COORDINATOR, {})
    await coordinator.initialize()

    # Create specialized agents
    detector = factory.create_agent(
        AgentType.THREAT_DETECTOR,
        {"threshold": 0.7, "patterns": []}
    )
    await detector.initialize()

    responder = factory.create_agent(
        AgentType.INCIDENT_RESPONDER,
        {}
    )
    await responder.initialize()

    # Register agents with coordinator
    await coordinator.register_agent(detector)
    await coordinator.register_agent(responder)

    print("\n✓ System initialized with 3 agents")
    print(f"  - Coordinator: {coordinator.agent_id[:8]}")
    print(f"  - Threat Detector: {detector.agent_id[:8]}")
    print(f"  - Incident Responder: {responder.agent_id[:8]}")

    # Simulate threat detection
    print("\n→ Simulating threat detection...")

    traffic_data = {
        "source_ip": "192.168.1.100",
        "dest_ip": "10.0.0.50",
        "dest_port": 4444,
        "packet_size": 2000,
        "connections_per_second": 150
    }

    threat = await detector.analyze_traffic(traffic_data)

    if threat:
        print(f"  ⚠ Threat detected: {threat.event_type}")
        print(f"    Severity: {threat.severity}")
        print(f"    Source: {threat.source_ip}")

        # Send to responder
        message = await detector.send_message(
            responder.agent_id,
            "threat_detected",
            {"threat": threat.__dict__},
            MessagePriority.HIGH
        )

        await responder.receive_message(message)
        response = await responder.process_message(message)

        if response:
            print(f"  ✓ Response executed:")
            actions = response.payload["response"]["actions_taken"]
            for action in actions:
                print(f"    - {action}")

    # Show system status
    print("\n→ System Status:")
    status = coordinator.get_system_status()
    for agent_info in status["agents"]:
        print(f"  Agent: {agent_info['type']}")
        print(f"    State: {agent_info['state']}")
        print(f"    Messages: {agent_info['metrics']['messages_sent']} sent, "
              f"{agent_info['metrics']['messages_received']} received")

    print("\n→ Factory Status:")
    factory_status = factory.get_factory_status()
    print(f"  Total agents: {factory_status['total_agents']}")
    print(f"  Active agents: {factory_status['active_agents']}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

# Run demo
if __name__ == "__main__":
    asyncio.run(demo_system())