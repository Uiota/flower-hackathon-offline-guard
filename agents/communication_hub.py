#!/usr/bin/env python3
"""
Communication Hub for UIOTA Offline Guard

Central communication and task relay system for coordinating
between different Guardian agents in the ecosystem.
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import uuid
import weakref

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Represents a message between agents."""
    id: str
    source_agent: str
    target_agent: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: str
    priority: str = "normal"  # low, normal, high, critical
    requires_response: bool = False
    correlation_id: Optional[str] = None

@dataclass
class TaskRequest:
    """Represents a task request from one agent to another."""
    id: str
    source_agent: str
    target_agent: str
    task_type: str
    task_data: Dict[str, Any]
    priority: str
    timeout: int
    callback: Optional[Callable] = None
    timestamp: str = ""
    status: str = "pending"  # pending, in_progress, completed, failed, timeout

@dataclass
class AgentInfo:
    """Information about a registered agent."""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    status: str  # online, offline, busy, error
    last_seen: str
    message_handler: Optional[Callable] = None
    task_handler: Optional[Callable] = None

class CommunicationHub:
    """
    Central hub for agent communication and task coordination.
    Manages message routing, task delegation, and status tracking.
    """

    def __init__(self):
        """Initialize the communication hub."""
        self.hub_dir = Path.home() / ".uiota" / "communication"
        self.message_log_dir = self.hub_dir / "messages"
        self.task_log_dir = self.hub_dir / "tasks"

        # Agent registry
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_instances: Dict[str, weakref.ReferenceType] = {}

        # Message and task queues
        self.message_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.task_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.pending_tasks: Dict[str, TaskRequest] = {}

        # Message routing and delivery
        self.message_history: deque = deque(maxlen=10000)
        self.delivery_stats: Dict[str, int] = defaultdict(int)

        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        # Runtime state
        self._running = False
        self._hub_thread: Optional[threading.Thread] = None
        self._message_workers: List[threading.Thread] = []

        # Configuration
        self.config = {
            "delivery": {
                "max_retries": 3,
                "retry_delay": 2,
                "batch_size": 10,
                "timeout_seconds": 30
            },
            "routing": {
                "enable_broadcast": True,
                "enable_priority_queue": True,
                "load_balancing": True
            },
            "logging": {
                "log_all_messages": True,
                "log_task_details": True,
                "rotate_logs": True
            }
        }

        self._init_directories()

        logger.info("CommunicationHub initialized")

    def _init_directories(self) -> None:
        """Create necessary directories for communication logging."""
        self.hub_dir.mkdir(parents=True, exist_ok=True)
        self.message_log_dir.mkdir(parents=True, exist_ok=True)
        self.task_log_dir.mkdir(parents=True, exist_ok=True)

    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str],
                      agent_instance: Any = None, message_handler: Callable = None,
                      task_handler: Callable = None) -> bool:
        """
        Register an agent with the communication hub.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (e.g., 'security_monitor', 'development')
            capabilities: List of capabilities/tasks the agent can handle
            agent_instance: Reference to the actual agent instance
            message_handler: Function to handle incoming messages
            task_handler: Function to handle incoming tasks

        Returns:
            True if registration successful
        """
        try:
            agent_info = AgentInfo(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities,
                status="online",
                last_seen=datetime.now().isoformat(),
                message_handler=message_handler,
                task_handler=task_handler
            )

            self.agents[agent_id] = agent_info

            if agent_instance:
                self.agent_instances[agent_id] = weakref.ref(agent_instance)

            # Initialize queues
            self.message_queues[agent_id] = deque(maxlen=1000)
            self.task_queues[agent_id] = deque(maxlen=500)

            # Trigger registration event
            self._trigger_event("agent_registered", {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "capabilities": capabilities
            })

            logger.info(f"Agent registered: {agent_id} ({agent_type})")
            return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the communication hub.

        Args:
            agent_id: Agent identifier to unregister

        Returns:
            True if unregistration successful
        """
        try:
            if agent_id in self.agents:
                self.agents[agent_id].status = "offline"

                # Clean up queues after a delay to allow message delivery
                threading.Timer(60, lambda: self._cleanup_agent_queues(agent_id)).start()

                # Trigger unregistration event
                self._trigger_event("agent_unregistered", {"agent_id": agent_id})

                logger.info(f"Agent unregistered: {agent_id}")
                return True
            else:
                logger.warning(f"Agent not found for unregistration: {agent_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False

    def _cleanup_agent_queues(self, agent_id: str) -> None:
        """Clean up queues for unregistered agent."""
        if agent_id in self.message_queues:
            del self.message_queues[agent_id]
        if agent_id in self.task_queues:
            del self.task_queues[agent_id]
        if agent_id in self.agents:
            del self.agents[agent_id]
        if agent_id in self.agent_instances:
            del self.agent_instances[agent_id]

    def send_message(self, source_agent: str, target_agent: str, message_type: str,
                    payload: Dict[str, Any], priority: str = "normal",
                    requires_response: bool = False) -> str:
        """
        Send a message from one agent to another.

        Args:
            source_agent: ID of sending agent
            target_agent: ID of receiving agent (or "broadcast" for all)
            message_type: Type of message
            payload: Message data
            priority: Message priority
            requires_response: Whether message requires a response

        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())

        message = Message(
            id=message_id,
            source_agent=source_agent,
            target_agent=target_agent,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now().isoformat(),
            priority=priority,
            requires_response=requires_response
        )

        # Route message
        if target_agent == "broadcast":
            self._broadcast_message(message)
        else:
            self._route_message(message)

        # Log message
        self._log_message(message)

        return message_id

    def send_task_request(self, source_agent: str, target_agent: str, task_type: str,
                         task_data: Dict[str, Any], priority: str = "normal",
                         timeout: int = 300, callback: Callable = None) -> str:
        """
        Send a task request to another agent.

        Args:
            source_agent: ID of requesting agent
            target_agent: ID of target agent (or "auto" for capability-based routing)
            task_type: Type of task to execute
            task_data: Task parameters
            priority: Task priority
            timeout: Task timeout in seconds
            callback: Callback function for task completion

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())

        # Auto-route based on capabilities if target is "auto"
        if target_agent == "auto":
            target_agent = self._find_capable_agent(task_type)
            if not target_agent:
                logger.error(f"No agent found capable of handling task: {task_type}")
                return task_id

        task_request = TaskRequest(
            id=task_id,
            source_agent=source_agent,
            target_agent=target_agent,
            task_type=task_type,
            task_data=task_data,
            priority=priority,
            timeout=timeout,
            callback=callback,
            timestamp=datetime.now().isoformat()
        )

        # Store pending task
        self.pending_tasks[task_id] = task_request

        # Queue task
        self._queue_task(task_request)

        # Log task
        self._log_task(task_request)

        return task_id

    def _find_capable_agent(self, task_type: str) -> Optional[str]:
        """Find an agent capable of handling a specific task type."""
        for agent_id, agent_info in self.agents.items():
            if (agent_info.status == "online" and
                task_type in agent_info.capabilities):
                return agent_id
        return None

    def _route_message(self, message: Message) -> None:
        """Route a message to its target agent."""
        target_agent = message.target_agent

        if target_agent not in self.agents:
            logger.warning(f"Target agent not found: {target_agent}")
            return

        # Add to message queue with priority ordering
        queue = self.message_queues[target_agent]

        if self.config["routing"]["enable_priority_queue"]:
            # Insert based on priority
            priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
            msg_priority = priority_order.get(message.priority, 2)

            inserted = False
            for i, existing_msg in enumerate(queue):
                existing_priority = priority_order.get(existing_msg.priority, 2)
                if msg_priority < existing_priority:
                    queue.insert(i, message)
                    inserted = True
                    break

            if not inserted:
                queue.append(message)
        else:
            queue.append(message)

        # Update delivery stats
        self.delivery_stats["messages_routed"] += 1

    def _broadcast_message(self, message: Message) -> None:
        """Broadcast a message to all online agents."""
        if not self.config["routing"]["enable_broadcast"]:
            logger.warning("Broadcast messaging is disabled")
            return

        online_agents = [
            agent_id for agent_id, agent_info in self.agents.items()
            if agent_info.status == "online" and agent_id != message.source_agent
        ]

        for agent_id in online_agents:
            # Create individual message for each target
            individual_message = Message(
                id=str(uuid.uuid4()),
                source_agent=message.source_agent,
                target_agent=agent_id,
                message_type=message.message_type,
                payload=message.payload,
                timestamp=message.timestamp,
                priority=message.priority,
                requires_response=message.requires_response,
                correlation_id=message.id
            )

            self.message_queues[agent_id].append(individual_message)

        self.delivery_stats["broadcasts_sent"] += 1

    def _queue_task(self, task_request: TaskRequest) -> None:
        """Queue a task request for processing."""
        target_agent = task_request.target_agent

        if target_agent not in self.agents:
            logger.warning(f"Target agent not found for task: {target_agent}")
            task_request.status = "failed"
            return

        # Add to task queue
        self.task_queues[target_agent].append(task_request)
        task_request.status = "queued"

        self.delivery_stats["tasks_queued"] += 1

    def get_messages(self, agent_id: str, limit: int = 10) -> List[Message]:
        """
        Get pending messages for an agent.

        Args:
            agent_id: Agent identifier
            limit: Maximum number of messages to return

        Returns:
            List of messages
        """
        if agent_id not in self.message_queues:
            return []

        messages = []
        queue = self.message_queues[agent_id]

        for _ in range(min(limit, len(queue))):
            if queue:
                messages.append(queue.popleft())

        # Update agent last seen
        if agent_id in self.agents:
            self.agents[agent_id].last_seen = datetime.now().isoformat()

        return messages

    def get_tasks(self, agent_id: str, limit: int = 5) -> List[TaskRequest]:
        """
        Get pending tasks for an agent.

        Args:
            agent_id: Agent identifier
            limit: Maximum number of tasks to return

        Returns:
            List of task requests
        """
        if agent_id not in self.task_queues:
            return []

        tasks = []
        queue = self.task_queues[agent_id]

        for _ in range(min(limit, len(queue))):
            if queue:
                task = queue.popleft()
                task.status = "in_progress"
                tasks.append(task)

        # Update agent status
        if agent_id in self.agents:
            self.agents[agent_id].last_seen = datetime.now().isoformat()
            if tasks:
                self.agents[agent_id].status = "busy"

        return tasks

    def complete_task(self, task_id: str, result: Dict[str, Any] = None,
                     error: str = None) -> bool:
        """
        Mark a task as completed.

        Args:
            task_id: Task identifier
            result: Task result data
            error: Error message if task failed

        Returns:
            True if task completion recorded successfully
        """
        if task_id not in self.pending_tasks:
            logger.warning(f"Task not found: {task_id}")
            return False

        task = self.pending_tasks[task_id]

        if error:
            task.status = "failed"
            logger.error(f"Task {task_id} failed: {error}")
        else:
            task.status = "completed"
            logger.info(f"Task {task_id} completed successfully")

        # Call callback if provided
        if task.callback:
            try:
                task.callback(task_id, result, error)
            except Exception as e:
                logger.error(f"Task callback error: {e}")

        # Send completion message to source agent
        self.send_message(
            source_agent="hub",
            target_agent=task.source_agent,
            message_type="task_completion",
            payload={
                "task_id": task_id,
                "status": task.status,
                "result": result,
                "error": error
            }
        )

        # Log task completion
        self._log_task_completion(task, result, error)

        # Clean up
        del self.pending_tasks[task_id]

        return True

    def get_agent_status(self, agent_id: str = None) -> Dict[str, Any]:
        """
        Get status of specific agent or all agents.

        Args:
            agent_id: Specific agent ID, or None for all agents

        Returns:
            Agent status information
        """
        if agent_id:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                return {
                    "agent_id": agent_id,
                    "type": agent.agent_type,
                    "status": agent.status,
                    "capabilities": agent.capabilities,
                    "last_seen": agent.last_seen,
                    "pending_messages": len(self.message_queues[agent_id]),
                    "pending_tasks": len(self.task_queues[agent_id])
                }
            else:
                return {"error": f"Agent not found: {agent_id}"}
        else:
            return {
                "total_agents": len(self.agents),
                "online_agents": len([a for a in self.agents.values() if a.status == "online"]),
                "agents": {
                    agent_id: {
                        "type": agent.agent_type,
                        "status": agent.status,
                        "last_seen": agent.last_seen,
                        "pending_messages": len(self.message_queues[agent_id]),
                        "pending_tasks": len(self.task_queues[agent_id])
                    }
                    for agent_id, agent in self.agents.items()
                }
            }

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication hub statistics."""
        return {
            "delivery_stats": dict(self.delivery_stats),
            "message_history_size": len(self.message_history),
            "pending_tasks": len(self.pending_tasks),
            "total_agents": len(self.agents),
            "active_queues": len([q for q in self.message_queues.values() if len(q) > 0])
        }

    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add an event handler for specific event types."""
        self.event_handlers[event_type].append(handler)

    def _trigger_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Trigger event handlers for a specific event type."""
        for handler in self.event_handlers[event_type]:
            try:
                handler(event_data)
            except Exception as e:
                logger.error(f"Event handler error for {event_type}: {e}")

    def _log_message(self, message: Message) -> None:
        """Log message to persistent storage."""
        if not self.config["logging"]["log_all_messages"]:
            return

        try:
            # Add to memory history
            self.message_history.append(message)

            # Log to file
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = self.message_log_dir / f"messages_{timestamp}.jsonl"

            with open(log_file, 'a') as f:
                f.write(json.dumps(asdict(message)) + '\n')

        except Exception as e:
            logger.error(f"Failed to log message: {e}")

    def _log_task(self, task_request: TaskRequest) -> None:
        """Log task request to persistent storage."""
        if not self.config["logging"]["log_task_details"]:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = self.task_log_dir / f"tasks_{timestamp}.jsonl"

            task_data = asdict(task_request)
            # Remove callback from logging (not serializable)
            task_data.pop("callback", None)

            with open(log_file, 'a') as f:
                f.write(json.dumps(task_data) + '\n')

        except Exception as e:
            logger.error(f"Failed to log task: {e}")

    def _log_task_completion(self, task: TaskRequest, result: Dict[str, Any],
                           error: str) -> None:
        """Log task completion."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = self.task_log_dir / f"task_completions_{timestamp}.jsonl"

            completion_data = {
                "task_id": task.id,
                "completion_time": datetime.now().isoformat(),
                "status": task.status,
                "result": result,
                "error": error
            }

            with open(log_file, 'a') as f:
                f.write(json.dumps(completion_data) + '\n')

        except Exception as e:
            logger.error(f"Failed to log task completion: {e}")

    def _hub_loop(self) -> None:
        """Main hub processing loop."""
        logger.info("Communication hub processing started")

        while self._running:
            try:
                # Process message delivery
                self._process_pending_deliveries()

                # Check for task timeouts
                self._check_task_timeouts()

                # Update agent status
                self._update_agent_status()

                # Clean up old data
                self._cleanup_old_data()

                time.sleep(1)  # Process every second

            except Exception as e:
                logger.error(f"Hub processing error: {e}")
                time.sleep(5)

        logger.info("Communication hub processing stopped")

    def _process_pending_deliveries(self) -> None:
        """Process any pending message deliveries."""
        for agent_id, agent_info in self.agents.items():
            if agent_info.status != "online":
                continue

            # Process messages with handlers
            if agent_info.message_handler and self.message_queues[agent_id]:
                messages = list(self.message_queues[agent_id])
                self.message_queues[agent_id].clear()

                for message in messages:
                    try:
                        agent_info.message_handler(message)
                        self.delivery_stats["messages_delivered"] += 1
                    except Exception as e:
                        logger.error(f"Message delivery error to {agent_id}: {e}")

            # Process tasks with handlers
            if agent_info.task_handler and self.task_queues[agent_id]:
                tasks = list(self.task_queues[agent_id])
                self.task_queues[agent_id].clear()

                for task in tasks:
                    try:
                        agent_info.task_handler(task)
                        self.delivery_stats["tasks_delivered"] += 1
                    except Exception as e:
                        logger.error(f"Task delivery error to {agent_id}: {e}")

    def _check_task_timeouts(self) -> None:
        """Check for and handle task timeouts."""
        current_time = datetime.now()

        timed_out_tasks = []
        for task_id, task in self.pending_tasks.items():
            task_time = datetime.fromisoformat(task.timestamp)
            if (current_time - task_time).total_seconds() > task.timeout:
                timed_out_tasks.append(task_id)

        for task_id in timed_out_tasks:
            self.complete_task(task_id, error="Task timeout")

    def _update_agent_status(self) -> None:
        """Update agent status based on last seen time."""
        current_time = datetime.now()
        timeout_threshold = timedelta(minutes=5)

        for agent_id, agent_info in self.agents.items():
            last_seen = datetime.fromisoformat(agent_info.last_seen)
            if current_time - last_seen > timeout_threshold:
                if agent_info.status == "online":
                    agent_info.status = "offline"
                    logger.warning(f"Agent {agent_id} marked as offline due to inactivity")

    def _cleanup_old_data(self) -> None:
        """Clean up old data to prevent memory leaks."""
        # Clean up old message history (keep last 1000)
        while len(self.message_history) > 1000:
            self.message_history.popleft()

        # Clean up completed tasks older than 1 hour
        current_time = datetime.now()
        old_tasks = []

        for task_id, task in self.pending_tasks.items():
            if task.status in ["completed", "failed"]:
                task_time = datetime.fromisoformat(task.timestamp)
                if (current_time - task_time).total_seconds() > 3600:
                    old_tasks.append(task_id)

        for task_id in old_tasks:
            del self.pending_tasks[task_id]

    def start(self) -> None:
        """Start the communication hub."""
        if self._running:
            logger.warning("Communication hub is already running")
            return

        self._running = True
        self._hub_thread = threading.Thread(target=self._hub_loop, daemon=True)
        self._hub_thread.start()

        logger.info("CommunicationHub started")

    def stop(self) -> None:
        """Stop the communication hub."""
        if not self._running:
            return

        self._running = False

        if self._hub_thread:
            self._hub_thread.join(timeout=10)

        logger.info("CommunicationHub stopped")

def create_communication_hub() -> CommunicationHub:
    """Factory function to create a configured CommunicationHub."""
    return CommunicationHub()

if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    hub = create_communication_hub()

    try:
        hub.start()

        # Register demo agents
        hub.register_agent("demo_agent_1", "test", ["demo_task"])
        hub.register_agent("demo_agent_2", "test", ["demo_task"])

        # Send demo messages
        msg_id = hub.send_message(
            source_agent="demo_agent_1",
            target_agent="demo_agent_2",
            message_type="greeting",
            payload={"message": "Hello from agent 1"}
        )

        # Send demo task
        task_id = hub.send_task_request(
            source_agent="demo_agent_1",
            target_agent="demo_agent_2",
            task_type="demo_task",
            task_data={"input": "test data"}
        )

        time.sleep(5)

        # Get status
        status = hub.get_agent_status()
        stats = hub.get_communication_stats()

        print(f"Agent Status: {json.dumps(status, indent=2)}")
        print(f"Communication Stats: {json.dumps(stats, indent=2)}")

    except KeyboardInterrupt:
        print("Demo interrupted")
    finally:
        hub.stop()