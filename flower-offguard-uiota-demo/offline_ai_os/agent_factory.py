#!/usr/bin/env python3
"""
Agent Factory System for Offline AI OS
Dynamic agent creation with capability injection and resource management
"""

import asyncio
import json
import yaml
import psutil
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

from base_agent import BaseAgent, AgentCapabilities, AgentState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== RESOURCE MANAGEMENT ====================

@dataclass
class ResourceRequirements:
    """Resource requirements for an agent"""
    cpu_cores: float = 0.5  # CPU cores (can be fractional)
    memory_mb: int = 512     # RAM in MB
    gpu_memory_mb: int = 0   # GPU memory in MB (0 = no GPU)
    disk_mb: int = 100       # Disk space in MB
    network_bandwidth_mbps: int = 10  # Network bandwidth in Mbps


@dataclass
class SystemResources:
    """Available system resources"""
    total_cpu_cores: int
    available_cpu_cores: float
    total_memory_mb: int
    available_memory_mb: int
    total_gpu_memory_mb: int
    available_gpu_memory_mb: int
    total_disk_mb: int
    available_disk_mb: int
    network_bandwidth_mbps: int


class ResourceManager:
    """
    Manages system resources and allocations for agents
    """

    def __init__(self):
        self.allocated_resources: Dict[str, ResourceRequirements] = {}
        self._update_system_resources()
        logger.info("ResourceManager initialized")

    def _update_system_resources(self):
        """Update current system resource availability"""
        # CPU
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        available_cpu = cpu_count * (1 - cpu_percent / 100)

        # Memory
        memory = psutil.virtual_memory()
        total_memory_mb = memory.total // (1024 * 1024)
        available_memory_mb = memory.available // (1024 * 1024)

        # Disk
        disk = psutil.disk_usage('/')
        total_disk_mb = disk.total // (1024 * 1024)
        available_disk_mb = disk.free // (1024 * 1024)

        # GPU (simplified - would need pynvml for real GPU detection)
        total_gpu_memory_mb = 0
        available_gpu_memory_mb = 0

        # Network (simplified)
        network_bandwidth_mbps = 1000  # Assume 1 Gbps

        self.system_resources = SystemResources(
            total_cpu_cores=cpu_count,
            available_cpu_cores=available_cpu,
            total_memory_mb=total_memory_mb,
            available_memory_mb=available_memory_mb,
            total_gpu_memory_mb=total_gpu_memory_mb,
            available_gpu_memory_mb=available_gpu_memory_mb,
            total_disk_mb=total_disk_mb,
            available_disk_mb=available_disk_mb,
            network_bandwidth_mbps=network_bandwidth_mbps
        )

    def can_allocate(self, requirements: ResourceRequirements) -> bool:
        """Check if resources can be allocated"""
        self._update_system_resources()

        # Calculate currently allocated resources
        allocated_cpu = sum(r.cpu_cores for r in self.allocated_resources.values())
        allocated_memory = sum(r.memory_mb for r in self.allocated_resources.values())
        allocated_gpu = sum(r.gpu_memory_mb for r in self.allocated_resources.values())
        allocated_disk = sum(r.disk_mb for r in self.allocated_resources.values())

        # Check if requirements can be met
        can_allocate_cpu = (allocated_cpu + requirements.cpu_cores) <= self.system_resources.total_cpu_cores
        can_allocate_memory = (allocated_memory + requirements.memory_mb) <= self.system_resources.available_memory_mb
        can_allocate_gpu = (allocated_gpu + requirements.gpu_memory_mb) <= self.system_resources.total_gpu_memory_mb
        can_allocate_disk = (allocated_disk + requirements.disk_mb) <= self.system_resources.available_disk_mb

        return can_allocate_cpu and can_allocate_memory and can_allocate_gpu and can_allocate_disk

    def allocate(self, agent_id: str, requirements: ResourceRequirements) -> bool:
        """Allocate resources for an agent"""
        if not self.can_allocate(requirements):
            logger.warning(f"Cannot allocate resources for {agent_id}")
            return False

        self.allocated_resources[agent_id] = requirements
        logger.info(f"Allocated resources for {agent_id}: {requirements}")
        return True

    def deallocate(self, agent_id: str):
        """Deallocate resources for an agent"""
        if agent_id in self.allocated_resources:
            del self.allocated_resources[agent_id]
            logger.info(f"Deallocated resources for {agent_id}")

    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics"""
        self._update_system_resources()

        allocated_cpu = sum(r.cpu_cores for r in self.allocated_resources.values())
        allocated_memory = sum(r.memory_mb for r in self.allocated_resources.values())
        allocated_gpu = sum(r.gpu_memory_mb for r in self.allocated_resources.values())
        allocated_disk = sum(r.disk_mb for r in self.allocated_resources.values())

        return {
            "system": asdict(self.system_resources),
            "allocated": {
                "cpu_cores": allocated_cpu,
                "memory_mb": allocated_memory,
                "gpu_memory_mb": allocated_gpu,
                "disk_mb": allocated_disk,
                "agents": len(self.allocated_resources)
            },
            "utilization": {
                "cpu_percent": (allocated_cpu / self.system_resources.total_cpu_cores) * 100,
                "memory_percent": (allocated_memory / self.system_resources.total_memory_mb) * 100,
                "disk_percent": (allocated_disk / self.system_resources.total_disk_mb) * 100
            }
        }


# ==================== CAPABILITY LOADING ====================

class CapabilityLoader:
    """
    Loads agent capabilities from YAML definitions
    """

    def __init__(self, capabilities_dir: str = "offline_ai_os/capabilities"):
        self.capabilities_dir = Path(capabilities_dir)
        self.capabilities_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_capabilities: Dict[str, Dict] = {}
        self._create_default_capabilities()
        logger.info(f"CapabilityLoader initialized with dir: {capabilities_dir}")

    def _create_default_capabilities(self):
        """Create default capability definitions"""
        default_capabilities = {
            "threat_detection": {
                "name": "threat_detection",
                "category": "detection",
                "description": "Detect security threats and anomalies",
                "skills": [
                    "signature_matching",
                    "anomaly_detection",
                    "behavioral_analysis",
                    "network_intrusion_detection"
                ],
                "dependencies": ["network_access", "log_analysis"],
                "resource_requirements": {
                    "cpu_cores": 1.0,
                    "memory_mb": 1024,
                    "gpu_memory_mb": 0,
                    "disk_mb": 500
                }
            },
            "malware_analysis": {
                "name": "malware_analysis",
                "category": "analysis",
                "description": "Analyze malware samples and behavior",
                "skills": [
                    "static_analysis",
                    "dynamic_analysis",
                    "sandbox_execution",
                    "reverse_engineering"
                ],
                "dependencies": ["sandboxing", "disassembler"],
                "resource_requirements": {
                    "cpu_cores": 2.0,
                    "memory_mb": 4096,
                    "gpu_memory_mb": 0,
                    "disk_mb": 5000
                }
            },
            "incident_response": {
                "name": "incident_response",
                "category": "response",
                "description": "Respond to security incidents",
                "skills": [
                    "threat_mitigation",
                    "system_isolation",
                    "evidence_collection",
                    "remediation"
                ],
                "dependencies": ["system_control", "network_control"],
                "resource_requirements": {
                    "cpu_cores": 0.5,
                    "memory_mb": 512,
                    "gpu_memory_mb": 0,
                    "disk_mb": 1000
                }
            },
            "threat_intelligence": {
                "name": "threat_intelligence",
                "category": "learning",
                "description": "Gather and analyze threat intelligence",
                "skills": [
                    "ioc_collection",
                    "threat_correlation",
                    "pattern_learning",
                    "predictive_analysis"
                ],
                "dependencies": ["database", "ml_framework"],
                "resource_requirements": {
                    "cpu_cores": 1.5,
                    "memory_mb": 2048,
                    "gpu_memory_mb": 2048,
                    "disk_mb": 10000
                }
            },
            "secure_communication": {
                "name": "secure_communication",
                "category": "communication",
                "description": "Secure inter-agent communication",
                "skills": [
                    "encryption",
                    "message_signing",
                    "key_exchange",
                    "secure_channels"
                ],
                "dependencies": ["cryptography"],
                "resource_requirements": {
                    "cpu_cores": 0.5,
                    "memory_mb": 256,
                    "gpu_memory_mb": 0,
                    "disk_mb": 100
                }
            }
        }

        for cap_name, cap_data in default_capabilities.items():
            cap_file = self.capabilities_dir / f"{cap_name}.yaml"
            if not cap_file.exists():
                with open(cap_file, 'w') as f:
                    yaml.dump(cap_data, f, default_flow_style=False)

    def load_capability(self, capability_name: str) -> Optional[Dict]:
        """Load a capability definition"""
        if capability_name in self.loaded_capabilities:
            return self.loaded_capabilities[capability_name]

        cap_file = self.capabilities_dir / f"{capability_name}.yaml"
        if not cap_file.exists():
            logger.warning(f"Capability file not found: {capability_name}")
            return None

        with open(cap_file, 'r') as f:
            capability = yaml.safe_load(f)

        self.loaded_capabilities[capability_name] = capability
        logger.info(f"Loaded capability: {capability_name}")
        return capability

    def load_all_capabilities(self) -> Dict[str, Dict]:
        """Load all available capabilities"""
        for cap_file in self.capabilities_dir.glob("*.yaml"):
            cap_name = cap_file.stem
            self.load_capability(cap_name)

        return self.loaded_capabilities

    def get_capability_requirements(self, capability_name: str) -> Optional[ResourceRequirements]:
        """Get resource requirements for a capability"""
        capability = self.load_capability(capability_name)
        if not capability:
            return None

        req_data = capability.get("resource_requirements", {})
        return ResourceRequirements(**req_data)


# ==================== BLUEPRINT REGISTRY ====================

@dataclass
class AgentBlueprint:
    """Blueprint for creating an agent"""
    name: str
    agent_type: str
    description: str
    capabilities: List[str]
    config: Dict[str, Any]
    resource_requirements: ResourceRequirements


class BlueprintRegistry:
    """
    Manages agent blueprints
    """

    def __init__(self, blueprints_dir: str = "offline_ai_os/blueprints"):
        self.blueprints_dir = Path(blueprints_dir)
        self.blueprints_dir.mkdir(parents=True, exist_ok=True)
        self.blueprints: Dict[str, AgentBlueprint] = {}
        self._create_default_blueprints()
        logger.info(f"BlueprintRegistry initialized with dir: {blueprints_dir}")

    def _create_default_blueprints(self):
        """Create default agent blueprints"""
        default_blueprints = {
            "threat_detector": {
                "name": "threat_detector",
                "agent_type": "ThreatDetectorAgent",
                "description": "Detects security threats and anomalies",
                "capabilities": ["threat_detection", "secure_communication"],
                "config": {
                    "detection_threshold": 0.85,
                    "scan_interval_seconds": 60,
                    "alert_priority": "high"
                },
                "resource_requirements": {
                    "cpu_cores": 1.0,
                    "memory_mb": 1024,
                    "gpu_memory_mb": 0,
                    "disk_mb": 500
                }
            },
            "malware_analyzer": {
                "name": "malware_analyzer",
                "agent_type": "MalwareAnalyzerAgent",
                "description": "Analyzes malware samples",
                "capabilities": ["malware_analysis", "secure_communication"],
                "config": {
                    "sandbox_enabled": True,
                    "analysis_timeout_seconds": 300,
                    "max_concurrent_samples": 3
                },
                "resource_requirements": {
                    "cpu_cores": 2.0,
                    "memory_mb": 4096,
                    "gpu_memory_mb": 0,
                    "disk_mb": 5000
                }
            },
            "incident_responder": {
                "name": "incident_responder",
                "agent_type": "IncidentResponderAgent",
                "description": "Responds to security incidents",
                "capabilities": ["incident_response", "secure_communication"],
                "config": {
                    "auto_mitigation": False,
                    "response_time_seconds": 10,
                    "escalation_threshold": "critical"
                },
                "resource_requirements": {
                    "cpu_cores": 0.5,
                    "memory_mb": 512,
                    "gpu_memory_mb": 0,
                    "disk_mb": 1000
                }
            },
            "intelligence_analyst": {
                "name": "intelligence_analyst",
                "agent_type": "IntelligenceAnalystAgent",
                "description": "Analyzes threat intelligence",
                "capabilities": ["threat_intelligence", "secure_communication"],
                "config": {
                    "learning_rate": 0.001,
                    "model_update_interval_hours": 24,
                    "correlation_threshold": 0.75
                },
                "resource_requirements": {
                    "cpu_cores": 1.5,
                    "memory_mb": 2048,
                    "gpu_memory_mb": 2048,
                    "disk_mb": 10000
                }
            }
        }

        for bp_name, bp_data in default_blueprints.items():
            bp_file = self.blueprints_dir / f"{bp_name}.json"
            if not bp_file.exists():
                with open(bp_file, 'w') as f:
                    json.dump(bp_data, f, indent=2)

    def load_blueprint(self, blueprint_name: str) -> Optional[AgentBlueprint]:
        """Load an agent blueprint"""
        if blueprint_name in self.blueprints:
            return self.blueprints[blueprint_name]

        bp_file = self.blueprints_dir / f"{blueprint_name}.json"
        if not bp_file.exists():
            logger.warning(f"Blueprint file not found: {blueprint_name}")
            return None

        with open(bp_file, 'r') as f:
            bp_data = json.load(f)

        # Convert resource requirements
        req_data = bp_data.pop("resource_requirements")
        resource_requirements = ResourceRequirements(**req_data)

        blueprint = AgentBlueprint(
            **bp_data,
            resource_requirements=resource_requirements
        )

        self.blueprints[blueprint_name] = blueprint
        logger.info(f"Loaded blueprint: {blueprint_name}")
        return blueprint

    def register_blueprint(self, blueprint: AgentBlueprint):
        """Register a new blueprint"""
        self.blueprints[blueprint.name] = blueprint

        # Save to file
        bp_file = self.blueprints_dir / f"{blueprint.name}.json"
        bp_data = {
            "name": blueprint.name,
            "agent_type": blueprint.agent_type,
            "description": blueprint.description,
            "capabilities": blueprint.capabilities,
            "config": blueprint.config,
            "resource_requirements": asdict(blueprint.resource_requirements)
        }

        with open(bp_file, 'w') as f:
            json.dump(bp_data, f, indent=2)

        logger.info(f"Registered blueprint: {blueprint.name}")

    def list_blueprints(self) -> List[str]:
        """List all available blueprints"""
        return list(self.blueprints.keys())


# ==================== AGENT FACTORY ====================

class AgentFactory:
    """
    Advanced agent factory with capability injection and resource management
    """

    def __init__(self):
        self.resource_manager = ResourceManager()
        self.capability_loader = CapabilityLoader()
        self.blueprint_registry = BlueprintRegistry()
        self.created_agents: Dict[str, BaseAgent] = {}

        # Load all capabilities and blueprints
        self.capability_loader.load_all_capabilities()
        for bp_file in self.blueprint_registry.blueprints_dir.glob("*.json"):
            self.blueprint_registry.load_blueprint(bp_file.stem)

        logger.info("AgentFactory initialized")

    def create_agent_from_blueprint(self, blueprint_name: str,
                                    agent_id: Optional[str] = None) -> Optional[BaseAgent]:
        """Create an agent from a blueprint"""
        # Load blueprint
        blueprint = self.blueprint_registry.load_blueprint(blueprint_name)
        if not blueprint:
            logger.error(f"Blueprint not found: {blueprint_name}")
            return None

        # Check resource availability
        if not self.resource_manager.can_allocate(blueprint.resource_requirements):
            logger.error(f"Insufficient resources for {blueprint_name}")
            return None

        # Create agent (placeholder - would instantiate actual agent class)
        # In real implementation, would use dynamic class loading
        from base_agent import ThreatDetectorAgent

        agent = ThreatDetectorAgent(agent_id=agent_id, config=blueprint.config)

        # Inject capabilities
        for cap_name in blueprint.capabilities:
            capability = self.capability_loader.load_capability(cap_name)
            if capability:
                category = capability.get("category", "detection")
                for skill in capability.get("skills", []):
                    agent.add_capability(category, skill)

        # Allocate resources
        self.resource_manager.allocate(agent.agent_id, blueprint.resource_requirements)

        # Track agent
        self.created_agents[agent.agent_id] = agent

        logger.info(f"Created agent {agent.agent_id} from blueprint {blueprint_name}")
        return agent

    def create_custom_agent(self, agent_type: Type[BaseAgent],
                          capabilities: List[str],
                          config: Dict[str, Any],
                          agent_id: Optional[str] = None) -> Optional[BaseAgent]:
        """Create a custom agent with specific capabilities"""
        # Calculate total resource requirements
        total_requirements = ResourceRequirements()
        for cap_name in capabilities:
            cap_req = self.capability_loader.get_capability_requirements(cap_name)
            if cap_req:
                total_requirements.cpu_cores += cap_req.cpu_cores
                total_requirements.memory_mb += cap_req.memory_mb
                total_requirements.gpu_memory_mb += cap_req.gpu_memory_mb
                total_requirements.disk_mb += cap_req.disk_mb

        # Check resource availability
        if not self.resource_manager.can_allocate(total_requirements):
            logger.error(f"Insufficient resources for custom agent")
            return None

        # Create agent
        agent = agent_type(agent_id=agent_id, config=config)

        # Inject capabilities
        for cap_name in capabilities:
            capability = self.capability_loader.load_capability(cap_name)
            if capability:
                category = capability.get("category", "detection")
                for skill in capability.get("skills", []):
                    agent.add_capability(category, skill)

        # Allocate resources
        self.resource_manager.allocate(agent.agent_id, total_requirements)

        # Track agent
        self.created_agents[agent.agent_id] = agent

        logger.info(f"Created custom agent {agent.agent_id}")
        return agent

    async def destroy_agent(self, agent_id: str):
        """Destroy an agent and free its resources"""
        if agent_id not in self.created_agents:
            logger.warning(f"Agent not found: {agent_id}")
            return

        agent = self.created_agents[agent_id]

        # Terminate agent
        await agent.terminate()

        # Deallocate resources
        self.resource_manager.deallocate(agent_id)

        # Remove from tracking
        del self.created_agents[agent_id]

        logger.info(f"Destroyed agent {agent_id}")

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID"""
        return self.created_agents.get(agent_id)

    def list_agents(self) -> List[str]:
        """List all created agents"""
        return list(self.created_agents.keys())

    def get_factory_status(self) -> Dict[str, Any]:
        """Get factory status"""
        return {
            "created_agents": len(self.created_agents),
            "agents": [
                {
                    "agent_id": agent_id,
                    "agent_type": agent.agent_type,
                    "state": agent.state.value,
                    "capabilities": agent.capabilities.all_capabilities()
                }
                for agent_id, agent in self.created_agents.items()
            ],
            "resource_usage": self.resource_manager.get_resource_usage(),
            "available_blueprints": self.blueprint_registry.list_blueprints(),
            "loaded_capabilities": list(self.capability_loader.loaded_capabilities.keys())
        }


# ==================== DEMO ====================

async def demo_agent_factory():
    """Demonstrate agent factory system"""
    print("=" * 80)
    print("AGENT FACTORY SYSTEM DEMO")
    print("=" * 80)

    # Create factory
    factory = AgentFactory()
    print(f"\n✓ Agent Factory initialized")

    # Show available blueprints
    print(f"\n📋 Available Blueprints:")
    for bp_name in factory.blueprint_registry.list_blueprints():
        bp = factory.blueprint_registry.load_blueprint(bp_name)
        print(f"   • {bp_name}: {bp.description}")

    # Show available capabilities
    print(f"\n🎯 Available Capabilities:")
    for cap_name, cap_data in factory.capability_loader.loaded_capabilities.items():
        print(f"   • {cap_name}: {cap_data['description']}")

    # Show resource availability
    print(f"\n💻 System Resources:")
    resources = factory.resource_manager.get_resource_usage()
    print(f"   CPU Cores: {resources['system']['total_cpu_cores']}")
    print(f"   Memory: {resources['system']['total_memory_mb']} MB")
    print(f"   Disk: {resources['system']['total_disk_mb']} MB")

    # Create agents from blueprints
    print(f"\n🤖 Creating Agents from Blueprints...")

    detector = factory.create_agent_from_blueprint("threat_detector")
    if detector:
        await detector.initialize()
        await detector.start()
        print(f"✓ Created and started: {detector}")
        print(f"  Capabilities: {detector.capabilities.all_capabilities()}")

    responder = factory.create_agent_from_blueprint("incident_responder")
    if responder:
        await responder.initialize()
        await responder.start()
        print(f"✓ Created and started: {responder}")
        print(f"  Capabilities: {responder.capabilities.all_capabilities()}")

    # Show factory status
    print(f"\n📊 Factory Status:")
    status = factory.get_factory_status()
    print(f"   Total Agents: {status['created_agents']}")
    print(f"   Resource Utilization:")
    print(f"      CPU: {status['resource_usage']['utilization']['cpu_percent']:.1f}%")
    print(f"      Memory: {status['resource_usage']['utilization']['memory_percent']:.1f}%")
    print(f"      Disk: {status['resource_usage']['utilization']['disk_percent']:.1f}%")

    # Test agent execution
    if detector:
        print(f"\n🔍 Testing Threat Detector Agent...")
        result = await detector.execute_task({
            "source_ip": "192.168.1.100",
            "suspicious_pattern": "sql_injection_attempt"
        })
        print(f"✓ Detection result: {result['result']}")

    # Cleanup
    print(f"\n🧹 Cleaning up agents...")
    for agent_id in factory.list_agents():
        await factory.destroy_agent(agent_id)
    print(f"✓ All agents destroyed")

    print("\n" + "=" * 80)
    print("AGENT FACTORY DEMO COMPLETE")
    print("=" * 80)
    print("\n🎯 Key Features Demonstrated:")
    print("   ✓ Dynamic agent creation from blueprints")
    print("   ✓ Capability injection system")
    print("   ✓ Resource management and allocation")
    print("   ✓ Agent lifecycle management")
    print("   ✓ Blueprint and capability registries")


if __name__ == "__main__":
    asyncio.run(demo_agent_factory())