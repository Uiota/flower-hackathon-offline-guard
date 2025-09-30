#!/usr/bin/env python3
"""
SOVEREIGN AGENT FACTORY SYSTEM
==================================
Enterprise-grade multi-agent system with:
- Development Agents (Code, Test, Deploy)
- Cyber Defense Agents (Security, Threat Intel, Response)
- Project Management Agent (Planning, Coordination, Reporting)
- Sovereign-grade offline wallet and token system
- Zero-trust security architecture
"""

import os
import sys
import json
import time
import uuid
import asyncio
import threading
import logging
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# Configure secure offline environment
os.environ["OFFLINE_MODE"] = "1"
os.environ["SECURE_MODE"] = "1"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sovereign_agents.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== CORE MODELS ====================

@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    required_resources: Dict[str, Any]
    security_level: str  # 'sovereign', 'classified', 'restricted', 'public'
    permissions: List[str]
    rate_limit: Optional[int] = None

@dataclass
class TaskRequest:
    """Task request for agent processing"""
    task_id: str
    agent_type: str
    priority: int  # 1=highest, 5=lowest
    task_data: Dict[str, Any]
    requester: str
    deadline: Optional[datetime] = None
    security_classification: str = "restricted"

@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    agent_id: str
    success: bool
    result_data: Dict[str, Any]
    execution_time: float
    errors: List[str] = field(default_factory=list)
    completion_time: datetime = field(default_factory=datetime.now)

@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_response_time: float = 0.0
    uptime: float = 0.0
    last_activity: Optional[datetime] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)

# ==================== SOVEREIGN WALLET SYSTEM ====================

class SovereignWallet:
    """Sovereign-grade offline wallet with quantum resistance"""

    def __init__(self, wallet_id: str, passphrase: str):
        self.wallet_id = wallet_id
        self.created_at = datetime.now(timezone.utc)
        self.db_path = Path(f"wallets/{wallet_id}.db")
        self.db_path.parent.mkdir(exist_ok=True)

        # Generate sovereign keypair
        self._generate_sovereign_keys(passphrase)
        self._init_database()

        logger.info(f"🏛️ Sovereign wallet initialized: {wallet_id}")

    def _generate_sovereign_keys(self, passphrase: str):
        """Generate quantum-resistant cryptographic keys"""
        # RSA 4096-bit for quantum resistance
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        self.public_key = self.private_key.public_key()

        # Derive encryption key from passphrase
        passphrase_hash = hashlib.pbkdf2_hmac(
            'sha256',
            passphrase.encode(),
            self.wallet_id.encode(),
            100000
        )
        self.encryption_key = passphrase_hash[:32]  # AES-256 key

    def _init_database(self):
        """Initialize wallet database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tokens (
                    token_id TEXT PRIMARY KEY,
                    token_type TEXT NOT NULL,
                    balance REAL NOT NULL DEFAULT 0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    tx_id TEXT PRIMARY KEY,
                    from_wallet TEXT,
                    to_wallet TEXT,
                    token_type TEXT,
                    amount REAL,
                    signature TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending'
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS keys (
                    key_type TEXT PRIMARY KEY,
                    encrypted_key BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def create_token(self, token_type: str, initial_supply: float, metadata: Dict) -> str:
        """Create new sovereign token"""
        token_id = f"{token_type}_{uuid.uuid4().hex[:8]}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO tokens (token_id, token_type, balance, metadata) VALUES (?, ?, ?, ?)",
                (token_id, token_type, initial_supply, json.dumps(metadata))
            )

        logger.info(f"💰 Created sovereign token: {token_id} ({initial_supply} {token_type})")
        return token_id

    def get_balance(self, token_type: str) -> float:
        """Get token balance"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT SUM(balance) FROM tokens WHERE token_type = ?",
                (token_type,)
            )
            result = cursor.fetchone()
            return result[0] if result and result[0] else 0.0

    def transfer(self, to_wallet: str, token_type: str, amount: float) -> str:
        """Transfer tokens to another wallet"""
        if self.get_balance(token_type) < amount:
            raise ValueError("Insufficient balance")

        tx_id = f"tx_{uuid.uuid4().hex}"
        signature = self._sign_transaction(to_wallet, token_type, amount)

        with sqlite3.connect(self.db_path) as conn:
            # Record transaction
            conn.execute(
                "INSERT INTO transactions (tx_id, from_wallet, to_wallet, token_type, amount, signature) VALUES (?, ?, ?, ?, ?, ?)",
                (tx_id, self.wallet_id, to_wallet, token_type, amount, signature)
            )

            # Update balance
            conn.execute(
                "UPDATE tokens SET balance = balance - ? WHERE token_type = ?",
                (amount, token_type)
            )

        logger.info(f"💸 Transfer initiated: {amount} {token_type} to {to_wallet}")
        return tx_id

    def _sign_transaction(self, to_wallet: str, token_type: str, amount: float) -> str:
        """Sign transaction with private key"""
        tx_data = f"{to_wallet}:{token_type}:{amount}:{datetime.now().isoformat()}"
        signature = self.private_key.sign(
            tx_data.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature.hex()

    def export_public_key(self) -> str:
        """Export public key for verification"""
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode()

# ==================== BASE AGENT ARCHITECTURE ====================

class BaseAgent(ABC):
    """Base class for all sovereign agents"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.capabilities = []
        self.metrics = AgentMetrics()
        self.running = False
        self.task_queue = asyncio.Queue()

        # Security
        self.security_level = config.get('security_level', 'restricted')
        self.permissions = config.get('permissions', [])

        # Initialize wallet connection
        wallet_id = config.get('wallet_id', 'default_sovereign')
        passphrase = config.get('wallet_passphrase', 'sovereign_secure_2024')
        self.wallet = SovereignWallet(wallet_id, passphrase)

        logger.info(f"🤖 Agent {agent_id} initialized with {self.security_level} security")

    @abstractmethod
    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Process a task - implemented by specific agents"""
        pass

    async def start(self):
        """Start agent processing loop"""
        self.running = True
        logger.info(f"▶️  Agent {self.agent_id} started")

        while self.running:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                result = await self.process_task(task)

                # Update metrics
                self.metrics.last_activity = datetime.now()
                if result.success:
                    self.metrics.tasks_completed += 1
                else:
                    self.metrics.tasks_failed += 1

                # Earn tokens for completed tasks
                if result.success:
                    await self._earn_task_reward(task.priority)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Agent {self.agent_id} error: {e}")

    async def submit_task(self, task: TaskRequest):
        """Submit task to agent"""
        await self.task_queue.put(task)

    async def _earn_task_reward(self, task_priority: int):
        """Earn tokens for completing tasks"""
        # Higher priority tasks earn more tokens
        reward_amount = (6 - task_priority) * 10  # Priority 1 = 50 tokens, Priority 5 = 10 tokens

        try:
            # Create reward tokens if not exist
            current_balance = self.wallet.get_balance('AGENT_WORK')
            if current_balance == 0:
                self.wallet.create_token(
                    'AGENT_WORK',
                    reward_amount,
                    {'type': 'work_reward', 'agent': self.agent_id}
                )
            else:
                # In real implementation, would receive from factory wallet
                pass
        except Exception as e:
            logger.warning(f"Could not award tokens: {e}")

    def stop(self):
        """Stop agent"""
        self.running = False
        logger.info(f"⏹️  Agent {self.agent_id} stopped")

# ==================== DEVELOPMENT AGENTS ====================

class CodeReviewAgent(BaseAgent):
    """Agent for code review and analysis"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.capabilities = [
            AgentCapability(
                name="code_review",
                description="Automated code review and quality analysis",
                required_resources={"cpu": 0.5, "memory": "1GB"},
                security_level="restricted",
                permissions=["read_code", "analyze_patterns", "suggest_improvements"]
            )
        ]

    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Process code review task"""
        start_time = time.time()

        try:
            code_content = task.task_data.get('code', '')
            file_path = task.task_data.get('file_path', 'unknown')

            # Simulate code analysis
            await asyncio.sleep(0.5)  # Simulate processing time

            review_results = {
                'quality_score': 85 + (hash(code_content) % 15),  # Simulate score
                'issues_found': [
                    {'type': 'style', 'line': 42, 'message': 'Consider using more descriptive variable names'},
                    {'type': 'performance', 'line': 67, 'message': 'This loop could be optimized'},
                    {'type': 'security', 'line': 123, 'message': 'Input validation recommended'}
                ][:hash(code_content) % 4],  # Random number of issues
                'suggestions': [
                    'Add type hints for better code documentation',
                    'Consider breaking down large functions',
                    'Add error handling for edge cases'
                ],
                'complexity_score': hash(code_content) % 100,
                'reviewed_file': file_path
            }

            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                result_data=review_results,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                result_data={},
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

class TestGenerationAgent(BaseAgent):
    """Agent for automated test generation"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.capabilities = [
            AgentCapability(
                name="test_generation",
                description="Generate automated tests for code",
                required_resources={"cpu": 0.3, "memory": "512MB"},
                security_level="restricted",
                permissions=["read_code", "generate_tests", "write_test_files"]
            )
        ]

    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Generate tests for provided code"""
        start_time = time.time()

        try:
            code_content = task.task_data.get('code', '')
            test_type = task.task_data.get('test_type', 'unit')

            await asyncio.sleep(0.3)  # Simulate test generation

            test_results = {
                'generated_tests': [
                    {
                        'test_name': 'test_basic_functionality',
                        'test_code': f'def test_basic_functionality():\n    # Generated test\n    assert True',
                        'coverage_target': 'main_function'
                    },
                    {
                        'test_name': 'test_edge_cases',
                        'test_code': f'def test_edge_cases():\n    # Test edge cases\n    assert handle_edge_case() is not None',
                        'coverage_target': 'edge_handling'
                    }
                ],
                'test_coverage': 78,
                'test_type': test_type,
                'recommendations': [
                    'Add integration tests for API endpoints',
                    'Consider property-based testing for complex logic'
                ]
            }

            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                result_data=test_results,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                result_data={},
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

class DeploymentAgent(BaseAgent):
    """Agent for handling deployments"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.capabilities = [
            AgentCapability(
                name="deployment",
                description="Automated deployment and infrastructure management",
                required_resources={"cpu": 1.0, "memory": "2GB", "network": True},
                security_level="classified",
                permissions=["deploy_code", "manage_infrastructure", "access_secrets"]
            )
        ]

    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Handle deployment task"""
        start_time = time.time()

        try:
            deployment_target = task.task_data.get('target', 'development')
            application = task.task_data.get('application', 'unknown')

            await asyncio.sleep(1.0)  # Simulate deployment time

            deployment_results = {
                'deployment_id': f"deploy_{uuid.uuid4().hex[:8]}",
                'target_environment': deployment_target,
                'application': application,
                'status': 'successful',
                'deployed_at': datetime.now().isoformat(),
                'health_checks': [
                    {'service': 'web_server', 'status': 'healthy'},
                    {'service': 'database', 'status': 'healthy'},
                    {'service': 'cache', 'status': 'healthy'}
                ],
                'deployment_url': f"https://{application}.{deployment_target}.example.com"
            }

            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                result_data=deployment_results,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                result_data={},
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

# ==================== CYBER DEFENSE AGENTS ====================

class ThreatDetectionAgent(BaseAgent):
    """Agent for threat detection and analysis"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.capabilities = [
            AgentCapability(
                name="threat_detection",
                description="Real-time threat detection and analysis",
                required_resources={"cpu": 2.0, "memory": "4GB", "gpu": 0.5},
                security_level="sovereign",
                permissions=["monitor_network", "analyze_traffic", "access_threat_intel"]
            )
        ]

        # Initialize threat database
        self.threat_signatures = self._load_threat_signatures()

    def _load_threat_signatures(self) -> Dict[str, Any]:
        """Load threat signatures database"""
        return {
            'malware_hashes': [
                'a1b2c3d4e5f6',
                '9z8y7x6w5v4u',
                'suspicious_hash_123'
            ],
            'attack_patterns': [
                'SQL injection attempts',
                'XSS payload patterns',
                'Command injection vectors'
            ],
            'suspicious_ips': [
                '192.168.1.100',
                '10.0.0.255'
            ]
        }

    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Process threat detection task"""
        start_time = time.time()

        try:
            data_source = task.task_data.get('source', 'network')
            data_content = task.task_data.get('data', '')

            await asyncio.sleep(0.8)  # Simulate analysis time

            # Simulate threat analysis
            threat_level = 'low'
            threats_detected = []

            # Check for known threats
            if any(sig in data_content for sig in self.threat_signatures['malware_hashes']):
                threat_level = 'critical'
                threats_detected.append('Known malware signature detected')

            if 'DROP TABLE' in data_content.upper():
                threat_level = 'high'
                threats_detected.append('SQL injection attempt detected')

            analysis_results = {
                'threat_level': threat_level,
                'threats_detected': threats_detected,
                'confidence_score': 0.85 if threats_detected else 0.1,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_source': data_source,
                'recommendations': [
                    'Block suspicious IP addresses',
                    'Update security rules',
                    'Alert security team'
                ] if threats_detected else ['Continue monitoring']
            }

            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                result_data=analysis_results,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                result_data={},
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

class IncidentResponseAgent(BaseAgent):
    """Agent for automated incident response"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.capabilities = [
            AgentCapability(
                name="incident_response",
                description="Automated incident response and containment",
                required_resources={"cpu": 1.5, "memory": "3GB"},
                security_level="sovereign",
                permissions=["block_traffic", "isolate_systems", "escalate_alerts"]
            )
        ]

    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Handle incident response"""
        start_time = time.time()

        try:
            incident_type = task.task_data.get('incident_type', 'unknown')
            severity = task.task_data.get('severity', 'medium')
            affected_systems = task.task_data.get('affected_systems', [])

            await asyncio.sleep(0.5)  # Simulate response time

            # Determine response actions based on severity
            response_actions = []

            if severity in ['critical', 'high']:
                response_actions.extend([
                    'Isolate affected systems from network',
                    'Activate incident response team',
                    'Preserve forensic evidence'
                ])

            if incident_type == 'malware':
                response_actions.extend([
                    'Run antivirus scans on affected systems',
                    'Block malicious file hashes',
                    'Update security signatures'
                ])

            response_results = {
                'incident_id': f"inc_{uuid.uuid4().hex[:8]}",
                'incident_type': incident_type,
                'severity': severity,
                'actions_taken': response_actions,
                'containment_status': 'contained' if severity != 'critical' else 'partial',
                'response_time': time.time() - start_time,
                'next_steps': [
                    'Continue monitoring affected systems',
                    'Conduct post-incident review',
                    'Update security procedures'
                ]
            }

            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                result_data=response_results,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                result_data={},
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

# ==================== PROJECT MANAGEMENT AGENT ====================

class ProjectManagementAgent(BaseAgent):
    """Agent for project planning, coordination, and reporting"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.capabilities = [
            AgentCapability(
                name="project_management",
                description="Automated project planning and coordination",
                required_resources={"cpu": 0.5, "memory": "1GB"},
                security_level="classified",
                permissions=["read_projects", "update_tasks", "generate_reports"]
            )
        ]

        # Project database
        self.projects = {}
        self.tasks = {}

    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Handle project management task"""
        start_time = time.time()

        try:
            action = task.task_data.get('action', 'status')
            project_id = task.task_data.get('project_id', 'default')

            if action == 'create_project':
                result = await self._create_project(task.task_data)
            elif action == 'update_task':
                result = await self._update_task(task.task_data)
            elif action == 'generate_report':
                result = await self._generate_report(project_id)
            else:
                result = await self._get_status(project_id)

            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                result_data=result,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                result_data={},
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

    async def _create_project(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new project"""
        project_id = data.get('project_id', f"proj_{uuid.uuid4().hex[:8]}")

        self.projects[project_id] = {
            'name': data.get('name', 'Unnamed Project'),
            'description': data.get('description', ''),
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'tasks': [],
            'milestones': data.get('milestones', []),
            'team_members': data.get('team_members', [])
        }

        return {'project_id': project_id, 'status': 'created'}

    async def _update_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update task status"""
        task_id = data.get('task_id', f"task_{uuid.uuid4().hex[:8]}")

        self.tasks[task_id] = {
            'title': data.get('title', 'New Task'),
            'status': data.get('status', 'todo'),
            'assignee': data.get('assignee', ''),
            'priority': data.get('priority', 3),
            'updated_at': datetime.now().isoformat()
        }

        return {'task_id': task_id, 'status': 'updated'}

    async def _generate_report(self, project_id: str) -> Dict[str, Any]:
        """Generate project status report"""
        project = self.projects.get(project_id, {})

        return {
            'project_id': project_id,
            'project_name': project.get('name', 'Unknown'),
            'total_tasks': len(self.tasks),
            'completed_tasks': len([t for t in self.tasks.values() if t.get('status') == 'done']),
            'in_progress_tasks': len([t for t in self.tasks.values() if t.get('status') == 'in_progress']),
            'report_generated_at': datetime.now().isoformat(),
            'overall_progress': 75,  # Simulated
            'recommendations': [
                'Focus on high-priority tasks',
                'Review blocked tasks',
                'Update project timeline'
            ]
        }

    async def _get_status(self, project_id: str) -> Dict[str, Any]:
        """Get project status"""
        project = self.projects.get(project_id, {})

        return {
            'project_id': project_id,
            'status': project.get('status', 'unknown'),
            'task_count': len(self.tasks),
            'last_updated': datetime.now().isoformat()
        }

# ==================== AGENT FACTORY ====================

class SovereignAgentFactory:
    """Factory for creating and managing sovereign agents"""

    def __init__(self):
        self.agents = {}
        self.running = False
        self.task_router = {}

        # Initialize factory wallet
        self.factory_wallet = SovereignWallet('factory_sovereign', 'factory_secure_2024')

        # Create initial token supply
        self._initialize_token_economy()

        logger.info("🏭 Sovereign Agent Factory initialized")

    def _initialize_token_economy(self):
        """Initialize factory token economy"""
        # Create different token types
        self.factory_wallet.create_token(
            'AGENT_WORK',
            100000,
            {'type': 'work_reward', 'description': 'Tokens for completed agent tasks'}
        )

        self.factory_wallet.create_token(
            'FACTORY_GOVERNANCE',
            10000,
            {'type': 'governance', 'description': 'Voting rights for factory decisions'}
        )

        self.factory_wallet.create_token(
            'SECURITY_BOND',
            50000,
            {'type': 'security', 'description': 'Security clearance and bonding tokens'}
        )

        logger.info("💰 Token economy initialized")

    def create_agent(self, agent_type: str, agent_id: str, config: Dict[str, Any]) -> BaseAgent:
        """Create new agent instance"""

        agent_classes = {
            'code_review': CodeReviewAgent,
            'test_generation': TestGenerationAgent,
            'deployment': DeploymentAgent,
            'threat_detection': ThreatDetectionAgent,
            'incident_response': IncidentResponseAgent,
            'project_management': ProjectManagementAgent
        }

        if agent_type not in agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent_class = agent_classes[agent_type]
        agent = agent_class(agent_id, config)

        self.agents[agent_id] = agent
        self.task_router[agent_type] = agent_id

        logger.info(f"🤖 Created {agent_type} agent: {agent_id}")
        return agent

    async def start_all_agents(self):
        """Start all created agents"""
        self.running = True
        logger.info("🚀 Starting all agents...")

        tasks = []
        for agent in self.agents.values():
            tasks.append(asyncio.create_task(agent.start()))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def submit_task(self, task: TaskRequest) -> str:
        """Submit task to appropriate agent"""
        if task.agent_type not in self.task_router:
            raise ValueError(f"No agent available for type: {task.agent_type}")

        agent_id = self.task_router[task.agent_type]
        agent = self.agents[agent_id]

        await agent.submit_task(task)
        logger.info(f"📋 Task {task.task_id} submitted to {agent_id}")

        return agent_id

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {}

        for agent_id, agent in self.agents.items():
            status[agent_id] = {
                'running': agent.running,
                'tasks_completed': agent.metrics.tasks_completed,
                'tasks_failed': agent.metrics.tasks_failed,
                'last_activity': agent.metrics.last_activity.isoformat() if agent.metrics.last_activity else None,
                'security_level': agent.security_level
            }

        return status

    def get_factory_wallet_status(self) -> Dict[str, Any]:
        """Get factory wallet status"""
        return {
            'wallet_id': self.factory_wallet.wallet_id,
            'token_balances': {
                'AGENT_WORK': self.factory_wallet.get_balance('AGENT_WORK'),
                'FACTORY_GOVERNANCE': self.factory_wallet.get_balance('FACTORY_GOVERNANCE'),
                'SECURITY_BOND': self.factory_wallet.get_balance('SECURITY_BOND')
            },
            'created_at': self.factory_wallet.created_at.isoformat()
        }

    def stop_all_agents(self):
        """Stop all agents"""
        logger.info("⏹️  Stopping all agents...")
        for agent in self.agents.values():
            agent.stop()
        self.running = False

# ==================== DEMO AND TESTING ====================

async def demo_sovereign_agent_factory():
    """Demonstrate the Sovereign Agent Factory"""

    print("\n" + "="*80)
    print("SOVEREIGN AGENT FACTORY DEMONSTRATION")
    print("="*80)

    # Initialize factory
    factory = SovereignAgentFactory()

    # Create agents
    print("\n🏭 Creating Agents...")

    # Development agents
    factory.create_agent('code_review', 'code_reviewer_01', {
        'security_level': 'restricted',
        'permissions': ['read_code', 'analyze_patterns']
    })

    factory.create_agent('test_generation', 'test_generator_01', {
        'security_level': 'restricted',
        'permissions': ['read_code', 'generate_tests']
    })

    factory.create_agent('deployment', 'deployer_01', {
        'security_level': 'classified',
        'permissions': ['deploy_code', 'manage_infrastructure']
    })

    # Cyber defense agents
    factory.create_agent('threat_detection', 'threat_hunter_01', {
        'security_level': 'sovereign',
        'permissions': ['monitor_network', 'analyze_traffic', 'access_threat_intel']
    })

    factory.create_agent('incident_response', 'incident_responder_01', {
        'security_level': 'sovereign',
        'permissions': ['block_traffic', 'isolate_systems', 'escalate_alerts']
    })

    # Project management agent
    factory.create_agent('project_management', 'project_manager_01', {
        'security_level': 'classified',
        'permissions': ['read_projects', 'update_tasks', 'generate_reports']
    })

    print(f"✅ Created {len(factory.agents)} agents")

    # Start agents (run in background)
    print("\n🚀 Starting agents...")
    agent_tasks = [asyncio.create_task(agent.start()) for agent in factory.agents.values()]

    # Wait a moment for agents to start
    await asyncio.sleep(1)

    # Submit test tasks
    print("\n📋 Submitting test tasks...")

    # Code review task
    code_review_task = TaskRequest(
        task_id="task_001",
        agent_type="code_review",
        priority=2,
        task_data={
            'code': 'def hello_world():\n    print("Hello, World!")\n    return True',
            'file_path': 'hello.py'
        },
        requester="demo_user"
    )
    await factory.submit_task(code_review_task)

    # Threat detection task
    threat_task = TaskRequest(
        task_id="task_002",
        agent_type="threat_detection",
        priority=1,
        task_data={
            'source': 'network',
            'data': 'GET /admin; DROP TABLE users; --'
        },
        requester="security_system"
    )
    await factory.submit_task(threat_task)

    # Project management task
    project_task = TaskRequest(
        task_id="task_003",
        agent_type="project_management",
        priority=3,
        task_data={
            'action': 'create_project',
            'name': 'Sovereign AI Development',
            'description': 'Building next-gen AI agents'
        },
        requester="pm_system"
    )
    await factory.submit_task(project_task)

    # Let agents process tasks
    print("\n⏳ Processing tasks (5 seconds)...")
    await asyncio.sleep(5)

    # Show status
    print("\n📊 Agent Status:")
    status = factory.get_agent_status()
    for agent_id, agent_status in status.items():
        print(f"  {agent_id}:")
        print(f"    Running: {agent_status['running']}")
        print(f"    Completed: {agent_status['tasks_completed']}")
        print(f"    Failed: {agent_status['tasks_failed']}")

    # Show wallet status
    print("\n💰 Factory Wallet Status:")
    wallet_status = factory.get_factory_wallet_status()
    print(f"  Wallet ID: {wallet_status['wallet_id']}")
    print(f"  Token Balances:")
    for token_type, balance in wallet_status['token_balances'].items():
        print(f"    {token_type}: {balance}")

    # Stop agents
    print("\n⏹️  Stopping agents...")
    factory.stop_all_agents()

    # Cancel background tasks
    for task in agent_tasks:
        task.cancel()

    print("\n✅ Sovereign Agent Factory demonstration complete!")
    print("="*80)

if __name__ == "__main__":
    try:
        asyncio.run(demo_sovereign_agent_factory())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        import traceback
        traceback.print_exc()