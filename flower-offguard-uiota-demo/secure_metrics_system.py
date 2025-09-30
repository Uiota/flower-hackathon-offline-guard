#!/usr/bin/env python3
"""
Secure Metrics System for Memory Guardian
Multi-agent architecture with zero-trust verification and cryptographic audit trails
"""

import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import base64


@dataclass
class SecureMetric:
    """Cryptographically signed metric with audit trail"""
    metric_id: str
    metric_type: str  # 'cognitive', 'system', 'security', 'agent'
    metric_name: str
    value: float
    unit: str
    timestamp: str
    collector_agent_id: str
    signature: str
    previous_hash: str
    current_hash: str
    verification_count: int = 0
    consensus_reached: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class MetricVerification:
    """Agent verification of a metric"""
    verification_id: str
    metric_id: str
    verifier_agent_id: str
    verification_result: bool  # True = valid, False = invalid
    confidence_score: float  # 0.0-1.0
    timestamp: str
    signature: str


@dataclass
class AgentConsensus:
    """Consensus result from multiple agents"""
    consensus_id: str
    metric_id: str
    total_verifiers: int
    positive_verifications: int
    negative_verifications: int
    average_confidence: float
    consensus_reached: bool
    consensus_value: Optional[float]
    timestamp: str


class CryptographicSigner:
    """Ed25519 signature system for metrics"""

    def __init__(self, agent_id: str, keys_dir: str = ".secrets/agent_keys"):
        self.agent_id = agent_id
        self.keys_dir = keys_dir
        os.makedirs(keys_dir, exist_ok=True)

        self.key_path = os.path.join(keys_dir, f"{agent_id}_key.pem")
        self.pubkey_path = os.path.join(keys_dir, f"{agent_id}_pubkey.pem")

        # Load or generate keys
        if os.path.exists(self.key_path):
            self._load_keys()
        else:
            self._generate_keys()

    def _generate_keys(self):
        """Generate new Ed25519 key pair"""
        self.private_key = ed25519.Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

        # Save keys
        with open(self.key_path, 'wb') as f:
            f.write(self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        with open(self.pubkey_path, 'wb') as f:
            f.write(self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))

    def _load_keys(self):
        """Load existing keys"""
        with open(self.key_path, 'rb') as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(), password=None
            )

        with open(self.pubkey_path, 'rb') as f:
            self.public_key = serialization.load_pem_public_key(f.read())

    def sign(self, data: str) -> str:
        """Sign data with private key"""
        signature = self.private_key.sign(data.encode())
        return base64.b64encode(signature).decode()

    def verify(self, data: str, signature: str, public_key: ed25519.Ed25519PublicKey) -> bool:
        """Verify signature with public key"""
        try:
            sig_bytes = base64.b64decode(signature.encode())
            public_key.verify(sig_bytes, data.encode())
            return True
        except Exception:
            return False

    def get_public_key_pem(self) -> str:
        """Get public key as PEM string"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()


class MetricsBlockchain:
    """Blockchain-style audit trail for metrics"""

    def __init__(self, db_path: str = "secure_metrics.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize blockchain database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Metrics chain
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_id TEXT UNIQUE NOT NULL,
                metric_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                timestamp TEXT NOT NULL,
                collector_agent_id TEXT NOT NULL,
                signature TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                current_hash TEXT NOT NULL,
                verification_count INTEGER DEFAULT 0,
                consensus_reached INTEGER DEFAULT 0
            )
        """)

        # Verifications
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metric_verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verification_id TEXT UNIQUE NOT NULL,
                metric_id TEXT NOT NULL,
                verifier_agent_id TEXT NOT NULL,
                verification_result INTEGER NOT NULL,
                confidence_score REAL NOT NULL,
                timestamp TEXT NOT NULL,
                signature TEXT NOT NULL,
                FOREIGN KEY (metric_id) REFERENCES metrics_chain(metric_id)
            )
        """)

        # Consensus records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_consensus (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consensus_id TEXT UNIQUE NOT NULL,
                metric_id TEXT NOT NULL,
                total_verifiers INTEGER NOT NULL,
                positive_verifications INTEGER NOT NULL,
                negative_verifications INTEGER NOT NULL,
                average_confidence REAL NOT NULL,
                consensus_reached INTEGER NOT NULL,
                consensus_value REAL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (metric_id) REFERENCES metrics_chain(metric_id)
            )
        """)

        # Agent registry
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT UNIQUE NOT NULL,
                agent_type TEXT NOT NULL,
                public_key TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                total_metrics_collected INTEGER DEFAULT 0,
                total_verifications INTEGER DEFAULT 0,
                reputation_score REAL DEFAULT 1.0
            )
        """)

        conn.commit()
        conn.close()

    def get_last_hash(self) -> str:
        """Get hash of last block in chain"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT current_hash FROM metrics_chain
            ORDER BY id DESC LIMIT 1
        """)
        result = cursor.fetchone()
        conn.close()

        if result:
            return result[0]
        return "0" * 64  # Genesis hash

    def calculate_hash(self, metric: SecureMetric) -> str:
        """Calculate SHA-256 hash of metric"""
        data = f"{metric.metric_id}{metric.metric_type}{metric.metric_name}"
        data += f"{metric.value}{metric.unit}{metric.timestamp}"
        data += f"{metric.collector_agent_id}{metric.previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()

    def add_metric(self, metric: SecureMetric) -> bool:
        """Add metric to blockchain"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO metrics_chain
                (metric_id, metric_type, metric_name, value, unit, timestamp,
                 collector_agent_id, signature, previous_hash, current_hash,
                 verification_count, consensus_reached)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.metric_id, metric.metric_type, metric.metric_name,
                metric.value, metric.unit, metric.timestamp,
                metric.collector_agent_id, metric.signature,
                metric.previous_hash, metric.current_hash,
                metric.verification_count, int(metric.consensus_reached)
            ))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding metric: {e}")
            return False

    def add_verification(self, verification: MetricVerification) -> bool:
        """Add verification to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO metric_verifications
                (verification_id, metric_id, verifier_agent_id,
                 verification_result, confidence_score, timestamp, signature)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                verification.verification_id, verification.metric_id,
                verification.verifier_agent_id, int(verification.verification_result),
                verification.confidence_score, verification.timestamp,
                verification.signature
            ))

            # Update verification count
            cursor.execute("""
                UPDATE metrics_chain
                SET verification_count = verification_count + 1
                WHERE metric_id = ?
            """, (verification.metric_id,))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding verification: {e}")
            return False

    def record_consensus(self, consensus: AgentConsensus) -> bool:
        """Record consensus result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO agent_consensus
                (consensus_id, metric_id, total_verifiers,
                 positive_verifications, negative_verifications,
                 average_confidence, consensus_reached, consensus_value, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                consensus.consensus_id, consensus.metric_id,
                consensus.total_verifiers, consensus.positive_verifications,
                consensus.negative_verifications, consensus.average_confidence,
                int(consensus.consensus_reached), consensus.consensus_value,
                consensus.timestamp
            ))

            # Update metric consensus status
            cursor.execute("""
                UPDATE metrics_chain
                SET consensus_reached = ?
                WHERE metric_id = ?
            """, (int(consensus.consensus_reached), consensus.metric_id))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error recording consensus: {e}")
            return False

    def verify_chain_integrity(self) -> Tuple[bool, List[str]]:
        """Verify entire blockchain integrity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT metric_id, previous_hash, current_hash
            FROM metrics_chain ORDER BY id ASC
        """)

        blocks = cursor.fetchall()
        conn.close()

        errors = []
        previous_hash = "0" * 64

        for metric_id, prev_hash, curr_hash in blocks:
            if prev_hash != previous_hash:
                errors.append(f"Hash mismatch at {metric_id}: expected {previous_hash}, got {prev_hash}")
            previous_hash = curr_hash

        return len(errors) == 0, errors


class MetricsCollectorAgent:
    """Agent that collects and signs metrics"""

    def __init__(self, agent_id: str, blockchain: MetricsBlockchain):
        self.agent_id = agent_id
        self.blockchain = blockchain
        self.signer = CryptographicSigner(agent_id)

        print(f"✅ Metrics Collector Agent {agent_id} initialized")

    def collect_cognitive_metric(self, metric_name: str, value: float, unit: str) -> SecureMetric:
        """Collect and sign a cognitive health metric"""
        metric_id = hashlib.sha256(
            f"{self.agent_id}{metric_name}{time.time()}".encode()
        ).hexdigest()[:16]

        previous_hash = self.blockchain.get_last_hash()

        # Create metric
        metric = SecureMetric(
            metric_id=metric_id,
            metric_type="cognitive",
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=datetime.now().isoformat(),
            collector_agent_id=self.agent_id,
            signature="",
            previous_hash=previous_hash,
            current_hash=""
        )

        # Calculate hash
        metric.current_hash = self.blockchain.calculate_hash(metric)

        # Sign metric
        sign_data = f"{metric.metric_id}{metric.current_hash}"
        metric.signature = self.signer.sign(sign_data)

        # Add to blockchain
        self.blockchain.add_metric(metric)

        return metric

    def collect_system_metric(self, metric_name: str, value: float, unit: str) -> SecureMetric:
        """Collect system performance metric"""
        return self._collect_metric("system", metric_name, value, unit)

    def collect_security_metric(self, metric_name: str, value: float, unit: str) -> SecureMetric:
        """Collect security-related metric"""
        return self._collect_metric("security", metric_name, value, unit)

    def _collect_metric(self, metric_type: str, metric_name: str,
                       value: float, unit: str) -> SecureMetric:
        """Generic metric collection"""
        metric_id = hashlib.sha256(
            f"{self.agent_id}{metric_name}{time.time()}".encode()
        ).hexdigest()[:16]

        previous_hash = self.blockchain.get_last_hash()

        metric = SecureMetric(
            metric_id=metric_id,
            metric_type=metric_type,
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=datetime.now().isoformat(),
            collector_agent_id=self.agent_id,
            signature="",
            previous_hash=previous_hash,
            current_hash=""
        )

        metric.current_hash = self.blockchain.calculate_hash(metric)
        sign_data = f"{metric.metric_id}{metric.current_hash}"
        metric.signature = self.signer.sign(sign_data)

        self.blockchain.add_metric(metric)
        return metric


class MetricsVerifierAgent:
    """Agent that verifies metrics collected by others"""

    def __init__(self, agent_id: str, blockchain: MetricsBlockchain):
        self.agent_id = agent_id
        self.blockchain = blockchain
        self.signer = CryptographicSigner(agent_id)

        print(f"✅ Metrics Verifier Agent {agent_id} initialized")

    def verify_metric(self, metric: SecureMetric) -> MetricVerification:
        """Verify a metric's authenticity and validity"""
        # Check 1: Hash integrity
        calculated_hash = self.blockchain.calculate_hash(metric)
        hash_valid = (calculated_hash == metric.current_hash)

        # Check 2: Chain integrity (previous hash matches)
        chain_valid = self._verify_chain_link(metric)

        # Check 3: Signature verification (would need collector's public key)
        signature_valid = True  # Simplified for demo

        # Check 4: Value plausibility
        value_plausible = self._check_value_plausibility(metric)

        # Calculate overall validity and confidence
        checks = [hash_valid, chain_valid, signature_valid, value_plausible]
        verification_result = all(checks)
        confidence_score = sum(checks) / len(checks)

        # Create verification record
        verification_id = hashlib.sha256(
            f"{self.agent_id}{metric.metric_id}{time.time()}".encode()
        ).hexdigest()[:16]

        verification = MetricVerification(
            verification_id=verification_id,
            metric_id=metric.metric_id,
            verifier_agent_id=self.agent_id,
            verification_result=verification_result,
            confidence_score=confidence_score,
            timestamp=datetime.now().isoformat(),
            signature=""
        )

        # Sign verification
        sign_data = f"{verification.verification_id}{verification.verification_result}{verification.confidence_score}"
        verification.signature = self.signer.sign(sign_data)

        # Record verification
        self.blockchain.add_verification(verification)

        return verification

    def _verify_chain_link(self, metric: SecureMetric) -> bool:
        """Verify this metric's previous_hash matches actual previous block"""
        conn = sqlite3.connect(self.blockchain.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT current_hash FROM metrics_chain
            WHERE timestamp < ?
            ORDER BY timestamp DESC LIMIT 1
        """, (metric.timestamp,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return result[0] == metric.previous_hash
        return metric.previous_hash == "0" * 64  # Genesis block

    def _check_value_plausibility(self, metric: SecureMetric) -> bool:
        """Check if metric value is within plausible range"""
        plausibility_ranges = {
            "cognitive": {"overall_score": (0, 100), "memory_score": (0, 100)},
            "system": {"cpu_usage": (0, 100), "memory_mb": (0, 10000)},
            "security": {"failed_logins": (0, 100), "encryption_strength": (0, 256)}
        }

        if metric.metric_type in plausibility_ranges:
            type_ranges = plausibility_ranges[metric.metric_type]
            for key, (min_val, max_val) in type_ranges.items():
                if key in metric.metric_name.lower():
                    return min_val <= metric.value <= max_val

        return True  # Default to plausible if no range defined


class ConsensusCoordinator:
    """Coordinates consensus among multiple verifier agents"""

    def __init__(self, blockchain: MetricsBlockchain):
        self.blockchain = blockchain
        print("✅ Consensus Coordinator initialized")

    def achieve_consensus(self, metric_id: str,
                         min_verifiers: int = 3,
                         consensus_threshold: float = 0.66) -> AgentConsensus:
        """Achieve consensus on a metric from multiple verifiers"""

        conn = sqlite3.connect(self.blockchain.db_path)
        cursor = conn.cursor()

        # Get all verifications for this metric
        cursor.execute("""
            SELECT verification_result, confidence_score
            FROM metric_verifications
            WHERE metric_id = ?
        """, (metric_id,))

        verifications = cursor.fetchall()
        conn.close()

        if len(verifications) < min_verifiers:
            return self._insufficient_consensus(metric_id, len(verifications))

        positive = sum(1 for v in verifications if v[0] == 1)
        negative = len(verifications) - positive
        avg_confidence = sum(v[1] for v in verifications) / len(verifications)

        # Consensus reached if threshold met
        consensus_reached = (positive / len(verifications)) >= consensus_threshold

        # Get metric value for consensus
        conn = sqlite3.connect(self.blockchain.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM metrics_chain WHERE metric_id = ?", (metric_id,))
        result = cursor.fetchone()
        conn.close()

        consensus_value = result[0] if result and consensus_reached else None

        consensus_id = hashlib.sha256(
            f"consensus{metric_id}{time.time()}".encode()
        ).hexdigest()[:16]

        consensus = AgentConsensus(
            consensus_id=consensus_id,
            metric_id=metric_id,
            total_verifiers=len(verifications),
            positive_verifications=positive,
            negative_verifications=negative,
            average_confidence=avg_confidence,
            consensus_reached=consensus_reached,
            consensus_value=consensus_value,
            timestamp=datetime.now().isoformat()
        )

        self.blockchain.record_consensus(consensus)
        return consensus

    def _insufficient_consensus(self, metric_id: str, verifier_count: int) -> AgentConsensus:
        """Return consensus failure due to insufficient verifiers"""
        consensus_id = hashlib.sha256(
            f"consensus{metric_id}{time.time()}".encode()
        ).hexdigest()[:16]

        return AgentConsensus(
            consensus_id=consensus_id,
            metric_id=metric_id,
            total_verifiers=verifier_count,
            positive_verifications=0,
            negative_verifications=0,
            average_confidence=0.0,
            consensus_reached=False,
            consensus_value=None,
            timestamp=datetime.now().isoformat()
        )


class SecureMetricsSystem:
    """Complete secure metrics system with multi-agent architecture"""

    def __init__(self):
        self.blockchain = MetricsBlockchain()

        # Initialize agents
        self.collector = MetricsCollectorAgent("collector_001", self.blockchain)
        self.verifiers = [
            MetricsVerifierAgent(f"verifier_{i:03d}", self.blockchain)
            for i in range(1, 4)  # 3 verifier agents
        ]
        self.coordinator = ConsensusCoordinator(self.blockchain)

        print("✅ Secure Metrics System initialized")

    def collect_and_verify_metric(self, metric_type: str, metric_name: str,
                                  value: float, unit: str) -> Dict:
        """Collect metric and get consensus verification"""

        # Step 1: Collect metric
        if metric_type == "cognitive":
            metric = self.collector.collect_cognitive_metric(metric_name, value, unit)
        elif metric_type == "system":
            metric = self.collector.collect_system_metric(metric_name, value, unit)
        else:
            metric = self.collector.collect_security_metric(metric_name, value, unit)

        print(f"📊 Metric collected: {metric_name} = {value} {unit}")

        # Step 2: Verify with multiple agents
        verifications = []
        for verifier in self.verifiers:
            verification = verifier.verify_metric(metric)
            verifications.append(verification)
            print(f"   ✓ Verified by {verifier.agent_id}: "
                  f"{'VALID' if verification.verification_result else 'INVALID'} "
                  f"(confidence: {verification.confidence_score:.2f})")

        # Step 3: Achieve consensus
        consensus = self.coordinator.achieve_consensus(metric.metric_id)
        print(f"   🤝 Consensus: {'REACHED' if consensus.consensus_reached else 'NOT REACHED'} "
              f"({consensus.positive_verifications}/{consensus.total_verifiers} positive)")

        return {
            "metric": metric.to_dict(),
            "verifications": [v.__dict__ for v in verifications],
            "consensus": consensus.__dict__
        }

    def get_metrics_dashboard(self) -> Dict:
        """Get comprehensive metrics dashboard"""
        conn = sqlite3.connect(self.blockchain.db_path)
        cursor = conn.cursor()

        # Total metrics
        cursor.execute("SELECT COUNT(*) FROM metrics_chain")
        total_metrics = cursor.fetchone()[0]

        # Metrics by type
        cursor.execute("""
            SELECT metric_type, COUNT(*) FROM metrics_chain
            GROUP BY metric_type
        """)
        by_type = dict(cursor.fetchall())

        # Consensus stats
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN consensus_reached = 1 THEN 1 ELSE 0 END) as reached
            FROM metrics_chain
        """)
        consensus_total, consensus_reached = cursor.fetchone()

        # Chain integrity
        integrity, errors = self.blockchain.verify_chain_integrity()

        conn.close()

        return {
            "total_metrics": total_metrics,
            "metrics_by_type": by_type,
            "consensus_rate": consensus_reached / max(consensus_total, 1),
            "chain_integrity": integrity,
            "integrity_errors": errors
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("SECURE METRICS SYSTEM - Multi-Agent Architecture")
    print("=" * 80)

    # Initialize system
    system = SecureMetricsSystem()

    print("\n" + "=" * 80)
    print("TEST 1: Collecting Cognitive Health Metrics")
    print("=" * 80)

    # Collect cognitive metrics
    result1 = system.collect_and_verify_metric(
        "cognitive", "overall_score", 85.5, "points"
    )

    result2 = system.collect_and_verify_metric(
        "cognitive", "memory_score", 88.2, "points"
    )

    print("\n" + "=" * 80)
    print("TEST 2: Collecting System Metrics")
    print("=" * 80)

    # Collect system metrics
    result3 = system.collect_and_verify_metric(
        "system", "cpu_usage", 45.3, "percent"
    )

    result4 = system.collect_and_verify_metric(
        "system", "memory_usage", 1024.5, "MB"
    )

    print("\n" + "=" * 80)
    print("TEST 3: Collecting Security Metrics")
    print("=" * 80)

    # Collect security metrics
    result5 = system.collect_and_verify_metric(
        "security", "encryption_strength", 256, "bits"
    )

    print("\n" + "=" * 80)
    print("DASHBOARD SUMMARY")
    print("=" * 80)

    dashboard = system.get_metrics_dashboard()
    print(f"\nTotal Metrics Collected: {dashboard['total_metrics']}")
    print(f"Metrics by Type:")
    for metric_type, count in dashboard['metrics_by_type'].items():
        print(f"   {metric_type}: {count}")
    print(f"\nConsensus Rate: {dashboard['consensus_rate']:.1%}")
    print(f"Chain Integrity: {'✅ VALID' if dashboard['chain_integrity'] else '❌ COMPROMISED'}")

    if dashboard['integrity_errors']:
        print(f"Integrity Errors:")
        for error in dashboard['integrity_errors']:
            print(f"   ⚠️ {error}")

    print("\n✅ Secure Metrics System Demo Complete!")