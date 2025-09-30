#!/usr/bin/env python3
"""
Memory Guardian - Cognitive Health & Property Protection System
Integrates with LL TOKEN ecosystem and federated learning for Alzheimer's prevention
"""

import json
import os
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


@dataclass
class CognitiveAssessment:
    """Record of cognitive health assessment"""
    timestamp: str
    user_id: str
    memory_score: float  # 0-100
    reaction_time_ms: float
    pattern_recognition_score: float
    problem_solving_score: float
    overall_score: float
    baseline_deviation: float  # % change from baseline

    def to_dict(self):
        return asdict(self)


@dataclass
class PropertyRecord:
    """Secure property and asset record"""
    record_id: str
    record_type: str  # 'deed', 'will', 'financial', 'medical', 'other'
    title: str
    encrypted_content: str
    document_hash: str
    created_at: str
    last_accessed: str
    trusted_contacts: List[str]

    def to_dict(self):
        return asdict(self)


@dataclass
class TrustedContact:
    """Emergency contact with tiered access"""
    contact_id: str
    name: str
    relationship: str
    access_level: int  # 1=emergency only, 2=view, 3=full access
    phone: str
    email: str
    verification_code: str
    ll_token_address: Optional[str]
    reputation_score: float

    def to_dict(self):
        return asdict(self)


class MemoryGuardianDB:
    """Secure local database for Memory Guardian data"""

    def __init__(self, db_path: str = "memory_guardian.db"):
        self.db_path = db_path
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Cognitive assessments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                memory_score REAL,
                reaction_time_ms REAL,
                pattern_recognition_score REAL,
                problem_solving_score REAL,
                overall_score REAL,
                baseline_deviation REAL,
                fl_contribution_hash TEXT,
                tokens_earned REAL DEFAULT 0
            )
        """)

        # Property records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS property_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT UNIQUE NOT NULL,
                record_type TEXT NOT NULL,
                title TEXT NOT NULL,
                encrypted_content TEXT NOT NULL,
                document_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                trusted_contacts TEXT
            )
        """)

        # Trusted contacts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trusted_contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contact_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                relationship TEXT,
                access_level INTEGER DEFAULT 1,
                phone TEXT,
                email TEXT,
                verification_code TEXT,
                ll_token_address TEXT,
                reputation_score REAL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

        # User settings and baselines
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                cognitive_baseline REAL,
                encryption_key_salt TEXT NOT NULL,
                ll_token_wallet TEXT,
                fl_participation_enabled INTEGER DEFAULT 1,
                total_tokens_earned REAL DEFAULT 0,
                last_assessment TEXT
            )
        """)

        # Alert logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                resolved INTEGER DEFAULT 0,
                resolution_notes TEXT
            )
        """)

        self.conn.commit()

    def add_cognitive_assessment(self, assessment: CognitiveAssessment,
                                tokens_earned: float = 0) -> int:
        """Store cognitive assessment result"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO cognitive_assessments
            (timestamp, user_id, memory_score, reaction_time_ms,
             pattern_recognition_score, problem_solving_score,
             overall_score, baseline_deviation, tokens_earned)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (assessment.timestamp, assessment.user_id, assessment.memory_score,
              assessment.reaction_time_ms, assessment.pattern_recognition_score,
              assessment.problem_solving_score, assessment.overall_score,
              assessment.baseline_deviation, tokens_earned))
        self.conn.commit()
        return cursor.lastrowid

    def get_assessment_history(self, user_id: str, days: int = 30) -> List[Dict]:
        """Get cognitive assessment history"""
        cursor = self.conn.cursor()
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute("""
            SELECT * FROM cognitive_assessments
            WHERE user_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (user_id, cutoff_date))

        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def add_property_record(self, record: PropertyRecord):
        """Add encrypted property record"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO property_records
            (record_id, record_type, title, encrypted_content,
             document_hash, created_at, last_accessed, trusted_contacts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (record.record_id, record.record_type, record.title,
              record.encrypted_content, record.document_hash,
              record.created_at, record.last_accessed,
              json.dumps(record.trusted_contacts)))
        self.conn.commit()

    def add_trusted_contact(self, contact: TrustedContact):
        """Add trusted emergency contact"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trusted_contacts
            (contact_id, name, relationship, access_level, phone,
             email, verification_code, ll_token_address,
             reputation_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (contact.contact_id, contact.name, contact.relationship,
              contact.access_level, contact.phone, contact.email,
              contact.verification_code, contact.ll_token_address,
              contact.reputation_score, datetime.now().isoformat()))
        self.conn.commit()

    def log_alert(self, alert_type: str, severity: str, message: str):
        """Log security or health alert"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO alert_logs (timestamp, alert_type, severity, message)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().isoformat(), alert_type, severity, message))
        self.conn.commit()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class CognitiveHealthMonitor:
    """Monitor cognitive health and detect decline patterns"""

    def __init__(self, user_id: str, db: MemoryGuardianDB):
        self.user_id = user_id
        self.db = db
        self.baseline_score = self._calculate_baseline()

    def _calculate_baseline(self) -> float:
        """Calculate cognitive baseline from recent assessments"""
        history = self.db.get_assessment_history(self.user_id, days=90)
        if len(history) < 3:
            return 0.0  # Need more data

        # Average of first 10 assessments (initial baseline period)
        baseline_scores = [h['overall_score'] for h in history[:10]]
        return sum(baseline_scores) / len(baseline_scores)

    def assess_cognitive_state(self, assessment: CognitiveAssessment) -> Dict:
        """Evaluate current cognitive state vs baseline"""
        if self.baseline_score == 0:
            status = "establishing_baseline"
            risk_level = "unknown"
        else:
            deviation = ((assessment.overall_score - self.baseline_score)
                        / self.baseline_score * 100)
            assessment.baseline_deviation = deviation

            if deviation < -20:
                status = "significant_decline"
                risk_level = "high"
                self.db.log_alert("cognitive_decline", "high",
                                f"Significant cognitive decline detected: {deviation:.1f}%")
            elif deviation < -10:
                status = "mild_decline"
                risk_level = "medium"
                self.db.log_alert("cognitive_decline", "medium",
                                f"Mild cognitive decline detected: {deviation:.1f}%")
            elif deviation < -5:
                status = "monitoring"
                risk_level = "low"
            else:
                status = "healthy"
                risk_level = "none"

        return {
            "status": status,
            "risk_level": risk_level,
            "baseline_score": self.baseline_score,
            "current_score": assessment.overall_score,
            "deviation_percent": assessment.baseline_deviation,
            "recommendation": self._get_recommendation(status)
        }

    def _get_recommendation(self, status: str) -> str:
        """Get health recommendation based on status"""
        recommendations = {
            "establishing_baseline": "Continue daily exercises to establish your cognitive baseline",
            "healthy": "Great job! Continue your daily cognitive exercises",
            "monitoring": "Consider increasing exercise frequency and consulting a healthcare provider",
            "mild_decline": "Please consult with a healthcare provider for evaluation",
            "significant_decline": "URGENT: Contact your healthcare provider and trusted contacts immediately"
        }
        return recommendations.get(status, "Continue monitoring")


class PropertyVault:
    """Secure vault for important documents and property records"""

    def __init__(self, user_id: str, master_password: str, db: MemoryGuardianDB):
        self.user_id = user_id
        self.db = db
        self.encryption_key = self._derive_key(master_password)
        self.cipher = Fernet(self.encryption_key)

    def _derive_key(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from master password"""
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def add_document(self, record_type: str, title: str, content: str,
                    trusted_contacts: List[str] = None) -> str:
        """Add encrypted document to vault"""
        # Encrypt content
        encrypted_content = self.cipher.encrypt(content.encode())

        # Calculate document hash for integrity verification
        doc_hash = hashlib.sha256(content.encode()).hexdigest()

        # Create record
        record_id = hashlib.sha256(
            f"{self.user_id}{title}{time.time()}".encode()
        ).hexdigest()[:16]

        record = PropertyRecord(
            record_id=record_id,
            record_type=record_type,
            title=title,
            encrypted_content=encrypted_content.decode(),
            document_hash=doc_hash,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            trusted_contacts=trusted_contacts or []
        )

        self.db.add_property_record(record)
        return record_id

    def retrieve_document(self, record_id: str) -> Optional[str]:
        """Decrypt and retrieve document"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT encrypted_content, document_hash
            FROM property_records
            WHERE record_id = ?
        """, (record_id,))

        result = cursor.fetchone()
        if not result:
            return None

        encrypted_content, original_hash = result

        # Decrypt content
        try:
            decrypted = self.cipher.decrypt(encrypted_content.encode())

            # Verify integrity
            current_hash = hashlib.sha256(decrypted).hexdigest()
            if current_hash != original_hash:
                self.db.log_alert("integrity_violation", "critical",
                                f"Document hash mismatch for record {record_id}")
                return None

            # Update last accessed
            cursor.execute("""
                UPDATE property_records
                SET last_accessed = ?
                WHERE record_id = ?
            """, (datetime.now().isoformat(), record_id))
            self.db.conn.commit()

            return decrypted.decode()
        except Exception as e:
            self.db.log_alert("decryption_error", "high",
                            f"Failed to decrypt record {record_id}: {str(e)}")
            return None


class TokenRewardSystem:
    """Integrate with LL TOKEN system for rewards"""

    def __init__(self, wallet_address: str):
        self.wallet_address = wallet_address
        self.reward_rates = {
            "daily_assessment": {"LLT-EXP": 50, "LLT-EDU": 10},
            "baseline_established": {"LLT-EXP": 200, "LLT-REP": 100},
            "consistent_participation": {"LLT-EXP": 100, "LLT-REWARD": 25},
            "fl_contribution": {"LLT-REWARD": 50, "LLT-DATA": 25},
            "document_secured": {"LLT-REP": 10},
            "trusted_contact_verified": {"LLT-REP": 50}
        }

    def calculate_rewards(self, action: str, quality_multiplier: float = 1.0) -> Dict[str, float]:
        """Calculate token rewards for actions"""
        base_rewards = self.reward_rates.get(action, {})
        return {token: amount * quality_multiplier
                for token, amount in base_rewards.items()}

    def issue_rewards(self, action: str, quality: float = 1.0) -> Dict:
        """Issue tokens for completed actions"""
        rewards = self.calculate_rewards(action, quality)

        # In production, this would interact with actual token system
        # For now, we log the rewards
        return {
            "wallet": self.wallet_address,
            "action": action,
            "rewards": rewards,
            "timestamp": datetime.now().isoformat(),
            "status": "issued"
        }


class FederatedLearningContribution:
    """Contribute anonymous cognitive health data to FL network"""

    def __init__(self, user_id: str, db: MemoryGuardianDB):
        self.user_id = user_id
        self.db = db

    def prepare_fl_contribution(self, assessment: CognitiveAssessment) -> Dict:
        """Prepare anonymized assessment data for FL"""
        # Remove all identifying information
        anonymous_data = {
            "age_group": self._generalize_age(),
            "memory_score": assessment.memory_score,
            "reaction_time_ms": assessment.reaction_time_ms,
            "pattern_recognition_score": assessment.pattern_recognition_score,
            "problem_solving_score": assessment.problem_solving_score,
            "overall_score": assessment.overall_score,
            "timestamp_generalized": self._generalize_timestamp(assessment.timestamp)
        }

        # Create contribution hash for verification
        contribution_hash = hashlib.sha256(
            json.dumps(anonymous_data, sort_keys=True).encode()
        ).hexdigest()

        return {
            "data": anonymous_data,
            "contribution_hash": contribution_hash,
            "model_target": "alzheimers_early_detection_v1"
        }

    def _generalize_age(self) -> str:
        """Generalize age to 10-year buckets for privacy"""
        # This would get actual age from user profile
        # For now, return placeholder
        return "60-70"

    def _generalize_timestamp(self, timestamp: str) -> str:
        """Generalize timestamp to week for privacy"""
        dt = datetime.fromisoformat(timestamp)
        # Round to start of week
        week_start = dt - timedelta(days=dt.weekday())
        return week_start.strftime("%Y-W%W")


class MemoryGuardianApp:
    """Main application controller"""

    def __init__(self, user_id: str, master_password: str,
                 ll_token_wallet: str):
        self.user_id = user_id
        self.db = MemoryGuardianDB()
        self.cognitive_monitor = CognitiveHealthMonitor(user_id, self.db)
        self.property_vault = PropertyVault(user_id, master_password, self.db)
        self.token_system = TokenRewardSystem(ll_token_wallet)
        self.fl_contributor = FederatedLearningContribution(user_id, self.db)

        print(f"✅ Memory Guardian initialized for user {user_id}")
        print(f"🧠 Cognitive baseline: {self.cognitive_monitor.baseline_score:.1f}")

    def run_daily_assessment(self, assessment_data: Dict) -> Dict:
        """Run daily cognitive assessment"""
        # Create assessment object
        assessment = CognitiveAssessment(
            timestamp=datetime.now().isoformat(),
            user_id=self.user_id,
            memory_score=assessment_data['memory_score'],
            reaction_time_ms=assessment_data['reaction_time_ms'],
            pattern_recognition_score=assessment_data['pattern_recognition_score'],
            problem_solving_score=assessment_data['problem_solving_score'],
            overall_score=assessment_data['overall_score'],
            baseline_deviation=0.0
        )

        # Evaluate cognitive state
        evaluation = self.cognitive_monitor.assess_cognitive_state(assessment)

        # Calculate rewards
        quality = min(assessment.overall_score / 100, 1.0)
        rewards = self.token_system.issue_rewards("daily_assessment", quality)

        # Store assessment
        total_tokens = sum(rewards['rewards'].values())
        self.db.add_cognitive_assessment(assessment, total_tokens)

        # Contribute to FL if enabled
        fl_contribution = self.fl_contributor.prepare_fl_contribution(assessment)
        fl_rewards = self.token_system.issue_rewards("fl_contribution")

        return {
            "assessment": assessment.to_dict(),
            "evaluation": evaluation,
            "rewards": rewards,
            "fl_contribution": fl_contribution,
            "fl_rewards": fl_rewards
        }

    def secure_document(self, doc_type: str, title: str,
                       content: str, trusted_contacts: List[str] = None) -> Dict:
        """Securely store important document"""
        record_id = self.property_vault.add_document(
            doc_type, title, content, trusted_contacts
        )

        rewards = self.token_system.issue_rewards("document_secured")

        return {
            "record_id": record_id,
            "title": title,
            "type": doc_type,
            "secured_at": datetime.now().isoformat(),
            "rewards": rewards
        }

    def add_trusted_contact(self, name: str, relationship: str,
                          access_level: int, phone: str, email: str,
                          ll_token_address: str = None) -> Dict:
        """Add emergency contact"""
        contact = TrustedContact(
            contact_id=hashlib.sha256(f"{email}{time.time()}".encode()).hexdigest()[:12],
            name=name,
            relationship=relationship,
            access_level=access_level,
            phone=phone,
            email=email,
            verification_code=os.urandom(4).hex(),
            ll_token_address=ll_token_address,
            reputation_score=0.0
        )

        self.db.add_trusted_contact(contact)
        rewards = self.token_system.issue_rewards("trusted_contact_verified")

        return {
            "contact": contact.to_dict(),
            "rewards": rewards
        }

    def get_dashboard_summary(self) -> Dict:
        """Get comprehensive dashboard summary"""
        history = self.db.get_assessment_history(self.user_id, days=30)

        return {
            "user_id": self.user_id,
            "cognitive_baseline": self.cognitive_monitor.baseline_score,
            "total_assessments": len(history),
            "last_assessment": history[0] if history else None,
            "average_score_30d": sum(h['overall_score'] for h in history) / len(history) if history else 0,
            "trend": self._calculate_trend(history),
            "total_tokens_earned": sum(h.get('tokens_earned', 0) for h in history),
            "recent_alerts": self._get_recent_alerts()
        }

    def _calculate_trend(self, history: List[Dict]) -> str:
        """Calculate cognitive trend"""
        if len(history) < 2:
            return "insufficient_data"

        recent = history[:7]  # Last 7 assessments
        older = history[7:14] if len(history) > 7 else history

        recent_avg = sum(h['overall_score'] for h in recent) / len(recent)
        older_avg = sum(h['overall_score'] for h in older) / len(older)

        change = ((recent_avg - older_avg) / older_avg * 100)

        if change > 5:
            return "improving"
        elif change < -5:
            return "declining"
        else:
            return "stable"

    def _get_recent_alerts(self) -> List[Dict]:
        """Get recent unresolved alerts"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT * FROM alert_logs
            WHERE resolved = 0
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def shutdown(self):
        """Clean shutdown"""
        self.db.close()
        print("✅ Memory Guardian shutdown complete")


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("MEMORY GUARDIAN - Cognitive Health & Property Protection System")
    print("=" * 80)

    # Initialize app
    app = MemoryGuardianApp(
        user_id="user_demo_001",
        master_password="SecurePassword123!",
        ll_token_wallet="LL1234567890ABCDEF"
    )

    # Simulate daily assessment
    print("\n📊 Running daily cognitive assessment...")
    assessment_result = app.run_daily_assessment({
        "memory_score": 85.0,
        "reaction_time_ms": 450.0,
        "pattern_recognition_score": 88.0,
        "problem_solving_score": 82.0,
        "overall_score": 85.0
    })

    print(f"   Status: {assessment_result['evaluation']['status']}")
    print(f"   Risk Level: {assessment_result['evaluation']['risk_level']}")
    print(f"   Rewards: {assessment_result['rewards']['rewards']}")

    # Secure a document
    print("\n🔒 Securing important document...")
    doc_result = app.secure_document(
        doc_type="will",
        title="Last Will and Testament",
        content="This is a sample will document with important legal information...",
        trusted_contacts=["contact_001", "contact_002"]
    )
    print(f"   Document ID: {doc_result['record_id']}")
    print(f"   Rewards: {doc_result['rewards']['rewards']}")

    # Add trusted contact
    print("\n👥 Adding trusted emergency contact...")
    contact_result = app.add_trusted_contact(
        name="Jane Smith",
        relationship="daughter",
        access_level=3,
        phone="+1-555-0123",
        email="jane.smith@example.com",
        ll_token_address="LL_JANE_WALLET_123"
    )
    print(f"   Contact ID: {contact_result['contact']['contact_id']}")
    print(f"   Verification Code: {contact_result['contact']['verification_code']}")

    # Dashboard summary
    print("\n📈 Dashboard Summary:")
    summary = app.get_dashboard_summary()
    print(f"   Total Assessments: {summary['total_assessments']}")
    print(f"   30-Day Average Score: {summary['average_score_30d']:.1f}")
    print(f"   Trend: {summary['trend']}")
    print(f"   Total Tokens Earned: {summary['total_tokens_earned']:.1f}")

    app.shutdown()
    print("\n✅ Demo complete!")