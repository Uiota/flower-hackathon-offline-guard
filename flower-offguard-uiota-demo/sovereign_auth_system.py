#!/usr/bin/env python3
"""
Sovereign Authentication System
Advanced login system with biometric integration and wallet-based authentication
"""

import asyncio
import json
import hashlib
import hmac
import secrets
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import bcrypt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AuthSession:
    """Represents an active authentication session"""
    session_id: str
    wallet_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    permissions: List[str]
    biometric_verified: bool = False
    mfa_verified: bool = False

class BiometricAuthenticator:
    """Simulates advanced biometric authentication"""

    def __init__(self):
        self.facial_recognition_active = True
        self.voice_recognition_active = True
        self.fingerprint_active = True
        self.behavioral_analysis_active = True

    async def verify_facial_recognition(self, image_data: bytes) -> Dict[str, Any]:
        """Simulate facial recognition verification"""
        await asyncio.sleep(0.5)  # Simulate processing time

        # Simulate facial recognition processing
        confidence_score = 0.95 + (secrets.randbits(8) / 1000)

        return {
            "verified": confidence_score > 0.85,
            "confidence": min(confidence_score, 0.999),
            "liveness_check": True,
            "template_match": True,
            "processing_time": 0.5
        }

    async def verify_voice_recognition(self, audio_data: bytes) -> Dict[str, Any]:
        """Simulate voice recognition verification"""
        await asyncio.sleep(0.3)

        confidence_score = 0.92 + (secrets.randbits(8) / 1000)

        return {
            "verified": confidence_score > 0.80,
            "confidence": min(confidence_score, 0.999),
            "voice_pattern_match": True,
            "anti_spoofing": True,
            "processing_time": 0.3
        }

    async def verify_fingerprint(self, fingerprint_data: bytes) -> Dict[str, Any]:
        """Simulate fingerprint verification"""
        await asyncio.sleep(0.2)

        confidence_score = 0.98 + (secrets.randbits(8) / 1000)

        return {
            "verified": confidence_score > 0.90,
            "confidence": min(confidence_score, 0.999),
            "minutiae_match": True,
            "quality_score": 0.95,
            "processing_time": 0.2
        }

    async def behavioral_analysis(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavioral patterns for continuous authentication"""
        await asyncio.sleep(0.1)

        # Analyze typing patterns, mouse movements, etc.
        behavioral_score = 0.88 + (secrets.randbits(8) / 1000)

        return {
            "verified": behavioral_score > 0.75,
            "confidence": min(behavioral_score, 0.999),
            "typing_pattern_match": True,
            "mouse_dynamics_match": True,
            "device_fingerprint_match": True,
            "processing_time": 0.1
        }

class SovereignAuthSystem:
    """Advanced authentication system with wallet integration"""

    def __init__(self, db_path: str = "sovereign_auth.db"):
        self.db_path = db_path
        self.active_sessions: Dict[str, AuthSession] = {}
        self.biometric_auth = BiometricAuthenticator()
        self.session_timeout = timedelta(hours=8)
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)

        # Initialize database
        self._init_database()

        logger.info("🔐 Sovereign Authentication System initialized")

    def _init_database(self):
        """Initialize authentication database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    wallet_id TEXT UNIQUE NOT NULL,
                    public_key TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    permissions TEXT DEFAULT '[]',
                    biometric_templates TEXT DEFAULT '{}',
                    security_level INTEGER DEFAULT 1,
                    mfa_secret TEXT
                )
            ''')

            # Authentication attempts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS auth_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    ip_address TEXT,
                    attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN,
                    failure_reason TEXT,
                    biometric_used BOOLEAN DEFAULT FALSE,
                    session_id TEXT
                )
            ''')

            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id INTEGER,
                    wallet_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    ip_address TEXT,
                    user_agent TEXT,
                    biometric_verified BOOLEAN DEFAULT FALSE,
                    mfa_verified BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            conn.commit()
            logger.info("✅ Authentication database initialized")

    async def register_user(self, username: str, email: str, password: str,
                          wallet_id: str, public_key: str,
                          permissions: List[str] = None) -> Dict[str, Any]:
        """Register a new user with wallet integration"""

        if permissions is None:
            permissions = ["basic_access", "wallet_read"]

        # Hash password with bcrypt
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        # Generate MFA secret
        mfa_secret = secrets.token_urlsafe(32)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO users
                    (username, email, password_hash, wallet_id, public_key,
                     permissions, mfa_secret, security_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (username, email, password_hash, wallet_id, public_key,
                      json.dumps(permissions), mfa_secret, 2))

                user_id = cursor.lastrowid
                conn.commit()

                logger.info(f"✅ User registered: {username} (ID: {user_id})")

                return {
                    "success": True,
                    "user_id": user_id,
                    "username": username,
                    "wallet_id": wallet_id,
                    "mfa_secret": mfa_secret,
                    "permissions": permissions
                }

        except sqlite3.IntegrityError as e:
            logger.error(f"❌ Registration failed: {e}")
            return {
                "success": False,
                "error": "Username, email, or wallet_id already exists"
            }

    async def authenticate_user(self, username: str, password: str,
                              ip_address: str = "127.0.0.1",
                              biometric_data: Dict[str, bytes] = None) -> Dict[str, Any]:
        """Authenticate user with multi-factor authentication"""

        start_time = time.time()

        # Check for account lockout
        if await self._is_account_locked(username, ip_address):
            await self._log_auth_attempt(username, ip_address, False, "account_locked")
            return {
                "success": False,
                "error": "Account temporarily locked due to failed attempts",
                "lockout_remaining": 900  # 15 minutes in seconds
            }

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get user data
                cursor.execute('''
                    SELECT id, username, password_hash, wallet_id, public_key,
                           permissions, biometric_templates, security_level,
                           mfa_secret, is_active
                    FROM users WHERE username = ?
                ''', (username,))

                user_data = cursor.fetchone()

                if not user_data or not user_data[9]:  # is_active check
                    await self._log_auth_attempt(username, ip_address, False, "user_not_found")
                    return {
                        "success": False,
                        "error": "Invalid credentials"
                    }

                # Verify password
                if not bcrypt.checkpw(password.encode(), user_data[2].encode()):
                    await self._log_auth_attempt(username, ip_address, False, "invalid_password")
                    return {
                        "success": False,
                        "error": "Invalid credentials"
                    }

                # Biometric verification if provided
                biometric_verified = False
                if biometric_data and user_data[7] >= 2:  # security_level >= 2
                    biometric_result = await self._verify_biometrics(biometric_data)
                    biometric_verified = biometric_result["verified"]

                    if not biometric_verified and user_data[7] >= 3:  # Required for level 3+
                        await self._log_auth_attempt(username, ip_address, False, "biometric_failed")
                        return {
                            "success": False,
                            "error": "Biometric verification required"
                        }

                # Create session
                session_id = secrets.token_urlsafe(32)
                expires_at = datetime.now(timezone.utc) + self.session_timeout

                permissions = json.loads(user_data[5])

                session = AuthSession(
                    session_id=session_id,
                    wallet_id=user_data[3],
                    user_id=str(user_data[0]),
                    created_at=datetime.now(timezone.utc),
                    expires_at=expires_at,
                    permissions=permissions,
                    biometric_verified=biometric_verified,
                    mfa_verified=False  # Will be set after MFA verification
                )

                # Store session
                cursor.execute('''
                    INSERT INTO sessions
                    (session_id, user_id, wallet_id, expires_at, ip_address,
                     biometric_verified, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, user_data[0], user_data[3], expires_at,
                      ip_address, biometric_verified, True))

                # Update last login
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                ''', (user_data[0],))

                conn.commit()

                self.active_sessions[session_id] = session

                # Log successful attempt
                await self._log_auth_attempt(username, ip_address, True, "success", session_id)

                auth_time = time.time() - start_time

                logger.info(f"✅ User authenticated: {username} (Session: {session_id[:8]}...)")

                return {
                    "success": True,
                    "session_id": session_id,
                    "wallet_id": user_data[3],
                    "user_id": user_data[0],
                    "username": username,
                    "permissions": permissions,
                    "expires_at": expires_at.isoformat(),
                    "biometric_verified": biometric_verified,
                    "mfa_required": user_data[7] >= 2,
                    "mfa_secret": user_data[8] if user_data[7] >= 2 else None,
                    "security_level": user_data[7],
                    "auth_time": auth_time
                }

        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            await self._log_auth_attempt(username, ip_address, False, str(e))
            return {
                "success": False,
                "error": "Authentication system error"
            }

    async def verify_session(self, session_id: str) -> Dict[str, Any]:
        """Verify an active session"""

        if session_id not in self.active_sessions:
            # Check database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT s.session_id, s.user_id, s.wallet_id, s.expires_at,
                           s.biometric_verified, s.mfa_verified, u.username, u.permissions
                    FROM sessions s
                    JOIN users u ON s.user_id = u.id
                    WHERE s.session_id = ? AND s.is_active = TRUE
                ''', (session_id,))

                session_data = cursor.fetchone()

                if not session_data:
                    return {"valid": False, "error": "Session not found"}

                expires_at = datetime.fromisoformat(session_data[3])
                if expires_at < datetime.now(timezone.utc):
                    return {"valid": False, "error": "Session expired"}

                # Recreate session object
                permissions = json.loads(session_data[7])
                session = AuthSession(
                    session_id=session_data[0],
                    wallet_id=session_data[2],
                    user_id=str(session_data[1]),
                    created_at=datetime.now(timezone.utc),
                    expires_at=expires_at,
                    permissions=permissions,
                    biometric_verified=session_data[4],
                    mfa_verified=session_data[5]
                )

                self.active_sessions[session_id] = session

        session = self.active_sessions.get(session_id)

        if not session:
            return {"valid": False, "error": "Session not found"}

        if session.expires_at < datetime.now(timezone.utc):
            await self.logout_user(session_id)
            return {"valid": False, "error": "Session expired"}

        return {
            "valid": True,
            "session": {
                "session_id": session.session_id,
                "wallet_id": session.wallet_id,
                "user_id": session.user_id,
                "permissions": session.permissions,
                "biometric_verified": session.biometric_verified,
                "mfa_verified": session.mfa_verified,
                "expires_at": session.expires_at.isoformat()
            }
        }

    async def logout_user(self, session_id: str) -> Dict[str, Any]:
        """Logout user and invalidate session"""

        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        # Deactivate in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions SET is_active = FALSE WHERE session_id = ?
            ''', (session_id,))
            conn.commit()

        logger.info(f"✅ User logged out (Session: {session_id[:8]}...)")

        return {"success": True, "message": "Logged out successfully"}

    async def _verify_biometrics(self, biometric_data: Dict[str, bytes]) -> Dict[str, Any]:
        """Verify biometric data"""

        results = {}
        overall_verified = True

        if "facial" in biometric_data:
            facial_result = await self.biometric_auth.verify_facial_recognition(
                biometric_data["facial"]
            )
            results["facial"] = facial_result
            if not facial_result["verified"]:
                overall_verified = False

        if "voice" in biometric_data:
            voice_result = await self.biometric_auth.verify_voice_recognition(
                biometric_data["voice"]
            )
            results["voice"] = voice_result
            if not voice_result["verified"]:
                overall_verified = False

        if "fingerprint" in biometric_data:
            fingerprint_result = await self.biometric_auth.verify_fingerprint(
                biometric_data["fingerprint"]
            )
            results["fingerprint"] = fingerprint_result
            if not fingerprint_result["verified"]:
                overall_verified = False

        return {
            "verified": overall_verified,
            "results": results,
            "confidence": sum(r.get("confidence", 0) for r in results.values()) / len(results) if results else 0
        }

    async def _is_account_locked(self, username: str, ip_address: str) -> bool:
        """Check if account is locked due to failed attempts"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check failed attempts in the last lockout duration
            cutoff_time = datetime.now(timezone.utc) - self.lockout_duration

            cursor.execute('''
                SELECT COUNT(*) FROM auth_attempts
                WHERE (username = ? OR ip_address = ?)
                AND success = FALSE
                AND attempt_time > ?
            ''', (username, ip_address, cutoff_time))

            failed_attempts = cursor.fetchone()[0]

            return failed_attempts >= self.max_failed_attempts

    async def _log_auth_attempt(self, username: str, ip_address: str,
                               success: bool, failure_reason: str = None,
                               session_id: str = None):
        """Log authentication attempt"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO auth_attempts
                (username, ip_address, success, failure_reason, session_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, ip_address, success, failure_reason, session_id))

            conn.commit()

    async def get_user_profile(self, session_id: str) -> Dict[str, Any]:
        """Get user profile information"""

        session_check = await self.verify_session(session_id)
        if not session_check["valid"]:
            return {"success": False, "error": session_check["error"]}

        session = self.active_sessions[session_id]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT username, email, wallet_id, created_at, last_login,
                       permissions, security_level
                FROM users WHERE id = ?
            ''', (int(session.user_id),))

            user_data = cursor.fetchone()

            if not user_data:
                return {"success": False, "error": "User not found"}

            return {
                "success": True,
                "profile": {
                    "username": user_data[0],
                    "email": user_data[1],
                    "wallet_id": user_data[2],
                    "created_at": user_data[3],
                    "last_login": user_data[4],
                    "permissions": json.loads(user_data[5]),
                    "security_level": user_data[6],
                    "session_info": {
                        "biometric_verified": session.biometric_verified,
                        "mfa_verified": session.mfa_verified,
                        "expires_at": session.expires_at.isoformat()
                    }
                }
            }

    async def update_security_level(self, session_id: str, new_level: int) -> Dict[str, Any]:
        """Update user security level (requires admin permissions)"""

        session_check = await self.verify_session(session_id)
        if not session_check["valid"]:
            return {"success": False, "error": session_check["error"]}

        session = self.active_sessions[session_id]

        if "admin" not in session.permissions:
            return {"success": False, "error": "Admin permissions required"}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE users SET security_level = ? WHERE id = ?
            ''', (new_level, int(session.user_id)))

            conn.commit()

        return {"success": True, "message": f"Security level updated to {new_level}"}

# Demo function
async def demo_auth_system():
    """Demonstrate the sovereign authentication system"""

    print("\n🔐 SOVEREIGN AUTHENTICATION SYSTEM DEMO")
    print("=" * 50)

    # Initialize authentication system
    auth_system = SovereignAuthSystem("demo_auth.db")

    # Register a demo user
    print("\n📝 Registering demo user...")

    # Generate demo wallet data
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()

    registration_result = await auth_system.register_user(
        username="demo_user",
        email="demo@sovereign.io",
        password="SecurePassword123!",
        wallet_id="wallet_" + secrets.token_hex(8),
        public_key=public_key_pem,
        permissions=["basic_access", "wallet_read", "wallet_write", "agent_access"]
    )

    print(f"Registration: {registration_result}")

    # Authenticate user
    print("\n🔑 Authenticating user...")

    # Simulate biometric data
    biometric_data = {
        "facial": b"fake_facial_data",
        "fingerprint": b"fake_fingerprint_data"
    }

    auth_result = await auth_system.authenticate_user(
        username="demo_user",
        password="SecurePassword123!",
        ip_address="192.168.1.100",
        biometric_data=biometric_data
    )

    print(f"Authentication: {auth_result}")

    if auth_result["success"]:
        session_id = auth_result["session_id"]

        # Get user profile
        print("\n👤 Getting user profile...")
        profile_result = await auth_system.get_user_profile(session_id)
        print(f"Profile: {profile_result}")

        # Verify session
        print("\n✅ Verifying session...")
        session_result = await auth_system.verify_session(session_id)
        print(f"Session verification: {session_result}")

        # Logout
        print("\n🚪 Logging out...")
        logout_result = await auth_system.logout_user(session_id)
        print(f"Logout: {logout_result}")

    print("\n✅ Authentication system demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_auth_system())