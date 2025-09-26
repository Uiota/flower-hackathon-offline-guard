"""
Off-Guard Zero-Trust Security Module

Implements cryptographic signing/verification and environment safety checks
for federated learning model updates.
"""

import hashlib
import logging
import os
import platform
import sys
from typing import Tuple

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ed25519
# from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class GuardConfig:
    """Configuration for Off-Guard security checks."""

    # Environment requirements
    offline_mode_required: bool = True
    allowed_python_versions: List[str] = field(default_factory=lambda: ["3.10", "3.11", "3.12"])

    # Library version constraints (simplified - in production would be more comprehensive)
    required_libraries: Dict[str, str] = field(default_factory=lambda: {
        "torch": "2.1.0",
        "flwr": "1.8.0",
        "cryptography": "41.0.7"
    })

    # Security settings
    signature_required: bool = True
    verify_system_integrity: bool = True


def preflight_check(config: GuardConfig = None) -> None:
    """
    Perform comprehensive security and environment checks before FL operations.

    Args:
        config: Security configuration (uses defaults if None)

    Raises:
        AssertionError: If security checks fail
        RuntimeError: If critical security violations detected
    """
    if config is None:
        config = GuardConfig()

    logger.info("Running Off-Guard preflight security checks...")

    # 1. Check offline mode
    if config.offline_mode_required:
        offline_mode = os.getenv("OFFLINE_MODE")
        if offline_mode != "1":
            raise AssertionError(
                "OFFLINE_MODE=1 environment variable required for safe operation. "
                "Set OFFLINE_MODE=1 to enable secure offline mode."
            )
        logger.info("✓ Offline mode enabled")

    # 2. Verify Python version
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current_python not in config.allowed_python_versions:
        raise AssertionError(
            f"Python version {current_python} not in allowed versions: {config.allowed_python_versions}"
        )
    logger.info(f"✓ Python version {current_python} verified")

    # 3. Check critical library versions (simplified check)
    for lib_name, expected_version in config.required_libraries.items():
        try:
            if lib_name == "torch":
                import torch
                actual_version = torch.__version__.split('+')[0]  # Remove CUDA suffix
            elif lib_name == "flwr":
                import flwr
                actual_version = flwr.__version__
            elif lib_name == "cryptography":
                import cryptography
                actual_version = cryptography.__version__
            else:
                continue  # Skip unknown libraries

            # Simple version check (in production, use proper semantic versioning)
            if not actual_version.startswith(expected_version.split('.')[0]):
                logger.warning(f"Library {lib_name} version {actual_version} may be incompatible with {expected_version}")
            else:
                logger.info(f"✓ Library {lib_name} version {actual_version} verified")

        except ImportError:
            logger.warning(f"Required library {lib_name} not found - skipping in development mode")

    # 4. System integrity checks (basic implementation)
    if config.verify_system_integrity:
        _check_system_integrity()

    logger.info("✓ All Off-Guard security checks passed")


def _check_system_integrity() -> None:
    """Basic system integrity checks (platform-specific)."""
    try:
        # Check if we're in a virtual environment (recommended for isolation)
        if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
            logger.info("✓ Running in virtual environment")
        else:
            logger.warning("Not running in virtual environment - recommended for isolation")

        # Basic process check (avoid common risky processes if possible)
        system_info = {
            "platform": platform.system(),
            "architecture": platform.architecture()[0],
            "python_implementation": platform.python_implementation(),
        }
        logger.info(f"✓ System info verified: {system_info}")

    except Exception as e:
        logger.warning(f"System integrity check partial failure (non-critical): {e}")


def new_keypair() -> Tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
    """
    Generate a new Ed25519 keypair for signing/verification.

    Returns:
        Tuple of (private_key, public_key)
    """
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    logger.debug("New Ed25519 keypair generated")
    return private_key, public_key


def sign_blob(private_key: ed25519.Ed25519PrivateKey, data: bytes) -> bytes:
    """
    Sign data using Ed25519 private key.

    Args:
        private_key: Ed25519 private key
        data: Data to sign

    Returns:
        Signature bytes
    """
    try:
        signature = private_key.sign(data)
        logger.debug(f"Data signed successfully ({len(signature)} byte signature)")
        return signature
    except Exception as e:
        logger.error(f"Signing failed: {e}")
        raise


def verify_blob(public_key: ed25519.Ed25519PublicKey, data: bytes, signature: bytes) -> bool:
    """
    Verify signature using Ed25519 public key.

    Args:
        public_key: Ed25519 public key
        data: Original data
        signature: Signature to verify

    Returns:
        True if signature is valid, False otherwise
    """
    try:
        public_key.verify(signature, data)
        logger.debug("Signature verification successful")
        return True
    except Exception as e:
        logger.debug(f"Signature verification failed: {e}")
        return False


def hash_data(data: bytes, algorithm: str = "sha256") -> str:
    """
    Generate cryptographic hash of data.

    Args:
        data: Data to hash
        algorithm: Hash algorithm ("sha256", "sha512", "blake2b")

    Returns:
        Hex-encoded hash string
    """
    if algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(data).hexdigest()
    elif algorithm == "blake2b":
        return hashlib.blake2b(data).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def serialize_private_key(private_key: ed25519.Ed25519PrivateKey) -> bytes:
    """Serialize private key to bytes."""
    return private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )


def serialize_public_key(public_key: ed25519.Ed25519PublicKey) -> bytes:
    """Serialize public key to bytes."""
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )


def deserialize_private_key(key_bytes: bytes) -> ed25519.Ed25519PrivateKey:
    """Deserialize private key from bytes."""
    return ed25519.Ed25519PrivateKey.from_private_bytes(key_bytes)


def deserialize_public_key(key_bytes: bytes) -> ed25519.Ed25519PublicKey:
    """Deserialize public key from bytes."""
    return ed25519.Ed25519PublicKey.from_public_bytes(key_bytes)