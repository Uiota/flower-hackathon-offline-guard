"""
Tests for Off-Guard security module
"""

import os
import pytest
import tempfile
from pathlib import Path

from src import guard


class TestGuard:
    """Test cases for guard module."""

    def setup_method(self):
        """Setup for each test."""
        # Set offline mode for tests
        os.environ["OFFLINE_MODE"] = "1"

    def test_preflight_check_passes(self):
        """Test that preflight check passes with proper environment."""
        # Should not raise exception
        guard.preflight_check()

    def test_preflight_check_fails_without_offline_mode(self):
        """Test that preflight check fails without OFFLINE_MODE."""
        # Remove offline mode
        if "OFFLINE_MODE" in os.environ:
            del os.environ["OFFLINE_MODE"]

        with pytest.raises(AssertionError, match="OFFLINE_MODE"):
            guard.preflight_check()

        # Restore for cleanup
        os.environ["OFFLINE_MODE"] = "1"

    def test_keypair_generation(self):
        """Test Ed25519 keypair generation."""
        private_key, public_key = guard.new_keypair()

        assert private_key is not None
        assert public_key is not None
        assert hasattr(private_key, 'sign')
        assert hasattr(public_key, 'verify')

    def test_sign_and_verify_success(self):
        """Test successful signing and verification."""
        private_key, public_key = guard.new_keypair()
        test_data = b"Hello, federated world!"

        # Sign data
        signature = guard.sign_blob(private_key, test_data)
        assert signature is not None
        assert len(signature) > 0

        # Verify signature
        is_valid = guard.verify_blob(public_key, test_data, signature)
        assert is_valid is True

    def test_verify_invalid_signature(self):
        """Test verification of invalid signature."""
        private_key1, public_key1 = guard.new_keypair()
        private_key2, public_key2 = guard.new_keypair()

        test_data = b"Test data"

        # Sign with key1, verify with key2 (should fail)
        signature = guard.sign_blob(private_key1, test_data)
        is_valid = guard.verify_blob(public_key2, test_data, signature)
        assert is_valid is False

    def test_verify_tampered_data(self):
        """Test verification fails with tampered data."""
        private_key, public_key = guard.new_keypair()
        original_data = b"Original data"
        tampered_data = b"Tampered data"

        # Sign original, verify tampered (should fail)
        signature = guard.sign_blob(private_key, original_data)
        is_valid = guard.verify_blob(public_key, tampered_data, signature)
        assert is_valid is False

    def test_key_serialization(self):
        """Test key serialization and deserialization."""
        private_key1, public_key1 = guard.new_keypair()

        # Serialize keys
        private_bytes = guard.serialize_private_key(private_key1)
        public_bytes = guard.serialize_public_key(public_key1)

        # Deserialize keys
        private_key2 = guard.deserialize_private_key(private_bytes)
        public_key2 = guard.deserialize_public_key(public_bytes)

        # Test that deserialized keys work
        test_data = b"Serialization test"
        signature = guard.sign_blob(private_key2, test_data)
        is_valid = guard.verify_blob(public_key2, test_data, signature)
        assert is_valid is True

    def test_hash_data(self):
        """Test data hashing."""
        test_data = b"Hash this data"

        # Test different algorithms
        hash_sha256 = guard.hash_data(test_data, "sha256")
        hash_sha512 = guard.hash_data(test_data, "sha512")
        hash_blake2b = guard.hash_data(test_data, "blake2b")

        assert len(hash_sha256) == 64  # SHA256 hex length
        assert len(hash_sha512) == 128  # SHA512 hex length
        assert len(hash_blake2b) == 128  # BLAKE2b hex length

        # Hashes should be different
        assert hash_sha256 != hash_sha512
        assert hash_sha256 != hash_blake2b

        # Same input should produce same hash
        hash_sha256_2 = guard.hash_data(test_data, "sha256")
        assert hash_sha256 == hash_sha256_2

    def test_guard_config(self):
        """Test GuardConfig validation."""
        config = guard.GuardConfig()

        # Test default values
        assert config.offline_mode_required is True
        assert "3.10" in config.allowed_python_versions
        assert "torch" in config.required_libraries

        # Test custom config
        custom_config = guard.GuardConfig(
            offline_mode_required=False,
            signature_required=False
        )
        assert custom_config.offline_mode_required is False
        assert custom_config.signature_required is False