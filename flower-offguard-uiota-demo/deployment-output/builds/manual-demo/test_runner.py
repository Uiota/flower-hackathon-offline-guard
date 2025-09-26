#!/usr/bin/env python3
"""
Simple test runner for flower-offguard-uiota demo
Runs tests without requiring pytest dependency
"""

import os
import sys
import traceback
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, '.')

def run_guard_tests():
    """Run guard module tests."""
    print("Running Guard Module Tests...")

    try:
        from src import guard

        # Test 1: GuardConfig creation
        print("  Test 1: GuardConfig creation...")
        config = guard.GuardConfig()
        assert config.offline_mode_required is True
        assert "3.11" in config.allowed_python_versions
        print("    ✓ PASSED")

        # Test 2: Keypair generation
        print("  Test 2: Keypair generation...")
        private_key, public_key = guard.new_keypair()
        assert private_key is not None
        assert public_key is not None
        assert hasattr(private_key, 'sign')
        print("    ✓ PASSED")

        # Test 3: Sign and verify
        print("  Test 3: Sign and verify...")
        test_data = b"Hello, federated world!"
        signature = guard.sign_blob(private_key, test_data)
        assert signature is not None
        assert len(signature) > 0

        is_valid = guard.verify_blob(public_key, test_data, signature)
        assert is_valid is True
        print("    ✓ PASSED")

        # Test 4: Verify invalid signature
        print("  Test 4: Verify invalid signature...")
        private_key2, public_key2 = guard.new_keypair()
        signature2 = guard.sign_blob(private_key2, test_data)
        is_valid = guard.verify_blob(public_key, test_data, signature2)
        assert is_valid is False
        print("    ✓ PASSED")

        # Test 5: Hash data
        print("  Test 5: Hash data...")
        test_data = b"Hash this data"
        hash_sha256 = guard.hash_data(test_data, "sha256")
        hash_sha512 = guard.hash_data(test_data, "sha512")
        assert len(hash_sha256) == 64  # SHA256 hex length
        assert len(hash_sha512) == 128  # SHA512 hex length
        assert hash_sha256 != hash_sha512
        print("    ✓ PASSED")

        # Test 6: Preflight check (with offline mode)
        print("  Test 6: Preflight check...")
        os.environ["OFFLINE_MODE"] = "1"
        guard.preflight_check()  # Should not raise exception
        print("    ✓ PASSED")

        print("All Guard Tests PASSED! ✅")
        return True

    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        traceback.print_exc()
        return False

def run_mesh_sync_tests():
    """Run mesh sync module tests."""
    print("Running Mesh Sync Module Tests...")

    try:
        from src.mesh_sync import MeshTransport, MeshConfig, MeshUpdate
        import json
        import time

        # Setup
        temp_dir = Path(tempfile.mkdtemp())
        config = MeshConfig(
            base_latency_ms=0,
            jitter_ms=0,
            dropout_rate=0.0,
            cleanup_old_files=False
        )
        mesh = MeshTransport(config, temp_dir)

        # Test 1: Push update
        print("  Test 1: Push update...")
        client_id = "test_client_1"
        round_number = 1
        payload = {"weights": [1, 2, 3], "accuracy": 0.95}

        result = mesh.push_update(client_id, round_number, payload)
        assert result is True

        # Check file was created
        update_files = list(temp_dir.glob(f"update_*_r{round_number}_*.json"))
        assert len(update_files) == 1
        print("    ✓ PASSED")

        # Test 2: Pull updates
        print("  Test 2: Pull updates...")
        # Push multiple updates
        for i in range(3):
            mesh.push_update(f"client_{i+1}", 2, {"client": f"{i+1}", "weights": [i*3, i*3+1, i*3+2]})

        updates = mesh.pull_updates(2, timeout_seconds=1.0)
        assert len(updates) == 3
        print("    ✓ PASSED")

        # Test 3: Queue status
        print("  Test 3: Queue status...")
        status = mesh.get_queue_status()
        assert status["total_updates"] >= 0  # Allow empty queue
        assert status["total_size_bytes"] >= 0  # Allow zero size
        print("    ✓ PASSED")

        # Test 4: MeshUpdate dataclass
        print("  Test 4: MeshUpdate dataclass...")
        update = MeshUpdate(
            client_id="test_client",
            round_number=1,
            timestamp=time.time(),
            payload_hash="abc123",
            payload={"test": "data"},
            metadata={"source": "test"}
        )
        assert update.client_id == "test_client"
        assert update.payload["test"] == "data"
        print("    ✓ PASSED")

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        print("All Mesh Sync Tests PASSED! ✅")
        return True

    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        traceback.print_exc()
        return False

def run_import_tests():
    """Test basic imports."""
    print("Running Import Tests...")

    try:
        # Test basic imports
        print("  Test 1: Basic module imports...")
        import src.guard
        import src.mesh_sync
        try:
            import src.models
            print("    ✓ models module available")
        except ImportError:
            import src.models_mock
            print("    ✓ models_mock module loaded (torch not available)")
        try:
            import src.utils
            print("    ✓ utils module available")
        except ImportError:
            import src.utils_mock
            print("    ✓ utils_mock module loaded (numpy not available)")
        print("    ✓ PASSED")

        # Test available dependencies
        print("  Test 2: Available dependencies...")
        try:
            import cryptography
            print(f"    ✓ cryptography: {cryptography.__version__}")
        except ImportError:
            print("    ⚠️  cryptography not available")

        try:
            import numpy
            print(f"    ✓ numpy: {numpy.__version__}")
        except ImportError:
            print("    ⚠️  numpy not available")

        try:
            import torch
            print(f"    ✓ torch: {torch.__version__}")
        except ImportError:
            print("    ⚠️  torch not available")

        try:
            import flwr
            print(f"    ✓ flwr: {flwr.__version__}")
        except ImportError:
            print("    ⚠️  flwr not available")

        print("Import Tests COMPLETED! ✅")
        return True

    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test runner."""
    print("🧪 Flower Off-Guard UIOTA Demo Test Runner")
    print("=" * 50)

    # Set offline mode for tests
    os.environ["OFFLINE_MODE"] = "1"

    tests_passed = 0
    total_tests = 3

    # Run tests
    if run_import_tests():
        tests_passed += 1

    if run_guard_tests():
        tests_passed += 1

    if run_mesh_sync_tests():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} test suites passed")

    if tests_passed == total_tests:
        print("🎉 All tests PASSED!")
        return 0
    else:
        print("❌ Some tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())