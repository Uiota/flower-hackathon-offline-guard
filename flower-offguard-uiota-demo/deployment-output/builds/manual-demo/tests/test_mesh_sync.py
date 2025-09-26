"""
Tests for UIOTA mesh synchronization module
"""

import json
import tempfile
import time
from pathlib import Path
import pytest

from src.mesh_sync import MeshTransport, MeshConfig, MeshUpdate


class TestMeshSync:
    """Test cases for mesh synchronization."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = MeshConfig(
            base_latency_ms=0,  # No latency for tests
            jitter_ms=0,
            dropout_rate=0.0,  # No dropouts for tests
            cleanup_old_files=False
        )
        self.mesh = MeshTransport(self.config, self.temp_dir)

    def teardown_method(self):
        """Cleanup after each test."""
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_push_update_success(self):
        """Test successful update push."""
        client_id = "test_client_1"
        round_number = 1
        payload = {"weights": [1, 2, 3], "accuracy": 0.95}

        result = self.mesh.push_update(client_id, round_number, payload)
        assert result is True

        # Check file was created
        update_files = list(self.temp_dir.glob(f"update_*_r{round_number}_*.json"))
        assert len(update_files) == 1

        # Check file content
        with open(update_files[0]) as f:
            data = json.load(f)

        assert data["client_id"] == client_id
        assert data["round_number"] == round_number
        assert data["payload"] == payload

    def test_pull_updates_success(self):
        """Test successful update pull."""
        # Push multiple updates
        payloads = [
            {"client": "1", "weights": [1, 2, 3]},
            {"client": "2", "weights": [4, 5, 6]},
            {"client": "3", "weights": [7, 8, 9]},
        ]

        round_number = 2
        for i, payload in enumerate(payloads):
            self.mesh.push_update(f"client_{i+1}", round_number, payload)

        # Pull updates
        updates = self.mesh.pull_updates(round_number, timeout_seconds=1.0)

        assert len(updates) == 3

        # Check updates content
        client_ids = [update.client_id for update in updates]
        assert "client_1" in client_ids
        assert "client_2" in client_ids
        assert "client_3" in client_ids

        # Check payloads
        for update in updates:
            assert "weights" in update.payload
            assert "client" in update.payload

    def test_pull_updates_with_limit(self):
        """Test pulling updates with max limit."""
        # Push 5 updates
        round_number = 3
        for i in range(5):
            payload = {"client": f"{i}", "data": f"test_{i}"}
            self.mesh.push_update(f"client_{i}", round_number, payload)

        # Pull with limit of 3
        updates = self.mesh.pull_updates(round_number, max_updates=3, timeout_seconds=1.0)

        assert len(updates) == 3

    def test_pull_updates_wrong_round(self):
        """Test pulling updates for non-existent round."""
        # Push updates for round 1
        self.mesh.push_update("client_1", 1, {"data": "test"})

        # Try to pull updates for round 2
        updates = self.mesh.pull_updates(2, timeout_seconds=0.5)

        assert len(updates) == 0

    def test_dropout_simulation(self):
        """Test network dropout simulation."""
        # Configure high dropout rate
        config = MeshConfig(
            base_latency_ms=0,
            dropout_rate=1.0  # 100% dropout
        )
        mesh = MeshTransport(config, self.temp_dir)

        # Try to push update (should be dropped)
        result = mesh.push_update("client_1", 1, {"data": "test"})
        assert result is False

        # Check no file was created
        update_files = list(self.temp_dir.glob("update_*.json"))
        assert len(update_files) == 0

    def test_latency_simulation(self):
        """Test latency simulation."""
        # Configure latency
        config = MeshConfig(
            base_latency_ms=100,
            jitter_ms=50,
            dropout_rate=0.0
        )
        mesh = MeshTransport(config, self.temp_dir)

        # Measure push time
        start_time = time.time()
        result = mesh.push_update("client_1", 1, {"data": "test"})
        elapsed_time = time.time() - start_time

        assert result is True
        assert elapsed_time >= 0.1  # At least base latency
        assert elapsed_time <= 0.2  # Not more than base + max jitter

    def test_queue_status(self):
        """Test queue status reporting."""
        # Push some updates
        for i in range(3):
            self.mesh.push_update(f"client_{i}", 1, {"data": f"test_{i}"})

        for i in range(2):
            self.mesh.push_update(f"client_{i}", 2, {"data": f"test_{i}"})

        # Get status
        status = self.mesh.get_queue_status()

        assert status["total_updates"] == 5
        assert status["rounds_pending"][1] == 3
        assert status["rounds_pending"][2] == 2
        assert status["total_size_bytes"] > 0

    def test_clear_queue(self):
        """Test queue clearing."""
        # Push updates for different rounds
        self.mesh.push_update("client_1", 1, {"data": "round1"})
        self.mesh.push_update("client_2", 2, {"data": "round2"})
        self.mesh.push_update("client_3", 2, {"data": "round2"})

        # Clear specific round
        cleared = self.mesh.clear_queue(round_number=2)
        assert cleared == 2

        # Check remaining updates
        status = self.mesh.get_queue_status()
        assert status["total_updates"] == 1
        assert 1 in status["rounds_pending"]
        assert 2 not in status["rounds_pending"]

        # Clear all
        cleared_all = self.mesh.clear_queue()
        assert cleared_all == 1

        status_after = self.mesh.get_queue_status()
        assert status_after["total_updates"] == 0

    def test_mesh_update_dataclass(self):
        """Test MeshUpdate dataclass."""
        update = MeshUpdate(
            client_id="test_client",
            round_number=1,
            timestamp=time.time(),
            payload_hash="abc123",
            payload={"test": "data"},
            metadata={"source": "test"}
        )

        assert update.client_id == "test_client"
        assert update.round_number == 1
        assert update.payload["test"] == "data"
        assert update.metadata["source"] == "test"

    def test_corrupted_file_handling(self):
        """Test handling of corrupted update files."""
        # Create corrupted file manually
        corrupted_file = self.temp_dir / "update_client1_r1_123456789.json"
        with open(corrupted_file, 'w') as f:
            f.write("invalid json content {")

        # Try to pull updates (should handle corruption gracefully)
        updates = self.mesh.pull_updates(1, timeout_seconds=0.5)

        # Should return empty list without crashing
        assert len(updates) == 0

        # Corrupted file should be moved to corrupted directory
        assert not corrupted_file.exists()
        corrupted_dir = self.temp_dir / "corrupted"
        if corrupted_dir.exists():
            assert len(list(corrupted_dir.glob("*.json"))) == 1