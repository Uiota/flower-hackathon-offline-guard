"""
UIOTA Mesh Synchronization Mock

Simulates offline/delayed mesh network transport using local file queue system.
This module provides the abstraction layer for swapping in real transport mechanisms
like LoRa, QR codes, or other offline communication methods.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Default queue directory
MESH_QUEUE_DIR = Path("./mesh_queue")


@dataclass
class MeshUpdate:
    """Represents a mesh network update packet."""
    client_id: str
    round_number: int
    timestamp: float
    payload_hash: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class MeshConfig:
    """Configuration for mesh network simulation."""
    base_latency_ms: int = 50
    jitter_ms: int = 25
    dropout_rate: float = 0.1
    max_queue_size: int = 1000
    cleanup_old_files: bool = True
    file_retention_hours: int = 24


class MeshTransport:
    """
    Mock UIOTA mesh transport layer.

    This class simulates the behavior of an offline mesh network by using
    a local file-based queue system. In production, this would be replaced
    with actual LoRa, satellite, or other offline communication protocols.
    """

    def __init__(self, config: MeshConfig = None, queue_dir: Path = None):
        self.config = config or MeshConfig()
        self.queue_dir = queue_dir or MESH_QUEUE_DIR

        # Initialize queue directory
        self.queue_dir.mkdir(exist_ok=True)
        logger.info(f"Mesh transport initialized - Queue: {self.queue_dir}")

        # Cleanup old files if enabled
        if self.config.cleanup_old_files:
            self._cleanup_old_files()

    def push_update(
        self,
        client_id: str,
        round_number: int,
        payload: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Push an update to the mesh network.

        Args:
            client_id: Identifier of the sending client
            round_number: FL round number
            payload: Update data (model weights, etc.)
            metadata: Additional metadata

        Returns:
            True if successful, False if dropped due to network conditions
        """
        # Simulate network dropout
        if random.random() < self.config.dropout_rate:
            logger.debug(f"Update from client {client_id} dropped due to network conditions")
            return False

        # Simulate network latency
        latency = self.config.base_latency_ms + random.randint(0, self.config.jitter_ms)
        if latency > 0:
            time.sleep(latency / 1000.0)  # Convert ms to seconds

        try:
            # Create mesh update packet
            update = MeshUpdate(
                client_id=client_id,
                round_number=round_number,
                timestamp=time.time(),
                payload_hash=self._hash_payload(payload),
                payload=payload,
                metadata=metadata or {}
            )

            # Generate unique filename
            filename = f"update_c{client_id}_r{round_number}_{int(update.timestamp * 1000)}.json"
            filepath = self.queue_dir / filename

            # Write to queue
            with open(filepath, 'w') as f:
                json.dump(asdict(update), f, indent=2)

            logger.debug(f"Update pushed: {filename} (latency: {latency}ms)")
            return True

        except Exception as e:
            logger.error(f"Failed to push update from client {client_id}: {e}")
            return False

    def pull_updates(
        self,
        round_number: int,
        max_updates: Optional[int] = None,
        timeout_seconds: float = 5.0
    ) -> List[MeshUpdate]:
        """
        Pull available updates for a specific round from the mesh network.

        Args:
            round_number: FL round to retrieve updates for
            max_updates: Maximum number of updates to retrieve
            timeout_seconds: How long to wait for updates

        Returns:
            List of MeshUpdate objects
        """
        start_time = time.time()
        collected_updates = []
        processed_files = set()

        logger.debug(f"Pulling updates for round {round_number} (timeout: {timeout_seconds}s)")

        while time.time() - start_time < timeout_seconds:
            try:
                # Find matching update files
                pattern = f"update_*_r{round_number}_*.json"
                matching_files = list(self.queue_dir.glob(pattern))

                # Process new files
                for filepath in matching_files:
                    if filepath.name in processed_files:
                        continue

                    try:
                        with open(filepath, 'r') as f:
                            update_data = json.load(f)

                        # Convert back to MeshUpdate object
                        update = MeshUpdate(**update_data)
                        collected_updates.append(update)
                        processed_files.add(filepath.name)

                        # Remove processed file
                        filepath.unlink(missing_ok=True)
                        logger.debug(f"Processed update: {filepath.name}")

                        # Check if we've collected enough
                        if max_updates and len(collected_updates) >= max_updates:
                            logger.debug(f"Collected maximum updates ({max_updates})")
                            break

                    except Exception as e:
                        logger.error(f"Failed to process update file {filepath}: {e}")
                        # Move corrupted file instead of deleting
                        try:
                            corrupted_path = self.queue_dir / "corrupted" / filepath.name
                            corrupted_path.parent.mkdir(exist_ok=True)
                            filepath.rename(corrupted_path)
                        except Exception:
                            filepath.unlink(missing_ok=True)

                # Break if we have enough updates
                if max_updates and len(collected_updates) >= max_updates:
                    break

                # Short sleep to avoid busy waiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error during update pull: {e}")
                break

        elapsed_time = time.time() - start_time
        logger.info(f"Pulled {len(collected_updates)} updates for round {round_number} "
                   f"in {elapsed_time:.2f}s")

        return collected_updates

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics."""
        try:
            all_files = list(self.queue_dir.glob("update_*.json"))

            # Group by round
            round_counts = {}
            total_size = 0
            oldest_timestamp = None
            newest_timestamp = None

            for filepath in all_files:
                try:
                    # Extract round from filename
                    parts = filepath.stem.split('_')
                    round_num = int(parts[2][1:])  # Remove 'r' prefix
                    timestamp = int(parts[3]) / 1000.0  # Convert ms to seconds

                    round_counts[round_num] = round_counts.get(round_num, 0) + 1
                    total_size += filepath.stat().st_size

                    if oldest_timestamp is None or timestamp < oldest_timestamp:
                        oldest_timestamp = timestamp
                    if newest_timestamp is None or timestamp > newest_timestamp:
                        newest_timestamp = timestamp

                except (ValueError, IndexError):
                    continue

            return {
                "total_updates": len(all_files),
                "total_size_bytes": total_size,
                "rounds_pending": round_counts,
                "oldest_update": oldest_timestamp,
                "newest_update": newest_timestamp,
                "queue_directory": str(self.queue_dir),
                "config": asdict(self.config)
            }

        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {"error": str(e)}

    def clear_queue(self, round_number: Optional[int] = None) -> int:
        """
        Clear updates from the queue.

        Args:
            round_number: If specified, only clear updates for this round

        Returns:
            Number of files cleared
        """
        try:
            if round_number is not None:
                pattern = f"update_*_r{round_number}_*.json"
            else:
                pattern = "update_*.json"

            files_to_clear = list(self.queue_dir.glob(pattern))
            cleared_count = 0

            for filepath in files_to_clear:
                try:
                    filepath.unlink()
                    cleared_count += 1
                except Exception as e:
                    logger.error(f"Failed to clear file {filepath}: {e}")

            if round_number:
                logger.info(f"Cleared {cleared_count} updates for round {round_number}")
            else:
                logger.info(f"Cleared {cleared_count} updates from entire queue")

            return cleared_count

        except Exception as e:
            logger.error(f"Failed to clear queue: {e}")
            return 0

    def _hash_payload(self, payload: Dict[str, Any]) -> str:
        """Generate hash of payload for integrity checking."""
        import hashlib
        payload_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload_str.encode()).hexdigest()[:16]  # Short hash

    def _cleanup_old_files(self) -> None:
        """Remove old update files based on retention policy."""
        try:
            current_time = time.time()
            retention_seconds = self.config.file_retention_hours * 3600

            old_files = []
            for filepath in self.queue_dir.glob("update_*.json"):
                try:
                    file_age = current_time - filepath.stat().st_mtime
                    if file_age > retention_seconds:
                        old_files.append(filepath)
                except Exception:
                    continue

            # Remove old files
            cleaned_count = 0
            for filepath in old_files:
                try:
                    filepath.unlink()
                    cleaned_count += 1
                except Exception as e:
                    logger.error(f"Failed to cleanup old file {filepath}: {e}")

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old update files")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Global mesh transport instance (for convenience)
_global_mesh_transport = None


def get_mesh_transport(config: MeshConfig = None) -> MeshTransport:
    """Get global mesh transport instance."""
    global _global_mesh_transport
    if _global_mesh_transport is None:
        _global_mesh_transport = MeshTransport(config)
    return _global_mesh_transport


# Convenience functions for backward compatibility
def push_update(client_id: str, round_number: int, payload: Dict[str, Any], **kwargs) -> bool:
    """Push update using global mesh transport."""
    return get_mesh_transport().push_update(client_id, round_number, payload, **kwargs)


def pull_updates(round_number: int, max_items: Optional[int] = None, timeout_s: float = 5.0) -> List[Dict[str, Any]]:
    """Pull updates using global mesh transport (returns dict format for compatibility)."""
    updates = get_mesh_transport().pull_updates(round_number, max_items, timeout_s)
    return [asdict(update) for update in updates]