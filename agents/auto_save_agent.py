#!/usr/bin/env python3
"""
Auto-Save Agent for UIOTA Offline Guard

Automatically saves progress, configurations, and state changes
across all other agents in the system. Ensures data persistence
and recovery capabilities.
"""

import asyncio
import json
import os
import shutil
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from threading import Thread, Event
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class SaveState:
    """Represents a saveable state snapshot."""
    timestamp: str
    agent_id: str
    state_type: str
    data: Dict[str, Any]
    checksum: str

class AutoSaveAgent:
    """
    Continuously monitors and saves the state of all agents in the system.
    Provides automatic backup, recovery, and state synchronization.
    """

    def __init__(self, save_interval: int = 30, max_backups: int = 100):
        """
        Initialize the auto-save agent.

        Args:
            save_interval: Seconds between automatic saves
            max_backups: Maximum number of backup files to retain
        """
        self.save_interval = save_interval
        self.max_backups = max_backups
        self.save_dir = Path.home() / ".uiota" / "auto_saves"
        self.state_file = self.save_dir / "current_state.json"
        self.backup_dir = self.save_dir / "backups"

        # Internal state
        self._stop_event = Event()
        self._save_thread: Optional[Thread] = None
        self._agents_state: Dict[str, SaveState] = {}
        self._watched_paths: List[Path] = []

        # Initialize directories
        self._init_directories()

        # Load existing state
        self._load_state()

        logger.info(f"AutoSaveAgent initialized - save interval: {save_interval}s")

    def _init_directories(self) -> None:
        """Create necessary directories for saving."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Create config directory if it doesn't exist
        config_dir = Path.home() / ".uiota" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("Auto-save directories initialized")

    def _load_state(self) -> None:
        """Load existing state from disk."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)

                # Convert loaded data back to SaveState objects
                for agent_id, state_data in data.get('agents', {}).items():
                    self._agents_state[agent_id] = SaveState(**state_data)

                logger.info(f"Loaded state for {len(self._agents_state)} agents")
            else:
                logger.info("No existing state found, starting fresh")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            self._agents_state = {}

    def _save_state(self) -> None:
        """Save current state to disk."""
        try:
            # Prepare data for serialization
            save_data = {
                'timestamp': datetime.now().isoformat(),
                'agents': {
                    agent_id: asdict(state)
                    for agent_id, state in self._agents_state.items()
                },
                'watched_paths': [str(p) for p in self._watched_paths]
            }

            # Create backup before saving
            if self.state_file.exists():
                self._create_backup()

            # Write current state
            with open(self.state_file, 'w') as f:
                json.dump(save_data, f, indent=2)

            logger.debug(f"State saved for {len(self._agents_state)} agents")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _create_backup(self) -> None:
        """Create a timestamped backup of the current state."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"state_backup_{timestamp}.json"

            # Copy current state to backup
            shutil.copy2(self.state_file, backup_file)

            # Clean up old backups
            self._cleanup_old_backups()

            logger.debug(f"Backup created: {backup_file}")

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")

    def _cleanup_old_backups(self) -> None:
        """Remove old backup files to maintain max_backups limit."""
        try:
            backup_files = sorted(
                self.backup_dir.glob("state_backup_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            # Remove excess backups
            for backup_file in backup_files[self.max_backups:]:
                backup_file.unlink()
                logger.debug(f"Removed old backup: {backup_file}")

        except Exception as e:
            logger.error(f"Failed to cleanup backups: {e}")

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def register_agent(self, agent_id: str, initial_state: Dict[str, Any] = None) -> None:
        """
        Register an agent for auto-saving.

        Args:
            agent_id: Unique identifier for the agent
            initial_state: Initial state data to save
        """
        if initial_state is None:
            initial_state = {}

        state = SaveState(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            state_type="registration",
            data=initial_state,
            checksum=self._calculate_checksum(initial_state)
        )

        self._agents_state[agent_id] = state
        logger.info(f"Registered agent for auto-save: {agent_id}")

    def update_agent_state(self, agent_id: str, state_data: Dict[str, Any],
                          state_type: str = "update") -> None:
        """
        Update the state of a registered agent.

        Args:
            agent_id: Agent identifier
            state_data: New state data
            state_type: Type of state update
        """
        state = SaveState(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            state_type=state_type,
            data=state_data,
            checksum=self._calculate_checksum(state_data)
        )

        self._agents_state[agent_id] = state
        logger.debug(f"Updated state for agent: {agent_id}")

    def add_watched_path(self, path: Path) -> None:
        """
        Add a file or directory path to monitor for changes.

        Args:
            path: Path to monitor
        """
        if path not in self._watched_paths:
            self._watched_paths.append(path)
            logger.info(f"Added watched path: {path}")

    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the current state of an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent state data or None if not found
        """
        if agent_id in self._agents_state:
            return self._agents_state[agent_id].data
        return None

    def list_registered_agents(self) -> List[str]:
        """Get list of all registered agent IDs."""
        return list(self._agents_state.keys())

    def _auto_save_loop(self) -> None:
        """Main auto-save loop running in background thread."""
        logger.info("Auto-save loop started")

        while not self._stop_event.is_set():
            try:
                # Save current state
                self._save_state()

                # Check for file changes in watched paths
                self._check_watched_paths()

                # Wait for next save interval
                self._stop_event.wait(self.save_interval)

            except Exception as e:
                logger.error(f"Error in auto-save loop: {e}")
                time.sleep(5)  # Brief pause before retrying

        logger.info("Auto-save loop stopped")

    def _check_watched_paths(self) -> None:
        """Check watched paths for changes and save if needed."""
        for path in self._watched_paths:
            try:
                if path.exists():
                    # Simple modification time check
                    mtime = path.stat().st_mtime
                    agent_id = f"file_watcher_{path.name}"

                    # Check if this file has changed
                    current_state = self.get_agent_state(agent_id)
                    if current_state is None or current_state.get('mtime', 0) != mtime:
                        self.update_agent_state(
                            agent_id,
                            {
                                'path': str(path),
                                'mtime': mtime,
                                'last_check': datetime.now().isoformat()
                            },
                            'file_change'
                        )
                        logger.debug(f"Detected change in watched file: {path}")

            except Exception as e:
                logger.error(f"Error checking watched path {path}: {e}")

    def start(self) -> None:
        """Start the auto-save agent."""
        if self._save_thread is None or not self._save_thread.is_alive():
            self._stop_event.clear()
            self._save_thread = Thread(target=self._auto_save_loop, daemon=True)
            self._save_thread.start()
            logger.info("AutoSaveAgent started")
        else:
            logger.warning("AutoSaveAgent is already running")

    def stop(self) -> None:
        """Stop the auto-save agent."""
        if self._save_thread and self._save_thread.is_alive():
            self._stop_event.set()
            self._save_thread.join(timeout=10)
            logger.info("AutoSaveAgent stopped")

        # Final save
        self._save_state()

    def force_save(self) -> None:
        """Force an immediate save of all agent states."""
        self._save_state()
        logger.info("Forced save completed")

    def restore_from_backup(self, backup_file: Optional[Path] = None) -> bool:
        """
        Restore state from a backup file.

        Args:
            backup_file: Specific backup file to restore from.
                        If None, uses the most recent backup.

        Returns:
            True if restoration was successful
        """
        try:
            if backup_file is None:
                # Find most recent backup
                backup_files = sorted(
                    self.backup_dir.glob("state_backup_*.json"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                if not backup_files:
                    logger.error("No backup files found")
                    return False
                backup_file = backup_files[0]

            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return False

            # Load backup data
            with open(backup_file, 'r') as f:
                data = json.load(f)

            # Restore agent states
            self._agents_state = {}
            for agent_id, state_data in data.get('agents', {}).items():
                self._agents_state[agent_id] = SaveState(**state_data)

            # Restore watched paths
            self._watched_paths = [Path(p) for p in data.get('watched_paths', [])]

            # Save restored state as current
            self._save_state()

            logger.info(f"Successfully restored from backup: {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False

def create_auto_save_agent(save_interval: int = 30) -> AutoSaveAgent:
    """
    Factory function to create and configure an AutoSaveAgent.

    Args:
        save_interval: Seconds between automatic saves

    Returns:
        Configured AutoSaveAgent instance
    """
    agent = AutoSaveAgent(save_interval=save_interval)

    # Add common paths to watch
    project_root = Path.cwd()
    agent.add_watched_path(project_root / ".guardian" / "config.yaml")
    agent.add_watched_path(project_root / "flower-offguard-uiota-demo")

    return agent

if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    agent = create_auto_save_agent(save_interval=10)

    # Register some demo agents
    agent.register_agent("demo_agent_1", {"status": "active", "count": 0})
    agent.register_agent("demo_agent_2", {"mode": "learning", "progress": 0.5})

    # Start auto-saving
    agent.start()

    try:
        # Simulate agent activity
        for i in range(5):
            time.sleep(3)
            agent.update_agent_state("demo_agent_1", {"status": "active", "count": i})
            agent.update_agent_state("demo_agent_2", {"mode": "learning", "progress": 0.5 + i * 0.1})
            print(f"Updated agents at iteration {i}")

        print("Auto-save demo completed successfully")

    except KeyboardInterrupt:
        print("Demo interrupted")
    finally:
        agent.stop()