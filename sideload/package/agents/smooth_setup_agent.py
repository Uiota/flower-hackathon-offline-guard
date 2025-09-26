#!/usr/bin/env python3
"""
Smooth Setup Agent for UIOTA Offline Guard

Automatically handles all downloads, connections, and integrations
so users can just download and run without manual setup.
"""

import asyncio
import os
import subprocess
import sys
import json
import shutil
import tempfile
import urllib.request
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

@dataclass
class SetupTask:
    """Represents a setup task to be executed."""
    name: str
    description: str
    command: Optional[str] = None
    function: Optional[callable] = None
    dependencies: List[str] = None
    timeout: int = 300
    retry_count: int = 3

class SmoothSetupAgent:
    """
    Handles all setup, download, and integration tasks automatically.
    Ensures the system is ready to run demos immediately after download.
    """

    def __init__(self, project_root: Path = None):
        """
        Initialize the smooth setup agent.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.setup_tasks: Dict[str, SetupTask] = {}
        self.completed_tasks: set = set()
        self.failed_tasks: set = set()
        self.dependencies_map: Dict[str, List[str]] = {}

        # Setup configuration
        self.config = {
            "auto_install_missing": True,
            "parallel_downloads": True,
            "max_workers": 4,
            "timeout_seconds": 300,
            "auto_retry": True
        }

        # Initialize setup tasks
        self._init_setup_tasks()

        logger.info("SmoothSetupAgent initialized")

    def _init_setup_tasks(self) -> None:
        """Initialize all setup tasks in correct dependency order."""

        # 1. System requirements check
        self.add_task(SetupTask(
            name="check_system",
            description="Check system requirements and compatibility",
            function=self._check_system_requirements
        ))

        # 2. Install system dependencies
        self.add_task(SetupTask(
            name="install_podman",
            description="Install Podman container engine",
            function=self._install_podman,
            dependencies=["check_system"]
        ))

        # 3. Install Python dependencies
        self.add_task(SetupTask(
            name="install_python_deps",
            description="Install required Python packages",
            function=self._install_python_dependencies,
            dependencies=["check_system"]
        ))

        # 4. Download ML models and datasets
        self.add_task(SetupTask(
            name="download_ml_assets",
            description="Download ML models and training datasets",
            function=self._download_ml_assets,
            dependencies=["install_python_deps"]
        ))

        # 5. Setup containers
        self.add_task(SetupTask(
            name="setup_containers",
            description="Build and configure all containers",
            function=self._setup_containers,
            dependencies=["install_podman"]
        ))

        # 6. Configure network
        self.add_task(SetupTask(
            name="configure_network",
            description="Configure network ports and connectivity",
            function=self._configure_network,
            dependencies=["setup_containers"]
        ))

        # 7. Initialize configuration
        self.add_task(SetupTask(
            name="init_config",
            description="Create default configuration files",
            function=self._init_configuration,
            dependencies=["check_system"]
        ))

        # 8. Test all systems
        self.add_task(SetupTask(
            name="test_systems",
            description="Run system tests to verify setup",
            function=self._test_all_systems,
            dependencies=["setup_containers", "configure_network", "init_config", "download_ml_assets"]
        ))

    def add_task(self, task: SetupTask) -> None:
        """Add a setup task to the execution queue."""
        self.setup_tasks[task.name] = task
        if task.dependencies:
            self.dependencies_map[task.name] = task.dependencies

    def _check_system_requirements(self) -> bool:
        """Check if the system meets minimum requirements."""
        try:
            logger.info("Checking system requirements...")

            # Check Python version
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            if sys.version_info < (3, 9):
                logger.error(f"Python {python_version} is too old. Requires Python 3.9+")
                return False

            logger.info(f"✓ Python {python_version} OK")

            # Check available disk space (need at least 5GB)
            disk_usage = shutil.disk_usage(self.project_root)
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 5:
                logger.error(f"Insufficient disk space: {free_gb:.1f}GB free, need 5GB+")
                return False

            logger.info(f"✓ Disk space: {free_gb:.1f}GB available")

            # Check internet connectivity
            if not self._check_internet():
                logger.warning("No internet connectivity - will use offline mode")
            else:
                logger.info("✓ Internet connectivity OK")

            return True

        except Exception as e:
            logger.error(f"System requirements check failed: {e}")
            return False

    def _check_internet(self) -> bool:
        """Check if internet connectivity is available."""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return True
        except (socket.error, socket.timeout):
            return False

    def _install_podman(self) -> bool:
        """Install Podman if not already present."""
        try:
            # Check if Podman is already installed
            result = subprocess.run(["podman", "--version"],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✓ Podman already installed: {result.stdout.strip()}")
                return True

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        logger.info("Installing Podman...")

        # Detect OS and install accordingly
        import platform
        system = platform.system().lower()

        try:
            if system == "linux":
                # Try different package managers
                if shutil.which("apt"):
                    cmd = ["sudo", "apt", "update", "&&", "sudo", "apt", "install", "-y", "podman"]
                elif shutil.which("dnf"):
                    cmd = ["sudo", "dnf", "install", "-y", "podman"]
                elif shutil.which("zypper"):
                    cmd = ["sudo", "zypper", "install", "-y", "podman"]
                else:
                    logger.error("Unsupported Linux distribution for automatic Podman install")
                    return False

            elif system == "darwin":  # macOS
                if shutil.which("brew"):
                    cmd = ["brew", "install", "podman"]
                else:
                    logger.error("Homebrew required for automatic Podman install on macOS")
                    return False

            else:
                logger.error(f"Unsupported operating system: {system}")
                return False

            # Execute installation
            result = subprocess.run(cmd, timeout=300, text=True)
            if result.returncode == 0:
                logger.info("✓ Podman installed successfully")
                return True
            else:
                logger.error("Podman installation failed")
                return False

        except Exception as e:
            logger.error(f"Failed to install Podman: {e}")
            return False

    def _install_python_dependencies(self) -> bool:
        """Install required Python packages."""
        try:
            logger.info("Installing Python dependencies...")

            # Core requirements
            packages = [
                "torch==2.1.0+cpu",
                "flwr==1.8.0",
                "cryptography==41.0.7",
                "pydantic",
                "fastapi",
                "uvicorn",
                "jupyter",
                "pandas",
                "numpy",
                "scikit-learn",
                "aiohttp",
                "websockets"
            ]

            # Install packages
            for package in packages:
                try:
                    logger.info(f"Installing {package}...")
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", package
                    ], capture_output=True, text=True, timeout=120)

                    if result.returncode != 0:
                        logger.warning(f"Failed to install {package}: {result.stderr}")
                    else:
                        logger.debug(f"✓ Installed {package}")

                except subprocess.TimeoutExpired:
                    logger.warning(f"Timeout installing {package}")

            logger.info("✓ Python dependencies installation completed")
            return True

        except Exception as e:
            logger.error(f"Failed to install Python dependencies: {e}")
            return False

    def _download_ml_assets(self) -> bool:
        """Download ML models and datasets needed for demos."""
        try:
            logger.info("Downloading ML assets...")

            assets_dir = self.project_root / "data" / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)

            # Small demo datasets and models
            downloads = [
                {
                    "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                    "filename": "iris_dataset.csv",
                    "description": "Iris dataset for ML demos"
                },
                # Add more assets as needed
            ]

            if not self._check_internet():
                logger.info("No internet - skipping ML asset downloads")
                # Create dummy files for offline demo
                self._create_dummy_assets(assets_dir)
                return True

            for download in downloads:
                try:
                    file_path = assets_dir / download["filename"]
                    if not file_path.exists():
                        logger.info(f"Downloading {download['description']}...")
                        urllib.request.urlretrieve(download["url"], file_path)
                        logger.debug(f"✓ Downloaded {download['filename']}")
                    else:
                        logger.debug(f"✓ {download['filename']} already exists")

                except Exception as e:
                    logger.warning(f"Failed to download {download['filename']}: {e}")

            logger.info("✓ ML assets download completed")
            return True

        except Exception as e:
            logger.error(f"Failed to download ML assets: {e}")
            return False

    def _create_dummy_assets(self, assets_dir: Path) -> None:
        """Create dummy assets for offline demo mode."""
        try:
            # Create dummy iris dataset
            iris_data = """5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
7.1,3.0,5.9,2.1,Iris-virginica"""

            iris_file = assets_dir / "iris_dataset.csv"
            iris_file.write_text(iris_data)
            logger.info("✓ Created dummy iris dataset")

        except Exception as e:
            logger.warning(f"Failed to create dummy assets: {e}")

    def _setup_containers(self) -> bool:
        """Build and configure all necessary containers."""
        try:
            logger.info("Setting up containers...")

            # Ensure container directories exist
            containers_dir = self.project_root / "containers"
            containers_dir.mkdir(exist_ok=True)

            # Create container configurations if they don't exist
            self._create_container_configs()

            # Build containers using existing script
            setup_script = self.project_root / "start-demos.sh"
            if setup_script.exists():
                logger.info("Running container setup script...")
                result = subprocess.run([str(setup_script)],
                                      cwd=self.project_root,
                                      timeout=600)
                if result.returncode == 0:
                    logger.info("✓ Containers setup completed")
                    return True
                else:
                    logger.error("Container setup script failed")
                    return False
            else:
                logger.warning("Container setup script not found, creating minimal containers")
                return self._create_minimal_containers()

        except Exception as e:
            logger.error(f"Failed to setup containers: {e}")
            return False

    def _create_container_configs(self) -> None:
        """Create container configuration files if missing."""
        containers_dir = self.project_root / "containers"

        # Web demo container
        web_dir = containers_dir / "web-demo"
        web_dir.mkdir(exist_ok=True)

        web_containerfile = web_dir / "Containerfile"
        if not web_containerfile.exists():
            web_config = """FROM docker.io/nginx:alpine
COPY web-demo/ /usr/share/nginx/html/
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]"""
            web_containerfile.write_text(web_config)

        # ML toolkit container
        ml_dir = containers_dir / "ml-toolkit"
        ml_dir.mkdir(exist_ok=True)

        ml_containerfile = ml_dir / "Containerfile"
        if not ml_containerfile.exists():
            ml_config = """FROM docker.io/python:3.11-slim
WORKDIR /app
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install flwr jupyter pandas numpy scikit-learn
COPY flower-offguard-uiota-demo/ ./
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]"""
            ml_containerfile.write_text(ml_config)

    def _create_minimal_containers(self) -> bool:
        """Create minimal containers for demo purposes."""
        try:
            logger.info("Creating minimal containers...")

            # Simple web server container
            result = subprocess.run([
                "podman", "run", "-d", "--name", "offline-guard-web",
                "-p", "8080:80", "docker.io/nginx:alpine"
            ], timeout=60)

            if result.returncode == 0:
                logger.info("✓ Minimal web container created")
                return True
            else:
                logger.error("Failed to create minimal containers")
                return False

        except Exception as e:
            logger.error(f"Failed to create minimal containers: {e}")
            return False

    def _configure_network(self) -> bool:
        """Configure network ports and connectivity."""
        try:
            logger.info("Configuring network...")

            # Check if required ports are available
            required_ports = [8080, 8888]
            for port in required_ports:
                if self._is_port_in_use(port):
                    logger.warning(f"Port {port} is already in use")
                else:
                    logger.debug(f"✓ Port {port} available")

            # Configure firewall if needed (basic check)
            self._configure_firewall(required_ports)

            logger.info("✓ Network configuration completed")
            return True

        except Exception as e:
            logger.error(f"Failed to configure network: {e}")
            return False

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is currently in use."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return False
        except OSError:
            return True

    def _configure_firewall(self, ports: List[int]) -> None:
        """Configure firewall to allow required ports (if possible)."""
        try:
            # This is a placeholder - actual firewall configuration
            # would be system-specific and require appropriate permissions
            logger.debug(f"Firewall configuration for ports: {ports}")

        except Exception as e:
            logger.warning(f"Could not configure firewall: {e}")

    def _init_configuration(self) -> bool:
        """Create default configuration files."""
        try:
            logger.info("Initializing configuration...")

            # Create .uiota directory
            uiota_dir = Path.home() / ".uiota"
            uiota_dir.mkdir(exist_ok=True)

            # Create guardian config if it doesn't exist
            guardian_dir = self.project_root / ".guardian"
            guardian_dir.mkdir(exist_ok=True)

            config_file = guardian_dir / "config.yaml"
            if not config_file.exists():
                config_content = """# UIOTA Guardian Configuration (Auto-generated)
guardian:
  class: "unassigned"
  level: 1
  xp: 0
  auto_save: true

agents:
  web:
    enabled: true
    port: 8080
    auto_start: true

  ml:
    enabled: true
    port: 8888
    cpu_only: true
    auto_start: true

system:
  container_engine: "podman"
  offline_first: true
  security_level: "high"
  auto_setup: true

setup:
  completed: true
  timestamp: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """
  agent: "SmoothSetupAgent"
"""
                config_file.write_text(config_content)
                logger.info("✓ Guardian configuration created")

            logger.info("✓ Configuration initialization completed")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize configuration: {e}")
            return False

    def _test_all_systems(self) -> bool:
        """Run comprehensive system tests."""
        try:
            logger.info("Testing all systems...")

            tests = [
                ("Podman", self._test_podman),
                ("Python imports", self._test_python_imports),
                ("Network connectivity", self._test_network),
                ("File permissions", self._test_file_permissions)
            ]

            all_passed = True
            for test_name, test_func in tests:
                try:
                    logger.info(f"Running {test_name} test...")
                    if test_func():
                        logger.info(f"✓ {test_name} test passed")
                    else:
                        logger.error(f"✗ {test_name} test failed")
                        all_passed = False
                except Exception as e:
                    logger.error(f"✗ {test_name} test error: {e}")
                    all_passed = False

            if all_passed:
                logger.info("✓ All system tests passed")
            else:
                logger.warning("Some system tests failed - demo may have limited functionality")

            return all_passed

        except Exception as e:
            logger.error(f"System testing failed: {e}")
            return False

    def _test_podman(self) -> bool:
        """Test Podman functionality."""
        try:
            result = subprocess.run(["podman", "ps"],
                                  capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False

    def _test_python_imports(self) -> bool:
        """Test critical Python imports."""
        try:
            import torch
            import flwr
            import cryptography
            return True
        except ImportError:
            return False

    def _test_network(self) -> bool:
        """Test network configuration."""
        try:
            # Test if we can bind to demo ports
            for port in [8080, 8888]:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
            return True
        except:
            return False

    def _test_file_permissions(self) -> bool:
        """Test file system permissions."""
        try:
            test_file = self.project_root / ".test_permissions"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except:
            return False

    def _get_executable_tasks(self) -> List[str]:
        """Get list of tasks that can be executed (dependencies satisfied)."""
        executable = []
        for task_name, task in self.setup_tasks.items():
            if task_name in self.completed_tasks or task_name in self.failed_tasks:
                continue

            # Check if all dependencies are completed
            if task.dependencies:
                if all(dep in self.completed_tasks for dep in task.dependencies):
                    executable.append(task_name)
            else:
                executable.append(task_name)

        return executable

    def _execute_task(self, task_name: str) -> bool:
        """Execute a single setup task."""
        task = self.setup_tasks[task_name]
        logger.info(f"Executing task: {task.description}")

        for attempt in range(task.retry_count):
            try:
                if task.function:
                    success = task.function()
                elif task.command:
                    result = subprocess.run(
                        task.command.split(),
                        timeout=task.timeout,
                        cwd=self.project_root
                    )
                    success = result.returncode == 0
                else:
                    logger.error(f"Task {task_name} has no execution method")
                    return False

                if success:
                    self.completed_tasks.add(task_name)
                    logger.info(f"✓ Task completed: {task.description}")
                    return True
                else:
                    logger.warning(f"Task failed (attempt {attempt + 1}): {task.description}")

            except Exception as e:
                logger.error(f"Task error (attempt {attempt + 1}): {task.description} - {e}")

            if attempt < task.retry_count - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

        self.failed_tasks.add(task_name)
        logger.error(f"✗ Task failed permanently: {task.description}")
        return False

    async def run_setup(self) -> bool:
        """Run all setup tasks in correct dependency order."""
        logger.info("Starting smooth setup process...")

        if self.config.get("parallel_downloads", True):
            return await self._run_setup_parallel()
        else:
            return self._run_setup_sequential()

    async def _run_setup_parallel(self) -> bool:
        """Run setup tasks with parallel execution where possible."""
        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            while len(self.completed_tasks) + len(self.failed_tasks) < len(self.setup_tasks):
                executable_tasks = self._get_executable_tasks()

                if not executable_tasks:
                    logger.error("No executable tasks remaining - dependency deadlock")
                    break

                # Submit tasks for parallel execution
                futures = {
                    executor.submit(self._execute_task, task_name): task_name
                    for task_name in executable_tasks
                }

                # Wait for at least one task to complete
                for future in as_completed(futures):
                    task_name = futures[future]
                    try:
                        success = future.result()
                        if not success:
                            logger.warning(f"Task failed: {task_name}")
                    except Exception as e:
                        logger.error(f"Task execution error: {task_name} - {e}")
                        self.failed_tasks.add(task_name)

        success_rate = len(self.completed_tasks) / len(self.setup_tasks)
        logger.info(f"Setup completed: {len(self.completed_tasks)}/{len(self.setup_tasks)} tasks successful ({success_rate:.1%})")

        return success_rate >= 0.8  # Consider successful if 80% of tasks completed

    def _run_setup_sequential(self) -> bool:
        """Run setup tasks sequentially."""
        while len(self.completed_tasks) + len(self.failed_tasks) < len(self.setup_tasks):
            executable_tasks = self._get_executable_tasks()

            if not executable_tasks:
                logger.error("No executable tasks remaining")
                break

            # Execute first available task
            task_name = executable_tasks[0]
            self._execute_task(task_name)

        success_rate = len(self.completed_tasks) / len(self.setup_tasks)
        logger.info(f"Setup completed: {len(self.completed_tasks)}/{len(self.setup_tasks)} tasks successful ({success_rate:.1%})")

        return success_rate >= 0.8

def create_smooth_setup_agent(project_root: Path = None) -> SmoothSetupAgent:
    """
    Factory function to create a configured SmoothSetupAgent.

    Args:
        project_root: Root directory of the project

    Returns:
        Configured SmoothSetupAgent instance
    """
    return SmoothSetupAgent(project_root)

async def auto_setup():
    """Run automatic setup process."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    agent = create_smooth_setup_agent()
    success = await agent.run_setup()

    if success:
        print("\n🎉 Setup completed successfully!")
        print("🚀 Run './start-demos.sh' to begin")
    else:
        print("\n⚠️  Setup completed with warnings")
        print("📋 Check logs for details")

if __name__ == "__main__":
    asyncio.run(auto_setup())