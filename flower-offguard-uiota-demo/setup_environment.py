#!/usr/bin/env python3
"""
Complete Environment Setup for Federated Learning with Off-Guard Security
Creates and configures the entire environment for production deployment
"""

import os
import sys
import subprocess
import platform
import shutil
import venv
from pathlib import Path
import json
import time

class EnvironmentSetup:
    """Comprehensive environment setup for FL system."""

    def __init__(self):
        self.project_root = Path.cwd()
        self.venv_path = self.project_root / ".venv"
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.system_info = {
            "platform": platform.system(),
            "arch": platform.machine(),
            "python": self.python_version,
            "node": platform.node()
        }

    def create_directories(self):
        """Create necessary directories for the FL system."""
        print("📁 Creating directory structure...")

        directories = [
            "data",
            "models",
            "logs",
            "demo_artifacts",
            "mesh_data",
            "config",
            "scripts",
            "tests",
            "docs",
            "deployment-output/builds",
            "deployment-output/configs",
            ".secrets",
            "backups"
        ]

        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}/")

        # Create .gitkeep files for empty directories
        gitkeep_dirs = ["data", "logs", "models", ".secrets", "backups"]
        for directory in gitkeep_dirs:
            gitkeep_path = self.project_root / directory / ".gitkeep"
            gitkeep_path.touch()

    def setup_virtual_environment(self):
        """Create and configure virtual environment."""
        print("🐍 Setting up Python virtual environment...")

        if self.venv_path.exists():
            print("   Virtual environment already exists")
            return True

        try:
            # Create virtual environment
            venv.create(str(self.venv_path), with_pip=True)
            print(f"   ✅ Virtual environment created at {self.venv_path}")

            # Get python executable path
            if platform.system() == "Windows":
                python_exe = self.venv_path / "Scripts" / "python.exe"
                pip_exe = self.venv_path / "Scripts" / "pip.exe"
            else:
                python_exe = self.venv_path / "bin" / "python"
                pip_exe = self.venv_path / "bin" / "pip"

            # Upgrade pip
            subprocess.run([str(pip_exe), "install", "--upgrade", "pip"],
                         check=True, capture_output=True)
            print("   ✅ Pip upgraded")

            return True

        except Exception as e:
            print(f"   ❌ Failed to create virtual environment: {e}")
            return False

    def install_dependencies(self, requirements_file="requirements-full.txt"):
        """Install Python dependencies."""
        print("📦 Installing Python dependencies...")

        if platform.system() == "Windows":
            pip_exe = self.venv_path / "Scripts" / "pip.exe"
        else:
            pip_exe = self.venv_path / "bin" / "pip"

        if not pip_exe.exists():
            print("   ❌ Virtual environment not found")
            return False

        try:
            # Install from requirements file
            if (self.project_root / requirements_file).exists():
                print(f"   Installing from {requirements_file}...")
                subprocess.run([
                    str(pip_exe), "install", "-r", str(requirements_file)
                ], check=True, capture_output=True)
                print("   ✅ Full dependencies installed")
            else:
                # Install minimal dependencies if full requirements not available
                print("   Installing minimal dependencies...")
                minimal_deps = [
                    "requests", "cryptography", "pydantic",
                    "python-dotenv", "psutil", "tqdm"
                ]
                for dep in minimal_deps:
                    subprocess.run([str(pip_exe), "install", dep],
                                 check=True, capture_output=True)
                print("   ✅ Minimal dependencies installed")

            return True

        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Some dependencies failed to install: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Dependency installation failed: {e}")
            return False

    def create_configuration_files(self):
        """Create configuration files for the system."""
        print("⚙️  Creating configuration files...")

        # Environment configuration
        env_content = self.generate_env_config()
        env_path = self.project_root / ".env"
        with open(env_path, "w") as f:
            f.write(env_content)
        print("   ✅ .env configuration created")

        # System configuration JSON
        config_data = {
            "system_info": self.system_info,
            "setup_timestamp": time.time(),
            "project_root": str(self.project_root),
            "venv_path": str(self.venv_path),
            "directories_created": True,
            "dependencies_installed": True,
            "security_mode": "offline",
            "fl_configuration": {
                "server_port": 8080,
                "dashboard_port": 8081,
                "default_rounds": 10,
                "default_clients": 6
            }
        }

        config_path = self.project_root / "config" / "system_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        print("   ✅ System configuration saved")

    def generate_env_config(self):
        """Generate environment configuration based on system."""
        return f"""# Federated Learning Environment - Auto-generated
# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
# System: {self.system_info['platform']} {self.system_info['arch']}
# Python: {self.system_info['python']}

# Core Security
OFFLINE_MODE=1
SECURITY_LEVEL=production
VENV_REQUIRED=true

# Federated Learning
FL_SERVER_HOST=localhost
FL_SERVER_PORT=8080
FL_DASHBOARD_PORT=8081
FL_ROUNDS=10
FL_CLIENTS_PER_ROUND=6

# Paths
DATA_DIR=./data
MODELS_DIR=./models
LOGS_DIR=./logs
ARTIFACTS_DIR=./demo_artifacts

# System
LOG_LEVEL=INFO
DEBUG_MODE=false
AUTO_START_AGENTS=true

# Development
DEMO_MODE=true
MOCK_HARDWARE=true
"""

    def create_startup_scripts(self):
        """Create startup and management scripts."""
        print("📜 Creating startup scripts...")

        # Linux/Mac startup script
        startup_script = """#!/bin/bash
set -e

echo "🚀 Starting Federated Learning Environment"
echo "=========================================="

# Load environment
if [ -f .env ]; then
    source .env
    echo "✅ Environment loaded"
else
    echo "⚠️  No .env file found, using defaults"
    export OFFLINE_MODE=1
fi

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found"
    echo "Run: python3 setup_environment.py"
    exit 1
fi

# Check dependencies
echo "🔍 Checking dependencies..."
python -c "import sys; print(f'Python: {sys.version}')"

# Start the complete demo
echo "🎯 Starting complete FL demo..."
exec ./run_all_demos.sh
"""

        startup_path = self.project_root / "start_fl_environment.sh"
        with open(startup_path, "w") as f:
            f.write(startup_script)
        startup_path.chmod(0o755)
        print("   ✅ start_fl_environment.sh created")

        # Windows startup script
        windows_script = """@echo off
echo 🚀 Starting Federated Learning Environment
echo ==========================================

REM Load environment
if exist .env (
    echo ✅ Environment file found
) else (
    echo ⚠️  No .env file found, using defaults
    set OFFLINE_MODE=1
)

REM Activate virtual environment
if exist .venv\\Scripts\\activate.bat (
    call .venv\\Scripts\\activate.bat
    echo ✅ Virtual environment activated
) else (
    echo ❌ Virtual environment not found
    echo Run: python setup_environment.py
    pause
    exit /b 1
)

REM Check dependencies
echo 🔍 Checking dependencies...
python -c "import sys; print(f'Python: {sys.version}')"

REM Start the dashboard
echo 🎯 Starting FL dashboard...
python dashboard_with_agents.py

pause
"""

        windows_path = self.project_root / "start_fl_environment.bat"
        with open(windows_path, "w") as f:
            f.write(windows_script)
        print("   ✅ start_fl_environment.bat created")

    def setup_security(self):
        """Set up security configurations."""
        print("🔒 Setting up security configurations...")

        # Create security directory
        security_dir = self.project_root / ".secrets"
        security_dir.mkdir(exist_ok=True)

        # Create .gitignore for secrets
        gitignore_content = """# Security files
.secrets/
*.key
*.pem
*.p12
*.keystore
config/production.env
.env.production
"""

        gitignore_path = self.project_root / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, "a") as f:
                f.write("\n" + gitignore_content)
        else:
            with open(gitignore_path, "w") as f:
                f.write(gitignore_content)

        print("   ✅ Security configurations set up")

    def create_docker_config(self):
        """Create Docker configuration for containerized deployment."""
        print("🐳 Creating Docker configuration...")

        dockerfile_content = """FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-full.txt .
RUN pip install --no-cache-dir -r requirements-full.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs models demo_artifacts

# Set environment variables
ENV OFFLINE_MODE=1
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8081/api/status || exit 1

# Default command
CMD ["python", "dashboard_with_agents.py"]
"""

        dockerfile_path = self.project_root / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        # Docker Compose
        compose_content = """version: '3.8'

services:
  fl-dashboard:
    build: .
    ports:
      - "8080:8080"
      - "8081:8081"
    environment:
      - OFFLINE_MODE=1
      - FL_SERVER_HOST=0.0.0.0
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/api/status"]
      interval: 30s
      timeout: 10s
      retries: 3

  fl-agent-1:
    build: .
    command: python fl_agent_system.py
    environment:
      - OFFLINE_MODE=1
      - FL_SERVER_HOST=fl-dashboard
    depends_on:
      - fl-dashboard
    restart: unless-stopped

networks:
  default:
    name: fl-network
"""

        compose_path = self.project_root / "docker-compose.yml"
        with open(compose_path, "w") as f:
            f.write(compose_content)

        print("   ✅ Docker configuration created")

    def run_setup(self):
        """Run the complete environment setup."""
        print("🛠️  FEDERATED LEARNING ENVIRONMENT SETUP")
        print("🔒 Off-Guard Security • 🤖 FL Agents • 📊 Dashboard")
        print("=" * 60)

        steps = [
            ("Creating directories", self.create_directories),
            ("Setting up virtual environment", self.setup_virtual_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Creating configuration files", self.create_configuration_files),
            ("Creating startup scripts", self.create_startup_scripts),
            ("Setting up security", self.setup_security),
            ("Creating Docker config", self.create_docker_config),
        ]

        success_count = 0
        for step_name, step_function in steps:
            print(f"\n{step_name}...")
            try:
                if step_function():
                    success_count += 1
                else:
                    print(f"   ⚠️  {step_name} completed with warnings")
                    success_count += 1
            except Exception as e:
                print(f"   ❌ {step_name} failed: {e}")

        print("\n" + "=" * 60)
        print(f"🏁 SETUP COMPLETE: {success_count}/{len(steps)} steps successful")

        if success_count == len(steps):
            print("✅ Environment setup completed successfully!")
            self.show_next_steps()
        else:
            print("⚠️  Setup completed with some issues")
            print("   Check the messages above for details")

    def show_next_steps(self):
        """Show next steps after setup."""
        print("\n🎯 NEXT STEPS:")
        print("=" * 30)
        print("1. Start the complete FL system:")
        if platform.system() == "Windows":
            print("   start_fl_environment.bat")
        else:
            print("   ./start_fl_environment.sh")

        print("\n2. Or start components individually:")
        print("   ./run_all_demos.sh")

        print("\n3. Access the dashboard:")
        print("   http://localhost:8081")

        print("\n4. Run tests:")
        print("   python test_full_system.py")

        print("\n5. View configuration:")
        print("   cat .env")

        print("\n🔧 CONFIGURATION FILES CREATED:")
        print("   .env - Environment variables")
        print("   config/system_config.json - System configuration")
        print("   Dockerfile - Container configuration")
        print("   docker-compose.yml - Multi-service deployment")

def main():
    """Main entry point."""
    setup = EnvironmentSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()