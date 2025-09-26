#!/usr/bin/env python3
"""
Demo Builder Agent - Automates building the Flower Off-Guard UIOTA demo package.

This agent handles:
- Virtual environment creation
- Dependency installation
- Test execution
- Demo packaging as distributable format
- Installation script generation
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='🔧 [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class DemoBuilderAgent:
    """Main demo builder automation agent."""

    def __init__(self, project_root: Path, output_dir: Path = None):
        self.project_root = Path(project_root).resolve()
        self.demo_dir = self.project_root / "flower-offguard-uiota-demo"
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "dist"
        self.build_dir = self.output_dir / "build"

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.build_dir.mkdir(parents=True, exist_ok=True)

        # Build configuration
        self.config = {
            "demo_name": "flower-offguard-uiota-demo",
            "version": "1.0.0",
            "python_version": "3.11",
            "platforms": ["linux", "darwin", "win32"],
            "package_formats": ["tar.gz", "zip"],
        }

    def validate_environment(self) -> bool:
        """Validate build environment and dependencies."""
        logger.info("Validating build environment...")

        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            return False

        # Check demo directory exists
        if not self.demo_dir.exists():
            logger.error(f"Demo directory not found: {self.demo_dir}")
            return False

        # Check requirements.txt exists
        requirements_file = self.demo_dir / "requirements.txt"
        if not requirements_file.exists():
            logger.error(f"Requirements file not found: {requirements_file}")
            return False

        # Check source directory
        src_dir = self.demo_dir / "src"
        if not src_dir.exists():
            logger.error(f"Source directory not found: {src_dir}")
            return False

        logger.info("✅ Environment validation passed")
        return True

    def create_virtual_environment(self) -> Tuple[Path, bool]:
        """Create and setup virtual environment for demo."""
        logger.info("Creating virtual environment...")

        venv_path = self.build_dir / "venv"

        # Remove existing venv if present
        if venv_path.exists():
            logger.info("Removing existing virtual environment...")
            shutil.rmtree(venv_path)

        try:
            # Create virtual environment
            subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], check=True, capture_output=True, text=True)

            # Get python executable path
            if sys.platform == "win32":
                python_exe = venv_path / "Scripts" / "python.exe"
                pip_exe = venv_path / "Scripts" / "pip.exe"
            else:
                python_exe = venv_path / "bin" / "python"
                pip_exe = venv_path / "bin" / "pip"

            # Upgrade pip
            subprocess.run([
                str(pip_exe), "install", "--upgrade", "pip"
            ], check=True, capture_output=True, text=True)

            logger.info(f"✅ Virtual environment created: {venv_path}")
            return venv_path, True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            return venv_path, False

    def install_dependencies(self, venv_path: Path) -> bool:
        """Install demo dependencies in virtual environment."""
        logger.info("Installing demo dependencies...")

        if sys.platform == "win32":
            pip_exe = venv_path / "Scripts" / "pip.exe"
        else:
            pip_exe = venv_path / "bin" / "pip"

        requirements_file = self.demo_dir / "requirements.txt"

        try:
            # Install demo requirements
            result = subprocess.run([
                str(pip_exe), "install", "-r", str(requirements_file)
            ], capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                logger.error(f"Dependency installation failed: {result.stderr}")
                return False

            # Install additional build tools
            subprocess.run([
                str(pip_exe), "install", "wheel", "setuptools", "twine"
            ], check=True, capture_output=True, text=True)

            logger.info("✅ Dependencies installed successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Dependency installation timed out")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False

    def run_tests(self, venv_path: Path) -> bool:
        """Run demo tests if they exist."""
        logger.info("Running demo tests...")

        tests_dir = self.demo_dir / "tests"
        if not tests_dir.exists():
            logger.info("No tests directory found, skipping tests")
            return True

        if sys.platform == "win32":
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"

        try:
            # Run pytest if available
            result = subprocess.run([
                str(python_exe), "-m", "pytest", str(tests_dir), "-v"
            ], capture_output=True, text=True, cwd=self.demo_dir, timeout=120)

            if result.returncode != 0:
                logger.warning(f"Some tests failed: {result.stdout}")
                logger.warning(f"Test errors: {result.stderr}")
                return False

            logger.info("✅ All tests passed")
            return True

        except FileNotFoundError:
            logger.info("pytest not available, skipping test execution")
            return True
        except subprocess.TimeoutExpired:
            logger.error("Test execution timed out")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Test execution failed: {e}")
            return False

    def create_package_structure(self) -> Path:
        """Create standardized package structure."""
        logger.info("Creating package structure...")

        package_dir = self.build_dir / self.config["demo_name"]

        # Remove existing package directory
        if package_dir.exists():
            shutil.rmtree(package_dir)

        package_dir.mkdir(parents=True)

        # Copy source files
        src_dest = package_dir / "src"
        shutil.copytree(self.demo_dir / "src", src_dest)

        # Copy requirements
        shutil.copy2(self.demo_dir / "requirements.txt", package_dir)

        # Copy tests if they exist
        tests_src = self.demo_dir / "tests"
        if tests_src.exists():
            shutil.copytree(tests_src, package_dir / "tests")

        # Create README
        readme_content = self._generate_readme()
        (package_dir / "README.md").write_text(readme_content)

        # Create setup script
        setup_content = self._generate_setup_script()
        (package_dir / "setup.py").write_text(setup_content)

        # Create installation scripts
        self._create_installation_scripts(package_dir)

        # Create configuration files
        self._create_config_files(package_dir)

        logger.info(f"✅ Package structure created: {package_dir}")
        return package_dir

    def _generate_readme(self) -> str:
        """Generate README.md for the demo package."""
        return f"""# {self.config["demo_name"].title()}

## Overview

Flower Off-Guard UIOTA Demo - A comprehensive federated learning demonstration with security features and mesh networking.

### Features

- 🌸 Flower AI federated learning integration
- 🛡️ Off-Guard security framework
- 🌐 UIOTA mesh networking
- 🔒 Differential privacy support
- 🚀 CPU-optimized for offline use

### Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Demo Server:**
   ```bash
   python src/server.py --rounds 5 --clients-per-round 10
   ```

3. **Run Demo Client:**
   ```bash
   python src/client.py --server-address localhost:8080
   ```

### Installation Scripts

- `install.sh` - Unix/Linux installation
- `install.bat` - Windows installation
- `install_dev.sh` - Development environment setup

### Configuration

Edit `config/demo.yaml` to customize:
- Dataset selection (MNIST, CIFAR-10)
- Federated learning strategy
- Security settings
- Network simulation parameters

### Documentation

- `docs/API.md` - API documentation
- `docs/SECURITY.md` - Security model
- `docs/EXAMPLES.md` - Usage examples

### Requirements

- Python 3.8+
- CPU-only (no GPU required)
- 2GB RAM minimum
- Internet connection for initial setup

### Support

Built for the Offline Guard project: https://github.com/uiota/offline-guard

Version: {self.config["version"]}
"""

    def _generate_setup_script(self) -> str:
        """Generate setup.py for pip installation."""
        return f'''#!/usr/bin/env python3
"""Setup script for {self.config["demo_name"]}."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip() for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="{self.config["demo_name"]}",
    version="{self.config["version"]}",
    description="Flower Off-Guard UIOTA Federated Learning Demo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="UIOTA Team",
    author_email="dev@uiota.org",
    url="https://github.com/uiota/offline-guard",
    packages=find_packages(),
    package_dir={{"": "src"}},
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    keywords="federated-learning flower ai security uiota",
    entry_points={{
        "console_scripts": [
            "{self.config["demo_name"]}-server=server:main",
            "{self.config["demo_name"]}-client=client:main",
        ],
    }},
    include_package_data=True,
    package_data={{
        "": ["*.yaml", "*.json", "*.txt", "*.md"],
    }},
    extras_require={{
        "dev": ["pytest>=7.4.3", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    }},
)
'''

    def _create_installation_scripts(self, package_dir: Path) -> None:
        """Create platform-specific installation scripts."""

        # Unix/Linux install script
        install_sh = package_dir / "install.sh"
        install_sh.write_text("""#!/bin/bash
set -e

echo "🌸 Installing Flower Off-Guard UIOTA Demo..."
echo "============================================"

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3, 8)" || {
    echo "❌ Python 3.8+ required"
    exit 1
}

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run setup
echo "⚙️ Installing demo package..."
pip install -e .

echo ""
echo "✅ Installation complete!"
echo ""
echo "🚀 Quick start:"
echo "   source venv/bin/activate"
echo "   python src/server.py"
echo ""
echo "📖 See README.md for full documentation"
""")
        install_sh.chmod(0o755)

        # Windows install script
        install_bat = package_dir / "install.bat"
        install_bat.write_text("""@echo off
echo 🌸 Installing Flower Off-Guard UIOTA Demo...
echo ============================================

REM Check Python
python --version >nul 2>&1 || (
    echo ❌ Python not found in PATH
    exit /b 1
)

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv
call venv\\Scripts\\activate.bat

REM Install dependencies
echo 📚 Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Run setup
echo ⚙️ Installing demo package...
pip install -e .

echo.
echo ✅ Installation complete!
echo.
echo 🚀 Quick start:
echo    venv\\Scripts\\activate.bat
echo    python src\\server.py
echo.
echo 📖 See README.md for full documentation
pause
""")

        # Development install script
        install_dev_sh = package_dir / "install_dev.sh"
        install_dev_sh.write_text("""#!/bin/bash
set -e

echo "🔧 Installing Flower Off-Guard UIOTA Demo (Development Mode)..."
echo "============================================================="

# Run standard install
./install.sh

# Activate environment
source venv/bin/activate

# Install development dependencies
echo "🛠️ Installing development tools..."
pip install -e ".[dev,docs]"

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v || echo "⚠️ Some tests failed"

echo ""
echo "✅ Development environment ready!"
echo ""
echo "🔧 Development commands:"
echo "   pytest tests/           # Run tests"
echo "   black src/             # Format code"
echo "   flake8 src/            # Lint code"
echo "   mypy src/              # Type checking"
echo ""
""")
        install_dev_sh.chmod(0o755)

    def _create_config_files(self, package_dir: Path) -> None:
        """Create configuration files for the demo."""

        config_dir = package_dir / "config"
        config_dir.mkdir(exist_ok=True)

        # Demo configuration
        demo_config = config_dir / "demo.yaml"
        demo_config.write_text("""# Flower Off-Guard UIOTA Demo Configuration

# Federated Learning Settings
federated_learning:
  strategy: "fedavg"  # fedavg, fedprox
  rounds: 5
  clients_per_round: 10
  min_clients: 2

# Dataset Settings
dataset:
  name: "mnist"  # mnist, cifar10
  batch_size: 32
  validation_split: 0.1

# Security Settings
security:
  differential_privacy: false
  noise_multiplier: 1.0
  max_grad_norm: 1.0
  enable_encryption: true

# Network Settings
network:
  server_address: "localhost:8080"
  mesh_enabled: true
  p2p_discovery: true
  latency_simulation:
    base_ms: 50
    jitter_ms: 25
    dropout_rate: 0.1

# Performance Settings
performance:
  cpu_only: true
  num_workers: 2
  memory_limit_gb: 2

# Logging
logging:
  level: "INFO"
  file: "demo.log"
  console: true
""")

        # Docker configuration
        dockerfile = package_dir / "Dockerfile"
        dockerfile.write_text("""FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy demo source
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user
RUN useradd -m demo && chown -R demo:demo /app
USER demo

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "src/server.py", "--config", "config/demo.yaml"]
""")

    def create_distributions(self, package_dir: Path) -> List[Path]:
        """Create distribution packages in multiple formats."""
        logger.info("Creating distribution packages...")

        distributions = []
        package_name = f"{self.config['demo_name']}-{self.config['version']}"

        for fmt in self.config["package_formats"]:
            if fmt == "tar.gz":
                dist_path = self.output_dir / f"{package_name}.tar.gz"
                with tarfile.open(dist_path, "w:gz") as tar:
                    tar.add(package_dir, arcname=self.config["demo_name"])
                distributions.append(dist_path)

            elif fmt == "zip":
                dist_path = self.output_dir / f"{package_name}.zip"
                with zipfile.ZipFile(dist_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for file_path in package_dir.rglob("*"):
                        if file_path.is_file():
                            arcname = self.config["demo_name"] / file_path.relative_to(package_dir)
                            zip_file.write(file_path, arcname)
                distributions.append(dist_path)

        logger.info(f"✅ Created {len(distributions)} distribution packages")
        return distributions

    def generate_checksums(self, distributions: List[Path]) -> Path:
        """Generate checksums for distribution files."""
        logger.info("Generating checksums...")

        import hashlib

        checksums_file = self.output_dir / "CHECKSUMS.txt"

        with open(checksums_file, "w") as f:
            f.write(f"# Checksums for {self.config['demo_name']} {self.config['version']}\\n")
            f.write(f"# Generated on {__import__('datetime').datetime.now().isoformat()}\\n\\n")

            for dist_path in distributions:
                # Calculate SHA256
                sha256_hash = hashlib.sha256()
                with open(dist_path, "rb") as file:
                    for chunk in iter(lambda: file.read(4096), b""):
                        sha256_hash.update(chunk)

                f.write(f"SHA256({dist_path.name}) = {sha256_hash.hexdigest()}\\n")

        logger.info(f"✅ Checksums saved to: {checksums_file}")
        return checksums_file

    def cleanup_build(self) -> None:
        """Clean up build directory."""
        logger.info("Cleaning up build directory...")

        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)

        logger.info("✅ Build cleanup complete")

    def build(self, skip_tests: bool = False, cleanup: bool = True) -> Dict:
        """Main build process."""
        logger.info("🚀 Starting demo build process...")

        build_results = {
            "success": False,
            "distributions": [],
            "checksums_file": None,
            "errors": []
        }

        try:
            # Validate environment
            if not self.validate_environment():
                build_results["errors"].append("Environment validation failed")
                return build_results

            # Create virtual environment
            venv_path, venv_success = self.create_virtual_environment()
            if not venv_success:
                build_results["errors"].append("Virtual environment creation failed")
                return build_results

            # Install dependencies
            if not self.install_dependencies(venv_path):
                build_results["errors"].append("Dependency installation failed")
                return build_results

            # Run tests
            if not skip_tests and not self.run_tests(venv_path):
                build_results["errors"].append("Tests failed")
                return build_results

            # Create package structure
            package_dir = self.create_package_structure()

            # Create distributions
            distributions = self.create_distributions(package_dir)
            build_results["distributions"] = distributions

            # Generate checksums
            checksums_file = self.generate_checksums(distributions)
            build_results["checksums_file"] = checksums_file

            # Cleanup
            if cleanup:
                self.cleanup_build()

            build_results["success"] = True
            logger.info("🎉 Demo build completed successfully!")

        except Exception as e:
            logger.error(f"Build failed with error: {e}")
            build_results["errors"].append(str(e))

        return build_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Flower Off-Guard UIOTA Demo Builder")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output-dir", help="Output directory for distributions")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test execution")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep build directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize builder
    builder = DemoBuilderAgent(
        project_root=args.project_root,
        output_dir=args.output_dir
    )

    # Run build
    results = builder.build(
        skip_tests=args.skip_tests,
        cleanup=not args.no_cleanup
    )

    # Print results
    if results["success"]:
        logger.info("📦 Build Summary:")
        logger.info(f"   Distributions: {len(results['distributions'])}")
        for dist in results["distributions"]:
            logger.info(f"   - {dist}")
        if results["checksums_file"]:
            logger.info(f"   Checksums: {results['checksums_file']}")
        return 0
    else:
        logger.error("❌ Build failed!")
        for error in results["errors"]:
            logger.error(f"   - {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())