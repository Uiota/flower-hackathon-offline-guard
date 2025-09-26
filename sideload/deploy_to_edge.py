#!/usr/bin/env python3
"""
Edge Device Sideload Deployment Script
Prepares and deploys the UIOTA Offline Guard system to edge devices
"""

import os
import sys
import json
import shutil
import subprocess
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime

class EdgeDeviceDeployer:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.sideload_dir = self.project_root / "sideload"
        self.package_dir = self.sideload_dir / "package"
        self.manifest = {
            "name": "UIOTA Offline Guard",
            "version": "1.0.0",
            "description": "Sovereign AI ecosystem with Guardian agents",
            "created": datetime.now().isoformat(),
            "arch": "universal",
            "os": "linux",
            "requirements": {
                "python": ">=3.8",
                "memory": "512MB",
                "storage": "2GB",
                "network": "optional"
            }
        }

    def create_package_structure(self):
        """Create the sideload package directory structure"""
        print("🗂️  Creating package structure...")

        # Clean and create package directory
        if self.package_dir.exists():
            shutil.rmtree(self.package_dir)
        self.package_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        dirs = [
            "agents", "web", "scripts", "config", "docs", "presentation"
        ]

        for dir_name in dirs:
            (self.package_dir / dir_name).mkdir(exist_ok=True)

        print("✅ Package structure created")

    def copy_core_files(self):
        """Copy essential system files to package"""
        print("📁 Copying core system files...")

        # Core agent files
        agents_src = self.project_root / "agents"
        if agents_src.exists():
            shutil.copytree(agents_src, self.package_dir / "agents", dirs_exist_ok=True)

        # Web demo files
        web_src = self.project_root / "web-demo"
        if web_src.exists():
            shutil.copytree(web_src, self.package_dir / "web", dirs_exist_ok=True)

        # Essential Python scripts
        essential_scripts = [
            "interactive_control.py",
            "simple_web_dashboard.py",
            "show_agent_demo.py",
            "test_agent_system.py",
            "start_everything.sh",
            "requirements.txt"
        ]

        for script in essential_scripts:
            src_file = self.project_root / script
            if src_file.exists():
                shutil.copy2(src_file, self.package_dir / "scripts")

        # Copy presentation
        presentation_src = self.project_root / "presentation"
        if presentation_src.exists():
            shutil.copytree(presentation_src, self.package_dir / "presentation", dirs_exist_ok=True)

        print("✅ Core files copied")

    def create_edge_launcher(self):
        """Create edge device launcher script"""
        print("🚀 Creating edge device launcher...")

        launcher_content = '''#!/bin/bash
# UIOTA Offline Guard - Edge Device Launcher
# Auto-configures and starts the Guardian Agent system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🛡️  UIOTA Offline Guard - Edge Device Deployment"
echo "================================================="
echo ""

# Check system requirements
echo "🔍 Checking system requirements..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $python_version detected"

# Check available memory
mem_available=$(free -m | awk 'NR==2{printf "%.0f", $7}')
if [ "$mem_available" -lt 512 ]; then
    echo "⚠️  Warning: Low memory ($mem_available MB). 512MB+ recommended"
else
    echo "✅ Memory: $mem_available MB available"
fi

# Check disk space
disk_available=$(df . | awk 'NR==2{print $4}')
if [ "$disk_available" -lt 2097152 ]; then # 2GB in KB
    echo "⚠️  Warning: Low disk space. 2GB+ recommended"
else
    echo "✅ Disk space: $(($disk_available / 1024))MB available"
fi

echo ""

# Install Python dependencies if needed
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    python3 -m pip install --user -q -r requirements.txt
    echo "✅ Dependencies installed"
fi

# Make scripts executable
echo "🔧 Setting up permissions..."
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x scripts/*.py 2>/dev/null || true

# Start the system
echo ""
echo "🚀 Starting UIOTA Offline Guard system..."
echo ""

# Check what interface to use
if [ "$1" = "web" ]; then
    echo "🌐 Starting web dashboard..."
    cd scripts && python3 simple_web_dashboard.py
elif [ "$1" = "demo" ]; then
    echo "🎬 Starting live demonstration..."
    cd scripts && python3 show_agent_demo.py
elif [ "$1" = "test" ]; then
    echo "🧪 Running system tests..."
    cd scripts && python3 test_agent_system.py
else
    echo "🎛️  Starting interactive control interface..."
    cd scripts && python3 interactive_control.py
fi
'''

        launcher_path = self.package_dir / "launch_guardian.sh"
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)

        # Make executable
        os.chmod(launcher_path, 0o755)

        print("✅ Edge launcher created")

    def create_deployment_config(self):
        """Create deployment configuration files"""
        print("⚙️  Creating deployment configuration...")

        # Main manifest
        manifest_path = self.package_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

        # Edge device config
        edge_config = {
            "deployment": {
                "mode": "edge",
                "offline_first": True,
                "auto_start": True,
                "interfaces": ["web", "cli", "demo"]
            },
            "agents": {
                "security_monitor": {"enabled": True, "priority": "high"},
                "development_agent": {"enabled": True, "priority": "medium"},
                "communication_hub": {"enabled": True, "priority": "high"},
                "debug_monitor": {"enabled": True, "priority": "low"}
            },
            "network": {
                "offline_mode": True,
                "p2p_enabled": True,
                "web_port": 8888,
                "api_port": 8889
            },
            "security": {
                "threat_detection": True,
                "file_monitoring": True,
                "resource_monitoring": True,
                "auto_quarantine": True
            }
        }

        config_path = self.package_dir / "config" / "edge_config.json"
        with open(config_path, 'w') as f:
            json.dump(edge_config, f, indent=2)

        # Installation instructions
        install_instructions = '''# UIOTA Offline Guard - Edge Device Installation

## Quick Start
1. Extract the package to your edge device
2. Run: `./launch_guardian.sh`
3. Follow the on-screen prompts

## Available Interfaces
- **Interactive CLI**: `./launch_guardian.sh` (default)
- **Web Dashboard**: `./launch_guardian.sh web`
- **Live Demo**: `./launch_guardian.sh demo`
- **System Test**: `./launch_guardian.sh test`

## System Requirements
- Python 3.8+
- 512MB+ RAM (1GB recommended)
- 2GB+ storage
- Linux/Unix-based OS

## Features
- ✅ 100% offline operation
- ✅ Real-time security monitoring
- ✅ AI-powered threat detection
- ✅ Code quality analysis
- ✅ P2P communication
- ✅ Guardian agent coordination

## Troubleshooting
If you encounter issues:
1. Check system requirements
2. Ensure Python 3.8+ is installed
3. Verify permissions on launch script
4. Check available memory and disk space

For support, see the documentation in the `docs/` directory.
'''

        readme_path = self.package_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(install_instructions)

        print("✅ Configuration files created")

    def create_verification_script(self):
        """Create package verification script"""
        print("🔐 Creating verification script...")

        verify_script = '''#!/usr/bin/env python3
"""
Package Verification Script
Verifies the integrity and completeness of the sideload package
"""

import os
import sys
import hashlib
import json
from pathlib import Path

def calculate_file_hash(filepath):
    """Calculate SHA256 hash of a file"""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_package():
    """Verify package integrity"""
    script_dir = Path(__file__).parent

    print("🔍 Verifying UIOTA Offline Guard package...")
    print("=" * 45)

    # Check required directories
    required_dirs = ["agents", "scripts", "config", "web", "docs", "presentation"]
    missing_dirs = []

    for dir_name in required_dirs:
        dir_path = script_dir / dir_name
        if dir_path.exists():
            print(f"✅ Directory: {dir_name}")
        else:
            print(f"❌ Missing: {dir_name}")
            missing_dirs.append(dir_name)

    # Check essential files
    essential_files = [
        "launch_guardian.sh",
        "manifest.json",
        "README.md",
        "config/edge_config.json"
    ]

    missing_files = []
    for file_path in essential_files:
        full_path = script_dir / file_path
        if full_path.exists():
            print(f"✅ File: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            missing_files.append(file_path)

    # Check manifest
    manifest_path = script_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            print(f"✅ Manifest: {manifest['name']} v{manifest['version']}")
        except Exception as e:
            print(f"❌ Manifest error: {e}")

    # Check launcher permissions
    launcher_path = script_dir / "launch_guardian.sh"
    if launcher_path.exists():
        if os.access(launcher_path, os.X_OK):
            print("✅ Launcher: Executable")
        else:
            print("⚠️  Launcher: Not executable (run 'chmod +x launch_guardian.sh')")

    print()

    if missing_dirs or missing_files:
        print("❌ Package verification FAILED")
        print("Missing components detected. Package may be corrupted.")
        return False
    else:
        print("✅ Package verification PASSED")
        print("All components present and ready for deployment.")
        return True

if __name__ == "__main__":
    success = verify_package()
    sys.exit(0 if success else 1)
'''

        verify_path = self.package_dir / "verify_package.py"
        with open(verify_path, 'w') as f:
            f.write(verify_script)

        # Make executable
        os.chmod(verify_path, 0o755)

        print("✅ Verification script created")

    def create_sideload_archive(self):
        """Create compressed sideload archive"""
        print("📦 Creating sideload archive...")

        # Create archive filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"uiota_offline_guard_sideload_{timestamp}.zip"
        archive_path = self.sideload_dir / archive_name

        # Create ZIP archive
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.package_dir):
                for file in files:
                    file_path = Path(root) / file
                    arc_path = file_path.relative_to(self.package_dir)
                    zipf.write(file_path, arc_path)

        # Calculate archive hash
        archive_hash = hashlib.sha256()
        with open(archive_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                archive_hash.update(chunk)

        # Create deployment info
        deployment_info = {
            "archive": archive_name,
            "size": archive_path.stat().st_size,
            "sha256": archive_hash.hexdigest(),
            "created": datetime.now().isoformat(),
            "contains": {
                "agents": "Guardian agent system",
                "scripts": "Deployment and control scripts",
                "web": "Web interface files",
                "presentation": "Digital presentation",
                "config": "Edge device configuration"
            },
            "deployment": {
                "extract": f"unzip {archive_name}",
                "verify": "python3 verify_package.py",
                "launch": "./launch_guardian.sh"
            }
        }

        info_path = self.sideload_dir / f"deployment_info_{timestamp}.json"
        with open(info_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)

        print(f"✅ Archive created: {archive_name}")
        print(f"📊 Size: {archive_path.stat().st_size // 1024} KB")
        print(f"🔐 SHA256: {archive_hash.hexdigest()[:16]}...")

        return archive_path, info_path

    def deploy(self):
        """Execute the complete deployment process"""
        print("🛡️  UIOTA Offline Guard - Sideload Package Creator")
        print("=" * 50)
        print()

        try:
            # Create package structure
            self.create_package_structure()

            # Copy core files
            self.copy_core_files()

            # Create edge-specific components
            self.create_edge_launcher()
            self.create_deployment_config()
            self.create_verification_script()

            # Create final archive
            archive_path, info_path = self.create_sideload_archive()

            print()
            print("🎉 Sideload package creation completed!")
            print()
            print("📦 Package Location:")
            print(f"   Archive: {archive_path}")
            print(f"   Info: {info_path}")
            print()
            print("🚀 Deployment Instructions:")
            print("   1. Transfer archive to edge device")
            print("   2. Extract: unzip <archive_name>")
            print("   3. Verify: python3 verify_package.py")
            print("   4. Launch: ./launch_guardian.sh")
            print()
            print("✅ Ready for edge device deployment!")

            return True

        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return False

def main():
    deployer = EdgeDeviceDeployer()
    success = deployer.deploy()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()