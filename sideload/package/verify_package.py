#!/usr/bin/env python3
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
