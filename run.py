#!/usr/bin/env python3
"""
UIOTA Offline Guard - One-Click Runner

The simplest possible way to run everything:
1. Download this project
2. Run: python run.py
3. Everything starts automatically!

No manual setup required.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Ensure we have a compatible Python version."""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade Python and try again")
        return False

    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} OK")
    return True

def ensure_agents_directory():
    """Create agents directory if it doesn't exist."""
    agents_dir = Path("agents")
    if not agents_dir.exists():
        agents_dir.mkdir()
        print("📁 Created agents directory")
    return True

def run_auto_launcher():
    """Run the auto demo launcher."""
    try:
        print("🚀 Starting UIOTA Auto Demo Launcher...")

        # Run the auto launcher
        result = subprocess.run([
            sys.executable, "auto_demo_launcher.py"
        ], check=False)

        return result.returncode == 0

    except FileNotFoundError:
        print("❌ auto_demo_launcher.py not found")
        print("   Make sure you're in the correct directory")
        return False
    except Exception as e:
        print(f"❌ Error running launcher: {e}")
        return False

def main():
    """Main entry point."""
    print("""
🛡️  UIOTA OFFLINE GUARD
═══════════════════════

🎯 One-click demo startup:
   Just run this script and everything happens automatically!

""")

    # Check prerequisites
    if not check_python_version():
        return 1

    if not ensure_agents_directory():
        return 1

    # Run the launcher
    success = run_auto_launcher()

    if success:
        print("\n🎉 Setup and demos completed!")
        print("🌐 Visit http://localhost:8080")
    else:
        print("\n⚠️  Issues encountered - check logs")

    return 0 if success else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(130)