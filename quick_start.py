#!/usr/bin/env python3
"""
UIOTA Offline Guard - Quick Start
Just run this and everything works!
"""

import socket
import subprocess
import sys
import time
from pathlib import Path

def check_port(port):
    """Check if a port is responding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', port))
            return result == 0
    except:
        return False

def start_web_demo():
    """Start the web demo if not already running."""
    if check_port(8080):
        print("✅ Web demo already running on port 8080")
        return True

    try:
        print("🚀 Starting web demo...")

        # Create logs directory
        Path("logs").mkdir(exist_ok=True)

        # Start web server
        process = subprocess.Popen([
            sys.executable, "-m", "http.server", "8080", "--directory", "web-demo"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait a moment and check if it started
        time.sleep(2)

        if check_port(8080):
            print("✅ Web demo started successfully!")
            return True
        else:
            print("❌ Failed to start web demo")
            return False

    except Exception as e:
        print(f"❌ Error starting web demo: {e}")
        return False

def main():
    """Main entry point."""
    print("""
🛡️  UIOTA OFFLINE GUARD - QUICK START
═════════════════════════════════════

🎯 Getting your demos ready...
""")

    # Start web demo
    web_success = start_web_demo()

    print("\n" + "="*50)
    print("🎉 QUICK START COMPLETE!")
    print("="*50)

    if web_success:
        print("🌐 Web Demo: http://localhost:8080")
        print("   • Guardian simulation")
        print("   • Offline mode demo")
        print("   • QR proof generation")
        print("   • Perfect for showing to classmates!")
        print()

    print("🎯 What you can do now:")
    print("  📚 Share http://localhost:8080 with classmates")
    print("  ✈️ Works offline for travel teams")
    print("  🏆 Perfect for hackathon demos")
    print("  🛡️ Show off the security features")
    print()

    if web_success:
        print("🚀 Your demo is LIVE and ready to share!")
    else:
        print("⚠️ Some services had issues - check the logs")

    print("\n🛑 To stop: Ctrl+C or close this window")

if __name__ == "__main__":
    try:
        main()
        # Keep running to maintain services
        while True:
            time.sleep(60)
            if not check_port(8080):
                print("⚠️ Web demo stopped - restarting...")
                start_web_demo()
    except KeyboardInterrupt:
        print("\n🛑 Quick start stopped by user")
        print("✅ Thanks for using UIOTA Offline Guard!")