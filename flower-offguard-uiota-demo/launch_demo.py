#!/usr/bin/env python3
"""
LL TOKEN OFFLINE - Demo Launcher
Launches the complete website demo with simulated model
"""

import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path

def launch_website_demo():
    """Launch the LL TOKEN OFFLINE website demo"""

    print("🚀" + "="*60 + "🚀")
    print("   LL TOKEN OFFLINE - Demo Launcher")
    print("   Quantum-Safe Wallet & Marketplace Demo")
    print("🚀" + "="*60 + "🚀")
    print()

    # Get the correct paths
    project_dir = Path(__file__).parent
    website_dir = project_dir / "website"

    if not website_dir.exists():
        print("❌ Website directory not found!")
        return False

    print("📁 Website directory found:", website_dir)
    print("🌐 Starting web server...")

    # Change to website directory
    os.chdir(website_dir)

    try:
        # Start the web server in background
        print("🔄 Launching HTTP server on port 8080...")

        # Try to open the website in browser
        website_url = "http://localhost:8080"
        print(f"🌐 Opening {website_url} in your browser...")

        # Open browser (this will work better from the launcher)
        time.sleep(1)
        webbrowser.open(website_url)

        print("✅ Demo launched successfully!")
        print()
        print("🎯 Available Features:")
        print("  💳 Wallet Tab - Quantum-safe token management")
        print("  🏪 Marketplace Tab - P2P trading & NFTs")
        print("  🌍 Metaverse Tab - Avatar & virtual land")
        print("  🏦 Staking Tab - Yield farming pools")
        print()
        print("🔍 Tips:")
        print("  • All features work offline-first")
        print("  • Check browser console for detailed logs")
        print("  • Navigate between tabs to explore features")
        print("  • Try the demo buttons and forms")
        print()
        print("🔗 Direct access: http://localhost:8080")
        print("⏹️  Press Ctrl+C to stop the demo")

        # Keep the server running
        try:
            subprocess.run([sys.executable, "-m", "http.server", "8080", "--bind", "localhost"])
        except KeyboardInterrupt:
            print("\n\n⏹️ Demo stopped by user")
            print("👋 Thank you for trying LL TOKEN OFFLINE!")

    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        return False

    return True

if __name__ == "__main__":
    launch_website_demo()