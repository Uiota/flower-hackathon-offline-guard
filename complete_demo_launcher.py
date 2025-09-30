#!/usr/bin/env python3
"""
Complete Off-Guard Demo Launcher
Integrated demo with all features: Flower FL, AI integration, encryption, and portal
"""

import asyncio
import subprocess
import threading
import time
import logging
import json
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional
import webbrowser
from datetime import datetime

class CompleteDemoLauncher:
    """Comprehensive demo launcher for Off-Guard platform"""

    def __init__(self):
        self.processes = {}
        self.demo_active = False
        self.web_server = None
        self.fl_demo = None
        self.demo_config = {
            "web_port": 8000,
            "fl_server_port": 8080,
            "auto_open_browser": True,
            "demo_duration": 300,  # 5 minutes
            "features": {
                "flower_fl": True,
                "ai_integration": True,
                "encryption": True,
                "portal_demo": True,
                "two_device_demo": True,
                "confidential_llm": True
            }
        }

    def print_banner(self):
        """Print startup banner"""
        banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🛡️  Off-Guard Platform                    ║
    ║              Complete Federated Learning Demo                ║
    ║                                                              ║
    ║  🌸 Flower Framework Integration                             ║
    ║  🤖 OpenAI + Anthropic AI Integration                       ║
    ║  🔐 End-to-End Encryption                                   ║
    ║  📱 Two-Device FL Demo                                      ║
    ║  💬 Confidential LLM Communication                         ║
    ║  🌐 Interactive Web Portal                                  ║
    ╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        required_packages = [
            "fastapi", "uvicorn", "flwr", "torch", "transformers",
            "cryptography", "websockets", "aiohttp"
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print("📦 Installing missing packages...")
            return self.install_dependencies(missing_packages)

        print("✅ All dependencies available")
        return True

    def install_dependencies(self, packages: List[str]) -> bool:
        """Install missing dependencies"""
        try:
            for package in packages:
                print(f"Installing {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package],
                             check=True, capture_output=True)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False

    def setup_environment(self):
        """Setup demo environment"""
        print("🔧 Setting up demo environment...")

        # Create demo directories
        demo_dirs = ["logs", "models", "data", "keys"]
        for dir_name in demo_dirs:
            Path(dir_name).mkdir(exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/demo.log'),
                logging.StreamHandler()
            ]
        )

        # Generate demo encryption keys
        from cryptography.fernet import Fernet
        demo_key = Fernet.generate_key()
        with open("keys/demo_encryption.key", "wb") as f:
            f.write(demo_key)

        print("✅ Environment setup complete")

    async def start_web_interface(self):
        """Start the main web interface"""
        print(f"🌐 Starting web interface on port {self.demo_config['web_port']}...")

        try:
            # Start web server
            process = subprocess.Popen([
                sys.executable, "web_interface.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes["web_server"] = process
            await asyncio.sleep(3)  # Allow server to start

            print(f"✅ Web interface started at http://localhost:{self.demo_config['web_port']}")

            # Auto-open browser
            if self.demo_config["auto_open_browser"]:
                try:
                    webbrowser.open(f"http://localhost:{self.demo_config['web_port']}")
                    print("🌐 Browser opened automatically")
                except Exception as e:
                    print(f"⚠️  Could not open browser: {e}")

        except Exception as e:
            print(f"❌ Web interface startup error: {e}")

    async def start_fl_server(self):
        """Start Flower federated learning server"""
        if not self.demo_config["features"]["flower_fl"]:
            return

        print("🌸 Starting Flower FL server...")

        try:
            # Start FL server in background
            fl_process = subprocess.Popen([
                sys.executable, "flower_fl_server.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes["fl_server"] = fl_process
            await asyncio.sleep(3)  # Allow server to start

            if fl_process.poll() is None:
                print("✅ Flower FL server started successfully")
            else:
                print("❌ Failed to start FL server")

        except Exception as e:
            print(f"❌ FL server startup error: {e}")

    async def run_two_device_demo(self):
        """Run the two-device federated learning demo"""
        if not self.demo_config["features"]["two_device_demo"]:
            return

        print("📱 Starting two-device FL demo...")

        try:
            # Start demo in background
            demo_process = subprocess.Popen([
                sys.executable, "two_device_demo.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes["two_device_demo"] = demo_process
            print("✅ Two-device demo started")

        except Exception as e:
            print(f"❌ Two-device demo error: {e}")

    async def run_confidential_llm_demo(self):
        """Run confidential LLM communication demo"""
        if not self.demo_config["features"]["confidential_llm"]:
            return

        print("🔐 Starting confidential LLM demo...")

        try:
            # Start demo in background
            llm_process = subprocess.Popen([
                sys.executable, "confidential_llm.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes["confidential_llm"] = llm_process
            print("✅ Confidential LLM demo started")

        except Exception as e:
            print(f"❌ Confidential LLM demo error: {e}")

    def show_demo_info(self):
        """Display demo information and URLs"""
        print("\n" + "="*60)
        print("🎯 DEMO INFORMATION")
        print("="*60)
        print(f"🌐 Main Interface:     http://localhost:{self.demo_config['web_port']}")
        print(f"📚 SDK Documentation: http://localhost:{self.demo_config['web_port']}/sdk")
        print(f"🎮 Portal Demo:       http://localhost:{self.demo_config['web_port']}/portal")
        print(f"📊 API Docs:          http://localhost:{self.demo_config['web_port']}/docs")
        print(f"🌸 FL Server:         localhost:{self.demo_config['fl_server_port']}")
        print("="*60)
        print("🔑 DEMO FEATURES:")
        print("  • Flower Federated Learning with encryption")
        print("  • OpenAI + Anthropic AI integration")
        print("  • Two-device secure training simulation")
        print("  • Confidential LLM offline communication")
        print("  • Interactive web portal with all functions")
        print("  • Complete SDK documentation")
        print("="*60)
        print("💡 USAGE TIPS:")
        print("  • Visit the portal for interactive demos")
        print("  • Check the SDK docs for implementation details")
        print("  • Use the chat interface to test AI integration")
        print("  • Monitor the two-device demo progress")
        print("="*60)

    async def monitor_demo(self):
        """Monitor demo status and provide updates"""
        start_time = time.time()

        while self.demo_active:
            current_time = time.time()
            elapsed = current_time - start_time

            # Status update every 30 seconds
            if int(elapsed) % 30 == 0:
                status = await self.get_demo_status()
                print(f"\n📊 Demo Status (Running {elapsed:.0f}s):")
                print(f"   Web Server: {'✅ Active' if status['web_active'] else '❌ Inactive'}")
                print(f"   FL Server: {'✅ Active' if status['fl_active'] else '❌ Inactive'}")
                print(f"   Active Processes: {len([p for p in self.processes.values() if p.poll() is None])}")

            await asyncio.sleep(1)

    async def get_demo_status(self) -> Dict:
        """Get current demo status"""
        status = {
            "web_active": "web_server" in self.processes and self.processes["web_server"].poll() is None,
            "fl_active": "fl_server" in self.processes and self.processes["fl_server"].poll() is None,
            "connections": 0,
            "features_active": self.demo_config["features"],
            "uptime": time.time(),
            "active_processes": len([p for p in self.processes.values() if p.poll() is None])
        }

        return status

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def shutdown(self):
        """Gracefully shutdown all demo components"""
        print("🔄 Shutting down demo components...")

        self.demo_active = False

        # Stop all processes
        for name, process in self.processes.items():
            if process.poll() is None:
                process.terminate()
                print(f"✅ {name} stopped")

        print("🎉 Demo shutdown complete")

    async def run_complete_demo(self):
        """Run the complete integrated demo"""
        try:
            self.demo_active = True

            # Setup
            self.print_banner()

            if not self.check_dependencies():
                return

            self.setup_environment()
            self.setup_signal_handlers()

            # Start core components
            await self.start_web_interface()

            if self.demo_config["features"]["flower_fl"]:
                await self.start_fl_server()

            # Start feature demos
            await self.run_two_device_demo()
            await self.run_confidential_llm_demo()

            # Show information
            self.show_demo_info()

            # Start monitoring
            print("\n🚀 Complete demo launched successfully!")
            print("📊 Monitoring demo status... (Ctrl+C to stop)")

            await self.monitor_demo()

        except KeyboardInterrupt:
            print("\n🛑 Demo interrupted by user")
        except Exception as e:
            print(f"❌ Demo error: {e}")
            logging.exception("Demo error")
        finally:
            await self.shutdown()

async def main():
    """Main entry point"""
    launcher = CompleteDemoLauncher()
    await launcher.run_complete_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        sys.exit(1)