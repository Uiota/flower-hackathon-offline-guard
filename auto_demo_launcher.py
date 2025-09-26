#!/usr/bin/env python3
"""
Auto Demo Launcher for UIOTA Offline Guard

One-click launcher that automatically:
1. Sets up the environment
2. Downloads all dependencies
3. Configures all services
4. Starts all demos
5. Saves progress continuously

Just run this file and everything starts automatically!
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Import our custom agents
sys.path.append(str(Path(__file__).parent / "agents"))

try:
    from auto_save_agent import create_auto_save_agent
    from smooth_setup_agent import create_smooth_setup_agent
except ImportError as e:
    print(f"Error importing agents: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

class AutoDemoLauncher:
    """
    Master orchestrator that coordinates all agents for seamless demo startup.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the auto demo launcher.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.logger = self._setup_logging()

        # Initialize agents
        self.auto_save_agent = None
        self.setup_agent = None

        # State tracking
        self.startup_complete = False
        self.services_running = False

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the launcher."""
        # Create logs directory
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Configure logging
        log_file = logs_dir / f"auto_demo_{int(time.time())}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Auto Demo Launcher starting - logs: {log_file}")
        return logger

    async def initialize_agents(self) -> bool:
        """Initialize all required agents."""
        try:
            self.logger.info("🤖 Initializing agents...")

            # Initialize auto-save agent
            self.auto_save_agent = create_auto_save_agent(save_interval=15)
            self.auto_save_agent.register_agent("demo_launcher", {
                "status": "initializing",
                "start_time": time.time(),
                "project_root": str(self.project_root)
            })

            # Initialize smooth setup agent
            self.setup_agent = create_smooth_setup_agent(self.project_root)

            # Start auto-save agent
            self.auto_save_agent.start()

            self.logger.info("✅ All agents initialized")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize agents: {e}")
            return False

    async def run_full_setup(self) -> bool:
        """Run the complete setup process."""
        try:
            self.logger.info("🚀 Starting full system setup...")

            # Update auto-save state
            if self.auto_save_agent:
                self.auto_save_agent.update_agent_state("demo_launcher", {
                    "status": "setting_up",
                    "phase": "system_setup",
                    "timestamp": time.time()
                })

            # Run smooth setup
            setup_success = await self.setup_agent.run_setup()

            if setup_success:
                self.logger.info("✅ Setup completed successfully")

                # Update state
                if self.auto_save_agent:
                    self.auto_save_agent.update_agent_state("demo_launcher", {
                        "status": "setup_complete",
                        "setup_success": True,
                        "timestamp": time.time()
                    })

                return True
            else:
                self.logger.warning("⚠️ Setup completed with warnings")
                return True  # Continue with limited functionality

        except Exception as e:
            self.logger.error(f"❌ Setup failed: {e}")
            return False

    async def start_demo_services(self) -> bool:
        """Start all demo services."""
        try:
            self.logger.info("🎬 Starting demo services...")

            # Update auto-save state
            if self.auto_save_agent:
                self.auto_save_agent.update_agent_state("demo_launcher", {
                    "status": "starting_services",
                    "timestamp": time.time()
                })

            # Check if start-demos.sh exists and is executable
            start_script = self.project_root / "start-demos.sh"
            if start_script.exists():
                self.logger.info("📜 Running start-demos.sh script...")

                import subprocess
                result = await asyncio.create_subprocess_exec(
                    str(start_script),
                    cwd=self.project_root,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await result.communicate()

                if result.returncode == 0:
                    self.logger.info("✅ Demo services started successfully")
                    self.services_running = True

                    # Log service outputs
                    if stdout:
                        self.logger.info(f"Service output: {stdout.decode()}")

                    # Update state
                    if self.auto_save_agent:
                        self.auto_save_agent.update_agent_state("demo_launcher", {
                            "status": "services_running",
                            "services_started": True,
                            "timestamp": time.time()
                        })

                    return True
                else:
                    self.logger.error(f"❌ Demo services failed to start: {stderr.decode()}")
                    return False

            else:
                self.logger.warning("📜 start-demos.sh not found, creating minimal demo...")
                return await self._start_minimal_demo()

        except Exception as e:
            self.logger.error(f"❌ Failed to start demo services: {e}")
            return False

    async def _start_minimal_demo(self) -> bool:
        """Start a minimal demo when full setup isn't available."""
        try:
            self.logger.info("🔧 Creating minimal demo setup...")

            # Create a simple web server
            import subprocess

            # Try to start a simple Python HTTP server
            result = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "http.server", "8080",
                cwd=self.project_root / "web-demo" if (self.project_root / "web-demo").exists() else self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Give it a moment to start
            await asyncio.sleep(2)

            if result.returncode is None:  # Still running
                self.logger.info("✅ Minimal web demo started on port 8080")
                self.services_running = True
                return True
            else:
                self.logger.error("❌ Failed to start minimal demo")
                return False

        except Exception as e:
            self.logger.error(f"❌ Minimal demo startup failed: {e}")
            return False

    async def show_demo_info(self) -> None:
        """Display information about running demos."""
        try:
            self.logger.info("📋 Demo Information")
            self.logger.info("=" * 50)

            if self.services_running:
                self.logger.info("🌐 Web Demo: http://localhost:8080")
                self.logger.info("🧠 ML Toolkit: http://localhost:8888 (if available)")
                self.logger.info("🤖 Discord Bot: Running in background")
                self.logger.info("")
                self.logger.info("🎯 Perfect for:")
                self.logger.info("  📚 Classmate collaboration")
                self.logger.info("  ✈️ Travel team coordination")
                self.logger.info("  🏆 Hackathon demonstrations")
                self.logger.info("  🛡️ Offline-first development")
            else:
                self.logger.info("⚠️ Services not fully running - check logs above")

            self.logger.info("")
            self.logger.info("🛑 To stop all demos: ./stop-demos.sh")
            self.logger.info("📊 Logs directory: ./logs/")

        except Exception as e:
            self.logger.error(f"❌ Failed to show demo info: {e}")

    async def monitor_services(self) -> None:
        """Monitor running services and maintain health."""
        try:
            self.logger.info("👁️ Starting service monitoring...")

            while self.services_running:
                # Update auto-save state
                if self.auto_save_agent:
                    self.auto_save_agent.update_agent_state("demo_launcher", {
                        "status": "monitoring",
                        "uptime": time.time() - self.auto_save_agent.get_agent_state("demo_launcher").get("start_time", time.time()),
                        "timestamp": time.time()
                    })

                # Check service health (basic port checks)
                healthy_services = await self._check_service_health()

                if not healthy_services:
                    self.logger.warning("⚠️ Some services may have stopped")

                # Wait before next check
                await asyncio.sleep(30)

        except asyncio.CancelledError:
            self.logger.info("🛑 Service monitoring stopped")
        except Exception as e:
            self.logger.error(f"❌ Service monitoring error: {e}")

    async def _check_service_health(self) -> bool:
        """Check if services are still running."""
        try:
            import socket

            # Check if ports are still in use (basic health check)
            ports_to_check = [8080, 8888]
            healthy_count = 0

            for port in ports_to_check:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        result = s.connect_ex(('localhost', port))
                        if result == 0:  # Port is open
                            healthy_count += 1
                except:
                    pass

            return healthy_count > 0

        except Exception:
            return False

    async def cleanup(self) -> None:
        """Clean up resources on shutdown."""
        try:
            self.logger.info("🧹 Cleaning up...")

            # Update final state
            if self.auto_save_agent:
                self.auto_save_agent.update_agent_state("demo_launcher", {
                    "status": "shutting_down",
                    "timestamp": time.time()
                })

                # Force final save
                self.auto_save_agent.force_save()

                # Stop auto-save agent
                self.auto_save_agent.stop()

            self.logger.info("✅ Cleanup completed")

        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")

    async def run(self) -> bool:
        """Run the complete auto demo launcher process."""
        try:
            self.logger.info("🛡️ UIOTA AUTO DEMO LAUNCHER")
            self.logger.info("=" * 40)
            self.logger.info("🚀 One-click setup and demo startup")
            self.logger.info("")

            # Step 1: Initialize agents
            if not await self.initialize_agents():
                return False

            # Step 2: Run full setup
            if not await self.run_full_setup():
                return False

            # Step 3: Start demo services
            if not await self.start_demo_services():
                return False

            # Step 4: Show demo information
            await self.show_demo_info()

            # Step 5: Monitor services
            self.startup_complete = True

            try:
                await self.monitor_services()
            except KeyboardInterrupt:
                self.logger.info("🛑 Shutdown requested by user")

            return True

        except Exception as e:
            self.logger.error(f"❌ Auto demo launcher failed: {e}")
            return False

        finally:
            await self.cleanup()

def print_banner():
    """Print a nice banner for the launcher."""
    banner = """
🛡️  UIOTA OFFLINE GUARD - AUTO DEMO LAUNCHER
═══════════════════════════════════════════════════

🤖 Automatically sets up everything you need:
   • Downloads all dependencies
   • Configures all services
   • Starts all demos
   • Saves progress continuously

🎯 Perfect for:
   📚 Classmate collaboration
   ✈️  Travel team coordination
   🏆 Hackathon demonstrations
   🛡️  Offline-first development

🚀 Just sit back and relax - everything happens automatically!

"""
    print(banner)

async def main():
    """Main entry point for the auto demo launcher."""
    print_banner()

    try:
        launcher = AutoDemoLauncher()
        success = await launcher.run()

        if success:
            print("\n🎉 All demos are running successfully!")
            print("🌐 Visit http://localhost:8080 to start using the system")
        else:
            print("\n⚠️ Setup completed with some issues")
            print("📋 Check the logs for more details")

    except KeyboardInterrupt:
        print("\n🛑 Launcher interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Make sure we're in the right directory
    script_dir = Path(__file__).parent
    if script_dir != Path.cwd():
        print(f"📁 Changing to project directory: {script_dir}")
        import os
        os.chdir(script_dir)

    # Run the launcher
    asyncio.run(main())