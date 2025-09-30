#!/usr/bin/env python3
"""
Complete Federated Learning Demo Launcher
Runs all components together for the full Off-Guard FL experience:
- Enhanced Dashboard with real-time monitoring
- FL Agent System with multiple clients
- Network simulation and security monitoring
- All previous demo variants for comparison
"""

import subprocess
import signal
import sys
import time
import os
import threading
import webbrowser
from pathlib import Path

# Set environment
os.environ["OFFLINE_MODE"] = "1"

class CompleteDemoLauncher:
    """Launches and manages all FL demo components."""

    def __init__(self):
        self.processes = []
        self.running = False

    def launch_component(self, name, command, delay=0):
        """Launch a demo component."""
        if delay > 0:
            time.sleep(delay)

        print(f"🚀 Starting {name}...")

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy()
            )

            self.processes.append({
                'name': name,
                'process': process,
                'command': ' '.join(command)
            })

            print(f"✅ {name} started (PID: {process.pid})")
            return True

        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return False

    def monitor_processes(self):
        """Monitor all running processes."""
        while self.running:
            time.sleep(5)

            for proc_info in self.processes:
                if proc_info['process'].poll() is not None:
                    print(f"⚠️  {proc_info['name']} stopped (exit code: {proc_info['process'].returncode})")

            time.sleep(10)

    def start_complete_demo(self):
        """Start the complete federated learning demo."""
        print("🎯 COMPLETE FEDERATED LEARNING DEMO LAUNCHER")
        print("🔒 Off-Guard Security • 🤖 Real Agents • 📊 Live Dashboard")
        print("=" * 70)

        self.running = True

        # Component 1: Enhanced Dashboard with Real Agents (Main Interface)
        success1 = self.launch_component(
            "Enhanced Dashboard + FL Agents",
            [sys.executable, "dashboard_with_agents.py"],
            delay=0
        )

        # Component 2: Fast Real Demo (Standalone FL Training)
        success2 = self.launch_component(
            "Fast Real FL Demo",
            [sys.executable, "demo_fast_real.py"],
            delay=3
        )

        # Component 3: Complete Working FL (Advanced Simulation)
        success3 = self.launch_component(
            "Complete Working FL",
            [sys.executable, "final-working-fl.py"],
            delay=5
        )

        # Component 4: Network FL Simulator (Network Analysis)
        success4 = self.launch_component(
            "Network FL Simulator",
            [sys.executable, "network-fl-simulator.py"],
            delay=7
        )

        # Component 5: Basic Demo (Lightweight Reference)
        success5 = self.launch_component(
            "Basic FL Demo",
            [sys.executable, "demo_basic.py"],
            delay=10
        )

        # Summary
        total_started = sum([success1, success2, success3, success4, success5])
        print(f"\n📊 Demo Launch Summary:")
        print(f"   ✅ Started: {total_started}/5 components")
        print(f"   🖥️  Total Processes: {len(self.processes)}")

        if total_started >= 3:
            print("🎉 Core demo components running successfully!")
        else:
            print("⚠️  Some components failed to start")

        # Wait for components to initialize
        print("\n⏳ Waiting for components to initialize...")
        time.sleep(8)

        # Display access information
        self.show_access_info()

        # Start monitoring
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()

        # Auto-open dashboard
        try:
            print("🌐 Opening dashboard in browser...")
            webbrowser.open('http://localhost:8081')
        except:
            pass

        return total_started

    def show_access_info(self):
        """Show how to access all demo components."""
        print("\n🔗 ACCESS INFORMATION")
        print("=" * 50)
        print("📊 Main Dashboard (Enhanced): http://localhost:8081")
        print("   • Real-time FL monitoring")
        print("   • Live agent metrics")
        print("   • Security status")
        print("   • Interactive controls")

        print("\n🚀 Demo Components Running:")
        for i, proc_info in enumerate(self.processes, 1):
            status = "🟢 Running" if proc_info['process'].poll() is None else "🔴 Stopped"
            print(f"   {i}. {proc_info['name']}: {status}")

        print("\n🛠️  API Endpoints:")
        print("   • GET  /api/metrics - Training metrics")
        print("   • GET  /api/status - System status")
        print("   • GET  /api/start-agents - Start FL agents")
        print("   • GET  /api/stop-agents - Stop FL agents")

        print("\n🧪 Test Commands:")
        print("   python3 test_full_system.py  # Comprehensive test")
        print("   curl http://localhost:8081/api/metrics | jq  # Live metrics")

    def run_interactive_mode(self):
        """Run interactive monitoring mode."""
        print("\n🎮 INTERACTIVE MODE")
        print("Commands: status, metrics, test, stop, help")
        print("Press Ctrl+C to exit")

        try:
            while self.running:
                try:
                    cmd = input("\nFL-Demo> ").strip().lower()

                    if cmd == "status":
                        self.show_status()
                    elif cmd == "metrics":
                        self.show_metrics()
                    elif cmd == "test":
                        self.run_test()
                    elif cmd == "stop":
                        print("Stopping all components...")
                        break
                    elif cmd == "help":
                        self.show_help()
                    elif cmd == "":
                        continue
                    else:
                        print(f"Unknown command: {cmd}. Type 'help' for options.")

                except EOFError:
                    break
                except KeyboardInterrupt:
                    print("\nExiting interactive mode...")
                    break

        except Exception as e:
            print(f"Interactive mode error: {e}")

    def show_status(self):
        """Show current status of all components."""
        print("\n📊 SYSTEM STATUS")
        print("-" * 30)

        active = 0
        for proc_info in self.processes:
            if proc_info['process'].poll() is None:
                print(f"🟢 {proc_info['name']}: Running (PID: {proc_info['process'].pid})")
                active += 1
            else:
                print(f"🔴 {proc_info['name']}: Stopped")

        print(f"\nTotal: {active}/{len(self.processes)} components active")

    def show_metrics(self):
        """Show current FL metrics."""
        print("\n📈 CURRENT METRICS")
        print("-" * 30)

        try:
            import requests
            response = requests.get("http://localhost:8081/api/metrics", timeout=3)
            metrics = response.json()

            global_metrics = metrics.get("global_metrics", {})
            print(f"Round: {global_metrics.get('round')}")
            print(f"Accuracy: {global_metrics.get('accuracy', 0):.2%}")
            print(f"Loss: {global_metrics.get('loss', 0):.3f}")
            print(f"Active Clients: {global_metrics.get('active_clients')}/{global_metrics.get('total_clients')}")

        except Exception as e:
            print(f"❌ Could not fetch metrics: {e}")

    def run_test(self):
        """Run system test."""
        print("\n🧪 Running system test...")
        try:
            result = subprocess.run([sys.executable, "test_full_system.py"],
                                  capture_output=True, text=True, timeout=30)
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
        except Exception as e:
            print(f"Test failed: {e}")

    def show_help(self):
        """Show help information."""
        print("\n📚 HELP - Available Commands:")
        print("  status  - Show status of all components")
        print("  metrics - Show current FL training metrics")
        print("  test    - Run comprehensive system test")
        print("  stop    - Stop all components and exit")
        print("  help    - Show this help message")

    def cleanup(self):
        """Clean up all processes."""
        print("\n🧹 Cleaning up processes...")

        self.running = False

        for proc_info in self.processes:
            try:
                if proc_info['process'].poll() is None:
                    print(f"⏹️  Stopping {proc_info['name']}...")
                    proc_info['process'].terminate()

                    # Wait for graceful shutdown
                    try:
                        proc_info['process'].wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"🔥 Force killing {proc_info['name']}...")
                        proc_info['process'].kill()

            except Exception as e:
                print(f"Error stopping {proc_info['name']}: {e}")

        print("✅ All processes stopped")

def main():
    """Main entry point."""
    launcher = CompleteDemoLauncher()

    def signal_handler(sig, frame):
        print("\n⏹️  Shutdown signal received...")
        launcher.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start all components
        components_started = launcher.start_complete_demo()

        if components_started == 0:
            print("❌ No components started successfully")
            return 1

        # Run interactive mode
        launcher.run_interactive_mode()

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
    finally:
        launcher.cleanup()

    print("🏁 Complete FL Demo finished")
    return 0

if __name__ == "__main__":
    sys.exit(main())