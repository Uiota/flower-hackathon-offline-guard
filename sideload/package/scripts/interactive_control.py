#!/usr/bin/env python3
"""
Interactive Guardian Agent Control Interface

Real-time command-line interface to interact with and control
the Guardian Agent System.
"""

import sys
import time
import json
from pathlib import Path

# Add agents to path
sys.path.insert(0, str(Path(__file__).parent / "agents"))

from agents.agent_orchestrator import create_agent_orchestrator

class InteractiveControl:
    """Interactive control interface for Guardian agents."""

    def __init__(self):
        self.orchestrator = None
        self.running = False

    def print_banner(self):
        """Print the control interface banner."""
        banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                   🎛️ GUARDIAN AGENT INTERACTIVE CONTROL                       ║
║                                                                               ║
║  Real-time command and control interface for cybersecurity and dev agents    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def print_menu(self):
        """Print the main menu."""
        print("\n🎛️ CONTROL MENU:")
        print("─" * 50)
        print("1. 🚀 Start Agent System")
        print("2. ⏹️  Stop Agent System")
        print("3. 📊 Show System Status")
        print("4. 🛡️ Security Monitoring")
        print("5. 💻 Development Analysis")
        print("6. 📡 Communication Status")
        print("7. 🧪 Run Tests")
        print("8. 🔄 Restart Agent")
        print("9. 📈 Live Metrics")
        print("0. ❌ Exit")
        print("─" * 50)

    def get_user_choice(self):
        """Get user menu choice."""
        try:
            choice = input("\n🎯 Enter your choice (0-9): ").strip()
            return choice
        except KeyboardInterrupt:
            return "0"

    def start_system(self):
        """Start the Guardian Agent System."""
        if self.orchestrator:
            print("⚠️  System is already running!")
            return

        print("\n🚀 Starting Guardian Agent System...")
        print("═" * 50)

        self.orchestrator = create_agent_orchestrator()
        success = self.orchestrator.initialize_system()

        if success:
            self.running = True
            print("✅ Guardian Agent System is now OPERATIONAL!")

            # Show quick status
            time.sleep(1)
            status = self.orchestrator.get_system_status()
            total_agents = status.get('orchestrator', {}).get('total_agents', 0)
            running_agents = status.get('orchestrator', {}).get('running_agents', 0)

            print(f"🤖 Agents Started: {running_agents}/{total_agents}")
            print("🔗 All systems ready for operation")
        else:
            print("❌ Failed to start the system!")

    def stop_system(self):
        """Stop the Guardian Agent System."""
        if not self.orchestrator:
            print("⚠️  System is not running!")
            return

        print("\n⏹️ Stopping Guardian Agent System...")
        print("═" * 50)

        self.orchestrator.stop_all_agents()
        self.orchestrator = None
        self.running = False

        print("✅ System stopped successfully")
        print("💾 All agent states have been saved")

    def show_system_status(self):
        """Show comprehensive system status."""
        if not self.orchestrator:
            print("❌ System is not running! Start it first.")
            return

        print("\n📊 SYSTEM STATUS REPORT")
        print("═" * 50)

        status = self.orchestrator.get_system_status()

        # Orchestrator status
        orch_status = status.get('orchestrator', {})
        print(f"🎛️ Orchestrator: {'🟢 Running' if orch_status.get('running') else '🔴 Stopped'}")
        print(f"🤖 Total Agents: {orch_status.get('total_agents', 0)}")
        print(f"✅ Running Agents: {orch_status.get('running_agents', 0)}")

        # Agent details
        print("\n🤖 AGENT STATUS:")
        agent_status = status.get('agent_status', {})

        agent_icons = {
            'communication_hub': '📡',
            'debug_monitor': '🔍',
            'auto_save': '💾',
            'security_monitor': '🛡️',
            'development': '💻'
        }

        for agent_id, agent_data in agent_status.items():
            icon = agent_icons.get(agent_id, '🤖')
            status_text = agent_data.get('status', 'unknown')
            status_emoji = "🟢" if status_text == 'running' else "🔴" if status_text == 'stopped' else "🟡"

            agent_name = agent_id.replace('_', ' ').title()
            print(f"   {icon} {agent_name}: {status_emoji} {status_text}")

        # Communication hub details
        comm_status = status.get('communication_hub', {})
        if comm_status.get('active'):
            stats = comm_status.get('stats', {})
            print(f"\n📡 Communication Hub:")
            print(f"   • Active Agents: {stats.get('total_agents', 0)}")
            print(f"   • Message History: {stats.get('message_history_size', 0)}")

    def show_security_monitoring(self):
        """Show security monitoring details."""
        if not self.orchestrator or 'security_monitor' not in self.orchestrator.agents:
            print("❌ Security monitor not available!")
            return

        print("\n🛡️ SECURITY MONITORING REPORT")
        print("═" * 50)

        security_agent = self.orchestrator.agents['security_monitor']
        sec_status = security_agent.get_security_status()

        print(f"🔒 Status: {sec_status.get('status', 'unknown')}")
        print(f"📂 Monitored Files: {sec_status.get('monitored_files', 0)}")
        print(f"🏗️ Baselines Established: {'✅ Yes' if sec_status.get('baselines_established') else '❌ No'}")

        # Recent events
        events = sec_status.get('recent_events', [])
        print(f"\n🚨 Recent Security Events ({len(events)}):")

        if events:
            for i, event in enumerate(events[-5:], 1):  # Show last 5
                event_type = event.get('event_type', 'unknown')
                severity = event.get('severity', 'info')
                description = event.get('description', 'No description')

                severity_icon = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🔵',
                    'info': '⚪'
                }.get(severity, '⚪')

                print(f"   {i}. {severity_icon} [{severity.upper()}] {description}")
        else:
            print("   ✅ No security events detected")

        # Show real-time system metrics
        try:
            import psutil
            print(f"\n📊 System Metrics:")
            print(f"   • CPU Usage: {psutil.cpu_percent():.1f}%")
            print(f"   • Memory Usage: {psutil.virtual_memory().percent:.1f}%")
            print(f"   • Disk Usage: {(psutil.disk_usage('/').used / psutil.disk_usage('/').total * 100):.1f}%")
        except:
            print("   📊 System metrics unavailable")

    def show_development_analysis(self):
        """Show development agent analysis."""
        if not self.orchestrator or 'development' not in self.orchestrator.agents:
            print("❌ Development agent not available!")
            return

        print("\n💻 DEVELOPMENT ANALYSIS REPORT")
        print("═" * 50)

        dev_agent = self.orchestrator.agents['development']
        dev_status = dev_agent.get_status()
        summary = dev_status.get('summary', {})

        print(f"📁 Files Monitored: {summary.get('total_files_monitored', 0)}")
        print(f"🐛 Total Issues: {summary.get('total_issues', 0)}")
        print(f"🚨 Critical Issues: {summary.get('critical_issues', 0)}")

        # Run live analysis
        print("\n🔍 Running live code analysis...")
        analysis = dev_agent.run_manual_analysis()

        print(f"\n📊 Analysis Results:")
        print(f"   • Files Analyzed: {analysis.get('analyzed_files', 0)}")

        issues_by_severity = analysis.get('issues_by_severity', {})
        total_issues = sum(issues_by_severity.values())

        if total_issues > 0:
            print(f"   • Total Issues Found: {total_issues}")
            for severity, count in issues_by_severity.items():
                if count > 0:
                    severity_icon = {
                        'critical': '🔴',
                        'high': '🟠',
                        'medium': '🟡',
                        'low': '🔵'
                    }.get(severity, '⚪')
                    print(f"   • {severity_icon} {severity.title()} Issues: {count}")
        else:
            print("   ✅ No issues found")

        # Code quality assessment
        critical_issues = issues_by_severity.get('critical', 0)
        if critical_issues == 0:
            print("\n✅ Code Quality: EXCELLENT")
        elif critical_issues < 5:
            print("\n🟡 Code Quality: GOOD")
        else:
            print("\n🔴 Code Quality: NEEDS ATTENTION")

    def show_communication_status(self):
        """Show communication hub status."""
        if not self.orchestrator or 'communication_hub' not in self.orchestrator.agents:
            print("❌ Communication hub not available!")
            return

        print("\n📡 COMMUNICATION STATUS")
        print("═" * 50)

        hub = self.orchestrator.agents['communication_hub']
        hub_status = hub.get_agent_status()
        stats = hub.get_communication_stats()

        print(f"🔗 Total Registered Agents: {hub_status.get('total_agents', 0)}")
        print(f"🟢 Online Agents: {hub_status.get('online_agents', 0)}")

        # Communication statistics
        delivery_stats = stats.get('delivery_stats', {})
        print(f"\n📈 Communication Statistics:")

        for stat_name, value in delivery_stats.items():
            stat_display = stat_name.replace('_', ' ').title()
            print(f"   • {stat_display}: {value}")

        # Show agent details
        agents = hub_status.get('agents', {})
        if agents:
            print(f"\n🤖 Registered Agents:")
            for agent_id, agent_info in agents.items():
                status_emoji = "🟢" if agent_info.get('status') == 'online' else "🔴"
                print(f"   • {status_emoji} {agent_id}: {agent_info.get('status', 'unknown')}")

    def run_tests(self):
        """Run comprehensive system tests."""
        if not self.orchestrator:
            print("❌ System is not running! Start it first.")
            return

        print("\n🧪 RUNNING COMPREHENSIVE TESTS")
        print("═" * 50)

        print("🔬 Executing test suite...")
        test_results = self.orchestrator.run_comprehensive_test()

        if 'error' in test_results:
            print(f"❌ Test execution failed: {test_results['error']}")
            return

        summary = test_results.get('summary', {})
        success_rate = summary.get('success_rate', 0)

        print(f"\n🎯 Test Results:")
        print(f"   • Total Tests: {summary.get('total', 0)}")
        print(f"   • Passed: {summary.get('passed', 0)}")
        print(f"   • Failed: {summary.get('failed', 0)}")
        print(f"   • Errors: {summary.get('errors', 0)}")
        print(f"   • Success Rate: {success_rate:.1f}%")

        # Assessment
        if success_rate >= 95:
            print("\n🎉 EXCELLENT - All systems performing optimally!")
        elif success_rate >= 85:
            print("\n✅ GOOD - Systems operational with minor issues")
        elif success_rate >= 70:
            print("\n🟡 FAIR - Some systems need attention")
        else:
            print("\n🔴 POOR - Critical issues detected")

    def restart_agent(self):
        """Restart a specific agent."""
        if not self.orchestrator:
            print("❌ System is not running! Start it first.")
            return

        print("\n🔄 RESTART AGENT")
        print("═" * 50)

        # Show available agents
        status = self.orchestrator.get_system_status()
        agent_status = status.get('agent_status', {})

        print("Available agents:")
        agents = list(agent_status.keys())
        for i, agent_id in enumerate(agents, 1):
            agent_name = agent_id.replace('_', ' ').title()
            status_text = agent_status[agent_id].get('status', 'unknown')
            print(f"   {i}. {agent_name} ({status_text})")

        try:
            choice = input("\nEnter agent number to restart (or 0 to cancel): ").strip()

            if choice == "0":
                return

            agent_index = int(choice) - 1
            if 0 <= agent_index < len(agents):
                agent_id = agents[agent_index]
                agent_name = agent_id.replace('_', ' ').title()

                print(f"\n🔄 Restarting {agent_name}...")
                success = self.orchestrator.restart_agent(agent_id)

                if success:
                    print(f"✅ {agent_name} restarted successfully")
                else:
                    print(f"❌ Failed to restart {agent_name}")
            else:
                print("❌ Invalid selection")

        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input")

    def show_live_metrics(self):
        """Show live system metrics."""
        if not self.orchestrator:
            print("❌ System is not running! Start it first.")
            return

        print("\n📈 LIVE SYSTEM METRICS")
        print("═" * 50)
        print("Press Ctrl+C to stop monitoring\n")

        try:
            import psutil

            while True:
                # Clear previous metrics (simple version)
                print("\r" + " " * 80, end="\r")

                # Get current metrics
                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()

                # Get agent status
                status = self.orchestrator.get_system_status()
                running_agents = status.get('orchestrator', {}).get('running_agents', 0)
                total_agents = status.get('orchestrator', {}).get('total_agents', 0)

                # Display metrics
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] 🖥️ CPU: {cpu:5.1f}% | 🧠 Memory: {memory.percent:5.1f}% | 🤖 Agents: {running_agents}/{total_agents}", end="")

                time.sleep(2)

        except KeyboardInterrupt:
            print("\n\n📊 Live monitoring stopped")
        except ImportError:
            print("❌ System monitoring unavailable (psutil not installed)")

    def run(self):
        """Run the interactive control interface."""
        self.print_banner()

        try:
            while True:
                self.print_menu()
                choice = self.get_user_choice()

                if choice == "1":
                    self.start_system()
                elif choice == "2":
                    self.stop_system()
                elif choice == "3":
                    self.show_system_status()
                elif choice == "4":
                    self.show_security_monitoring()
                elif choice == "5":
                    self.show_development_analysis()
                elif choice == "6":
                    self.show_communication_status()
                elif choice == "7":
                    self.run_tests()
                elif choice == "8":
                    self.restart_agent()
                elif choice == "9":
                    self.show_live_metrics()
                elif choice == "0":
                    break
                else:
                    print("❌ Invalid choice! Please select 0-9.")

                input("\n⏸️ Press Enter to continue...")

        except KeyboardInterrupt:
            print("\n🛑 Control interface interrupted")
        finally:
            if self.orchestrator:
                print("\n🔄 Shutting down system...")
                self.stop_system()

            print("👋 Thank you for using Guardian Agent Control!")

if __name__ == "__main__":
    control = InteractiveControl()
    control.run()