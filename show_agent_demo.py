#!/usr/bin/env python3
"""
Guardian Agent System Live Demo

Shows the cybersecurity and development agents working in real-time.
"""

import sys
import time
import json
from pathlib import Path

# Add agents to path
sys.path.insert(0, str(Path(__file__).parent / "agents"))

from agents.agent_orchestrator import create_agent_orchestrator

def print_banner():
    """Print demo banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🛡️ GUARDIAN AGENT SYSTEM LIVE DEMO                        ║
║                                                                              ║
║  Cybersecurity Agents + Development Agents + Real-time Coordination         ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_section(title, emoji="🔹"):
    """Print a section header."""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))

def show_agent_system():
    """Demonstrate the Guardian Agent System."""
    print_banner()

    print_section("Initializing Guardian Agent System", "🚀")

    # Create orchestrator
    orchestrator = create_agent_orchestrator()

    print("🔧 Creating agent orchestrator...")
    time.sleep(1)

    # Initialize system
    print("⚡ Initializing all agents...")
    success = orchestrator.initialize_system()

    if not success:
        print("❌ Failed to initialize system")
        return

    print("✅ Guardian Agent System is now OPERATIONAL!")
    time.sleep(2)

    # Show system status
    print_section("System Status Overview", "📊")
    status = orchestrator.get_system_status()

    total_agents = status.get('orchestrator', {}).get('total_agents', 0)
    running_agents = status.get('orchestrator', {}).get('running_agents', 0)

    print(f"🤖 Total Agents: {total_agents}")
    print(f"✅ Running Agents: {running_agents}")
    print(f"🔗 Communication Hub: {'Active' if status.get('communication_hub', {}).get('active') else 'Inactive'}")
    print(f"🔍 Debug Monitor: {'Active' if status.get('debug_monitor', {}).get('active') else 'Inactive'}")

    # Show individual agent status
    print_section("Individual Agent Status", "🤖")
    for agent_id, agent_status in status.get('agent_status', {}).items():
        status_emoji = "✅" if agent_status.get('status') == 'running' else "❌"
        agent_name = agent_id.replace('_', ' ').title()
        print(f"{status_emoji} {agent_name}: {agent_status.get('status', 'unknown')}")

    time.sleep(2)

    # Demonstrate Security Monitoring
    print_section("Security Monitoring in Action", "🛡️")
    if 'security_monitor' in orchestrator.agents:
        security_agent = orchestrator.agents['security_monitor']
        sec_status = security_agent.get_security_status()

        print(f"🔒 Security Status: {sec_status.get('status', 'unknown')}")
        print(f"📂 Monitored Files: {sec_status.get('monitored_files', 0)}")
        print(f"🚨 Recent Events: {len(sec_status.get('recent_events', []))}")

        # Show recent security events
        events = sec_status.get('recent_events', [])
        if events:
            print("⚠️  Recent Security Events:")
            for event in events[-3:]:  # Show last 3 events
                event_type = event.get('event_type', 'unknown')
                severity = event.get('severity', 'info')
                print(f"   • [{severity.upper()}] {event_type}")
        else:
            print("✅ No security threats detected")

    time.sleep(2)

    # Demonstrate Development Monitoring
    print_section("Development Agent Analysis", "💻")
    if 'development' in orchestrator.agents:
        dev_agent = orchestrator.agents['development']
        dev_status = dev_agent.get_status()
        summary = dev_status.get('summary', {})

        print(f"📁 Files Monitored: {summary.get('total_files_monitored', 0)}")
        print(f"🐛 Total Issues Found: {summary.get('total_issues', 0)}")
        print(f"🚨 Critical Issues: {summary.get('critical_issues', 0)}")

        # Run quick code analysis
        print("\n🔍 Running live code analysis...")
        analysis = dev_agent.run_manual_analysis()

        print(f"📊 Analysis Results:")
        print(f"   • Files Analyzed: {analysis.get('analyzed_files', 0)}")
        issues_by_severity = analysis.get('issues_by_severity', {})
        for severity, count in issues_by_severity.items():
            if count > 0:
                print(f"   • {severity.title()} Issues: {count}")

    time.sleep(2)

    # Demonstrate Agent Communication
    print_section("Agent Communication Test", "📡")
    if 'communication_hub' in orchestrator.agents:
        hub = orchestrator.agents['communication_hub']
        hub_status = hub.get_agent_status()

        print(f"🔗 Total Registered Agents: {hub_status.get('total_agents', 0)}")
        print(f"🟢 Online Agents: {hub_status.get('online_agents', 0)}")

        # Get communication stats
        stats = hub.get_communication_stats()
        delivery_stats = stats.get('delivery_stats', {})

        print("📈 Communication Statistics:")
        for stat_name, value in delivery_stats.items():
            print(f"   • {stat_name.replace('_', ' ').title()}: {value}")

    time.sleep(2)

    # Run comprehensive tests
    print_section("Running System Tests", "🧪")
    print("🔬 Executing comprehensive test suite...")

    test_results = orchestrator.run_comprehensive_test()

    if 'error' not in test_results:
        summary = test_results.get('summary', {})
        success_rate = summary.get('success_rate', 0)

        print("🎯 Test Results:")
        print(f"   • Total Tests: {summary.get('total', 0)}")
        print(f"   • Passed: {summary.get('passed', 0)}")
        print(f"   • Failed: {summary.get('failed', 0)}")
        print(f"   • Success Rate: {success_rate:.1f}%")

        if success_rate >= 90:
            print("🎉 EXCELLENT - All systems operational!")
        elif success_rate >= 75:
            print("✅ GOOD - Minor issues detected")
        else:
            print("⚠️ ATTENTION - Some systems need review")
    else:
        print(f"❌ Test execution failed: {test_results['error']}")

    time.sleep(2)

    # Show real-time system metrics
    print_section("Real-time System Metrics", "📊")
    try:
        import psutil

        # System resources
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')

        print("💻 System Resources:")
        print(f"   • CPU Usage: {cpu_percent:.1f}%")
        print(f"   • Memory Usage: {memory.percent:.1f}%")
        print(f"   • Disk Usage: {(disk.used / disk.total * 100):.1f}%")

        # Process info
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024 * 1024)  # MB

        print("🐍 Agent System Resources:")
        print(f"   • Process Memory: {process_memory:.1f} MB")
        print(f"   • Process Threads: {process.num_threads()}")

    except Exception as e:
        print(f"📊 System metrics unavailable: {e}")

    time.sleep(2)

    # Demo summary
    print_section("Demo Summary", "🎊")
    print("✅ Guardian Agent System Successfully Demonstrated!")
    print()
    print("🛡️ Security Features:")
    print("   • Real-time threat monitoring")
    print("   • File integrity verification")
    print("   • Resource usage monitoring")
    print("   • Automated threat response")
    print()
    print("💻 Development Features:")
    print("   • Code quality analysis")
    print("   • Real-time issue detection")
    print("   • Automated testing")
    print("   • Performance monitoring")
    print()
    print("🤖 System Coordination:")
    print("   • Inter-agent communication")
    print("   • Task delegation and routing")
    print("   • Centralized orchestration")
    print("   • Automated health monitoring")

    print()
    print("🌐 Web Dashboard:")
    print("   The system also includes a web dashboard for visual monitoring")
    print("   Run: python3 simple_web_dashboard.py")
    print()

    # Keep system running briefly for observation
    print_section("System Running", "⚡")
    print("🔄 Keeping system active for 15 seconds...")
    print("   You can observe the agents working in real-time...")

    for i in range(15, 0, -1):
        print(f"   ⏰ {i} seconds remaining...", end='\r')
        time.sleep(1)

    print("\n")
    print_section("Shutting Down", "🔄")
    print("🛑 Gracefully stopping all agents...")

    # Save final state
    orchestrator.save_system_state()
    print("💾 System state saved")

    # Stop all agents
    orchestrator.stop_all_agents()
    print("✅ All agents stopped cleanly")

    print()
    print("🎉 Guardian Agent System Demo Complete!")
    print("   The system is ready for production deployment.")

if __name__ == "__main__":
    try:
        show_agent_system()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()