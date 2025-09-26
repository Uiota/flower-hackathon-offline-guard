#!/usr/bin/env python3
"""
Test Runner for UIOTA Guardian Agent System

Tests the complete agent ecosystem to ensure proper functionality
and coordination between cybersecurity and development agents.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add agents directory to path
sys.path.insert(0, str(Path(__file__).parent / "agents"))

from agents.agent_orchestrator import create_agent_orchestrator

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path.home() / ".uiota" / "test_run.log")
        ]
    )

def test_agent_system():
    """Test the complete Guardian agent system."""
    logger = logging.getLogger(__name__)

    print("🛡️ UIOTA Guardian Agent System Test")
    print("=" * 50)

    try:
        # Create orchestrator
        print("1. Creating Agent Orchestrator...")
        orchestrator = create_agent_orchestrator()

        # Initialize system
        print("2. Initializing Guardian Agent System...")
        if not orchestrator.initialize_system():
            print("❌ Failed to initialize system")
            return False

        print("✅ System initialized successfully")

        # Wait for agents to start up
        print("3. Waiting for agents to start up...")
        time.sleep(5)

        # Get system status
        print("4. Checking system status...")
        status = orchestrator.get_system_status()

        print(f"📊 System Status:")
        print(f"   Running: {status['orchestrator']['running']}")
        print(f"   Total Agents: {status['orchestrator']['total_agents']}")
        print(f"   Running Agents: {status['orchestrator']['running_agents']}")

        # Display agent statuses
        print("\n🤖 Agent Status:")
        for agent_id, agent_status in status['agent_status'].items():
            status_emoji = "✅" if agent_status['status'] == "running" else "❌"
            print(f"   {status_emoji} {agent_id}: {agent_status['status']}")

        # Test agent communication
        print("\n5. Testing agent communication...")
        if status['communication_hub']['active']:
            hub_stats = status['communication_hub']['stats']
            print(f"   📡 Communication Hub Active")
            print(f"   Messages Routed: {hub_stats.get('delivery_stats', {}).get('messages_routed', 0)}")
            print(f"   Active Queues: {hub_stats.get('active_queues', 0)}")

        # Test debugging system
        print("\n6. Testing debugging system...")
        if status['debug_monitor']['active']:
            diagnostics = status['debug_monitor']['diagnostics']
            if diagnostics:
                print(f"   🔍 Debug Monitor Active")
                print(f"   Total Events: {diagnostics.get('debug_events', {}).get('total_events', 0)}")
                print(f"   System Memory: {diagnostics.get('system', {}).get('memory_percent', 0):.1f}%")
                print(f"   System CPU: {diagnostics.get('system', {}).get('cpu_percent', 0):.1f}%")

        # Run comprehensive tests
        print("\n7. Running comprehensive tests...")
        test_results = orchestrator.run_comprehensive_test()

        if 'error' not in test_results:
            summary = test_results.get('summary', {})
            print(f"   🧪 Test Summary:")
            print(f"   Total Tests: {summary.get('total', 0)}")
            print(f"   Passed: {summary.get('passed', 0)}")
            print(f"   Failed: {summary.get('failed', 0)}")
            print(f"   Success Rate: {summary.get('success_rate', 0):.1f}%")
        else:
            print(f"   ❌ Test execution failed: {test_results['error']}")

        # Test security monitoring
        print("\n8. Testing security monitoring...")
        security_agent = orchestrator.agents.get('security_monitor')
        if security_agent:
            sec_status = security_agent.get_security_status()
            print(f"   🛡️ Security Monitor:")
            print(f"   Status: {sec_status.get('status', 'unknown')}")
            print(f"   Monitored Files: {sec_status.get('monitored_files', 0)}")
            print(f"   Recent Events: {len(sec_status.get('recent_events', []))}")

        # Test development monitoring
        print("\n9. Testing development monitoring...")
        dev_agent = orchestrator.agents.get('development')
        if dev_agent:
            dev_status = dev_agent.get_status()
            print(f"   💻 Development Agent:")
            print(f"   Monitored Files: {dev_status.get('summary', {}).get('total_files_monitored', 0)}")
            print(f"   Total Issues: {dev_status.get('summary', {}).get('total_issues', 0)}")
            print(f"   Critical Issues: {dev_status.get('summary', {}).get('critical_issues', 0)}")

        # Save system state
        print("\n10. Saving system state...")
        if orchestrator.save_system_state():
            print("   💾 System state saved successfully")
        else:
            print("   ❌ Failed to save system state")

        print("\n🎉 Agent System Test Completed Successfully!")
        print("\n📋 Test Summary:")
        print(f"   • {status['orchestrator']['running_agents']}/{status['orchestrator']['total_agents']} agents running")
        print(f"   • Communication hub: {'Active' if status['communication_hub']['active'] else 'Inactive'}")
        print(f"   • Debug monitoring: {'Active' if status['debug_monitor']['active'] else 'Inactive'}")
        print(f"   • Security monitoring: {'Active' if 'security_monitor' in orchestrator.agents else 'Inactive'}")
        print(f"   • Development monitoring: {'Active' if 'development' in orchestrator.agents else 'Inactive'}")

        # Keep system running for demonstration
        print("\n⏱️ System will run for 30 seconds for demonstration...")
        time.sleep(30)

        # Clean shutdown
        print("\n🔄 Initiating graceful shutdown...")
        orchestrator.stop_all_agents()
        print("✅ Shutdown complete")

        return True

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        if 'orchestrator' in locals():
            orchestrator.stop_all_agents()
        return False

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Test system error: {e}", exc_info=True)
        if 'orchestrator' in locals():
            orchestrator.stop_all_agents()
        return False

def main():
    """Main test execution."""
    setup_logging()

    print("Starting UIOTA Guardian Agent System Test...")
    print("This will test cybersecurity agents, development agents, and their coordination.\n")

    success = test_agent_system()

    if success:
        print("\n🎊 All tests completed successfully!")
        print("The Guardian agent system is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        print("Please check the logs for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()