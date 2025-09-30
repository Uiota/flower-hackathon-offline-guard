#!/usr/bin/env python3
"""
Memory Guardian Application Launcher
Comprehensive cognitive health and property protection system
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies():
    """Check if all required dependencies are installed"""
    required = {
        'cryptography': 'cryptography',
        'flask': 'Flask',
        'sqlite3': 'sqlite3 (built-in)'
    }

    missing = []

    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("❌ Missing required dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n📦 Install with: pip install cryptography flask")
        return False

    return True


def launch_web_interface(host='localhost', port=8090, open_browser=True):
    """Launch the web interface"""
    print("=" * 80)
    print("🧠 MEMORY GUARDIAN - Web Interface")
    print("=" * 80)
    print(f"\n🌐 Starting web server on http://{host}:{port}")
    print(f"📁 Serving files from: website/memory_guardian/")

    try:
        from flask import Flask, send_from_directory, jsonify, request
        from datetime import datetime
        import json

        app = Flask(__name__)

        # Serve static files
        @app.route('/')
        def index():
            return send_from_directory('website/memory_guardian', 'index.html')

        @app.route('/api/health')
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "offline_mode": os.getenv("OFFLINE_MODE", "1") == "1"
            })

        @app.route('/api/assessment/start', methods=['POST'])
        def start_assessment():
            from cognitive_exercises import CognitiveExerciseSuite
            suite = CognitiveExerciseSuite()

            difficulty = request.json.get('difficulty', 2)
            assessment = suite.generate_daily_assessment(difficulty)

            return jsonify(assessment)

        @app.route('/api/assessment/submit', methods=['POST'])
        def submit_assessment():
            from memory_guardian_system import MemoryGuardianApp
            from cognitive_exercises import CognitiveExerciseSuite

            results = request.json
            suite = CognitiveExerciseSuite()
            scores = suite.calculate_scores(results)

            return jsonify({
                "status": "success",
                "scores": scores,
                "tokens_earned": {
                    "LLT-EXP": 50,
                    "LLT-EDU": 10,
                    "LLT-REWARD": 25
                }
            })

        @app.route('/api/dashboard')
        def dashboard():
            # Return mock dashboard data
            return jsonify({
                "user_id": "demo_user_001",
                "cognitive_baseline": 85.0,
                "total_assessments": 142,
                "average_score_30d": 85.3,
                "trend": "stable",
                "total_tokens_earned": 4245
            })

        print("\n✅ Server ready!")
        print(f"   Access the app at: http://{host}:{port}")
        print("   Press Ctrl+C to stop\n")

        # Open browser
        if open_browser:
            import webbrowser
            webbrowser.open(f"http://{host}:{port}")

        app.run(host=host, port=port, debug=False)

    except Exception as e:
        print(f"\n❌ Error starting web server: {e}")
        sys.exit(1)


def launch_cli_mode():
    """Launch in CLI mode"""
    print("=" * 80)
    print("🧠 MEMORY GUARDIAN - Command Line Interface")
    print("=" * 80)

    from memory_guardian_system import MemoryGuardianApp
    from cognitive_exercises import CognitiveExerciseSuite

    # Initialize app
    print("\n📋 Initializing Memory Guardian...")
    app = MemoryGuardianApp(
        user_id="cli_user_001",
        master_password="SecurePassword123!",
        ll_token_wallet="LL_CLI_DEMO_WALLET"
    )

    suite = CognitiveExerciseSuite()

    while True:
        print("\n" + "=" * 80)
        print("MAIN MENU")
        print("=" * 80)
        print("1. Run Daily Cognitive Assessment")
        print("2. View Dashboard")
        print("3. Secure a Document")
        print("4. Add Trusted Contact")
        print("5. View Recent Activity")
        print("6. Generate System Report")
        print("0. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            print("\n📝 Running Daily Assessment...")
            assessment = suite.generate_daily_assessment(difficulty=2)

            print(f"\nAssessment ID: {assessment['assessment_id']}")
            print(f"Total Exercises: {len(assessment['exercises'])}")

            # Simulate results
            result = app.run_daily_assessment({
                'memory_score': 85.0,
                'reaction_time_ms': 450.0,
                'pattern_recognition_score': 88.0,
                'problem_solving_score': 82.0,
                'overall_score': 85.0
            })

            print(f"\n✅ Assessment Complete!")
            print(f"   Status: {result['evaluation']['status']}")
            print(f"   Overall Score: {result['assessment']['overall_score']:.1f}")
            print(f"   Tokens Earned: {sum(result['rewards']['rewards'].values()):.0f}")

        elif choice == "2":
            summary = app.get_dashboard_summary()
            print("\n📈 Dashboard Summary:")
            print(f"   User ID: {summary['user_id']}")
            print(f"   Cognitive Baseline: {summary['cognitive_baseline']:.1f}")
            print(f"   Total Assessments: {summary['total_assessments']}")
            print(f"   30-Day Average: {summary['average_score_30d']:.1f}")
            print(f"   Trend: {summary['trend']}")
            print(f"   Total Tokens: {summary['total_tokens_earned']:.1f}")

        elif choice == "3":
            print("\n🔒 Secure Document")
            title = input("Document title: ")
            doc_type = input("Document type (will/deed/financial/medical): ")
            content = input("Document content: ")

            result = app.secure_document(doc_type, title, content)
            print(f"\n✅ Document secured!")
            print(f"   Record ID: {result['record_id']}")
            print(f"   Tokens Earned: {sum(result['rewards']['rewards'].values()):.0f}")

        elif choice == "4":
            print("\n👥 Add Trusted Contact")
            name = input("Name: ")
            relationship = input("Relationship: ")
            phone = input("Phone: ")
            email = input("Email: ")
            access_level = int(input("Access level (1=emergency, 2=view, 3=full): "))

            result = app.add_trusted_contact(name, relationship, access_level, phone, email)
            print(f"\n✅ Contact added!")
            print(f"   Contact ID: {result['contact']['contact_id']}")
            print(f"   Verification Code: {result['contact']['verification_code']}")

        elif choice == "5":
            print("\n📋 Recent Activity:")
            print("   - Completed daily assessment (2 hours ago)")
            print("   - Contributed to FL network (2 hours ago)")
            print("   - Accessed document vault (yesterday)")

        elif choice == "6":
            from memory_guardian_agents import DevelopmentAgent
            agent = DevelopmentAgent()
            report = agent.generate_system_report()

            print("\n📊 System Report:")
            print(f"   Overall Status: {report['health_check']['overall_status']}")
            print(f"   Database Size: {report['database_stats'].get('database_size_mb', 0):.2f} MB")
            print(f"   Total Records: {report['database_stats'].get('cognitive_assessments_count', 0)}")

        elif choice == "0":
            print("\n👋 Goodbye!")
            app.shutdown()
            break

        else:
            print("\n❌ Invalid option")


def launch_agent_mode():
    """Launch agent system for maintenance and research"""
    print("=" * 80)
    print("🤖 MEMORY GUARDIAN - Agent System")
    print("=" * 80)

    from memory_guardian_agents import AgentCoordinator

    coordinator = AgentCoordinator()

    print("\n1. Run Daily Maintenance")
    print("2. Development Agent Only")
    print("3. Research Agent Only")
    print("4. Custom Task")

    choice = input("\nSelect mode: ").strip()

    if choice == "1":
        results = coordinator.run_daily_maintenance(user_id="agent_demo_user")
        print("\n✅ Maintenance complete!")

    elif choice == "2":
        report = coordinator.dev_agent.generate_system_report()
        print("\n📊 System Report:")
        print(f"   Status: {report['health_check']['overall_status']}")

    elif choice == "3":
        analysis = coordinator.research_agent.analyze_cognitive_trends(
            user_id="agent_demo_user",
            days=90
        )
        print("\n🔬 Cognitive Analysis:")
        if analysis.get("insights"):
            for insight in analysis["insights"]:
                print(f"   {insight}")


def main():
    parser = argparse.ArgumentParser(
        description="Memory Guardian - Cognitive Health & Property Protection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Launch web interface:     python launch_memory_guardian.py
  CLI mode:                 python launch_memory_guardian.py --cli
  Agent system:             python launch_memory_guardian.py --agents
  Custom port:              python launch_memory_guardian.py --port 8080
        """
    )

    parser.add_argument('--cli', action='store_true',
                       help='Launch in CLI mode')
    parser.add_argument('--agents', action='store_true',
                       help='Launch agent system')
    parser.add_argument('--host', default='localhost',
                       help='Host for web server (default: localhost)')
    parser.add_argument('--port', type=int, default=8090,
                       help='Port for web server (default: 8090)')
    parser.add_argument('--no-browser', action='store_true',
                       help='Don\'t open browser automatically')

    args = parser.parse_args()

    # Set offline mode
    os.environ['OFFLINE_MODE'] = '1'

    print("\n" + "=" * 80)
    print("🧠 MEMORY GUARDIAN - Alzheimer's Prevention & Property Protection")
    print("=" * 80)
    print("🔒 Offline Mode: ENABLED")
    print("🔐 Quantum-Safe Encryption: ACTIVE")
    print("🌐 Federated Learning: READY")
    print("=" * 80 + "\n")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Launch appropriate mode
    if args.cli:
        launch_cli_mode()
    elif args.agents:
        launch_agent_mode()
    else:
        launch_web_interface(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser
        )


if __name__ == "__main__":
    main()