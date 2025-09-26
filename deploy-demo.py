#!/usr/bin/env python3
"""
Deploy Demo - Main entry point for Flower Off-Guard UIOTA demo deployment.

This script provides a user-friendly interface to the deployment automation system,
with options for interactive and non-interactive deployment modes.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add deployment agents to path
sys.path.insert(0, str(Path(__file__).parent / "deployment-agents"))

try:
    from deployment_orchestrator_agent import DeploymentOrchestratorAgent
except ImportError as e:
    print(f"❌ Error importing deployment agents: {e}")
    print("📝 Make sure deployment-agents directory exists with all required scripts")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='🚀 [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class DemoDeploymentLauncher:
    """Interactive launcher for demo deployment."""

    def __init__(self):
        self.project_root = Path(__file__).parent.resolve()
        self.available_modes = {
            "complete": {
                "name": "Complete Deployment",
                "description": "Build demo, SDK, and website - full deployment pipeline",
                "stages": ["demo-builder", "sdk-packager", "website-generator"],
                "estimated_time": "15-30 minutes"
            },
            "sdk-only": {
                "name": "SDK Only",
                "description": "Create pip-installable SDK package only",
                "stages": ["demo-builder", "sdk-packager"],
                "estimated_time": "5-10 minutes"
            },
            "demo-only": {
                "name": "Demo Package Only",
                "description": "Create standalone demo packages only",
                "stages": ["demo-builder"],
                "estimated_time": "3-5 minutes"
            },
            "website-only": {
                "name": "Website Only",
                "description": "Generate download website (requires existing packages)",
                "stages": ["website-generator"],
                "estimated_time": "2-3 minutes"
            },
            "custom": {
                "name": "Custom Configuration",
                "description": "Choose specific stages and options",
                "stages": [],
                "estimated_time": "Variable"
            }
        }

    def show_banner(self):
        """Display welcome banner."""
        print("""
🌸 Flower Off-Guard UIOTA Demo Deployment System
═══════════════════════════════════════════════

🛡️  Automated deployment of federated learning demo with:
   • Demo package building with dependencies
   • SDK packaging for pip installation
   • Professional download website generation
   • Container integration support

""")

    def show_deployment_modes(self):
        """Display available deployment modes."""
        print("📋 Available Deployment Modes:")
        print("-" * 50)

        for key, mode in self.available_modes.items():
            print(f"{key:12} - {mode['name']}")
            print(f"{'':14} {mode['description']}")
            print(f"{'':14} Estimated time: {mode['estimated_time']}\n")

    def get_user_choice(self) -> str:
        """Get deployment mode choice from user."""
        while True:
            choice = input("Select deployment mode [complete]: ").strip().lower()

            if not choice:
                choice = "complete"

            if choice in self.available_modes:
                return choice
            elif choice in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                sys.exit(0)
            else:
                print(f"❌ Invalid choice: {choice}")
                print("Valid options:", ", ".join(self.available_modes.keys()))

    def configure_custom_deployment(self) -> Dict:
        """Configure custom deployment options."""
        print("\n🔧 Custom Deployment Configuration")
        print("-" * 40)

        config = {
            "stages": [],
            "options": {}
        }

        # Select stages
        print("Select stages to include:")
        all_stages = ["demo-builder", "sdk-packager", "website-generator"]

        for stage in all_stages:
            while True:
                include = input(f"Include {stage}? [Y/n]: ").strip().lower()
                if include in ["", "y", "yes"]:
                    config["stages"].append(stage)
                    break
                elif include in ["n", "no"]:
                    break
                else:
                    print("Please enter 'y' or 'n'")

        if not config["stages"]:
            print("❌ No stages selected, using complete deployment")
            config["stages"] = all_stages

        # Configure options
        print("\nAdditional options:")

        # Skip tests option
        while True:
            skip_tests = input("Skip tests during build? [y/N]: ").strip().lower()
            if skip_tests in ["", "n", "no"]:
                config["options"]["skip_tests"] = False
                break
            elif skip_tests in ["y", "yes"]:
                config["options"]["skip_tests"] = True
                break

        # Cleanup option
        while True:
            cleanup = input("Clean up build artifacts? [Y/n]: ").strip().lower()
            if cleanup in ["", "y", "yes"]:
                config["options"]["cleanup"] = True
                break
            elif cleanup in ["n", "no"]:
                config["options"]["cleanup"] = False
                break

        # Output directory
        output_dir = input("Output directory [deployment-output]: ").strip()
        if output_dir:
            config["options"]["output_dir"] = output_dir

        return config

    def confirm_deployment(self, mode: str, config: Dict = None) -> bool:
        """Confirm deployment configuration with user."""
        print(f"\n📋 Deployment Summary")
        print("-" * 30)

        if mode == "custom" and config:
            print(f"Mode: Custom")
            print(f"Stages: {', '.join(config['stages'])}")
            for key, value in config.get('options', {}).items():
                print(f"{key.replace('_', ' ').title()}: {value}")
        else:
            mode_info = self.available_modes[mode]
            print(f"Mode: {mode_info['name']}")
            print(f"Description: {mode_info['description']}")
            print(f"Estimated time: {mode_info['estimated_time']}")

        print(f"Project root: {self.project_root}")
        print()

        while True:
            confirm = input("Proceed with deployment? [Y/n]: ").strip().lower()
            if confirm in ["", "y", "yes"]:
                return True
            elif confirm in ["n", "no"]:
                return False
            else:
                print("Please enter 'y' or 'n'")

    def run_interactive_deployment(self) -> bool:
        """Run deployment in interactive mode."""
        self.show_banner()
        self.show_deployment_modes()

        # Get user choice
        mode = self.get_user_choice()
        print(f"\n✅ Selected: {self.available_modes[mode]['name']}")

        # Configure custom deployment if needed
        config = None
        if mode == "custom":
            config = self.configure_custom_deployment()

        # Confirm deployment
        if not self.confirm_deployment(mode, config):
            print("❌ Deployment cancelled by user")
            return False

        # Run deployment
        return self.run_deployment(mode, config)

    def run_deployment(self, mode: str, config: Dict = None) -> bool:
        """Run the actual deployment."""
        print(f"\n🚀 Starting {self.available_modes[mode]['name']}...")

        try:
            # Initialize orchestrator
            output_dir = None
            if config and config.get("options", {}).get("output_dir"):
                output_dir = Path(config["options"]["output_dir"])

            orchestrator = DeploymentOrchestratorAgent(
                project_root=self.project_root,
                output_dir=output_dir
            )

            # Configure orchestrator based on mode and options
            if config and "options" in config:
                if config["options"].get("cleanup") is not None:
                    orchestrator.config["cleanup_on_success"] = config["options"]["cleanup"]

            # Determine stages to skip
            skip_stages = []
            if mode == "custom" and config:
                all_stages = ["demo-builder", "sdk-packager", "website-generator"]
                skip_stages = [s for s in all_stages if s not in config["stages"]]
            elif mode != "complete":
                mode_stages = self.available_modes[mode]["stages"]
                all_stages = ["demo-builder", "sdk-packager", "website-generator"]
                skip_stages = [s for s in all_stages if s not in mode_stages]

            # Run deployment
            results = orchestrator.run_deployment(skip_stages=skip_stages)

            # Show results
            if results["success"]:
                self.show_success_summary(results)
                return True
            else:
                self.show_failure_summary(results)
                return False

        except KeyboardInterrupt:
            print("\n⏸️  Deployment interrupted by user")
            return False
        except Exception as e:
            logger.error(f"💥 Deployment failed with error: {e}")
            return False

    def show_success_summary(self, results: Dict):
        """Show successful deployment summary."""
        print("\n🎉 Deployment Completed Successfully!")
        print("=" * 50)

        execution_state = results.get("execution_state", {})
        artifacts = results.get("artifacts", {})

        print(f"📊 Stages completed: {len(execution_state.get('stages_completed', []))}")
        print(f"📁 Output directory: {results.get('output_dir')}")

        # Show artifacts
        if artifacts:
            print("\n📦 Generated Artifacts:")
            for artifact_type, files in artifacts.items():
                if files:
                    print(f"   {artifact_type.replace('_', ' ').title()}: {len(files)} files")

        # Show next steps
        print("\n🚀 Next Steps:")

        if artifacts.get('sdk_packages'):
            print("   📦 Install SDK:")
            wheel_files = [f for f in artifacts['sdk_packages'] if f.endswith('.whl')]
            if wheel_files:
                print(f"      pip install {Path(wheel_files[0]).name}")

        if artifacts.get('demo_packages'):
            print("   🎮 Run Demo:")
            print("      Extract package and run ./install.sh")

        if artifacts.get('website_files'):
            print("   🌐 Deploy Website:")
            print(f"      Open {results['output_dir']}/website/index.html")

        print("   📋 Check Report:")
        print(f"      View {results['output_dir']}/DEPLOYMENT_REPORT.md")

    def show_failure_summary(self, results: Dict):
        """Show failure summary."""
        print("\n❌ Deployment Failed!")
        print("=" * 30)

        execution_state = results.get("execution_state", {})
        errors = results.get("errors", [])

        if execution_state.get("stages_failed"):
            print(f"💥 Failed stages: {', '.join(execution_state['stages_failed'])}")

        if errors:
            print("\n🔍 Errors:")
            for error in errors:
                print(f"   - {error}")

        print(f"\n📋 Check logs: {results.get('output_dir')}/logs/")
        print("🔄 Try running with --resume to continue from last successful stage")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Flower Off-Guard UIOTA Demo Deployment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Deployment Modes:
  complete    - Full deployment (demo + SDK + website)
  sdk-only    - SDK package only
  demo-only   - Demo package only
  website-only- Website only
  custom      - Interactive custom configuration

Examples:
  python deploy-demo.py                     # Interactive mode
  python deploy-demo.py --mode complete     # Non-interactive complete
  python deploy-demo.py --mode sdk-only     # SDK only
  python deploy-demo.py --resume            # Resume failed deployment
        """
    )

    parser.add_argument(
        "--mode",
        choices=["complete", "sdk-only", "demo-only", "website-only", "custom"],
        help="Deployment mode"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for deployment artifacts"
    )

    parser.add_argument(
        "--skip-stages",
        nargs="*",
        choices=["demo-builder", "sdk-packager", "website-generator"],
        help="Skip specific deployment stages"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous deployment"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip tests during demo building"
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep build artifacts after completion"
    )

    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize launcher
    launcher = DemoDeploymentLauncher()

    # Interactive mode
    if args.interactive or (not args.mode and sys.stdin.isatty()):
        success = launcher.run_interactive_deployment()
        sys.exit(0 if success else 1)

    # Non-interactive mode
    if not args.mode:
        args.mode = "complete"

    print(f"🚀 Starting {args.mode} deployment...")

    try:
        # Initialize orchestrator
        orchestrator = DeploymentOrchestratorAgent(
            project_root=Path(__file__).parent,
            output_dir=Path(args.output_dir) if args.output_dir else None
        )

        # Configure based on arguments
        if args.no_cleanup:
            orchestrator.config["cleanup_on_success"] = False

        # Determine skip stages
        skip_stages = args.skip_stages or []

        # Add mode-specific skipped stages
        if args.mode == "sdk-only":
            skip_stages.append("website-generator")
        elif args.mode == "demo-only":
            skip_stages.extend(["sdk-packager", "website-generator"])
        elif args.mode == "website-only":
            skip_stages.extend(["demo-builder", "sdk-packager"])

        # Run deployment
        results = orchestrator.run_deployment(
            resume=args.resume,
            skip_stages=skip_stages
        )

        # Show results
        if results["success"]:
            launcher.show_success_summary(results)
            print(f"\n✅ {args.mode.title()} deployment completed successfully!")
            sys.exit(0)
        else:
            launcher.show_failure_summary(results)
            print(f"\n❌ {args.mode.title()} deployment failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏸️  Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()