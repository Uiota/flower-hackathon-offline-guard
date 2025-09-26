#!/usr/bin/env python3
"""
Deployment Orchestrator Agent - Orchestrates the entire demo deployment process.

This agent handles:
- Running all other agents in sequence
- Error recovery and retry logic
- File output management and staging
- Progress reporting and logging
- Integration with existing Podman infrastructure
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='🎯 [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentOrchestratorAgent:
    """Main deployment orchestration agent."""

    def __init__(self, project_root: Path, output_dir: Path = None):
        self.project_root = Path(project_root).resolve()
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "deployment-output"
        self.agents_dir = self.project_root / "deployment-agents"

        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "builds").mkdir(exist_ok=True)
        (self.output_dir / "distributions").mkdir(exist_ok=True)
        (self.output_dir / "website").mkdir(exist_ok=True)
        (self.output_dir / "sdk").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # Deployment configuration
        self.config = {
            "version": "1.0.0",
            "build_timestamp": datetime.now().isoformat(),
            "retry_attempts": 3,
            "timeout_seconds": 1800,  # 30 minutes
            "cleanup_on_success": True,
            "generate_checksums": True,
            "integration_tests": True,
        }

        # Agent execution order and dependencies
        self.deployment_pipeline = [
            {
                "name": "demo-builder",
                "script": "demo-builder-agent.py",
                "description": "Build demo package with dependencies",
                "required": True,
                "timeout": 600,
                "outputs": ["builds", "distributions"],
                "dependencies": []
            },
            {
                "name": "sdk-packager",
                "script": "sdk-packager-agent.py",
                "description": "Package demo as pip-installable SDK",
                "required": True,
                "timeout": 600,
                "outputs": ["sdk"],
                "dependencies": ["demo-builder"]
            },
            {
                "name": "website-generator",
                "script": "website-generator-agent.py",
                "description": "Generate download website",
                "required": False,
                "timeout": 300,
                "outputs": ["website"],
                "dependencies": ["demo-builder"]
            }
        ]

        # Execution state
        self.execution_state = {
            "started_at": None,
            "completed_at": None,
            "current_stage": None,
            "stages_completed": [],
            "stages_failed": [],
            "overall_success": False,
            "artifacts": {},
            "errors": []
        }

    def validate_environment(self) -> bool:
        """Validate deployment environment and prerequisites."""
        logger.info("🔍 Validating deployment environment...")

        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            return False

        # Check project structure
        if not self.project_root.exists():
            logger.error(f"Project root not found: {self.project_root}")
            return False

        # Check demo directory
        demo_dir = self.project_root / "flower-offguard-uiota-demo"
        if not demo_dir.exists():
            logger.error(f"Demo directory not found: {demo_dir}")
            return False

        # Check agents directory
        if not self.agents_dir.exists():
            logger.error(f"Deployment agents directory not found: {self.agents_dir}")
            return False

        # Check agent scripts exist
        for stage in self.deployment_pipeline:
            agent_script = self.agents_dir / stage["script"]
            if not agent_script.exists():
                logger.error(f"Agent script not found: {agent_script}")
                return False

        # Check dependencies
        required_tools = ["python3", "pip"]
        for tool in required_tools:
            if not shutil.which(tool):
                logger.error(f"Required tool not found: {tool}")
                return False

        # Check available disk space (minimum 1GB)
        try:
            statvfs = os.statvfs(self.output_dir)
            free_bytes = statvfs.f_frsize * statvfs.f_bavail
            if free_bytes < 1024 * 1024 * 1024:  # 1GB
                logger.warning(f"Low disk space: {free_bytes / (1024**3):.1f}GB available")
        except OSError:
            logger.warning("Could not check disk space")

        logger.info("✅ Environment validation passed")
        return True

    def setup_logging(self) -> None:
        """Setup comprehensive logging for deployment."""
        log_file = self.output_dir / "logs" / f"deployment-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)
        logging.getLogger().setLevel(logging.DEBUG)

        logger.info(f"📝 Logging to file: {log_file}")

    def save_execution_state(self) -> None:
        """Save current execution state to file."""
        state_file = self.output_dir / "deployment-state.json"
        with open(state_file, 'w') as f:
            json.dump(self.execution_state, f, indent=2, default=str)

    def load_execution_state(self) -> bool:
        """Load previous execution state if exists."""
        state_file = self.output_dir / "deployment-state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    self.execution_state.update(json.load(f))
                logger.info("📂 Loaded previous execution state")
                return True
            except Exception as e:
                logger.warning(f"Could not load execution state: {e}")
        return False

    def execute_agent(self, stage: Dict) -> Tuple[bool, Dict]:
        """Execute a single deployment agent."""
        agent_name = stage["name"]
        agent_script = self.agents_dir / stage["script"]

        logger.info(f"🚀 Executing {agent_name}...")
        self.execution_state["current_stage"] = agent_name

        # Prepare agent arguments
        agent_args = [
            sys.executable, str(agent_script),
            "--project-root", str(self.project_root),
            "--verbose"
        ]

        # Add stage-specific arguments
        if agent_name == "demo-builder":
            agent_args.extend([
                "--output-dir", str(self.output_dir / "distributions")
            ])
        elif agent_name == "sdk-packager":
            agent_args.extend([
                "--output-dir", str(self.output_dir / "sdk")
            ])
        elif agent_name == "website-generator":
            agent_args.extend([
                "--output-dir", str(self.output_dir / "website"),
                "--dist-dir", str(self.output_dir / "distributions")
            ])

        # Execute agent with timeout
        try:
            start_time = time.time()

            result = subprocess.run(
                agent_args,
                capture_output=True,
                text=True,
                timeout=stage.get("timeout", 600),
                cwd=self.project_root
            )

            execution_time = time.time() - start_time

            # Parse result
            agent_result = {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "outputs": []
            }

            # Check for expected outputs
            for output_type in stage.get("outputs", []):
                output_dir = self.output_dir / output_type
                if output_dir.exists() and any(output_dir.iterdir()):
                    agent_result["outputs"].append(output_type)

            if agent_result["success"]:
                logger.info(f"✅ {agent_name} completed successfully in {execution_time:.1f}s")
                self.execution_state["stages_completed"].append(agent_name)
            else:
                logger.error(f"❌ {agent_name} failed (exit code: {result.returncode})")
                logger.error(f"Error output: {result.stderr}")
                self.execution_state["stages_failed"].append(agent_name)

            return agent_result["success"], agent_result

        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {agent_name} timed out after {stage.get('timeout', 600)}s")
            return False, {"success": False, "error": "timeout"}

        except Exception as e:
            logger.error(f"💥 {agent_name} crashed: {e}")
            return False, {"success": False, "error": str(e)}

    def check_dependencies(self, stage: Dict) -> bool:
        """Check if stage dependencies are satisfied."""
        dependencies = stage.get("dependencies", [])

        for dep in dependencies:
            if dep not in self.execution_state["stages_completed"]:
                logger.error(f"❌ Dependency not satisfied: {dep} required for {stage['name']}")
                return False

        return True

    def retry_failed_stage(self, stage: Dict, attempt: int) -> Tuple[bool, Dict]:
        """Retry a failed stage with exponential backoff."""
        backoff_time = min(60, 2 ** attempt)  # Max 60 seconds
        logger.info(f"⏳ Retrying {stage['name']} in {backoff_time}s (attempt {attempt + 1})")

        time.sleep(backoff_time)
        return self.execute_agent(stage)

    def collect_artifacts(self) -> Dict:
        """Collect all generated artifacts."""
        logger.info("📦 Collecting deployment artifacts...")

        artifacts = {
            "demo_packages": [],
            "sdk_packages": [],
            "website_files": [],
            "logs": [],
            "checksums": []
        }

        # Collect demo packages
        dist_dir = self.output_dir / "distributions"
        if dist_dir.exists():
            artifacts["demo_packages"] = [
                str(f) for f in dist_dir.glob("*.tar.gz") if f.is_file()
            ] + [
                str(f) for f in dist_dir.glob("*.zip") if f.is_file()
            ]

        # Collect SDK packages
        sdk_dir = self.output_dir / "sdk"
        if sdk_dir.exists():
            artifacts["sdk_packages"] = [
                str(f) for f in sdk_dir.glob("*.whl") if f.is_file()
            ] + [
                str(f) for f in sdk_dir.glob("*.tar.gz") if f.is_file()
            ]

        # Collect website files
        website_dir = self.output_dir / "website"
        if website_dir.exists():
            artifacts["website_files"] = [str(website_dir / "index.html")]
            if (website_dir / "assets").exists():
                artifacts["website_files"].extend([
                    str(f) for f in (website_dir / "assets").glob("*") if f.is_file()
                ])

        # Collect logs
        logs_dir = self.output_dir / "logs"
        if logs_dir.exists():
            artifacts["logs"] = [
                str(f) for f in logs_dir.glob("*.log") if f.is_file()
            ]

        # Collect checksums
        for checksum_file in self.output_dir.rglob("CHECKSUMS.txt"):
            artifacts["checksums"].append(str(checksum_file))

        self.execution_state["artifacts"] = artifacts
        logger.info(f"📋 Collected {sum(len(v) for v in artifacts.values())} artifacts")

        return artifacts

    def generate_deployment_report(self) -> str:
        """Generate comprehensive deployment report."""
        logger.info("📊 Generating deployment report...")

        report_path = self.output_dir / "DEPLOYMENT_REPORT.md"

        # Calculate execution time
        start_time = self.execution_state.get("started_at")
        end_time = self.execution_state.get("completed_at")
        duration = "Unknown"

        if start_time and end_time:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            duration = str(end_dt - start_dt)

        # Generate report content
        report_content = f"""# Flower Off-Guard UIOTA Deployment Report

Generated: {datetime.now().isoformat()}

## Deployment Summary

- **Status**: {'✅ SUCCESS' if self.execution_state['overall_success'] else '❌ FAILED'}
- **Version**: {self.config['version']}
- **Duration**: {duration}
- **Stages Completed**: {len(self.execution_state['stages_completed'])}
- **Stages Failed**: {len(self.execution_state['stages_failed'])}

## Pipeline Execution

### Completed Stages
"""

        for stage in self.execution_state["stages_completed"]:
            report_content += f"- ✅ {stage}\\n"

        if self.execution_state["stages_failed"]:
            report_content += "\\n### Failed Stages\\n"
            for stage in self.execution_state["stages_failed"]:
                report_content += f"- ❌ {stage}\\n"

        # Add artifacts section
        artifacts = self.execution_state.get("artifacts", {})
        if artifacts:
            report_content += f"""

## Generated Artifacts

### Demo Packages ({len(artifacts.get('demo_packages', []))})
"""
            for package in artifacts.get('demo_packages', []):
                report_content += f"- `{Path(package).name}`\\n"

            report_content += f"""
### SDK Packages ({len(artifacts.get('sdk_packages', []))})
"""
            for package in artifacts.get('sdk_packages', []):
                report_content += f"- `{Path(package).name}`\\n"

            if artifacts.get('website_files'):
                report_content += f"""
### Website Files ({len(artifacts.get('website_files', []))})
- Website generated at: `{self.output_dir / 'website'}`
- Main page: `index.html`
- Assets directory with CSS, JS, and images
"""

        # Add installation instructions
        if artifacts.get('sdk_packages'):
            wheel_files = [p for p in artifacts['sdk_packages'] if p.endswith('.whl')]
            if wheel_files:
                report_content += f"""

## Installation Instructions

### SDK Installation

```bash
pip install {Path(wheel_files[0]).name}
```

### Demo Usage

```bash
# Start server
flower-offguard-server --dataset mnist --rounds 5

# Start client (in another terminal)
flower-offguard-client --client-id 1

# Interactive demo
flower-offguard-demo --interactive
```
"""

        if artifacts.get('demo_packages'):
            demo_packages = artifacts['demo_packages']
            if demo_packages:
                report_content += f"""
### Standalone Demo

```bash
# Extract package
tar -xzf {Path(demo_packages[0]).name}
cd flower-offguard-uiota-demo

# Install
./install.sh

# Run demo
python src/server.py
```
"""

        # Add container integration
        report_content += f"""

## Container Integration

The deployment can be integrated with the existing Podman infrastructure:

```bash
# Copy artifacts to container directory
cp {self.output_dir}/sdk/*.whl containers/ml-toolkit/

# Rebuild ML toolkit container
podman build -t offline-guard-ml-updated -f containers/ml-toolkit/Containerfile .

# Start with new demo
./start-demos.sh
```

## Configuration

Edit `config/demo.yaml` to customize:
- Dataset selection (MNIST, CIFAR-10)
- Federated learning strategy
- Security settings
- Network parameters

## Support

- 📚 Documentation: https://github.com/uiota/offline-guard
- 🐛 Issues: https://github.com/uiota/offline-guard/issues
- 📧 Contact: dev@uiota.org

---

Report generated by Deployment Orchestrator Agent v{self.config['version']}
"""

        # Write report
        report_path.write_text(report_content)
        logger.info(f"📋 Report saved to: {report_path}")

        return str(report_path)

    def cleanup_build_artifacts(self) -> None:
        """Clean up temporary build artifacts."""
        if not self.config.get("cleanup_on_success", True):
            return

        logger.info("🧹 Cleaning up build artifacts...")

        cleanup_dirs = [
            self.output_dir / "builds",
            self.output_dir / "temp"
        ]

        for cleanup_dir in cleanup_dirs:
            if cleanup_dir.exists():
                try:
                    shutil.rmtree(cleanup_dir)
                    logger.info(f"🗑️  Removed: {cleanup_dir}")
                except Exception as e:
                    logger.warning(f"Could not remove {cleanup_dir}: {e}")

    def integrate_with_podman(self) -> bool:
        """Integrate deployment with existing Podman infrastructure."""
        logger.info("🐳 Integrating with Podman infrastructure...")

        try:
            # Check if Podman is available
            if not shutil.which("podman"):
                logger.warning("Podman not found - skipping container integration")
                return True

            # Copy SDK packages to ML toolkit
            sdk_dir = self.output_dir / "sdk"
            ml_toolkit_dir = self.project_root / "containers" / "ml-toolkit"

            if sdk_dir.exists() and ml_toolkit_dir.exists():
                for wheel_file in sdk_dir.glob("*.whl"):
                    dest_file = ml_toolkit_dir / wheel_file.name
                    shutil.copy2(wheel_file, dest_file)
                    logger.info(f"📦 Copied SDK to ML toolkit: {wheel_file.name}")

            # Update container build scripts if needed
            # This would be expanded based on specific container requirements

            logger.info("✅ Podman integration completed")
            return True

        except Exception as e:
            logger.error(f"❌ Podman integration failed: {e}")
            return False

    def run_deployment(self, resume: bool = False, skip_stages: List[str] = None) -> Dict:
        """Run the complete deployment pipeline."""
        logger.info("🎯 Starting deployment orchestration...")

        self.execution_state["started_at"] = datetime.now().isoformat()
        skip_stages = skip_stages or []

        # Setup logging
        self.setup_logging()

        # Load previous state if resuming
        if resume:
            self.load_execution_state()

        try:
            # Validate environment
            if not self.validate_environment():
                self.execution_state["errors"].append("Environment validation failed")
                return self._finalize_deployment(False)

            # Execute pipeline stages
            for stage in self.deployment_pipeline:
                stage_name = stage["name"]

                # Skip if requested
                if stage_name in skip_stages:
                    logger.info(f"⏭️  Skipping {stage_name} (user requested)")
                    continue

                # Skip if already completed (resume mode)
                if resume and stage_name in self.execution_state["stages_completed"]:
                    logger.info(f"⏭️  Skipping {stage_name} (already completed)")
                    continue

                # Check dependencies
                if not self.check_dependencies(stage):
                    if stage["required"]:
                        self.execution_state["errors"].append(f"Required stage {stage_name} dependencies not met")
                        return self._finalize_deployment(False)
                    else:
                        logger.warning(f"⚠️  Skipping optional stage {stage_name} (dependencies not met)")
                        continue

                # Execute stage with retries
                success = False
                stage_result = None

                for attempt in range(self.config["retry_attempts"]):
                    if attempt > 0:
                        success, stage_result = self.retry_failed_stage(stage, attempt)
                    else:
                        success, stage_result = self.execute_agent(stage)

                    if success:
                        break
                else:
                    # All retry attempts failed
                    if stage["required"]:
                        self.execution_state["errors"].append(f"Required stage {stage_name} failed after {self.config['retry_attempts']} attempts")
                        return self._finalize_deployment(False)
                    else:
                        logger.warning(f"⚠️  Optional stage {stage_name} failed - continuing")

                # Save state after each stage
                self.save_execution_state()

            # Collect artifacts
            self.collect_artifacts()

            # Integrate with Podman
            self.integrate_with_podman()

            # Generate deployment report
            self.generate_deployment_report()

            # Cleanup if successful
            if self.execution_state["overall_success"]:
                self.cleanup_build_artifacts()

            return self._finalize_deployment(True)

        except KeyboardInterrupt:
            logger.info("⏸️  Deployment interrupted by user")
            self.execution_state["errors"].append("Deployment interrupted")
            return self._finalize_deployment(False)

        except Exception as e:
            logger.error(f"💥 Deployment failed with exception: {e}")
            self.execution_state["errors"].append(f"Unexpected error: {e}")
            return self._finalize_deployment(False)

    def _finalize_deployment(self, success: bool) -> Dict:
        """Finalize deployment and return results."""
        self.execution_state["completed_at"] = datetime.now().isoformat()
        self.execution_state["overall_success"] = success

        # Save final state
        self.save_execution_state()

        # Log summary
        if success:
            logger.info("🎉 Deployment completed successfully!")
        else:
            logger.error("❌ Deployment failed!")

        return {
            "success": success,
            "execution_state": self.execution_state,
            "output_dir": str(self.output_dir),
            "artifacts": self.execution_state.get("artifacts", {}),
            "errors": self.execution_state.get("errors", [])
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Flower Off-Guard UIOTA Deployment Orchestrator")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output-dir", help="Output directory for deployment artifacts")
    parser.add_argument("--resume", action="store_true", help="Resume previous deployment")
    parser.add_argument("--skip-stages", nargs="*", help="Skip specific stages")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize orchestrator
    orchestrator = DeploymentOrchestratorAgent(
        project_root=args.project_root,
        output_dir=args.output_dir
    )

    # Run deployment
    results = orchestrator.run_deployment(
        resume=args.resume,
        skip_stages=args.skip_stages
    )

    # Print summary
    if results["success"]:
        logger.info("🎯 Deployment Orchestration Summary:")
        logger.info(f"   Output directory: {results['output_dir']}")
        logger.info(f"   Stages completed: {len(results['execution_state']['stages_completed'])}")

        artifacts = results.get("artifacts", {})
        for artifact_type, files in artifacts.items():
            if files:
                logger.info(f"   {artifact_type}: {len(files)} files")

        logger.info("\\n🚀 Next steps:")
        logger.info("   1. Review DEPLOYMENT_REPORT.md")
        logger.info("   2. Test SDK installation")
        logger.info("   3. Deploy website (if generated)")
        logger.info("   4. Update container images")

        return 0
    else:
        logger.error("❌ Deployment orchestration failed!")
        for error in results.get("errors", []):
            logger.error(f"   - {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())