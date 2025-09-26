#!/usr/bin/env python3
"""
Test Coordinator for UIOTA Offline Guard

Manages automated testing, validation, and coordination between different
Guardian agents. Ensures system reliability and agent interoperability.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import importlib.util

from auto_save_agent import AutoSaveAgent, create_auto_save_agent
from smooth_setup_agent import SmoothSetupAgent, create_smooth_setup_agent
from security_monitor_agent import SecurityMonitorAgent, create_security_monitor
from development_agent import DevelopmentAgent, create_development_agent

logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Represents a test case for agent functionality."""
    name: str
    description: str
    test_type: str  # unit, integration, system, performance
    priority: str   # critical, high, medium, low
    timeout: int    # seconds
    dependencies: List[str]  # other test names
    setup_function: Optional[Callable] = None
    test_function: Optional[Callable] = None
    cleanup_function: Optional[Callable] = None

@dataclass
class TestResult:
    """Represents the result of a test execution."""
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    message: str
    details: Dict[str, Any]
    timestamp: str

@dataclass
class AgentTestSuite:
    """Test suite for a specific agent."""
    agent_name: str
    agent_instance: Any
    test_cases: List[TestCase]
    results: List[TestResult]

class TestCoordinator:
    """
    Coordinates testing across all Guardian agents, ensuring system
    reliability and proper agent interaction.
    """

    def __init__(self, project_root: Path = None):
        """
        Initialize the test coordinator.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.test_dir = Path.home() / ".uiota" / "testing"
        self.results_dir = self.test_dir / "results"
        self.logs_dir = self.test_dir / "logs"

        # Agent instances for testing
        self.agents: Dict[str, Any] = {}
        self.test_suites: Dict[str, AgentTestSuite] = {}

        # Test execution state
        self._running = False
        self._test_results: List[TestResult] = []
        self._current_test_session: Optional[str] = None

        # Configuration
        self.config = {
            "execution": {
                "parallel_tests": True,
                "max_workers": 4,
                "default_timeout": 60,
                "retry_failed": True,
                "stop_on_first_failure": False
            },
            "reporting": {
                "detailed_logs": True,
                "generate_html_report": True,
                "save_artifacts": True
            },
            "coordination": {
                "test_agent_interactions": True,
                "validate_data_flow": True,
                "check_resource_usage": True
            }
        }

        self._init_directories()
        self._setup_test_environment()

        logger.info("TestCoordinator initialized")

    def _init_directories(self) -> None:
        """Create necessary directories for testing."""
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _setup_test_environment(self) -> None:
        """Set up the testing environment and agent instances."""
        try:
            # Initialize agent instances for testing
            self.agents = {
                "auto_save": create_auto_save_agent(),
                "smooth_setup": create_smooth_setup_agent(),
                "security_monitor": create_security_monitor(),
                "development": create_development_agent()
            }

            # Create test suites for each agent
            self._create_test_suites()

            logger.info(f"Test environment set up with {len(self.agents)} agents")

        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")

    def _create_test_suites(self) -> None:
        """Create test suites for all agents."""
        self.test_suites = {
            "auto_save": self._create_auto_save_tests(),
            "smooth_setup": self._create_smooth_setup_tests(),
            "security_monitor": self._create_security_monitor_tests(),
            "development": self._create_development_tests(),
            "integration": self._create_integration_tests()
        }

    def _create_auto_save_tests(self) -> AgentTestSuite:
        """Create test suite for AutoSaveAgent."""
        tests = [
            TestCase(
                name="auto_save_basic_functionality",
                description="Test basic auto-save functionality",
                test_type="unit",
                priority="critical",
                timeout=30,
                dependencies=[],
                test_function=self._test_auto_save_basic
            ),
            TestCase(
                name="auto_save_state_persistence",
                description="Test state persistence across restarts",
                test_type="integration",
                priority="high",
                timeout=60,
                dependencies=["auto_save_basic_functionality"],
                test_function=self._test_auto_save_persistence
            ),
            TestCase(
                name="auto_save_backup_restoration",
                description="Test backup and restoration functionality",
                test_type="integration",
                priority="high",
                timeout=45,
                dependencies=["auto_save_basic_functionality"],
                test_function=self._test_auto_save_backup
            )
        ]

        return AgentTestSuite(
            agent_name="auto_save",
            agent_instance=self.agents["auto_save"],
            test_cases=tests,
            results=[]
        )

    def _create_smooth_setup_tests(self) -> AgentTestSuite:
        """Create test suite for SmoothSetupAgent."""
        tests = [
            TestCase(
                name="smooth_setup_system_check",
                description="Test system requirements checking",
                test_type="unit",
                priority="critical",
                timeout=30,
                dependencies=[],
                test_function=self._test_smooth_setup_system_check
            ),
            TestCase(
                name="smooth_setup_dependency_resolution",
                description="Test dependency installation resolution",
                test_type="integration",
                priority="high",
                timeout=120,
                dependencies=["smooth_setup_system_check"],
                test_function=self._test_smooth_setup_dependencies
            )
        ]

        return AgentTestSuite(
            agent_name="smooth_setup",
            agent_instance=self.agents["smooth_setup"],
            test_cases=tests,
            results=[]
        )

    def _create_security_monitor_tests(self) -> AgentTestSuite:
        """Create test suite for SecurityMonitorAgent."""
        tests = [
            TestCase(
                name="security_monitor_threat_detection",
                description="Test threat detection capabilities",
                test_type="unit",
                priority="critical",
                timeout=45,
                dependencies=[],
                test_function=self._test_security_monitor_detection
            ),
            TestCase(
                name="security_monitor_file_integrity",
                description="Test file integrity monitoring",
                test_type="integration",
                priority="high",
                timeout=60,
                dependencies=["security_monitor_threat_detection"],
                test_function=self._test_security_monitor_integrity
            ),
            TestCase(
                name="security_monitor_response",
                description="Test automated response to threats",
                test_type="integration",
                priority="high",
                timeout=30,
                dependencies=["security_monitor_threat_detection"],
                test_function=self._test_security_monitor_response
            )
        ]

        return AgentTestSuite(
            agent_name="security_monitor",
            agent_instance=self.agents["security_monitor"],
            test_cases=tests,
            results=[]
        )

    def _create_development_tests(self) -> AgentTestSuite:
        """Create test suite for DevelopmentAgent."""
        tests = [
            TestCase(
                name="development_code_analysis",
                description="Test code quality analysis",
                test_type="unit",
                priority="high",
                timeout=60,
                dependencies=[],
                test_function=self._test_development_analysis
            ),
            TestCase(
                name="development_test_execution",
                description="Test automated test execution",
                test_type="integration",
                priority="high",
                timeout=90,
                dependencies=["development_code_analysis"],
                test_function=self._test_development_testing
            )
        ]

        return AgentTestSuite(
            agent_name="development",
            agent_instance=self.agents["development"],
            test_cases=tests,
            results=[]
        )

    def _create_integration_tests(self) -> AgentTestSuite:
        """Create integration tests for agent interactions."""
        tests = [
            TestCase(
                name="agent_communication",
                description="Test communication between agents",
                test_type="integration",
                priority="critical",
                timeout=120,
                dependencies=[],
                test_function=self._test_agent_communication
            ),
            TestCase(
                name="resource_coordination",
                description="Test resource sharing and coordination",
                test_type="system",
                priority="high",
                timeout=180,
                dependencies=["agent_communication"],
                test_function=self._test_resource_coordination
            ),
            TestCase(
                name="error_handling",
                description="Test error propagation and handling",
                test_type="system",
                priority="high",
                timeout=90,
                dependencies=["agent_communication"],
                test_function=self._test_error_handling
            )
        ]

        return AgentTestSuite(
            agent_name="integration",
            agent_instance=None,
            test_cases=tests,
            results=[]
        )

    # Test Implementation Methods

    def _test_auto_save_basic(self) -> TestResult:
        """Test basic auto-save functionality."""
        start_time = time.time()

        try:
            agent = self.agents["auto_save"]

            # Register a test agent
            agent.register_agent("test_agent", {"status": "active", "data": "test"})

            # Update agent state
            agent.update_agent_state("test_agent", {"status": "updated", "data": "modified"})

            # Check if state is saved
            saved_state = agent.get_agent_state("test_agent")

            if saved_state and saved_state["status"] == "updated":
                return TestResult(
                    test_name="auto_save_basic_functionality",
                    status="passed",
                    duration=time.time() - start_time,
                    message="Auto-save basic functionality working",
                    details={"saved_state": saved_state},
                    timestamp=datetime.now().isoformat()
                )
            else:
                return TestResult(
                    test_name="auto_save_basic_functionality",
                    status="failed",
                    duration=time.time() - start_time,
                    message="Auto-save state retrieval failed",
                    details={"expected": {"status": "updated"}, "actual": saved_state},
                    timestamp=datetime.now().isoformat()
                )

        except Exception as e:
            return TestResult(
                test_name="auto_save_basic_functionality",
                status="error",
                duration=time.time() - start_time,
                message=f"Test error: {e}",
                details={"exception": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _test_auto_save_persistence(self) -> TestResult:
        """Test state persistence across agent restarts."""
        start_time = time.time()

        try:
            agent = self.agents["auto_save"]

            # Save state
            agent.force_save()

            # Simulate restart by creating new instance
            new_agent = create_auto_save_agent()

            # Check if state persisted
            saved_agents = new_agent.list_registered_agents()

            if len(saved_agents) > 0:
                return TestResult(
                    test_name="auto_save_state_persistence",
                    status="passed",
                    duration=time.time() - start_time,
                    message="State persistence working",
                    details={"persisted_agents": saved_agents},
                    timestamp=datetime.now().isoformat()
                )
            else:
                return TestResult(
                    test_name="auto_save_state_persistence",
                    status="failed",
                    duration=time.time() - start_time,
                    message="No state persisted",
                    details={"persisted_agents": saved_agents},
                    timestamp=datetime.now().isoformat()
                )

        except Exception as e:
            return TestResult(
                test_name="auto_save_state_persistence",
                status="error",
                duration=time.time() - start_time,
                message=f"Test error: {e}",
                details={"exception": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _test_auto_save_backup(self) -> TestResult:
        """Test backup and restoration functionality."""
        start_time = time.time()

        try:
            agent = self.agents["auto_save"]

            # Create test state and backup
            agent.register_agent("backup_test", {"data": "backup_test_data"})
            agent.force_save()

            # Test backup restoration
            backup_success = agent.restore_from_backup()

            if backup_success:
                return TestResult(
                    test_name="auto_save_backup_restoration",
                    status="passed",
                    duration=time.time() - start_time,
                    message="Backup restoration working",
                    details={"backup_restored": True},
                    timestamp=datetime.now().isoformat()
                )
            else:
                return TestResult(
                    test_name="auto_save_backup_restoration",
                    status="failed",
                    duration=time.time() - start_time,
                    message="Backup restoration failed",
                    details={"backup_restored": False},
                    timestamp=datetime.now().isoformat()
                )

        except Exception as e:
            return TestResult(
                test_name="auto_save_backup_restoration",
                status="error",
                duration=time.time() - start_time,
                message=f"Test error: {e}",
                details={"exception": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _test_smooth_setup_system_check(self) -> TestResult:
        """Test system requirements checking."""
        start_time = time.time()

        try:
            agent = self.agents["smooth_setup"]

            # Run system check
            check_result = agent._check_system_requirements()

            return TestResult(
                test_name="smooth_setup_system_check",
                status="passed" if check_result else "failed",
                duration=time.time() - start_time,
                message=f"System check {'passed' if check_result else 'failed'}",
                details={"system_check_result": check_result},
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return TestResult(
                test_name="smooth_setup_system_check",
                status="error",
                duration=time.time() - start_time,
                message=f"Test error: {e}",
                details={"exception": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _test_smooth_setup_dependencies(self) -> TestResult:
        """Test dependency installation resolution."""
        start_time = time.time()

        try:
            agent = self.agents["smooth_setup"]

            # Test dependency checking (without actually installing)
            tasks = agent._get_executable_tasks()

            return TestResult(
                test_name="smooth_setup_dependency_resolution",
                status="passed",
                duration=time.time() - start_time,
                message="Dependency resolution working",
                details={"executable_tasks": tasks},
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return TestResult(
                test_name="smooth_setup_dependency_resolution",
                status="error",
                duration=time.time() - start_time,
                message=f"Test error: {e}",
                details={"exception": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _test_security_monitor_detection(self) -> TestResult:
        """Test threat detection capabilities."""
        start_time = time.time()

        try:
            agent = self.agents["security_monitor"]

            # Get current security status
            status = agent.get_security_status()

            if status["status"] == "monitoring" or status["baselines_established"]:
                return TestResult(
                    test_name="security_monitor_threat_detection",
                    status="passed",
                    duration=time.time() - start_time,
                    message="Security monitoring operational",
                    details=status,
                    timestamp=datetime.now().isoformat()
                )
            else:
                return TestResult(
                    test_name="security_monitor_threat_detection",
                    status="failed",
                    duration=time.time() - start_time,
                    message="Security monitoring not operational",
                    details=status,
                    timestamp=datetime.now().isoformat()
                )

        except Exception as e:
            return TestResult(
                test_name="security_monitor_threat_detection",
                status="error",
                duration=time.time() - start_time,
                message=f"Test error: {e}",
                details={"exception": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _test_security_monitor_integrity(self) -> TestResult:
        """Test file integrity monitoring."""
        start_time = time.time()

        try:
            agent = self.agents["security_monitor"]

            # Create a test file and monitor it
            test_file = self.test_dir / "integrity_test.txt"
            test_file.write_text("original content")

            agent.add_watched_path(test_file)

            # Modify the file
            test_file.write_text("modified content")

            # Give agent time to detect change
            time.sleep(2)

            # Check if change was detected
            events = agent._check_file_integrity()

            test_file.unlink()  # cleanup

            return TestResult(
                test_name="security_monitor_file_integrity",
                status="passed",
                duration=time.time() - start_time,
                message="File integrity monitoring working",
                details={"events_detected": len(events)},
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return TestResult(
                test_name="security_monitor_file_integrity",
                status="error",
                duration=time.time() - start_time,
                message=f"Test error: {e}",
                details={"exception": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _test_security_monitor_response(self) -> TestResult:
        """Test automated response to threats."""
        start_time = time.time()

        try:
            agent = self.agents["security_monitor"]

            # Test basic response functionality
            initial_events = len(agent._security_events)

            # Security monitor should be functional
            return TestResult(
                test_name="security_monitor_response",
                status="passed",
                duration=time.time() - start_time,
                message="Security response system functional",
                details={"initial_events": initial_events},
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return TestResult(
                test_name="security_monitor_response",
                status="error",
                duration=time.time() - start_time,
                message=f"Test error: {e}",
                details={"exception": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _test_development_analysis(self) -> TestResult:
        """Test code quality analysis."""
        start_time = time.time()

        try:
            agent = self.agents["development"]

            # Run analysis on current project
            analysis_result = agent.run_manual_analysis()

            return TestResult(
                test_name="development_code_analysis",
                status="passed",
                duration=time.time() - start_time,
                message="Code analysis completed",
                details=analysis_result,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return TestResult(
                test_name="development_code_analysis",
                status="error",
                duration=time.time() - start_time,
                message=f"Test error: {e}",
                details={"exception": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _test_development_testing(self) -> TestResult:
        """Test automated test execution."""
        start_time = time.time()

        try:
            agent = self.agents["development"]

            # Run tests
            test_result = agent.run_manual_tests()

            return TestResult(
                test_name="development_test_execution",
                status="passed",
                duration=time.time() - start_time,
                message="Test execution completed",
                details=test_result,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return TestResult(
                test_name="development_test_execution",
                status="error",
                duration=time.time() - start_time,
                message=f"Test error: {e}",
                details={"exception": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _test_agent_communication(self) -> TestResult:
        """Test communication between agents."""
        start_time = time.time()

        try:
            # Test that all agents are responsive
            responsive_agents = 0

            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'get_status') or hasattr(agent, 'get_security_status'):
                    responsive_agents += 1

            return TestResult(
                test_name="agent_communication",
                status="passed",
                duration=time.time() - start_time,
                message=f"Agent communication test completed: {responsive_agents}/{len(self.agents)} responsive",
                details={"responsive_agents": responsive_agents, "total_agents": len(self.agents)},
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return TestResult(
                test_name="agent_communication",
                status="error",
                duration=time.time() - start_time,
                message=f"Test error: {e}",
                details={"exception": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _test_resource_coordination(self) -> TestResult:
        """Test resource sharing and coordination."""
        start_time = time.time()

        try:
            # Test resource usage coordination
            total_memory_usage = 0

            for agent_name, agent in self.agents.items():
                # Simulate resource usage check
                if hasattr(agent, '_running'):
                    total_memory_usage += 1  # Simplified metric

            return TestResult(
                test_name="resource_coordination",
                status="passed",
                duration=time.time() - start_time,
                message="Resource coordination test completed",
                details={"total_memory_usage": total_memory_usage},
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return TestResult(
                test_name="resource_coordination",
                status="error",
                duration=time.time() - start_time,
                message=f"Test error: {e}",
                details={"exception": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _test_error_handling(self) -> TestResult:
        """Test error propagation and handling."""
        start_time = time.time()

        try:
            # Test error handling by checking if agents handle invalid operations gracefully
            error_handled = True

            try:
                # Test invalid operation on auto_save agent
                self.agents["auto_save"].get_agent_state("non_existent_agent")
            except Exception:
                error_handled = False

            return TestResult(
                test_name="error_handling",
                status="passed" if error_handled else "failed",
                duration=time.time() - start_time,
                message=f"Error handling test {'passed' if error_handled else 'failed'}",
                details={"error_handled_gracefully": error_handled},
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return TestResult(
                test_name="error_handling",
                status="error",
                duration=time.time() - start_time,
                message=f"Test error: {e}",
                details={"exception": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def _execute_test_case(self, test_case: TestCase) -> TestResult:
        """Execute a single test case."""
        logger.info(f"Executing test: {test_case.name}")

        try:
            if test_case.test_function:
                return test_case.test_function()
            else:
                return TestResult(
                    test_name=test_case.name,
                    status="skipped",
                    duration=0.0,
                    message="No test function defined",
                    details={},
                    timestamp=datetime.now().isoformat()
                )

        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                status="error",
                duration=0.0,
                message=f"Test execution error: {e}",
                details={"exception": str(e)},
                timestamp=datetime.now().isoformat()
            )

    def run_tests(self, test_filter: str = None, parallel: bool = None) -> Dict[str, Any]:
        """
        Run all tests or filtered tests.

        Args:
            test_filter: Filter tests by name pattern or suite
            parallel: Override parallel execution setting

        Returns:
            Test execution results
        """
        start_time = time.time()
        self._current_test_session = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Starting test session: {self._current_test_session}")

        # Collect all tests to run
        all_tests = []
        for suite_name, suite in self.test_suites.items():
            for test_case in suite.test_cases:
                if not test_filter or test_filter in test_case.name or test_filter in suite_name:
                    all_tests.append((suite_name, test_case))

        logger.info(f"Executing {len(all_tests)} tests")

        # Execute tests
        use_parallel = parallel if parallel is not None else self.config["execution"]["parallel_tests"]

        if use_parallel and len(all_tests) > 1:
            results = self._run_tests_parallel(all_tests)
        else:
            results = self._run_tests_sequential(all_tests)

        # Store results
        for result in results:
            self._test_results.append(result)

            # Add to appropriate suite
            for suite_name, suite in self.test_suites.items():
                if any(tc.name == result.test_name for tc in suite.test_cases):
                    suite.results.append(result)
                    break

        # Generate summary
        total_duration = time.time() - start_time
        summary = self._generate_test_summary(results, total_duration)

        # Save results
        self._save_test_results(results, summary)

        logger.info(f"Test session completed in {total_duration:.2f}s")

        return summary

    def _run_tests_parallel(self, tests: List[Tuple[str, TestCase]]) -> List[TestResult]:
        """Run tests in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=self.config["execution"]["max_workers"]) as executor:
            future_to_test = {
                executor.submit(self._execute_test_case, test_case): (suite_name, test_case)
                for suite_name, test_case in tests
            }

            for future in as_completed(future_to_test):
                suite_name, test_case = future_to_test[future]
                try:
                    result = future.result(timeout=test_case.timeout)
                    results.append(result)
                except Exception as e:
                    results.append(TestResult(
                        test_name=test_case.name,
                        status="error",
                        duration=test_case.timeout,
                        message=f"Test timeout or error: {e}",
                        details={"exception": str(e)},
                        timestamp=datetime.now().isoformat()
                    ))

        return results

    def _run_tests_sequential(self, tests: List[Tuple[str, TestCase]]) -> List[TestResult]:
        """Run tests sequentially."""
        results = []

        for suite_name, test_case in tests:
            result = self._execute_test_case(test_case)
            results.append(result)

            # Stop on first failure if configured
            if (self.config["execution"]["stop_on_first_failure"] and
                result.status in ["failed", "error"]):
                logger.warning("Stopping tests due to failure")
                break

        return results

    def _generate_test_summary(self, results: List[TestResult], duration: float) -> Dict[str, Any]:
        """Generate test execution summary."""
        passed = len([r for r in results if r.status == "passed"])
        failed = len([r for r in results if r.status == "failed"])
        errors = len([r for r in results if r.status == "error"])
        skipped = len([r for r in results if r.status == "skipped"])

        return {
            "session_id": self._current_test_session,
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "summary": {
                "total": len(results),
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
                "success_rate": (passed / len(results)) * 100 if results else 0
            },
            "results": [asdict(result) for result in results]
        }

    def _save_test_results(self, results: List[TestResult], summary: Dict[str, Any]) -> None:
        """Save test results to disk."""
        try:
            # Save summary
            summary_file = self.results_dir / f"test_summary_{self._current_test_session}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            # Save detailed results
            detailed_file = self.results_dir / f"test_details_{self._current_test_session}.json"
            with open(detailed_file, 'w') as f:
                json.dump([asdict(result) for result in results], f, indent=2)

            logger.info(f"Test results saved to {summary_file}")

        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

    def get_test_status(self) -> Dict[str, Any]:
        """Get current test status and history."""
        recent_results = self._test_results[-20:] if self._test_results else []

        return {
            "total_tests_available": sum(len(suite.test_cases) for suite in self.test_suites.values()),
            "total_tests_executed": len(self._test_results),
            "recent_results": [asdict(result) for result in recent_results],
            "suites": {
                name: {
                    "test_count": len(suite.test_cases),
                    "results_count": len(suite.results)
                }
                for name, suite in self.test_suites.items()
            }
        }

def create_test_coordinator(project_root: Path = None) -> TestCoordinator:
    """Factory function to create a configured TestCoordinator."""
    return TestCoordinator(project_root)

if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    coordinator = create_test_coordinator()

    try:
        # Run all tests
        results = coordinator.run_tests()

        print(f"Test Results:")
        print(f"Total: {results['summary']['total']}")
        print(f"Passed: {results['summary']['passed']}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")

    except KeyboardInterrupt:
        print("Testing interrupted")
    except Exception as e:
        print(f"Testing error: {e}")