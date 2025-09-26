#!/usr/bin/env python3
"""
Development Agent for UIOTA Offline Guard

Automated code quality monitoring, testing, debugging assistance,
and development workflow management for the Guardian ecosystem.
"""

import asyncio
import ast
import json
import logging
import os
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import tempfile
import shutil

logger = logging.getLogger(__name__)

@dataclass
class CodeIssue:
    """Represents a code quality issue or bug."""
    file_path: str
    line_number: int
    issue_type: str  # syntax, logic, style, security, performance
    severity: str    # critical, high, medium, low
    description: str
    suggestion: Optional[str] = None
    auto_fixable: bool = False

@dataclass
class TestResult:
    """Represents test execution results."""
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    message: Optional[str] = None
    details: Dict[str, Any] = None

@dataclass
class BuildResult:
    """Represents build/compilation results."""
    success: bool
    duration: float
    errors: List[str]
    warnings: List[str]
    output: str

class DevelopmentAgent:
    """
    Automated development assistant providing code quality monitoring,
    testing coordination, and debugging support for Guardian agents.
    """

    def __init__(self, project_root: Path = None):
        """
        Initialize the development agent.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.dev_dir = Path.home() / ".uiota" / "development"
        self.reports_dir = self.dev_dir / "reports"
        self.temp_dir = self.dev_dir / "temp"

        # Monitoring state
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._code_issues: List[CodeIssue] = []
        self._test_history: List[TestResult] = []
        self._build_history: List[BuildResult] = []

        # File watching
        self._watched_files: Set[Path] = set()
        self._file_checksums: Dict[str, str] = {}

        # Configuration
        self.config = {
            "monitoring": {
                "check_interval": 10,
                "auto_test": True,
                "auto_lint": True,
                "auto_format": False
            },
            "quality": {
                "max_line_length": 100,
                "max_function_length": 50,
                "max_complexity": 10,
                "enforce_docstrings": True
            },
            "testing": {
                "auto_run_on_change": True,
                "parallel_execution": True,
                "coverage_threshold": 80
            },
            "tools": {
                "linter": "flake8",
                "formatter": "black",
                "type_checker": "mypy",
                "test_runner": "pytest"
            }
        }

        self._init_directories()
        self._discover_source_files()

        logger.info("DevelopmentAgent initialized")

    def _init_directories(self) -> None:
        """Create necessary directories for development operations."""
        self.dev_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _discover_source_files(self) -> None:
        """Discover and register source files for monitoring."""
        python_files = list(self.project_root.rglob("*.py"))

        # Filter out __pycache__ and other unimportant directories
        excluded_dirs = {"__pycache__", ".git", "venv", ".venv", "node_modules"}

        for py_file in python_files:
            if not any(excluded in str(py_file) for excluded in excluded_dirs):
                self._watched_files.add(py_file)
                self._file_checksums[str(py_file)] = self._calculate_file_checksum(py_file)

        logger.info(f"Monitoring {len(self._watched_files)} Python files")

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate checksum for file change detection."""
        try:
            import hashlib
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def _check_file_changes(self) -> List[Path]:
        """Check for modified files."""
        changed_files = []

        for file_path in self._watched_files:
            if not file_path.exists():
                continue

            current_checksum = self._calculate_file_checksum(file_path)
            stored_checksum = self._file_checksums.get(str(file_path), "")

            if current_checksum != stored_checksum:
                changed_files.append(file_path)
                self._file_checksums[str(file_path)] = current_checksum

        return changed_files

    def _analyze_code_quality(self, file_path: Path) -> List[CodeIssue]:
        """Analyze code quality for a specific file."""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse AST for analysis
            try:
                tree = ast.parse(content)
                issues.extend(self._analyze_ast(file_path, tree, content))
            except SyntaxError as e:
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=e.lineno or 0,
                    issue_type="syntax",
                    severity="critical",
                    description=f"Syntax error: {e.msg}",
                    auto_fixable=False
                ))

            # Additional checks
            issues.extend(self._check_style_issues(file_path, content))
            issues.extend(self._check_security_issues(file_path, content))

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")

        return issues

    def _analyze_ast(self, file_path: Path, tree: ast.AST, content: str) -> List[CodeIssue]:
        """Analyze AST for code quality issues."""
        issues = []
        lines = content.split('\n')

        class QualityVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Check function length
                func_lines = node.end_lineno - node.lineno if node.end_lineno else 0
                if func_lines > self.config["quality"]["max_function_length"]:
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        issue_type="style",
                        severity="medium",
                        description=f"Function '{node.name}' is too long ({func_lines} lines)",
                        suggestion=f"Consider breaking down function '{node.name}' into smaller functions"
                    ))

                # Check for docstring
                if self.config["quality"]["enforce_docstrings"]:
                    if not (node.body and isinstance(node.body[0], ast.Expr) and
                           isinstance(node.body[0].value, ast.Constant) and
                           isinstance(node.body[0].value.value, str)):
                        issues.append(CodeIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            issue_type="style",
                            severity="low",
                            description=f"Function '{node.name}' missing docstring",
                            suggestion=f"Add docstring to function '{node.name}'"
                        ))

                self.generic_visit(node)

            def visit_ClassDef(self, node):
                # Check for class docstring
                if self.config["quality"]["enforce_docstrings"]:
                    if not (node.body and isinstance(node.body[0], ast.Expr) and
                           isinstance(node.body[0].value, ast.Constant) and
                           isinstance(node.body[0].value.value, str)):
                        issues.append(CodeIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            issue_type="style",
                            severity="low",
                            description=f"Class '{node.name}' missing docstring",
                            suggestion=f"Add docstring to class '{node.name}'"
                        ))

                self.generic_visit(node)

        visitor = QualityVisitor()
        visitor.config = self.config
        visitor.visit(tree)

        return issues

    def _check_style_issues(self, file_path: Path, content: str) -> List[CodeIssue]:
        """Check for style issues."""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > self.config["quality"]["max_line_length"]:
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=i,
                    issue_type="style",
                    severity="low",
                    description=f"Line too long ({len(line)} > {self.config['quality']['max_line_length']})",
                    suggestion="Break long line into multiple lines",
                    auto_fixable=True
                ))

            # Check for trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=i,
                    issue_type="style",
                    severity="low",
                    description="Trailing whitespace",
                    suggestion="Remove trailing whitespace",
                    auto_fixable=True
                ))

        return issues

    def _check_security_issues(self, file_path: Path, content: str) -> List[CodeIssue]:
        """Check for potential security issues."""
        issues = []
        lines = content.split('\n')

        security_patterns = [
            ("eval(", "Dangerous use of eval()"),
            ("exec(", "Dangerous use of exec()"),
            ("shell=True", "Potential shell injection risk"),
            ("password", "Hardcoded password detected"),
            ("secret", "Hardcoded secret detected"),
            ("api_key", "Hardcoded API key detected")
        ]

        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            for pattern, message in security_patterns:
                if pattern in line_lower:
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=i,
                        issue_type="security",
                        severity="high",
                        description=message,
                        suggestion="Review and secure this code"
                    ))

        return issues

    def _run_external_linter(self, file_path: Path) -> List[CodeIssue]:
        """Run external linting tools."""
        issues = []

        try:
            # Run flake8 if available
            result = subprocess.run([
                "flake8", "--format=json", str(file_path)
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and result.stdout:
                try:
                    flake8_output = json.loads(result.stdout)
                    for item in flake8_output:
                        issues.append(CodeIssue(
                            file_path=item["filename"],
                            line_number=item["line_number"],
                            issue_type="style",
                            severity="low",
                            description=f"Flake8: {item['text']}"
                        ))
                except json.JSONDecodeError:
                    pass

        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("Flake8 not available or timed out")

        return issues

    def _run_tests(self, test_path: Path = None) -> List[TestResult]:
        """Run tests and return results."""
        results = []

        try:
            # Determine test command
            test_files = list(self.project_root.rglob("test_*.py"))
            test_files.extend(list(self.project_root.rglob("*_test.py")))

            if not test_files:
                logger.info("No test files found")
                return results

            # Run pytest if available
            cmd = ["python", "-m", "pytest", "--tb=short", "-v"]
            if test_path:
                cmd.append(str(test_path))

            start_time = time.time()
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=self.project_root, timeout=300
            )
            duration = time.time() - start_time

            # Parse pytest output (simplified)
            if "FAILED" in result.stdout or "ERROR" in result.stdout:
                results.append(TestResult(
                    test_name="pytest_run",
                    status="failed",
                    duration=duration,
                    message="Some tests failed",
                    details={"stdout": result.stdout, "stderr": result.stderr}
                ))
            else:
                results.append(TestResult(
                    test_name="pytest_run",
                    status="passed",
                    duration=duration,
                    message="All tests passed"
                ))

        except subprocess.TimeoutExpired:
            results.append(TestResult(
                test_name="pytest_run",
                status="error",
                duration=300,
                message="Tests timed out"
            ))
        except FileNotFoundError:
            logger.debug("pytest not available")

        return results

    def _run_build_check(self) -> BuildResult:
        """Run build/compilation checks."""
        start_time = time.time()
        errors = []
        warnings = []
        output = ""

        try:
            # Python syntax check for all files
            for file_path in self._watched_files:
                try:
                    with open(file_path, 'r') as f:
                        compile(f.read(), str(file_path), 'exec')
                except SyntaxError as e:
                    errors.append(f"{file_path}:{e.lineno}: {e.msg}")

            # Check imports
            result = subprocess.run([
                sys.executable, "-c",
                "import sys; sys.path.insert(0, '.'); "
                + "; ".join([f"import {f.stem}" for f in self._watched_files if f.name != "__init__.py"])
            ], capture_output=True, text=True, cwd=self.project_root, timeout=60)

            if result.stderr:
                for line in result.stderr.split('\n'):
                    if line.strip():
                        if "warning" in line.lower():
                            warnings.append(line.strip())
                        else:
                            errors.append(line.strip())

            output = result.stdout + result.stderr

        except subprocess.TimeoutExpired:
            errors.append("Build check timed out")
        except Exception as e:
            errors.append(f"Build check error: {e}")

        duration = time.time() - start_time

        return BuildResult(
            success=len(errors) == 0,
            duration=duration,
            errors=errors,
            warnings=warnings,
            output=output
        )

    def _auto_fix_issues(self, issues: List[CodeIssue]) -> int:
        """Automatically fix fixable issues."""
        fixed_count = 0

        fixable_issues = [issue for issue in issues if issue.auto_fixable]

        for issue in fixable_issues:
            try:
                if issue.issue_type == "style" and "trailing whitespace" in issue.description.lower():
                    self._fix_trailing_whitespace(Path(issue.file_path))
                    fixed_count += 1

            except Exception as e:
                logger.error(f"Failed to auto-fix issue in {issue.file_path}: {e}")

        return fixed_count

    def _fix_trailing_whitespace(self, file_path: Path) -> None:
        """Fix trailing whitespace in a file."""
        with open(file_path, 'r') as f:
            content = f.read()

        lines = content.split('\n')
        fixed_lines = [line.rstrip() for line in lines]

        with open(file_path, 'w') as f:
            f.write('\n'.join(fixed_lines))

    def _generate_report(self) -> Dict[str, Any]:
        """Generate development status report."""
        recent_issues = [asdict(issue) for issue in self._code_issues[-20:]]
        recent_tests = [asdict(test) for test in self._test_history[-10:]]
        recent_builds = [asdict(build) for build in self._build_history[-5:]]

        # Calculate metrics
        total_issues = len(self._code_issues)
        critical_issues = len([i for i in self._code_issues if i.severity == "critical"])
        auto_fixable = len([i for i in self._code_issues if i.auto_fixable])

        last_test_result = self._test_history[-1] if self._test_history else None
        last_build_result = self._build_history[-1] if self._build_history else None

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_files_monitored": len(self._watched_files),
                "total_issues": total_issues,
                "critical_issues": critical_issues,
                "auto_fixable_issues": auto_fixable,
                "last_test_status": last_test_result.status if last_test_result else "unknown",
                "last_build_status": "success" if last_build_result and last_build_result.success else "failed"
            },
            "recent_issues": recent_issues,
            "recent_tests": recent_tests,
            "recent_builds": recent_builds,
            "config": self.config
        }

    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save development report to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.reports_dir / f"dev_report_{timestamp}.json"

            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            # Keep only last 10 reports
            reports = sorted(self.reports_dir.glob("dev_report_*.json"))
            for old_report in reports[:-10]:
                old_report.unlink()

        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def _development_loop(self) -> None:
        """Main development monitoring loop."""
        logger.info("Development monitoring started")

        while self._running:
            try:
                # Check for file changes
                changed_files = self._check_file_changes()

                if changed_files:
                    logger.info(f"Detected changes in {len(changed_files)} files")

                    # Analyze changed files
                    for file_path in changed_files:
                        issues = self._analyze_code_quality(file_path)
                        self._code_issues.extend(issues)

                        # External linting
                        if self.config["monitoring"]["auto_lint"]:
                            external_issues = self._run_external_linter(file_path)
                            self._code_issues.extend(external_issues)

                    # Auto-fix if enabled
                    if self.config["monitoring"]["auto_format"]:
                        fixed = self._auto_fix_issues(self._code_issues)
                        if fixed > 0:
                            logger.info(f"Auto-fixed {fixed} issues")

                    # Run tests if enabled
                    if self.config["testing"]["auto_run_on_change"]:
                        test_results = self._run_tests()
                        self._test_history.extend(test_results)

                    # Run build check
                    build_result = self._run_build_check()
                    self._build_history.append(build_result)

                # Generate and save report periodically
                if len(self._code_issues) > 0 or len(self._test_history) > 0:
                    report = self._generate_report()
                    self._save_report(report)

                # Clean up old issues (keep last 100)
                self._code_issues = self._code_issues[-100:]

                time.sleep(self.config["monitoring"]["check_interval"])

            except Exception as e:
                logger.error(f"Development monitoring error: {e}")
                time.sleep(5)

        logger.info("Development monitoring stopped")

    def start(self) -> None:
        """Start the development agent."""
        if self._running:
            logger.warning("Development agent is already running")
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._development_loop, daemon=True)
        self._monitor_thread.start()

        logger.info("DevelopmentAgent started")

    def stop(self) -> None:
        """Stop the development agent."""
        if not self._running:
            return

        self._running = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)

        logger.info("DevelopmentAgent stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current development status."""
        return self._generate_report()

    def run_manual_analysis(self, file_path: Path = None) -> Dict[str, Any]:
        """Run manual code analysis on specific file or entire project."""
        if file_path:
            files_to_analyze = [file_path] if file_path in self._watched_files else []
        else:
            files_to_analyze = list(self._watched_files)

        all_issues = []
        for file in files_to_analyze:
            issues = self._analyze_code_quality(file)
            all_issues.extend(issues)

        return {
            "analyzed_files": len(files_to_analyze),
            "total_issues": len(all_issues),
            "issues_by_severity": {
                "critical": len([i for i in all_issues if i.severity == "critical"]),
                "high": len([i for i in all_issues if i.severity == "high"]),
                "medium": len([i for i in all_issues if i.severity == "medium"]),
                "low": len([i for i in all_issues if i.severity == "low"])
            },
            "issues": [asdict(issue) for issue in all_issues]
        }

    def run_manual_tests(self) -> Dict[str, Any]:
        """Run manual test execution."""
        test_results = self._run_tests()
        self._test_history.extend(test_results)

        return {
            "test_count": len(test_results),
            "results": [asdict(result) for result in test_results]
        }

def create_development_agent(project_root: Path = None) -> DevelopmentAgent:
    """Factory function to create a configured DevelopmentAgent."""
    return DevelopmentAgent(project_root)

if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    dev_agent = create_development_agent()

    try:
        dev_agent.start()

        # Run for demo period
        time.sleep(30)

        status = dev_agent.get_status()
        print(f"Development Status: {json.dumps(status, indent=2)}")

    except KeyboardInterrupt:
        print("Demo interrupted")
    finally:
        dev_agent.stop()