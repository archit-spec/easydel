# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import subprocess
import time
import typing as tp
from dataclasses import dataclass
from pathlib import Path

from easydel.utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Metrics from code quality evaluation."""
    linter_score: float = 0.0
    test_pass_rate: float = 0.0
    ci_success: bool = False
    linter_errors: int = 0
    test_failures: int = 0
    execution_time: float = 0.0
    linter_output: str = ""
    test_output: str = ""
    ci_output: str = ""


@dataclass
class CodeEvaluationResult:
    """Result from code quality evaluation."""
    before_metrics: QualityMetrics
    after_metrics: QualityMetrics
    improvement_score: float = 0.0
    overall_score: float = 0.0


class CodeQualityEvaluator:
    """
    Evaluates code quality by running linters, tests, and CI commands.
    Calculates before/after metrics for reward computation.
    """

    def __init__(
        self,
        enabled_linters: list[str] | None = None,
        linter_configs: dict[str, dict] | None = None,
        test_commands: list[str] | None = None,
        ci_commands: list[str] | None = None,
        test_timeout: int = 120,
        ci_timeout: int = 600,
    ):
        self.enabled_linters = enabled_linters or ["flake8", "pylint", "mypy"]
        self.linter_configs = linter_configs or {}
        self.test_commands = test_commands or ["python -m pytest", "npm test", "jest"]
        self.ci_commands = ci_commands or ["make ci", "npm run ci", "./ci.sh"]
        self.test_timeout = test_timeout
        self.ci_timeout = ci_timeout

    def _run_command(
        self,
        cmd: str | list[str],
        cwd: Path | str,
        timeout: int,
        env: dict[str, str] | None = None
    ) -> tuple[bool, str, str, float]:
        """
        Run a command and return success status, stdout, stderr, and execution time.
        """
        start_time = time.time()

        if isinstance(cmd, str):
            cmd = cmd.split()

        try:
            env_vars = os.environ.copy()
            if env:
                env_vars.update(env)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=env_vars,
            )

            stdout, stderr = process.communicate(timeout=timeout)
            success = process.returncode == 0
            execution_time = time.time() - start_time

            return success, stdout, stderr, execution_time

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return False, "", f"Command timed out after {timeout} seconds", execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            return False, "", str(e), execution_time

    def _run_linters(self, workspace: Path) -> tuple[float, int, str]:
        """
        Run enabled linters and return score, error count, and combined output.
        """
        total_score = 0.0
        total_errors = 0
        combined_output = ""

        for linter in self.enabled_linters:
            score, errors, output = self._run_single_linter(linter, workspace)
            total_score += score
            total_errors += errors
            combined_output += f"\n--- {linter.upper()} ---\n{output}"

        # Normalize score by number of linters
        if self.enabled_linters:
            avg_score = total_score / len(self.enabled_linters)
        else:
            avg_score = 1.0

        return avg_score, total_errors, combined_output

    def _run_single_linter(self, linter: str, workspace: Path) -> tuple[float, int, str]:
        """Run a single linter and return score, error count, and output."""
        linter_cmds = {
            "flake8": ["flake8", "--max-line-length=100", "--extend-ignore=E203,W503"],
            "pylint": ["pylint", "--output-format=text", "--reports=no"],
            "mypy": ["mypy", "--ignore-missing-imports"],
            "eslint": ["npx", "eslint", "--format=compact"],
            "prettier": ["npx", "prettier", "--check"],
        }

        if linter not in linter_cmds:
            logger.warning(f"Unknown linter: {linter}")
            return 1.0, 0, f"Linter {linter} not supported"

        cmd = linter_cmds[linter]

        # Apply custom config if available
        if linter in self.linter_configs:
            config = self.linter_configs[linter]
            # Apply config to command (simplified - could be extended)
            if "max-line-length" in config and linter == "flake8":
                cmd = ["flake8", f"--max-line-length={config['max-line-length']}"]

        success, stdout, stderr, _ = self._run_command(cmd, workspace, timeout=60)

        output = stdout + stderr

        if not success:
            # Parse error count from output
            error_count = self._parse_linter_errors(linter, output)
            # Score based on errors (lower is better, but we want higher scores for better code)
            score = max(0.0, 1.0 - min(error_count / 10.0, 1.0))  # Cap at 10 errors
        else:
            score = 1.0
            error_count = 0

        return score, error_count, output

    def _parse_linter_errors(self, linter: str, output: str) -> int:
        """Parse error count from linter output."""
        if linter == "flake8":
            # Count lines that look like errors
            lines = output.split('\n')
            return len([line for line in lines if ':' in line and line.split(':')[1].strip().isdigit()])
        elif linter == "pylint":
            # Look for "Your code has been rated" line
            match = re.search(r'Your code has been rated at ([0-9.]+)/10', output)
            if match:
                return int(10 - float(match.group(1)))  # Convert to error-like count
            return 0
        elif linter == "mypy":
            # Count error lines
            lines = output.split('\n')
            return len([line for line in lines if 'error:' in line.lower()])
        else:
            # Generic error counting
            lines = output.split('\n')
            return len([line for line in lines if 'error' in line.lower() or 'warning' in line.lower()])

    def _run_tests(self, workspace: Path) -> tuple[float, int, str]:
        """
        Run test commands and return pass rate, failure count, and combined output.
        """
        combined_output = ""
        total_passed = 0
        total_failed = 0

        for cmd in self.test_commands:
            success, stdout, stderr, _ = self._run_command(cmd, workspace, self.test_timeout)
            output = stdout + stderr
            combined_output += f"\n--- {cmd} ---\n{output}"

            if success:
                # Try to parse test results
                passed, failed = self._parse_test_results(cmd, output)
                total_passed += passed
                total_failed += failed
            else:
                total_failed += 1  # Assume at least one failure if command failed

        total_tests = total_passed + total_failed
        if total_tests > 0:
            pass_rate = total_passed / total_tests
        else:
            pass_rate = 1.0 if all(success for cmd in self.test_commands) else 0.0

        return pass_rate, total_failed, combined_output

    def _parse_test_results(self, cmd: str, output: str) -> tuple[int, int]:
        """Parse test results from command output."""
        if "pytest" in cmd:
            # Look for pytest summary
            match = re.search(r'(\d+) passed(?:, (\d+) failed)?', output)
            if match:
                passed = int(match.group(1))
                failed = int(match.group(2)) if match.group(2) else 0
                return passed, failed
        elif "jest" in cmd or "npm test" in cmd:
            # Look for jest summary
            match = re.search(r'Tests:\s*(\d+)\s*passed,\s*(\d+)\s*failed', output)
            if match:
                passed = int(match.group(1))
                failed = int(match.group(2))
                return passed, failed

        # Default: assume success means some tests passed
        return 1, 0

    def _run_ci(self, workspace: Path) -> tuple[bool, str]:
        """
        Run CI commands and return success status and combined output.
        """
        combined_output = ""

        for cmd in self.ci_commands:
            success, stdout, stderr, _ = self._run_command(cmd, workspace, self.ci_timeout)
            output = stdout + stderr
            combined_output += f"\n--- {cmd} ---\n{output}"

            if not success:
                return False, combined_output

        return True, combined_output

    def evaluate_quality(self, workspace: Path) -> QualityMetrics:
        """
        Evaluate code quality for a given workspace.
        """
        start_time = time.time()

        # Run linters
        linter_score, linter_errors, linter_output = self._run_linters(workspace)

        # Run tests
        test_pass_rate, test_failures, test_output = self._run_tests(workspace)

        # Run CI
        ci_success, ci_output = self._run_ci(workspace)

        execution_time = time.time() - start_time

        return QualityMetrics(
            linter_score=linter_score,
            test_pass_rate=test_pass_rate,
            ci_success=ci_success,
            linter_errors=linter_errors,
            test_failures=test_failures,
            execution_time=execution_time,
            linter_output=linter_output,
            test_output=test_output,
            ci_output=ci_output,
        )

    def evaluate_improvement(
        self,
        before_workspace: Path,
        after_workspace: Path
    ) -> CodeEvaluationResult:
        """
        Evaluate code quality before and after changes to compute improvement.
        """
        before_metrics = self.evaluate_quality(before_workspace)
        after_metrics = self.evaluate_quality(after_workspace)

        # Calculate improvement scores
        linter_improvement = after_metrics.linter_score - before_metrics.linter_score
        test_improvement = after_metrics.test_pass_rate - before_metrics.test_pass_rate
        ci_improvement = 1.0 if after_metrics.ci_success else 0.0
        ci_improvement -= 1.0 if before_metrics.ci_success else 0.0

        # Weighted improvement score
        improvement_score = (
            0.4 * linter_improvement +
            0.4 * test_improvement +
            0.2 * ci_improvement
        )

        # Overall score (after metrics weighted)
        overall_score = (
            0.4 * after_metrics.linter_score +
            0.4 * after_metrics.test_pass_rate +
            0.2 * (1.0 if after_metrics.ci_success else 0.0)
        )

        return CodeEvaluationResult(
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_score=improvement_score,
            overall_score=overall_score,
        )