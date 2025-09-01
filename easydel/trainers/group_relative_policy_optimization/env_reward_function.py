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
import shutil
import tempfile
import typing as tp
from pathlib import Path

from easydel.utils.helpers import get_logger

from .claude_code_executor import ClaudeCodeExecutor, ExecutionRequest
from .code_evaluator import CodeQualityEvaluator, CodeEvaluationResult
from .claude_code_config import ClaudeCodeConfig

logger = get_logger(__name__)


class EnvironmentRewardFunction:
    """
    Implements reward function that uses Claude Code execution and code quality evaluation.
    Handles workspace management and integrates with GRPO training loop.
    """

    def __init__(self, config: ClaudeCodeConfig):
        self.config = config

        # Initialize components
        self.executor = ClaudeCodeExecutor(
            claude_path=config.claude_code_path,
            max_parallel=config.max_parallel_executions,
            default_timeout=config.claude_code_timeout,
            workspace_root=config.claude_code_workspace_root,
            use_temp_workspace=config.use_temp_workspace,
            cleanup_workspaces=config.workspace_cleanup,
        )

        self.evaluator = CodeQualityEvaluator(
            enabled_linters=config.enabled_linters,
            linter_configs=config.linter_configs,
            test_commands=config.test_commands,
            ci_commands=config.ci_commands,
            test_timeout=config.test_timeout,
            ci_timeout=config.ci_timeout,
        )

        # Base workspace for comparison (if using real workspace)
        self.base_workspace = Path(config.claude_code_workspace_root) if config.claude_code_workspace_root else None

    def _prepare_workspace(self, workspace_path: str | None = None) -> Path:
        """Prepare a workspace for evaluation."""
        if workspace_path:
            return Path(workspace_path)
        elif self.base_workspace:
            return self.base_workspace
        else:
            # Create temp workspace
            return Path(tempfile.mkdtemp(prefix="env_reward_workspace_"))

    def _copy_base_to_workspace(self, base_path: Path, workspace: Path):
        """Copy base workspace contents to evaluation workspace."""
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, workspace / item.name)
                elif item.is_dir():
                    shutil.copytree(item, workspace / item.name, dirs_exist_ok=True)

    def compute_reward(
        self,
        prompts: list[str],
        completions: list[str],
        **kwargs
    ) -> list[float]:
        """
        Compute rewards for a batch of prompt-completion pairs using Claude Code execution.

        Args:
            prompts: List of prompts (should be coding tasks)
            completions: List of completions (code to execute)
            **kwargs: Additional arguments from training

        Returns:
            List of reward values for each prompt-completion pair
        """
        rewards = []

        # Prepare execution requests
        requests = []
        for prompt, completion in zip(prompts, completions):
            # Combine prompt and completion as the Claude Code command
            # The completion should be a coding task description
            claude_prompt = f"{completion.strip()}"

            request = ExecutionRequest(
                prompt=claude_prompt,
                workspace_path=str(self.base_workspace) if self.base_workspace else None,
                timeout=self.config.claude_code_timeout,
                capture_traces=self.config.capture_tool_traces,
            )
            requests.append(request)

        # Execute Claude Code commands in parallel
        results = self.executor.execute(requests)

        # Compute rewards for each result
        for i, result in enumerate(results):
            try:
                reward = self._compute_single_reward(result, prompts[i], completions[i])
                rewards.append(reward)
            except Exception as e:
                logger.warning(f"Failed to compute reward for trajectory {i}: {e}")
                rewards.append(self.config.baseline_reward)

        return rewards

    def _compute_single_reward(
        self,
        claude_result: "ClaudeCodeResult",
        prompt: str,
        completion: str
    ) -> float:
        """
        Compute reward for a single Claude Code execution result.
        """
        if not claude_result.success:
            # Failed execution gets baseline reward
            return self.config.baseline_reward

        # Prepare workspaces for before/after comparison
        base_workspace = self._prepare_workspace(claude_result.workspace_path)

        # For improvement calculation, we need before and after states
        # Since Claude Code modifies the workspace in-place, we need to:
        # 1. Create a copy of the base workspace
        # 2. Evaluate the copy (before state)
        # 3. Evaluate the modified workspace (after state)
        # 4. Compare the results

        if self.base_workspace and self.base_workspace.exists():
            # Create before workspace (copy of base)
            before_workspace = Path(tempfile.mkdtemp(prefix="before_workspace_"))
            self._copy_base_to_workspace(self.base_workspace, before_workspace)

            # After workspace is the one modified by Claude Code
            after_workspace = base_workspace

            # Evaluate improvement
            evaluation_result = self.evaluator.evaluate_improvement(before_workspace, after_workspace)

            # Clean up before workspace
            shutil.rmtree(before_workspace, ignore_errors=True)

            # Compute weighted reward
            reward = (
                self.config.linter_weight * evaluation_result.after_metrics.linter_score +
                self.config.test_weight * evaluation_result.after_metrics.test_pass_rate +
                self.config.ci_weight * (1.0 if evaluation_result.after_metrics.ci_success else 0.0)
            )

            # Add improvement bonus
            reward += evaluation_result.improvement_score * 0.1  # Small bonus for improvement

        else:
            # No base workspace, just evaluate the result workspace
            after_metrics = self.evaluator.evaluate_quality(base_workspace)

            reward = (
                self.config.linter_weight * after_metrics.linter_score +
                self.config.test_weight * after_metrics.test_pass_rate +
                self.config.ci_weight * (1.0 if after_metrics.ci_success else 0.0)
            )

        # Apply scaling and baseline
        reward = reward * self.config.reward_scale + self.config.baseline_reward

        # Log execution details
        logger.debug(
            f"Claude Code execution: success={claude_result.success}, "
            f"execution_time={claude_result.execution_time:.2f}s, "
            f"reward={reward:.4f}"
        )

        return float(reward)

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        **kwargs
    ) -> list[float]:
        """
        Callable interface for reward function.
        """
        return self.compute_reward(prompts, completions, **kwargs)

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.executor, '__del__'):
            self.executor.__del__()