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

import typing as tp
from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import hash_fn
from .grpo_config import GRPOConfig


@auto_pytree
class ClaudeCodeConfig(GRPOConfig):
    """
    Enhanced GRPO Configuration with Claude Code and environment parameters.
    Extends GRPOConfig with Claude Code execution and evaluation settings.
    """

    # Claude Code Configuration
    claude_code_enabled: bool = field(
        default=False,
        metadata={"help": "Whether to enable Claude Code execution for reward computation."},
    )

    claude_code_path: str = field(
        default="claude",
        metadata={"help": "Path to the Claude Code executable."},
    )

    claude_code_timeout: int = field(
        default=300,
        metadata={"help": "Timeout in seconds for Claude Code command execution."},
    )

    claude_code_workspace_root: str | None = field(
        default=None,
        metadata={"help": "Root directory for Claude Code workspace. If None, uses temp directories."},
    )

    # VLLM Configuration (for potential future integration)
    vllm_endpoint: str | None = field(
        default=None,
        metadata={"help": "VLLM server endpoint for Claude Code integration."},
    )

    # Code Quality Evaluation Weights
    linter_weight: float = field(
        default=0.3,
        metadata={"help": "Weight for linter score in total reward (0.0-1.0)."},
    )

    test_weight: float = field(
        default=0.4,
        metadata={"help": "Weight for test results in total reward (0.0-1.0)."},
    )

    ci_weight: float = field(
        default=0.3,
        metadata={"help": "Weight for CI pipeline results in total reward (0.0-1.0)."},
    )

    # Linter Configuration
    enabled_linters: list[str] = field(
        default_factory=lambda: ["flake8", "pylint", "mypy"],
        metadata={"help": "List of linters to run for code quality evaluation."},
    )

    linter_configs: dict[str, dict] = field(
        default_factory=dict,
        metadata={"help": "Configuration options for each linter."},
    )

    # Test Configuration
    test_commands: list[str] = field(
        default_factory=lambda: ["python -m pytest", "npm test", "jest"],
        metadata={"help": "Test commands to run for different frameworks."},
    )

    test_timeout: int = field(
        default=120,
        metadata={"help": "Timeout in seconds for test execution."},
    )

    # CI Configuration
    ci_commands: list[str] = field(
        default_factory=lambda: ["make ci", "npm run ci", "./ci.sh"],
        metadata={"help": "CI commands to run for project validation."},
    )

    ci_timeout: int = field(
        default=600,
        metadata={"help": "Timeout in seconds for CI execution."},
    )

    # Workspace Management
    use_temp_workspace: bool = field(
        default=True,
        metadata={"help": "Whether to use temporary workspaces for isolated execution."},
    )

    workspace_cleanup: bool = field(
        default=True,
        metadata={"help": "Whether to cleanup temporary workspaces after execution."},
    )

    max_parallel_executions: int = field(
        default=8,
        metadata={"help": "Maximum number of parallel Claude Code executions."},
    )

    # Reward Computation
    reward_scale: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for environment rewards."},
    )

    baseline_reward: float = field(
        default=0.0,
        metadata={"help": "Baseline reward added to all trajectories."},
    )

    # Execution Options
    capture_tool_traces: bool = field(
        default=True,
        metadata={"help": "Whether to capture Claude Code tool execution traces."},
    )

    parse_json_output: bool = field(
        default=True,
        metadata={"help": "Whether to parse JSON output from Claude Code."},
    )

    def __post_init__(self):
        """Post initialization to validate Claude Code configuration."""
        super().__post_init__()

        # Validate weights sum to 1.0
        total_weight = self.linter_weight + self.test_weight + self.ci_weight
        if not abs(total_weight - 1.0) < 1e-6:
            raise ValueError(f"Reward weights must sum to 1.0, got {total_weight}")

        # Validate timeouts
        if self.claude_code_timeout <= 0:
            raise ValueError("claude_code_timeout must be positive")
        if self.test_timeout <= 0:
            raise ValueError("test_timeout must be positive")
        if self.ci_timeout <= 0:
            raise ValueError("ci_timeout must be positive")

        # Validate parallel executions
        if self.max_parallel_executions <= 0:
            raise ValueError("max_parallel_executions must be positive")

        # Set default workspace if not provided
        if self.claude_code_workspace_root is None and not self.use_temp_workspace:
            raise ValueError("Must specify claude_code_workspace_root when use_temp_workspace=False")

    __hash__ = hash_fn