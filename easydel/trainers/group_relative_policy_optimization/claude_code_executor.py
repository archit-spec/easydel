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

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from easydel.utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class ClaudeCodeResult:
    """Result from Claude Code execution."""
    success: bool
    output: str
    error: str
    tool_traces: list[dict] | None = None
    execution_time: float = 0.0
    workspace_path: str | None = None


@dataclass
class ExecutionRequest:
    """Request for Claude Code execution."""
    prompt: str
    workspace_path: str | None = None
    timeout: int = 300
    capture_traces: bool = True


class ClaudeCodeExecutor:
    """
    Handles Claude Code command execution with support for parallel processing
    and workspace isolation.
    """

    def __init__(
        self,
        claude_path: str = "claude",
        max_parallel: int = 8,
        default_timeout: int = 300,
        workspace_root: str | None = None,
        use_temp_workspace: bool = True,
        cleanup_workspaces: bool = True,
    ):
        self.claude_path = claude_path
        self.max_parallel = max_parallel
        self.default_timeout = default_timeout
        self.workspace_root = Path(workspace_root) if workspace_root else None
        self.use_temp_workspace = use_temp_workspace
        self.cleanup_workspaces = cleanup_workspaces

        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="claude-exec")

        # Lock for workspace management
        self.workspace_lock = threading.Lock()

        # Track active workspaces for cleanup
        self.active_workspaces: set[Path] = set()

    def _create_workspace(self, base_workspace: Path | None = None) -> Path:
        """Create an isolated workspace for Claude Code execution."""
        with self.workspace_lock:
            if self.use_temp_workspace:
                workspace = Path(tempfile.mkdtemp(prefix="claude_workspace_"))
            else:
                if base_workspace:
                    workspace = Path(base_workspace)
                elif self.workspace_root:
                    workspace = self.workspace_root / f"claude_workspace_{int(time.time() * 1000)}"
                    workspace.mkdir(parents=True, exist_ok=True)
                else:
                    raise ValueError("No workspace path specified")

            self.active_workspaces.add(workspace)
            return workspace

    def _cleanup_workspace(self, workspace: Path):
        """Clean up a workspace after execution."""
        if not self.cleanup_workspaces or not self.use_temp_workspace:
            return

        with self.workspace_lock:
            try:
                if workspace.exists():
                    shutil.rmtree(workspace)
                self.active_workspaces.discard(workspace)
            except Exception as e:
                logger.warning(f"Failed to cleanup workspace {workspace}: {e}")

    def _execute_single(self, request: ExecutionRequest) -> ClaudeCodeResult:
        """Execute a single Claude Code command."""
        start_time = time.time()

        # Create or use workspace
        workspace = self._create_workspace(Path(request.workspace_path) if request.workspace_path else None)

        try:
            # Change to workspace directory
            original_cwd = os.getcwd()
            os.chdir(workspace)

            # Prepare command
            cmd = [self.claude_path, "-p", request.prompt]

            # Execute command
            env = os.environ.copy()
            env["CLAUDE_CODE_CAPTURE_TRACES"] = "1" if request.capture_traces else "0"

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=workspace,
            )

            try:
                stdout, stderr = process.communicate(timeout=request.timeout)
                success = process.returncode == 0
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                success = False
                stderr += f"\nTimeout after {request.timeout} seconds"

            # Parse tool traces if enabled
            tool_traces = None
            if request.capture_traces and success:
                tool_traces = self._parse_tool_traces(stdout)

            execution_time = time.time() - start_time

            return ClaudeCodeResult(
                success=success,
                output=stdout,
                error=stderr,
                tool_traces=tool_traces,
                execution_time=execution_time,
                workspace_path=str(workspace),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ClaudeCodeResult(
                success=False,
                output="",
                error=str(e),
                execution_time=execution_time,
                workspace_path=str(workspace),
            )
        finally:
            # Restore original directory
            os.chdir(original_cwd)

            # Schedule cleanup
            if self.cleanup_workspaces:
                threading.Thread(target=self._cleanup_workspace, args=(workspace,), daemon=True).start()

    def _parse_tool_traces(self, output: str) -> list[dict] | None:
        """Parse tool execution traces from Claude Code output."""
        try:
            # Look for JSON traces in output
            lines = output.split('\n')
            traces = []

            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        trace = json.loads(line)
                        if isinstance(trace, dict) and 'tool' in trace:
                            traces.append(trace)
                    except json.JSONDecodeError:
                        continue

            return traces if traces else None
        except Exception as e:
            logger.warning(f"Failed to parse tool traces: {e}")
            return None

    def execute(self, requests: list[ExecutionRequest]) -> list[ClaudeCodeResult]:
        """
        Execute multiple Claude Code commands in parallel.

        Args:
            requests: List of execution requests

        Returns:
            List of execution results in the same order as requests
        """
        if len(requests) == 1:
            # Single execution, no need for parallelism
            return [self._execute_single(requests[0])]

        # Submit parallel executions
        futures = []
        for request in requests:
            future = self.executor.submit(self._execute_single, request)
            futures.append(future)

        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=self.default_timeout + 10)  # Add buffer for collection
                results.append(result)
            except Exception as e:
                # Create error result
                results.append(ClaudeCodeResult(
                    success=False,
                    output="",
                    error=f"Execution failed: {str(e)}",
                    execution_time=0.0,
                ))

        return results

    async def execute_async(self, requests: list[ExecutionRequest]) -> list[ClaudeCodeResult]:
        """
        Async version of execute for integration with async training loops.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, requests)

    def __del__(self):
        """Cleanup executor and workspaces on destruction."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

        # Cleanup remaining workspaces
        with self.workspace_lock:
            for workspace in self.active_workspaces.copy():
                self._cleanup_workspace(workspace)