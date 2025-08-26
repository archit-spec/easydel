"""
Tool System for GSPO Training with Tool-Enabled Rollouts

This module provides a framework for enabling tool interactions during GSPO training rollouts.
It allows models to call external tools and receive results, which are then incorporated
into the conversation flow for more dynamic and capable policy learning.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from math_verify import parse, verify
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer


@dataclass
class ToolParameter:
    """Represents a parameter for a tool."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """Represents a tool definition with its parameters and metadata."""
    name: str
    description: str
    parameters: List[ToolParameter]
    strict: bool = False


@dataclass
class ToolCall:
    """Represents a tool call made by the model."""
    id: str
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    tool_call_id: str
    content: Any
    success: bool = True
    error: Optional[str] = None


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.parameters: List[ToolParameter] = []

    def add_parameter(self, name: str, type: str, description: str, required: bool = True, default: Any = None):
        """Add a parameter to the tool."""
        self.parameters.append(ToolParameter(name, type, description, required, default))

    def get_definition(self) -> ToolDefinition:
        """Get the tool definition for this tool."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters
        )

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with the given arguments."""
        pass


class CalculatorTool(BaseTool):
    """A calculator tool for mathematical computations."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="A calculator tool for performing mathematical computations. Supports basic arithmetic, algebra, and calculus operations."
        )
        self.add_parameter("expression", "string", "The mathematical expression to evaluate", required=True)

    def execute(self, expression: str) -> str:
        """Execute a mathematical calculation."""
        try:
            # Basic security: only allow safe mathematical operations
            allowed_names = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'len': len, 'pow': pow, 'divmod': divmod,
                'math': __import__('math'), 'sympy': __import__('sympy')
            }

            # Evaluate the expression in a restricted environment
            result = eval(expression, {"__builtins__": {}}, allowed_names)

            # Format the result nicely
            if isinstance(result, (int, float)):
                return f"\\boxed{{{result}}}"
            else:
                return str(result)

        except Exception as e:
            return f"Error in calculation: {str(e)}"


class SearchTool(BaseTool):
    """A search tool for looking up information."""

    def __init__(self):
        super().__init__(
            name="search",
            description="A search tool for looking up mathematical concepts, formulas, and definitions."
        )
        self.add_parameter("query", "string", "The search query", required=True)
        self.add_parameter("category", "string", "The category to search in (e.g., 'algebra', 'calculus', 'geometry')", required=False, default="general")

    def execute(self, query: str, category: str = "general") -> str:
        """Execute a search query."""
        # This is a simplified implementation - in practice, you'd connect to a real search API
        search_results = {
            "pythagorean theorem": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a² + b² = c²",
            "quadratic formula": "The quadratic formula is x = (-b ± √(b² - 4ac)) / (2a) for solving ax² + bx + c = 0",
            "derivative": "A derivative represents the instantaneous rate of change of a function at a specific point.",
            "integral": "An integral represents the accumulation of quantities or the area under a curve.",
        }

        query_lower = query.lower().strip()
        for key, result in search_results.items():
            if key in query_lower:
                return result

        return f"No specific information found for '{query}'. This is a simplified search tool for demonstration."


class ToolManager:
    """Manages available tools and their execution."""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register the default set of tools."""
        self.register_tool(CalculatorTool())
        self.register_tool(SearchTool())

    def register_tool(self, tool: BaseTool):
        """Register a new tool."""
        self.tools[tool.name] = tool

    def get_tool_definitions(self) -> List[ToolDefinition]:
        """Get definitions for all registered tools."""
        return [tool.get_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        if tool_call.tool_name not in self.tools:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=None,
                success=False,
                error=f"Tool '{tool_call.tool_name}' not found"
            )

        tool = self.tools[tool_call.tool_name]
        try:
            result = tool.execute(**tool_call.arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                content=result,
                success=True
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=None,
                success=False,
                error=str(e)
            )


class ToolConversationFormatter:
    """Handles formatting conversations with tool calls and results."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.tool_call_pattern = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)
        self.tool_result_pattern = re.compile(r'<tool_result>\s*(\{.*?\})\s*</tool_result>', re.DOTALL)

    def format_tool_call(self, tool_call: ToolCall) -> str:
        """Format a tool call for inclusion in the conversation."""
        call_data = {
            "id": tool_call.id,
            "tool_name": tool_call.tool_name,
            "arguments": tool_call.arguments
        }
        return f"<tool_call>{json.dumps(call_data)}</tool_call>"

    def format_tool_result(self, tool_result: ToolResult) -> str:
        """Format a tool result for inclusion in the conversation."""
        result_data = {
            "tool_call_id": tool_result.tool_call_id,
            "content": tool_result.content,
            "success": tool_result.success
        }
        if tool_result.error:
            result_data["error"] = tool_result.error
        return f"<tool_result>{json.dumps(result_data)}</tool_result>"

    def extract_tool_calls(self, text: str) -> List[ToolCall]:
        """Extract tool calls from generated text."""
        tool_calls = []
        matches = self.tool_call_pattern.findall(text)

        for match in matches:
            try:
                call_data = json.loads(match)
                tool_calls.append(ToolCall(
                    id=call_data["id"],
                    tool_name=call_data["tool_name"],
                    arguments=call_data["arguments"]
                ))
            except (json.JSONDecodeError, KeyError):
                continue

        return tool_calls

    def has_tool_calls(self, text: str) -> bool:
        """Check if the text contains tool calls."""
        return bool(self.tool_call_pattern.search(text))

    def clean_tool_markup(self, text: str) -> str:
        """Remove tool markup from text for display purposes."""
        text = self.tool_call_pattern.sub("", text)
        text = self.tool_result_pattern.sub("", text)
        return text.strip()


class ToolAwareRewardFunction:
    """A reward function that considers tool usage in evaluations."""

    def __init__(self, base_reward_fn: Callable, tool_manager: ToolManager):
        self.base_reward_fn = base_reward_fn
        self.tool_manager = tool_manager

    def __call__(self, prompts, completions, **kwargs):
        """Calculate rewards considering tool usage."""
        rewards = []

        for prompt, completion in zip(prompts, completions):
            # Extract tool calls from completion
            tool_calls = self._extract_tool_calls_from_completion(completion)

            # Calculate base reward - pass both prompts and completions to the base reward function
            base_reward = self.base_reward_fn(prompts=[prompt], completions=[completion], **kwargs)[0]

            # Apply tool usage bonuses/penalties
            tool_bonus = self._calculate_tool_bonus(tool_calls, completion)

            total_reward = base_reward + tool_bonus
            rewards.append(total_reward)

        return rewards

    def _extract_tool_calls_from_completion(self, completion) -> List[ToolCall]:
        """Extract tool calls from a completion."""
        formatter = ToolConversationFormatter(None)  # We don't need tokenizer for extraction
        if isinstance(completion, str):
            return formatter.extract_tool_calls(completion)
        elif isinstance(completion, dict) and "content" in completion:
            return formatter.extract_tool_calls(completion["content"])
        elif isinstance(completion, list) and len(completion) > 0:
            return formatter.extract_tool_calls(completion[0]["content"])
        return []

    def _calculate_tool_bonus(self, tool_calls: List[ToolCall], completion) -> float:
        """Calculate bonus/penalty for tool usage."""
        if not tool_calls:
            return 0.0

        bonus = 0.0

        # Bonus for using tools (encourages tool usage)
        tool_usage_bonus = 0.1 * len(tool_calls)

        # Bonus for successful tool usage
        successful_tools = sum(1 for call in tool_calls if call.tool_name in self.tool_manager.tools)
        success_bonus = 0.2 * successful_tools

        # Penalty for malformed tool calls
        malformed_penalty = -0.1 * max(0, len(tool_calls) - successful_tools)

        bonus = tool_usage_bonus + success_bonus + malformed_penalty

        return bonus


def create_tool_system() -> ToolManager:
    """Create and return a configured tool system."""
    return ToolManager()


def create_tool_aware_reward_function(base_reward_fn: Callable, tool_manager: ToolManager) -> ToolAwareRewardFunction:
    """Create a tool-aware reward function."""
    return ToolAwareRewardFunction(base_reward_fn, tool_manager)