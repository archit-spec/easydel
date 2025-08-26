"""
Complete Tool-Enabled GSPO Integration Example

This script demonstrates the complete integration of tool-enabled GSPO training,
including custom tools, enhanced reward functions, and full training pipeline.
"""

import json
from typing import List, Dict, Any
from tool_system import (
    ToolManager,
    BaseTool,
    ToolConversationFormatter,
    create_tool_system,
    create_tool_aware_reward_function
)
from tool_enabled_gspo import ToolEnabledGSPOTrainer, create_tool_enabled_system_prompt
import easydel as ed
from transformers import AutoTokenizer
from math_verify import parse, verify
import jax.numpy as jnp


class AdvancedCalculatorTool(BaseTool):
    """An advanced calculator tool with more capabilities."""

    def __init__(self):
        super().__init__(
            name="advanced_calculator",
            description="An advanced calculator supporting algebra, calculus, and symbolic math operations."
        )
        self.add_parameter("expression", "string", "The mathematical expression to evaluate", required=True)
        self.add_parameter("mode", "string", "Calculation mode: 'numeric', 'symbolic', or 'step_by_step'", required=False, default="numeric")

    def execute(self, expression: str, mode: str = "numeric") -> str:
        """Execute advanced mathematical calculations."""
        try:
            if mode == "step_by_step":
                return self._step_by_step_calculation(expression)
            elif mode == "symbolic":
                return self._symbolic_calculation(expression)
            else:
                return self._numeric_calculation(expression)
        except Exception as e:
            return f"Error in advanced calculation: {str(e)}"

    def _numeric_calculation(self, expression: str) -> str:
        """Perform numeric calculation."""
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'len': len, 'pow': pow, 'divmod': divmod,
            'math': __import__('math'), 'sympy': __import__('sympy')
        }

        result = eval(expression, {"__builtins__": {}}, allowed_names)

        if isinstance(result, (int, float)):
            return f"\\boxed{{{result}}}"
        else:
            return str(result)

    def _symbolic_calculation(self, expression: str) -> str:
        """Perform symbolic calculation using sympy."""
        try:
            import sympy as sp
            result = sp.sympify(expression)
            return f"\\boxed{{{sp.latex(result)}}}"
        except:
            return self._numeric_calculation(expression)

    def _step_by_step_calculation(self, expression: str) -> str:
        """Provide step-by-step calculation."""
        try:
            import sympy as sp
            expr = sp.sympify(expression)
            steps = []

            # Simplify if possible
            simplified = sp.simplify(expr)
            if simplified != expr:
                steps.append(f"Simplified: {expr} = {simplified}")

            # If it's an equation, try to solve it
            if '=' in expression:
                left, right = expression.split('=')
                left_expr = sp.sympify(left.strip())
                right_expr = sp.sympify(right.strip())
                solution = sp.solve(left_expr - right_expr)
                steps.append(f"Solution: {solution}")

            return " | ".join(steps) if steps else str(expr)

        except:
            return self._numeric_calculation(expression)


class GeometryTool(BaseTool):
    """A geometry tool for geometric calculations and properties."""

    def __init__(self):
        super().__init__(
            name="geometry_tool",
            description="A geometry tool for calculating areas, volumes, and properties of geometric shapes."
        )
        self.add_parameter("shape", "string", "The geometric shape (circle, triangle, rectangle, etc.)", required=True)
        self.add_parameter("operation", "string", "The operation to perform (area, perimeter, volume, etc.)", required=True)
        self.add_parameter("parameters", "string", "Parameters as comma-separated values", required=True)

    def execute(self, shape: str, operation: str, parameters: str) -> str:
        """Execute geometric calculations."""
        try:
            params = [float(p.strip()) for p in parameters.split(',')]

            if shape.lower() == "circle" and operation.lower() == "area":
                if len(params) != 1:
                    return "Circle area requires 1 parameter: radius"
                radius = params[0]
                area = 3.14159 * radius * radius
                return f"\\boxed{{{area}}}"

            elif shape.lower() == "triangle" and operation.lower() == "area":
                if len(params) != 2:
                    return "Triangle area requires 2 parameters: base, height"
                base, height = params
                area = 0.5 * base * height
                return f"\\boxed{{{area}}}"

            elif shape.lower() == "rectangle" and operation.lower() == "area":
                if len(params) != 2:
                    return "Rectangle area requires 2 parameters: length, width"
                length, width = params
                area = length * width
                return f"\\boxed{{{area}}}"

            else:
                return f"Unsupported operation: {operation} for shape: {shape}"

        except Exception as e:
            return f"Error in geometric calculation: {str(e)}"


def create_enhanced_tool_system() -> ToolManager:
    """Create an enhanced tool system with additional tools."""
    tool_manager = create_tool_system()

    # Register additional tools
    tool_manager.register_tool(AdvancedCalculatorTool())
    tool_manager.register_tool(GeometryTool())

    return tool_manager


def enhanced_format_reward(completions, **kwargs):
    """Enhanced format reward that checks for proper tool usage and answer format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards_list = []

    for content in completion_contents:
        matches = re.match(pattern, content)
        base_reward = 1.0 if matches else 0.0

        # Bonus for tool usage
        tool_usage_bonus = 0.2 if "<tool_call>" in content else 0.0

        # Bonus for proper tool result integration
        tool_result_bonus = 0.1 if "<tool_result>" in content else 0.0

        total_reward = base_reward + tool_usage_bonus + tool_result_bonus
        rewards_list.append(total_reward)

    return rewards_list


def enhanced_accuracy_reward(prompts, completions, batch, **kwargs):
    """Enhanced accuracy reward that considers tool-assisted solutions."""
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, solution in zip(completion_contents, solutions, strict=False):
        gold_parsed = parse(
            solution,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        answer_parsed = parse(
            content,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        if len(gold_parsed) != 0:
            try:
                accuracy = float(verify(answer_parsed, gold_parsed))
            except Exception:
                accuracy = 0.0
        else:
            accuracy = 1.0

        # Bonus for using tools to arrive at correct answer
        tool_bonus = 0.1 if "<tool_call>" in content and accuracy > 0.5 else 0.0

        # Bonus for complex calculations (heuristic)
        complexity_bonus = 0.05 if any(op in content for op in ["**", "sqrt", "integral", "derivative"]) else 0.0

        total_reward = accuracy + tool_bonus + complexity_bonus
        rewards.append(total_reward)

    return rewards


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    sample_problems = [
        {
            "problem": "Calculate the area of a circle with radius 5.",
            "solution": "\\boxed{78.54}"
        },
        {
            "problem": "Solve for x: 2x + 3 = 11",
            "solution": "\\boxed{4}"
        },
        {
            "problem": "Find the area of a triangle with base 6 and height 8.",
            "solution": "\\boxed{24}"
        },
        {
            "problem": "Calculate 15 squared plus 20 squared.",
            "solution": "\\boxed{625}"
        }
    ]

    return sample_problems


def demonstrate_full_integration():
    """Demonstrate the complete tool-enabled GSPO integration."""
    print("=== Tool-Enabled GSPO Full Integration Demo ===\n")

    # Create enhanced tool system
    print("1. Creating Enhanced Tool System...")
    tool_manager = create_enhanced_tool_system()

    print("Available Tools:")
    for tool_def in tool_manager.get_tool_definitions():
        print(f"  - {tool_def.name}: {tool_def.description}")
    print()

    # Create enhanced system prompt
    print("2. Creating Enhanced System Prompt...")
    tool_definitions = [tool.get_definition().__dict__ for tool in tool_manager.tools.values()]
    system_prompt = create_tool_enabled_system_prompt(tool_definitions)
    print(f"System prompt length: {len(system_prompt)} characters")
    print()

    # Demonstrate tool execution
    print("3. Demonstrating Tool Execution...")
    calc_tool = AdvancedCalculatorTool()
    result = calc_tool.execute("2 ** 8", mode="step_by_step")
    print(f"Advanced Calculator: 2^8 = {result}")

    geom_tool = GeometryTool()
    result = geom_tool.execute("circle", "area", "5")
    print(f"Geometry Tool: Circle area with r=5 = {result}")
    print()

    # Create tool-aware reward functions
    print("4. Creating Tool-Aware Reward Functions...")
    tool_aware_format_reward = create_tool_aware_reward_function(enhanced_format_reward, tool_manager)
    tool_aware_accuracy_reward = create_tool_aware_reward_function(enhanced_accuracy_reward, tool_manager)
    print("Reward functions created successfully")
    print()

    # Create sample data
    print("5. Preparing Sample Data...")
    sample_data = create_sample_dataset()
    print(f"Sample dataset: {len(sample_data)} problems")
    print()

    print("=== Integration Complete ===")
    print("The tool-enabled GSPO system is ready for training!")
    print("\nKey Features Demonstrated:")
    print("✓ Enhanced tool system with multiple tools")
    print("✓ Tool-aware reward functions")
    print("✓ Conversation formatting with tool calls")
    print("✓ System prompt enhancement")
    print("✓ Sample dataset preparation")
    print("\nTo use in training, initialize ToolEnabledGSPOTrainer with:")
    print("- tool_manager: Your configured tool manager")
    print("- reward_funcs: Tool-aware reward functions")
    print("- enable_tool_calling: True")
    print("- max_tool_iterations: 3 (recommended)")


def main():
    """Main demonstration function."""
    print("Tool-Enabled GSPO Training - Full Integration Demo")
    print("=" * 60)

    try:
        demonstrate_full_integration()

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("\nNext Steps:")
        print("1. Configure your model and dataset")
        print("2. Initialize ToolEnabledGSPOTrainer")
        print("3. Run training with tool interactions")
        print("4. Evaluate tool usage and performance")

    except Exception as e:
        print(f"Error during integration demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()