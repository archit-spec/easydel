"""
Tool-Enabled GSPO Training Demo

This script demonstrates how to use the tool-enabled GSPO trainer for training
models that can interact with external tools during rollouts.
"""

import json
from tool_system import (
    ToolManager,
    CalculatorTool,
    SearchTool,
    ToolConversationFormatter,
    create_tool_system
)
from tool_enabled_gspo import ToolEnabledGSPOTrainer, create_tool_enabled_system_prompt
import easydel as ed
from transformers import AutoTokenizer


def demonstrate_tool_system():
    """Demonstrate the tool system capabilities."""
    print("=== Tool System Demonstration ===\n")

    # Create tool manager
    tool_manager = create_tool_system()

    # Show available tools
    print("Available Tools:")
    for tool_def in tool_manager.get_tool_definitions():
        print(f"- {tool_def.name}: {tool_def.description}")
        if tool_def.parameters:
            print("  Parameters:")
            for param in tool_def.parameters:
                req = "required" if param.required else "optional"
                print(f"    - {param.name} ({param.type}, {req}): {param.description}")
        print()

    # Demonstrate calculator tool
    print("=== Calculator Tool Demo ===")
    calc_tool = CalculatorTool()
    result = calc_tool.execute(expression="2 * (3 + 4) ** 2")
    print(f"Expression: 2 * (3 + 4) ** 2")
    print(f"Result: {result}\n")

    # Demonstrate search tool
    print("=== Search Tool Demo ===")
    search_tool = SearchTool()
    result = search_tool.execute(query="pythagorean theorem", category="geometry")
    print(f"Query: pythagorean theorem")
    print(f"Result: {result}\n")


def demonstrate_conversation_formatting():
    """Demonstrate conversation formatting with tool calls."""
    print("=== Conversation Formatting Demo ===\n")

    # Create a mock tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

    # Create conversation formatter
    formatter = ToolConversationFormatter(tokenizer)

    # Example conversation with tool calls
    example_text = """
Let me solve this step by step.

First, I need to calculate the area of this triangle.
<tool_call>{"id": "call_1", "tool_name": "calculator", "arguments": {"expression": "0.5 * 3 * 4"}}</tool_call>

Now I need to look up the formula for the volume.
<tool_call>{"id": "call_2", "tool_name": "search", "arguments": {"query": "triangle area formula"}}</tool_call>

Based on the results, the area is 6.0 and the formula is correct.
<answer>6.0</answer>
"""

    print("Example conversation with tool calls:")
    print(example_text)

    # Extract tool calls
    tool_calls = formatter.extract_tool_calls(example_text)
    print(f"\nExtracted {len(tool_calls)} tool calls:")
    for i, call in enumerate(tool_calls, 1):
        print(f"{i}. {call.tool_name}: {call.arguments}")

    # Check for tool calls
    has_tools = formatter.has_tool_calls(example_text)
    print(f"\nContains tool calls: {has_tools}")

    # Clean markup
    clean_text = formatter.clean_tool_markup(example_text)
    print(f"\nClean text: {clean_text}\n")


def demonstrate_system_prompt():
    """Demonstrate the tool-enabled system prompt."""
    print("=== System Prompt Demo ===\n")

    # Create tool manager
    tool_manager = create_tool_system()

    # Get tool definitions
    tool_definitions = [tool.get_definition().__dict__ for tool in tool_manager.tools.values()]

    # Create enhanced system prompt
    system_prompt = create_tool_enabled_system_prompt(tool_definitions)

    print("Enhanced System Prompt:")
    print("=" * 50)
    print(system_prompt)
    print("=" * 50)


def demonstrate_tool_aware_reward():
    """Demonstrate tool-aware reward calculation."""
    print("=== Tool-Aware Reward Demo ===\n")

    from tool_system import create_tool_aware_reward_function

    # Create tool manager
    tool_manager = create_tool_system()

    # Simple base reward function
    def base_reward_function(prompts, completions, **kwargs):
        rewards = []
        for completion in completions:
            # Simple reward based on length (longer = better for demo)
            if isinstance(completion, str):
                reward = len(completion) * 0.01
            else:
                reward = 0.5
            rewards.append(reward)
        return rewards

    # Create tool-aware reward function
    tool_aware_reward = create_tool_aware_reward_function(base_reward_function, tool_manager)

    # Test completions with and without tools
    completion_without_tools = "The answer is 42."
    completion_with_tools = """
Let me calculate this step by step.
<tool_call>{"id": "call_1", "tool_name": "calculator", "arguments": {"expression": "6 * 7"}}</tool_call>
<tool_result>{"tool_call_id": "call_1", "content": "42", "success": true}</tool_result>
The answer is 42.
"""

    prompts = ["What is 6 times 7?"]

    # Calculate rewards
    reward_without = tool_aware_reward(prompts, [completion_without_tools])
    reward_with = tool_aware_reward(prompts, [completion_with_tools])

    print(f"Completion without tools: {completion_without_tools}")
    print(f"Reward: {reward_without[0]:.3f}")
    print()
    print(f"Completion with tools: {completion_with_tools.strip()}")
    print(f"Reward: {reward_with[0]:.3f}")
    print()
    print(f"Tool usage bonus: {reward_with[0] - reward_without[0]:.3f}")


def main():
    """Run all demonstrations."""
    print("Tool-Enabled GSPO Training Demonstration")
    print("=" * 50)

    try:
        demonstrate_tool_system()
        demonstrate_conversation_formatting()
        demonstrate_system_prompt()
        demonstrate_tool_aware_reward()

        print("\n=== Demo Complete ===")
        print("The tool-enabled GSPO trainer is ready to use!")
        print("To use it in training, run: python tool_enabled_gspo.py")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()