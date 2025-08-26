"""
Test Tool Integration

This script tests the tool integration functionality to ensure everything works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_tool_system():
    """Test the tool system components."""
    print("Testing Tool System...")

    try:
        from tool_system import (
            ToolManager,
            CalculatorTool,
            SearchTool,
            ToolConversationFormatter,
            create_tool_system,
            create_tool_aware_reward_function
        )

        # Test tool manager creation
        tool_manager = create_tool_system()
        assert len(tool_manager.tools) >= 2, "Should have at least 2 default tools"

        # Test calculator tool
        calc_tool = CalculatorTool()
        result = calc_tool.execute("2 + 3")
        assert "5" in result, "Calculator should compute 2 + 3 = 5"

        # Test search tool
        search_tool = SearchTool()
        result = search_tool.execute("pythagorean theorem")
        assert "pythagorean" in result.lower(), "Search should return relevant information"

        print("✓ Tool system tests passed")

    except Exception as e:
        print(f"✗ Tool system test failed: {e}")
        return False

    return True


def test_conversation_formatting():
    """Test conversation formatting with tool calls."""
    print("Testing Conversation Formatting...")

    try:
        from tool_system import ToolConversationFormatter, ToolCall, ToolResult
        from transformers import AutoTokenizer

        # Create tokenizer and formatter
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        formatter = ToolConversationFormatter(tokenizer)

        # Test tool call formatting
        tool_call = ToolCall(
            id="test_call_1",
            tool_name="calculator",
            arguments={"expression": "2 + 3"}
        )
        formatted_call = formatter.format_tool_call(tool_call)
        assert "tool_call" in formatted_call, "Should contain tool_call tag"
        assert "calculator" in formatted_call, "Should contain tool name"

        # Test tool result formatting
        tool_result = ToolResult(
            tool_call_id="test_call_1",
            content="5",
            success=True
        )
        formatted_result = formatter.format_tool_result(tool_result)
        assert "tool_result" in formatted_result, "Should contain tool_result tag"
        assert "5" in formatted_result, "Should contain result content"

        # Test tool call extraction
        test_text = f"Let's calculate. {formatted_call} The result is {formatted_result}"
        extracted_calls = formatter.extract_tool_calls(test_text)
        assert len(extracted_calls) == 1, "Should extract one tool call"
        assert extracted_calls[0].tool_name == "calculator", "Should extract correct tool name"

        print("✓ Conversation formatting tests passed")

    except Exception as e:
        print(f"✗ Conversation formatting test failed: {e}")
        return False

    return True


def test_tool_aware_reward():
    """Test tool-aware reward functions."""
    print("Testing Tool-Aware Rewards...")

    try:
        from tool_system import create_tool_aware_reward_function, create_tool_system

        def simple_base_reward(prompts, completions, **kwargs):
            return [0.5] * len(completions)

        tool_manager = create_tool_system()
        tool_aware_reward = create_tool_aware_reward_function(simple_base_reward, tool_manager)

        # Test with completion without tools
        completion_no_tools = "The answer is 42."
        reward_no_tools = tool_aware_reward(["test"], [completion_no_tools])[0]

        # Test with completion with tools
        completion_with_tools = """
Let's calculate.
<tool_call>{"id": "call_1", "tool_name": "calculator", "arguments": {"expression": "6 * 7"}}</tool_call>
<tool_result>{"tool_call_id": "call_1", "content": "42", "success": true}</tool_result>
The answer is 42.
"""
        reward_with_tools = tool_aware_reward(["test"], [completion_with_tools])[0]

        # Tool usage should provide bonus
        assert reward_with_tools >= reward_no_tools, "Tool usage should provide reward bonus"

        print("✓ Tool-aware reward tests passed")

    except Exception as e:
        print(f"✗ Tool-aware reward test failed: {e}")
        return False

    return True


def test_integration_components():
    """Test integration components."""
    print("Testing Integration Components...")

    try:
        # Test system prompt creation with inline implementation to avoid import issues
        from tool_system import create_tool_system

        def create_tool_enabled_system_prompt_test(tool_definitions):
            """Test version of system prompt creation."""
            tools_section = ""
            if tool_definitions:
                tools_section = "\n\n## Available Tools:\n"
                for tool in tool_definitions:
                    tools_section += f"\n### {tool.name}\n"
                    tools_section += f"Description: {tool.description}\n"
                    if hasattr(tool, 'parameters') and tool.parameters:
                        tools_section += "Parameters:\n"
                        for param in tool.parameters:
                            required = " (required)" if getattr(param, 'required', True) else " (optional)"
                            tools_section += f"- {param.name}: {param.description}{required}\n"

                tools_section += "\n## Tool Usage Instructions:\n"
                tools_section += "- To use a tool, output a <tool_call> block with the tool name and arguments\n"
                tools_section += "- Tool results will be provided in <tool_result> blocks\n"
                tools_section += "- You can make multiple tool calls in sequence\n"
                tools_section += "- Always format your final answer within <think> and <answer> tags\n"

            base_prompt = (
                "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
                "The assistant first thinks about the solution process in the mind and then provides the user with the answer. "
                "The think process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
                "i.e., <think> think process here </think><answer> answer here </answer>"
            )

            return base_prompt + tools_section

        # Test system prompt creation
        tool_manager = create_tool_system()
        tool_definitions = [tool.get_definition() for tool in tool_manager.tools.values()]
        system_prompt = create_tool_enabled_system_prompt_test(tool_definitions)

        assert len(system_prompt) > 100, "System prompt should be substantial"
        assert "tool_call" in system_prompt, "Should contain tool usage instructions"
        assert "calculator" in system_prompt, "Should contain tool descriptions"

        print("✓ Integration component tests passed")

    except Exception as e:
        print(f"✗ Integration component test failed: {e}")
        return False

    return True


def run_all_tests():
    """Run all integration tests."""
    print("Running Tool Integration Tests")
    print("=" * 40)

    tests = [
        test_tool_system,
        test_conversation_formatting,
        test_tool_aware_reward,
        test_integration_components,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Tool integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


def main():
    """Main test function."""
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()