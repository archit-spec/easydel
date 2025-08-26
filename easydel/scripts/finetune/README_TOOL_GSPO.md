# Tool-Enabled GSPO Training

This directory contains an enhanced version of the GSPO (Group Relative Policy Optimization) trainer that supports tool interactions during rollouts. This allows models to call external tools and receive results, enabling more dynamic and capable policy learning for mathematical reasoning and other tool-assisted tasks.

## Overview

The tool-enabled GSPO trainer extends the standard GSPO implementation with:

- **Tool System**: A framework for defining and executing external tools
- **Tool-Aware Generation**: Modified generation process that can handle tool calls and results
- **Enhanced Reward Functions**: Reward functions that consider tool usage and effectiveness
- **Conversation Formatting**: Support for tool calls and results in conversational format

## Files

- `tool_system.py`: Core tool system with tool definitions, execution framework, and conversation formatting
- `tool_enabled_gspo.py`: Enhanced GSPO trainer with tool interaction capabilities
- `tool_demo.py`: Demonstration script showing tool system capabilities
- `numinamath_gspo.py`: Original NuminaMath GSPO script (unchanged)
- `README_TOOL_GSPO.md`: This documentation file

## Tool System Architecture

### Core Components

1. **ToolManager**: Manages available tools and their execution
2. **BaseTool**: Abstract base class for implementing tools
3. **ToolConversationFormatter**: Handles formatting tool calls and results in conversations
4. **ToolAwareRewardFunction**: Reward function wrapper that considers tool usage

### Built-in Tools

- **CalculatorTool**: Performs mathematical calculations using Python's eval in a restricted environment
- **SearchTool**: Provides mathematical concepts, formulas, and definitions

### Tool Call Format

Tool calls are formatted as JSON within `<tool_call>` tags:

```json
<tool_call>
{
  "id": "call_1",
  "tool_name": "calculator",
  "arguments": {
    "expression": "2 * (3 + 4) ** 2"
  }
}
</tool_call>
```

Tool results are formatted as JSON within `<tool_result>` tags:

```json
<tool_result>
{
  "tool_call_id": "call_1",
  "content": "\\boxed{42}",
  "success": true
}
</tool_result>
```

## Usage

### Basic Usage

```python
from tool_enabled_gspo import create_tool_enabled_numina_math_gspo

# Create tool-enabled trainer
trainer, system_prompt = create_tool_enabled_numina_math_gspo(
    runtime_config=runtime_config,
    enable_tool_calling=True,
    max_tool_iterations=3,
)
```

### Custom Tools

```python
from tool_system import ToolManager, BaseTool

class CustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="custom_tool",
            description="A custom tool for specific tasks"
        )
        self.add_parameter("input", "string", "Input parameter", required=True)

    def execute(self, input: str) -> str:
        # Implement tool logic
        return f"Processed: {input}"

# Register custom tool
tool_manager = ToolManager()
tool_manager.register_tool(CustomTool())
```

### Data Tokenization

The system handles tool messages in the data tokenization process:

```python
def data_tokenize_fn(batch, tokenizer, tools):
    # Standard tokenization with tool support
    ids = tokenizer(
        batch["prompt"],
        return_tensors="np",
        padding="max_length",
        padding_side="left",
        max_length=max_prompt_length,
        truncation=True,
        add_special_tokens=False,
    )

    # Add solution tokenization for reward calculation
    ans = tokenizer(
        batch["solution"],
        return_tensors="np",
        padding="max_length",
        padding_side="left",
        max_length=max_prompt_length,
        truncation=True,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    ids.update({"solution_ids": ans["input_ids"]})
    return ids
```

## System Prompt Enhancement

The tool-enabled system automatically enhances the system prompt with:

- Available tool descriptions and parameters
- Tool usage instructions
- Formatting guidelines for tool calls and results

Example enhanced system prompt:

```
A conversation between User and Assistant. The user asks a question, and the Assistant solves it...

## Available Tools:

### calculator
Description: A calculator tool for performing mathematical computations...
Parameters:
- expression: The mathematical expression to evaluate (required)

## Tool Usage Instructions:
- To use a tool, output a <tool_call> block with the tool name and arguments
- Tool results will be provided in <tool_result> blocks
- Always format your final answer within <think> and <answer> tags
```

## Training Process

### Tool Interaction Flow

1. **Generation**: Model generates completion with potential tool calls
2. **Tool Detection**: System detects tool calls in generated text
3. **Tool Execution**: Tools are executed and results are formatted
4. **Result Integration**: Tool results are appended to the conversation
5. **Reward Calculation**: Tool-aware rewards are calculated based on tool usage and effectiveness

### Reward Function Enhancement

The tool-aware reward function provides bonuses for:

- **Tool Usage**: Encourages models to use available tools
- **Successful Tool Calls**: Rewards proper tool execution
- **Tool Effectiveness**: Considers whether tools helped solve the problem

## Configuration Options

### Tool-Enabled GSPO Trainer Parameters

- `tool_manager`: Custom tool manager instance (optional)
- `max_tool_iterations`: Maximum number of tool calls per rollout (default: 3)
- `enable_tool_calling`: Whether to enable tool interactions (default: True)

### Runtime Configuration

The system works with the existing `RunTimeConfig` from the original script, with additional tool-related parameters handled internally.

## Running the Demo

```bash
# Run the tool system demonstration
python tool_demo.py

# Run tool-enabled training (requires proper configuration)
python tool_enabled_gspo.py --model_name your_model --dataset_config your_config
```

## Example Workflow

1. **Setup**: Configure your model and dataset as usual
2. **Tool Registration**: Register any custom tools you need
3. **Training**: Use `ToolEnabledGSPOTrainer` instead of standard `GSPOTrainer`
4. **Inference**: Deploy the trained model with tool access

## Benefits

- **Enhanced Reasoning**: Models can access external knowledge and computation
- **Modular Tools**: Easy to add new tools without changing the core training loop
- **Safety**: Tools run in controlled environments with restricted execution
- **Flexibility**: Support for various tool types (calculators, search, APIs, etc.)

## Limitations

- **Tool Call Detection**: Relies on models learning the specific tool call format
- **Execution Overhead**: Tool calls add computational overhead during training
- **Context Length**: Tool results increase the context length requirements
- **Training Stability**: Tool interactions may affect training stability

## Future Enhancements

- **Tool Call Parsing**: More robust tool call detection and parsing
- **Tool Result Caching**: Cache tool results to reduce redundant computations
- **Multi-Tool Coordination**: Support for complex tool interaction patterns
- **Tool Learning**: Models that can learn to use new tools dynamically

## Troubleshooting

### Common Issues

1. **Tool calls not detected**: Ensure models are trained on the specific tool call format
2. **Tool execution errors**: Check tool implementations and error handling
3. **Memory issues**: Reduce `max_tool_iterations` or batch size
4. **Training instability**: Adjust tool usage rewards or disable tool calling temporarily

### Debugging

Enable detailed logging to see tool interactions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

The system provides detailed logs about tool calls, executions, and results during training.