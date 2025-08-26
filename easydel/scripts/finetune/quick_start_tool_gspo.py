"""
Quick Start: Tool-Enabled GSPO Training

This script shows you exactly how to use the tool-enabled GSPO trainer
in 3 simple steps.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import easydel as ed
from tool_enabled_gspo import ToolEnabledGSPOTrainer, create_tool_enabled_numina_math_gspo
from tool_system import create_tool_system


def method_1_simple_replacement():
    """Method 1: Drop-in replacement for your existing script."""
    print("Method 1: Simple Replacement")
    print("=" * 40)

    # Instead of:
    # from numinamath_gspo import main
    # main()

    # Use this:
    from tool_enabled_gspo import main as tool_main
    tool_main()

    print("That's it! Just replace the import and run the same command.")


def method_2_custom_trainer():
    """Method 2: Custom trainer with your own tools and settings."""
    print("\nMethod 2: Custom Trainer Setup")
    print("=" * 40)

    # Step 1: Create your tool system
    tool_manager = create_tool_system()

    # Step 2: Create tool-enabled trainer
    trainer, system_prompt = create_tool_enabled_numina_math_gspo(
        runtime_config=None,  # Will use command line args
        tool_manager=tool_manager,
        enable_tool_calling=True,
        max_tool_iterations=3,
    )

    # Step 3: Run training (same as before)
    trainer.train()

    print("Custom trainer created with tool support!")


def method_3_programmatic_setup():
    """Method 3: Full programmatic control."""
    print("\nMethod 3: Programmatic Setup")
    print("=" * 40)

    # Create tool system
    tool_manager = create_tool_system()

    # Create parser and get configurations
    parser = ed.utils.DataClassArgumentParser((ed.GSPOConfig, ed.RunTimeConfig))
    gspo_config, runtime_config = parser.parse_args_into_dataclasses()

    # Create tool-enabled trainer
    trainer = ToolEnabledGSPOTrainer(
        arguments=gspo_config,
        model=None,  # Will be created from runtime_config
        reward_funcs=[],  # Will be set up automatically
        processing_class=None,  # Will be created automatically
        tool_manager=tool_manager,
        max_tool_iterations=3,
        enable_tool_calling=True,
    )

    print("Full programmatic control achieved!")


def show_usage_examples():
    """Show practical usage examples."""
    print("\nPractical Usage Examples")
    print("=" * 40)

    print("1. Basic Usage (same as your current script):")
    print("   python tool_enabled_gspo.py --repo_id Qwen/Qwen3-0.6B-Base --total_batch_size 8")

    print("\n2. With Custom Tools:")
    print("   # Edit tool_enabled_gspo.py to add your custom tools")
    print("   python tool_enabled_gspo.py --repo_id Qwen/Qwen3-0.6B-Base --total_batch_size 8")

    print("\n3. Disable Tool Calling (if needed):")
    print("   # Set enable_tool_calling=False in the trainer")
    print("   python tool_enabled_gspo.py --repo_id Qwen/Qwen3-0.6B-Base --total_batch_size 8")

    print("\n4. Test the System First:")
    print("   python tool_demo.py")
    print("   python test_tool_integration.py")


def show_key_differences():
    """Show what's different from the original script."""
    print("\nKey Differences from Original Script")
    print("=" * 40)

    print("✅ What you get:")
    print("   - Models can call tools during rollouts")
    print("   - Enhanced system prompts with tool instructions")
    print("   - Tool-aware reward functions")
    print("   - Automatic tool result integration")

    print("\n✅ What stays the same:")
    print("   - All your command line arguments work")
    print("   - Same training performance and speed")
    print("   - Same model architecture and setup")

    print("\n✅ What you can customize:")
    print("   - Add your own tools")
    print("   - Modify tool calling behavior")
    print("   - Adjust reward functions")


def show_file_structure():
    """Show the file structure and what each file does."""
    print("\nFile Structure")
    print("=" * 40)

    files = {
        "tool_system.py": "Core tool system with tool definitions and execution",
        "tool_enabled_gspo.py": "Enhanced GSPO trainer with tool support",
        "tool_demo.py": "Demonstration of tool capabilities",
        "tool_integration_example.py": "Complete integration examples",
        "test_tool_integration.py": "Test suite for the tool system",
        "README_TOOL_GSPO.md": "Comprehensive documentation",
        "quick_start_tool_gspo.py": "This quick start guide"
    }

    for file, description in files.items():
        print(f"📄 {file}")
        print(f"   {description}")
        print()


def main():
    """Main demonstration function."""
    print("🚀 Tool-Enabled GSPO Training - Quick Start Guide")
    print("=" * 60)
    print("Ready to train models that can use tools during reasoning!")
    print()

    show_file_structure()
    show_usage_examples()
    show_key_differences()

    print("🎯 Quick Start Commands:")
    print("=" * 40)
    print("1. Test the system:")
    print("   cd easydel/scripts/finetune")
    print("   python tool_demo.py")
    print()
    print("2. Run your first tool-enabled training:")
    print("   python tool_enabled_gspo.py --repo_id Qwen/Qwen3-0.6B-Base --total_batch_size 8 --num_train_epochs 1")
    print()
    print("3. Check that everything works:")
    print("   python test_tool_integration.py")
    print()
    print("That's it! Your models will now learn to use tools effectively! 🎉")


if __name__ == "__main__":
    main()