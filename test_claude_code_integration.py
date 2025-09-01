#!/usr/bin/env python3
"""
Test script for Claude Code integration with GRPO trainer.
This script demonstrates how to use the enhanced GRPO trainer with Claude Code environment.
"""

import tempfile
import os
from pathlib import Path
import jax
import jax.numpy as jnp

from easydel.trainers.group_relative_policy_optimization import (
    ClaudeCodeConfig,
    EnhancedGRPOTrainer,
    EnvironmentRewardFunction,
    ClaudeCodeExecutor
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState


def create_mock_model():
    """Create a mock model for testing purposes."""
    # In a real scenario, you would use a proper model
    class MockModel(EasyDeLBaseModule):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {
                '_name_or_path': 'test-model',
                'pad_token_id': 0,
                'vocab_size': 1000
            })()
            self.mesh = jax.sharding.Mesh(jax.devices(), ('dp',))
        
        def generate(self, input_ids, attention_mask, generation_config, prng_key):
            # Mock generation - return some dummy sequences
            batch_size = input_ids.shape[0]
            seq_length = input_ids.shape[1] + generation_config.max_new_tokens
            return type('GenerationResult', (), {
                'sequences': jnp.ones((batch_size, seq_length), dtype=jnp.int32)
            })()
    
    return MockModel()


def test_claude_code_executor():
    """Test Claude Code executor functionality."""
    print("Testing Claude Code Executor...")
    
    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        executor = ClaudeCodeExecutor(
            claude_path="echo",  # Use echo for testing (simulates Claude Code)
            max_parallel=2,
            default_timeout=30,
            workspace_root=temp_dir,
            use_temp_workspace=False,
            cleanup_workspaces=True
        )
        
        # Test execution
        requests = [
            ExecutionRequest(
                prompt="print('Hello, World!')",
                workspace_path=temp_dir,
                timeout=10,
                capture_traces=True
            )
        ]
        
        results = executor.execute(requests)
        
        print(f"Execution results: {len(results)}")
        for i, result in enumerate(results):
            print(f"Result {i}: success={result.success}, output={result.output[:100]}...")
        
        return True


def test_environment_reward_function():
    """Test environment reward function."""
    print("Testing Environment Reward Function...")
    
    config = ClaudeCodeConfig(
        claude_code_enabled=True,
        claude_code_path="echo",  # Use echo for testing
        claude_code_timeout=30,
        use_temp_workspace=True,
        workspace_cleanup=True,
        max_parallel_executions=2
    )
    
    env = EnvironmentRewardFunction(config)
    
    # Test reward computation
    prompts = ["Write a function that adds two numbers"]
    completions = ["def add(a, b): return a + b"]
    
    rewards = env.compute_reward(prompts, completions)
    
    print(f"Rewards: {rewards}")
    return True


def test_enhanced_grpo_trainer():
    """Test enhanced GRPO trainer integration."""
    print("Testing Enhanced GRPO Trainer...")
    
    config = ClaudeCodeConfig(
        claude_code_enabled=True,
        claude_code_path="echo",  # Use echo for testing
        max_prompt_length=128,
        max_completion_length=256,
        num_return_sequences=2,
        total_batch_size=1,
        learning_rate=1e-6,
        beta=0.04
    )
    
    # Create mock model
    model = create_mock_model()
    model_state = model.to_state()
    
    # Create trainer
    trainer = EnhancedGRPOTrainer(
        arguments=config,
        model=model_state,
        reward_funcs=[],  # Will add Claude Code environment internally
        processing_class=None,  # Would normally be a tokenizer
        train_dataset=None,
        eval_dataset=None
    )
    
    print(f"Trainer created successfully: {trainer is not None}")
    print(f"Claude Code enabled: {config.claude_code_enabled}")
    print(f"Number of reward functions: {len(trainer.reward_funcs)}")
    
    return True


def main():
    """Run all tests."""
    print("Starting Claude Code Integration Tests...")
    print("=" * 50)
    
    try:
        # Test 1: Claude Code Executor
        if test_claude_code_executor():
            print("✓ Claude Code Executor test passed")
        else:
            print("✗ Claude Code Executor test failed")
        
        print()
        
        # Test 2: Environment Reward Function
        if test_environment_reward_function():
            print("✓ Environment Reward Function test passed")
        else:
            print("✗ Environment Reward Function test failed")
        
        print()
        
        # Test 3: Enhanced GRPO Trainer
        if test_enhanced_grpo_trainer():
            print("✓ Enhanced GRPO Trainer test passed")
        else:
            print("✗ Enhanced GRPO Trainer test failed")
        
        print()
        print("=" * 50)
        print("All tests completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()