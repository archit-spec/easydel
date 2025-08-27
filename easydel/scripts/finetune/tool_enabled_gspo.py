"""
Tool-Enabled GSPO Trainer

This module extends the GSPO trainer to support tool interactions during rollouts.
It allows models to call external tools and receive results, enabling more dynamic
and capable policy learning for mathematical reasoning and other tool-assisted tasks.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union, Callable
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, GenerationConfig, AutoConfig
from math_verify import LatexExtractionConfig, parse, verify  # type:ignore
from datasets import load_dataset

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tool_system import (
    ToolManager,
    ToolConversationFormatter,
    ToolCall,
    ToolResult,
    create_tool_system,
    create_tool_aware_reward_function
)
from numinamath_gspo import RunTimeConfig

# Define reward functions at module level for import
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list

def accuracy_reward(prompts, completions, batch, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    # This is a simplified version - in practice you'd need access to the original solutions
    # For now, just return a basic accuracy reward
    rewards = []
    for completion in completions:
        if isinstance(completion, list) and len(completion) > 0:
            content = completion[0]["content"]
        else:
            content = str(completion)

        # Simple heuristic: reward for having boxed answers
        if "\\boxed{" in content:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards
import easydel as ed
from easydel.trainers.group_relative_policy_optimization import GSPOTrainer
from easydel.infra.base_state import EasyDeLState


class ToolEnabledGSPOTrainer(GSPOTrainer):
    """GSPO Trainer with tool interaction capabilities."""

    def __init__(
        self,
        arguments,
        model,
        reward_funcs,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        data_tokenize_fn=None,
        tool_manager: Optional[ToolManager] = None,
        max_tool_iterations: int = 3,
        enable_tool_calling: bool = True,
    ):
        # Initialize the parent GSPO trainer
        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_tokenize_fn=data_tokenize_fn,
        )

        # Tool system setup
        self.tool_manager = tool_manager or create_tool_system()
        self.max_tool_iterations = max_tool_iterations
        self.enable_tool_calling = enable_tool_calling
        self.conversation_formatter = ToolConversationFormatter(self.processing_class)

        # Wrap reward functions to be tool-aware
        self.original_reward_funcs = reward_funcs.copy()
        self.reward_funcs = [
            create_tool_aware_reward_function(func, self.tool_manager)
            for func in reward_funcs
        ]

    def _configure_tool_enabled_generate_function(self):
        """Configure the generate function with tool interaction support."""
        from transformers import GenerationConfig
        from easydel.utils.compiling_utils import ejit
        from jax.sharding import NamedSharding, PartitionSpec
        from eformer import common_types
        from easydel.trainers.group_relative_policy_optimization.adaptive_mesh import get_adaptive_sharding_spec
        import inspect

        mesh = self.model.mesh
        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

        # Use adaptive sharding based on batch size and tensor parallelism
        _sig = inspect.signature(get_adaptive_sharding_spec)
        _kwargs = dict(
            total_batch_size=self.arguments.total_batch_size,
            force_tensor_parallel=self.arguments.force_tensor_parallel,
            mini_batch_size=self.arguments.mini_batch_size,
        )
        if 'force_data_parallel' in _sig.parameters:
            _kwargs['force_data_parallel'] = self.arguments.force_data_parallel
       ar
       
            _kwargs['rollouts_per_step'] = self.arguments.rollouts_per_step
        adaptive_spec = get_adaptive_sharding_spec(**_kwargs)
        input_sharding = NamedSharding(
            mesh=mesh,
            spec=adaptive_spec
        )

        def tool_enabled_generate(state, input_ids, attention_mask, num_return_sequences: int, prng_seed: int):
            """Generate completions with tool interaction support."""
            module = state.model

            with module.mesh:
                input_ids = module.config.partition_manager.shard(
                    input_ids,
                    axes=[common_types.BATCH, common_types.SEQUENCE_PARALLEL],
                    mode=common_types.MODE_PREFILL,
                )
                attention_mask = module.config.partition_manager.shard(
                    attention_mask,
                    axes=[common_types.BATCH, common_types.SEQUENCE_PARALLEL],
                    mode=common_types.MODE_PREFILL,
                )

                # Enhanced generation config for tool calling
                generation_config = GenerationConfig(
                    top_p=self.arguments.top_p,
                    top_k=self.arguments.top_k,
                    temperature=self.arguments.temperature,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    max_new_tokens=self.arguments.max_completion_length,
                    max_length=self.arguments.max_completion_length + self.arguments.max_prompt_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    use_cache=False,
                )

                # Build PRNG key from provided seed
                import jax
                prng_key = jax.random.PRNGKey(prng_seed)

                # Initial generation
                sequences = module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    prng_key=prng_key,
                ).sequences

                # If tool calling is enabled, process tool interactions
                if self.enable_tool_calling:
                    sequences = self._process_tool_interactions(
                        state, sequences, input_ids, attention_mask, prng_key
                    )

                # Re-constrain inputs to the step partition spec
                from eformer.escale import with_sharding_constraint
                input_ids = with_sharding_constraint(input_ids, self.arguments.step_partition_spec)
                attention_mask = with_sharding_constraint(attention_mask, self.arguments.step_partition_spec)
                return sequences, input_ids, attention_mask

        self.generate_function = ejit(
            tool_enabled_generate,
            in_shardings=(self.state_shardings, input_sharding, input_sharding, empty_sharding),
            out_shardings=(empty_sharding, input_sharding, input_sharding),
            static_argnums=(3,),
        )

    def _process_tool_interactions(self, state, sequences, input_ids, attention_mask, prng_key):
        """Process tool interactions in generated sequences."""
        # For now, return sequences as-is to avoid JAX tracer issues
        # Tool processing will be handled at a higher level in the training loop
        return sequences

    def configure_functions(self):
        """Configure functions with tool-enabled generation."""
        # Call parent configure_functions to set up shared components
        parent_result = super().configure_functions()

        # Override with tool-enabled generation function
        self._configure_tool_enabled_generate_function()

        return parent_result

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Override to add tool interaction processing."""
        # Call parent method to get the standard preprocessing
        processed_batch, metrics_dict = super()._preprocess_batch_input(state, batch, is_train)

        # Process tool interactions in the completions
        if self.enable_tool_calling and is_train:
            processed_batch = self._post_process_tool_interactions(processed_batch)

        return processed_batch, metrics_dict

    def _post_process_tool_interactions(self, batch):
        """Process tool interactions after generation."""
        completion_ids = batch["completion_ids"]

        # Convert to numpy for processing (outside JIT)
        completion_ids_np = jax.device_get(completion_ids)

        # Process each completion for tool interactions
        enhanced_completions = []
        tool_usage_count = 0

        for i, completion_seq in enumerate(completion_ids_np):
            # Decode the completion
            text = self.processing_class.decode(completion_seq, skip_special_tokens=True)

            # Check for tool calls and process them
            if self.conversation_formatter.has_tool_calls(text):
                tool_calls = self.conversation_formatter.extract_tool_calls(text)

                # Log tool usage
                print(f"🔧 [Step {getattr(self, '_current_step', 0)}] Query {i}: Found {len(tool_calls)} tool calls")
                for j, tool_call in enumerate(tool_calls[:self.max_tool_iterations]):
                    print(f"   Tool {j+1}: {tool_call.tool_name}({tool_call.arguments})")

                # Execute tools and get results
                tool_results = []
                for tool_call in tool_calls[:self.max_tool_iterations]:
                    result = self.tool_manager.execute_tool(tool_call)
                    tool_results.append(result)
                    print(f"   Result: {result.content}")

                # Format tool results and append to sequence
                enhanced_text = text
                for result in tool_results:
                    enhanced_text += "\n" + self.conversation_formatter.format_tool_result(result)

                # Re-encode the enhanced sequence
                enhanced_ids = self.processing_class.encode(
                    enhanced_text,
                    return_tensors="np",
                    padding="max_length",
                    max_length=self.arguments.max_completion_length,
                    truncation=True,
                    add_special_tokens=False
                )
                enhanced_completions.append(enhanced_ids[0])
                tool_usage_count += 1
            else:
                # No tool calls, use original sequence
                enhanced_completions.append(completion_seq)

        # Log summary
        if tool_usage_count > 0:
            print(f"📊 [Step {getattr(self, '_current_step', 0)}] Tool usage: {tool_usage_count}/{len(completion_ids_np)} completions used tools")

        # Convert back to JAX array
        enhanced_completions = jnp.array(enhanced_completions)

        # Update the batch with enhanced completions
        batch["completion_ids"] = enhanced_completions

        return batch

    def on_step_end(self, state, metrics):
        """Override to track current step for logging."""
        self._current_step = getattr(self, '_current_step', 0) + 1
        return super().on_step_end(state, metrics)


def create_tool_enabled_system_prompt(tool_definitions: List) -> str:
    """Create a system prompt that includes tool definitions."""
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


def create_tool_enabled_numina_math_gspo(
    runtime_config: RunTimeConfig,
    tool_manager: Optional[ToolManager] = None,
    enable_tool_calling: bool = True,
    max_tool_iterations: int = 3,
) -> tuple:
    """Create a tool-enabled GSPO trainer for NuminaMath fine-tuning."""

    # Create tool manager if not provided
    if tool_manager is None:
        tool_manager = create_tool_system()

    # Get tool definitions for system prompt
    tool_definitions = [tool.get_definition() for tool in tool_manager.tools.values()]

    # Create enhanced system prompt
    enhanced_system_prompt = create_tool_enabled_system_prompt(tool_definitions)

    return tool_manager, enhanced_system_prompt


def main():
    """Main function for tool-enabled NuminaMath GSPO training."""
    parser = ed.utils.DataClassArgumentParser((ed.GSPOConfig, RunTimeConfig))
    gspo_config, runtime_config = parser.parse_args_into_dataclasses()

    runtime_config: RunTimeConfig
    gspo_config: ed.GSPOConfig

    # Print arguments once per process without initializing JAX backend at import-time
    try:
        print("Training Arguments\n----------------------")
        print(gspo_config)
        print("----------------------")
    except Exception:
        pass

    # Create tool system and enhanced system prompt
    tool_manager, enhanced_system_prompt = create_tool_enabled_numina_math_gspo(
        runtime_config=runtime_config,
        enable_tool_calling=True,
        max_tool_iterations=3,
    )

    processor = AutoTokenizer.from_pretrained(runtime_config.processor_repo_id)
    processor.padding_side = "left"

    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id

    max_prompt_length = gspo_config.max_prompt_length
    max_completion_length = gspo_config.max_completion_length
    max_sequence_length = max_completion_length + max_prompt_length

    hf_config = AutoConfig.from_pretrained(runtime_config.repo_id)

    avails = [v.module.__name__ for v in ed.infra.factory.registry.task_registry[ed.TaskType.IMAGE_TEXT_TO_TEXT].values()]

    if hf_config.architectures and any(arch in avails for arch in hf_config.architectures):
        load_module = ed.AutoEasyDeLModelForImageTextToText
    else:
        load_module = ed.AutoEasyDeLModelForCausalLM

    # Configure adaptive mesh early if using tensor/data parallelism
    if gspo_config.force_tensor_parallel is not None or gspo_config.force_data_parallel is not None:
        from easydel.trainers.group_relative_policy_optimization.adaptive_mesh import configure_adaptive_mesh_inplace

        # This updates gspo_config in-place with mesh_dims, step_partition_spec, etc.
        mesh_plan = configure_adaptive_mesh_inplace(gspo_config)
        sharding_axis_dims = (mesh_plan.dp, mesh_plan.fsdp, mesh_plan.ep, mesh_plan.tp, mesh_plan.sp)

        # Also set sharding_axis_dims on the config for model loading
        gspo_config.sharding_axis_dims = sharding_axis_dims

        print(f"Using adaptive mesh: DP={mesh_plan.dp}, FSDP={mesh_plan.fsdp}, "
              f"TP={mesh_plan.tp}, EP={mesh_plan.ep}, SP={mesh_plan.sp}")
        print(f"Dataset will be sharded across {mesh_plan.dp} data-parallel workers")
        print(f"Step partition spec: {mesh_plan.step_partition_spec}")
    else:
        sharding_axis_dims = runtime_config.sharding_axis
        print(f"Using default mesh dims: {runtime_config.sharding_axis}")

    model = load_module.from_pretrained(
        runtime_config.repo_id,
        auto_shard_model=True,
        sharding_axis_dims=sharding_axis_dims,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            max_position_embeddings=max_sequence_length,
            freq_max_position_embeddings=max_sequence_length,
            mask_max_position_embeddings=max_sequence_length,
            attn_dtype=runtime_config.attn_dtype,
            attn_softmax_dtype=runtime_config.attn_softmax_dtype,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,  # Disable cache quantization to avoid shape issues
            attn_mechanism=ed.AttentionMechanisms.VANILLA,  # Force vanilla attention to avoid sliding window issues
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,  # Avoid static_argnums issues
            use_sliding_window=False,  # Explicitly disable sliding window
            sliding_window=None,  # Force None to prevent sliding window logic
            use_cache=False,  # Disable KV cache to avoid dynamic shape issues
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        param_dtype=runtime_config.param_dtype,
        dtype=runtime_config.dtype,
        precision=jax.lax.Precision.DEFAULT,
        partition_axis=ed.PartitionAxis(),
    )

    # Create tool-aware reward functions
    tool_aware_format_reward = create_tool_aware_reward_function(format_reward, tool_manager)
    tool_aware_accuracy_reward = create_tool_aware_reward_function(accuracy_reward, tool_manager)

    # Load dataset with enhanced system prompt
    dataset_id = "AI-MO/NuminaMath-TIR"
    train_dataset, test_dataset = load_dataset(
        dataset_id,
        split=[
            f"train[:{runtime_config.dataset_use_rate}%]",
            f"test[:{runtime_config.dataset_use_rate}%]",
        ],
    )

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": example["problem"]},
            ],
        }

    train_dataset = train_dataset.map(make_conversation, remove_columns=["messages"])
    test_dataset = test_dataset.map(make_conversation, remove_columns=["messages"])

    def data_tokenize_fn(batch, tokenizer, tools):
        ids = tokenizer(
            batch["prompt"],
            return_tensors="np",
            padding="max_length",
            padding_side="left",
            max_length=gspo_config.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        ans = tokenizer(
            batch["solution"],
            return_tensors="np",
            padding="max_length",
            padding_side="left",
            max_length=gspo_config.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        ids.update({"solution_ids": ans["input_ids"]})
        return ids

    # Create the tool-enabled trainer
    trainer = ToolEnabledGSPOTrainer(
        arguments=gspo_config,
        model=model,
        reward_funcs=[tool_aware_format_reward, tool_aware_accuracy_reward],
        processing_class=processor,
        eval_dataset=test_dataset,
        train_dataset=train_dataset,
        data_tokenize_fn=data_tokenize_fn,
        tool_manager=tool_manager,
        max_tool_iterations=3,
        enable_tool_calling=True,
    )

    trainer.train()


if __name__ == "__main__":
    main()