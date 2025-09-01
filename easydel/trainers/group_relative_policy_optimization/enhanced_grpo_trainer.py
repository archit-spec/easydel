
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
from functools import cached_property, partial
import inspect

import flax
import flax.nnx
import jax
from eformer import common_types
from eformer.escale import with_sharding_constraint
from flax import nnx as nn
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
from transformers import AutoTokenizer, GenerationConfig, ProcessorMixin

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.compiling_utils import ejit
from easydel.utils.helpers import capture_time, get_logger
from easydel.utils.traversals import deepcopy_model

from ..prompt_utils import apply_chat_template, is_conversational, maybe_apply_chat_template, maybe_extract_prompt
from ..trainer.trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_configurations import MetricsType
from ._fn import get_per_token_logps, grpo_step
from .grpo_config import GRPOConfig
from .claude_code_config import ClaudeCodeConfig
from .env_reward_function import EnvironmentRewardFunction

try:
    import wandb  # type:ignore
except ImportError:
    wandb = None

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
logger = get_logger(__name__)
RewardFunc = tp.Union[EasyDeLBaseModule, EasyDeLState, tp.Callable[[list, list], list[float]]]  # noqa


def _fileaf(x):
    return isinstance(x, jax.Array)


def delete_tree(pytree):
    return jax.tree_util.tree_map(
        lambda x: x.delete() if isinstance(x, jax.Array) else None,
        pytree,
        is_leaf=_fileaf,
    )


class EnhancedGRPOTrainer(Trainer):
    """
    Enhanced GRPO Trainer with Claude Code environment integration.
    Extends GRPOTrainer to support Claude Code execution and code quality evaluation.
    """

    arguments: ClaudeCodeConfig  # type hinting

    def __init__(
        self,
        arguments: ClaudeCodeConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc],
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType = None,
        reward_processing_classes: ProcessingClassType = None,
        data_tokenize_fn: tp.Callable | None = None,
    ):
        assert arguments is not None, (
            "You Have to pass `arguments` that will be used for training, but you have passed `arguments=None`"
        )
        assert isinstance(arguments, ClaudeCodeConfig), f"arguments type must be `ClaudeCodeConfig` but got {type(arguments)}"
        assert processing_class is not None, "processing_class must be specified to tokenize a DPO dataset."

        # Apply adaptive mesh configuration before storing arguments
        from .adaptive_mesh import configure_adaptive_mesh_inplace
        
        # Configure mesh and update arguments in-place
        if arguments.force_tensor_parallel is not None or arguments.force_data_parallel is not None:
            mesh_plan = configure_adaptive_mesh_inplace(arguments)
            
            # Ensure dataset sharding aligns with DP only
            try:
                # Dataset should be sharded across DP workers only
                arguments.grain_shard_count = mesh_plan.dp
                arguments.grain_shard_index = jax.process_index() % mesh_plan.dp
                
                if jax.process_index() == 0:
                    logger.info(
                        f"Configured mesh: DP={mesh_plan.dp}, FSDP={mesh_plan.fsdp}, "
                        f"TP={mesh_plan.tp}, dataset shards={mesh_plan.dp}"
                    )
            except Exception as e:
                logger.warning(f"Failed to configure dataset sharding: {e}")
                # Fallback to safe defaults
                arguments.grain_shard_count = 1
                arguments.grain_shard_index = 0
        else:
            # No forced parallelism, use default behavior
            pass
        
        self.arguments = arguments
        self.truncation_mode = arguments.truncation_mode
        self.processing_class = processing_class

        if not isinstance(model, EasyDeLState):
            model = model.to_state()

        self.ref_state = deepcopy_model(model=model)

        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.model.config._name_or_path,
                padding_side="left",
            )
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=model.model.mesh)

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs, strict=False)
        ):
            if isinstance(reward_func, EasyDeLBaseModule | EasyDeLState):
                if isinstance(reward_func, EasyDeLBaseModule):
                    reward_func = reward_func.to_state()
                    sharding = reward_func.shardings

                    @ejit(
                        static_argnums=(0,),
                        in_shardings=(
                            sharding.graphstate,
                            sharding.graphother,
                            empty_sharding,
                        ),
                        out_shardings=empty_sharding,
                    )
                    def apply_fn(gd, gs, gt, batch):
                        batch = with_sharding_constraint(
                            arr=batch,
                            sharding=self.arguments.step_partition_spec,
                        )
                        return nn.merge(gd, gs, gt)(**batch)

                    reward_func = reward_func.replace(apply_fn=apply_fn)

                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.model.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token

                reward_func.model.config.pad_token_id = reward_processing_class.pad_token_id
                if reward_processing_classes is not None:
                    reward_processing_classes[i] = reward_processing_class
                reward_funcs[i] = reward_func

        self.num_generations = arguments.num_return_sequences
        self.reward_processing_classes = reward_processing_classes
        self.reward_funcs = reward_funcs
        self.arguments = arguments
        self.processing_class = processing_class
        self.train_is_conversational = False
        self.eval_is_conversational = False
        self.data_tokenize_fn = data_tokenize_fn
        if train_dataset is not None:
            train_dataset = self._prepare_dataset(
                dataset=train_dataset,
                processing_class=processing_class,
                arguments=arguments,
                dataset_name="train",
            )
        if eval_dataset is not None:
            eval_dataset = self._prepare_dataset(
                dataset=eval_dataset,
                processing_class=processing_class,
                arguments=arguments,
                dataset_name="eval",
            )
        log_table = None
        if self.arguments.use_wandb and self.arguments.can_log_metrics and wandb is not None:
            log_table = wandb.Table(columns=["generations", "took", "length", "step"])
        else:
            log_table = None
        self.log_table = log_table

        # Initialize Claude Code environment if enabled
        self.claude_env = None
        if arguments.claude_code_enabled:
            self.claude_env = EnvironmentRewardFunction(arguments)
            # Add Claude Code environment as a reward function
            self.reward_funcs.append(self.claude_env)

        super().__init__(
            model_state=model,
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=None,
        )

    @cached_property
    def pad_token_id(self):
        if isinstance(self.processing_class, ProcessorMixin):
            pad_token_id = self.processing_class.tokenizer.pad_token_id
        else:
            pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is not None:
            return pad_token_id
        else:
            return self.eos_token_id[0]

    @cached_property
    def eos_token_id(self) -> list[int]:
        eos_ids = []
        # 1) Start with EOS from the primary processing class/tokenizer
        if isinstance(self.processing_class, ProcessorMixin):
            tokenizer = self.processing_class.tokenizer
            proc_eos_token_id = tokenizer.eos_token_id
        else:
            tokenizer = self.processing_class
            proc_eos_token_id = getattr(self.processing_class, "eos_token_id", None)

        if isinstance(proc_eos_token_id, int):
            proc_eos_token_id = [proc_eos_token_id]
        if isinstance(proc_eos_token_id, (list, tuple)):
            eos_ids.extend([t for t in proc_eos_token_id if t is not None])

        # 2) Include common Qwen end tokens if the tokenizer knows them
        special_tokens = [
            "<|im_end|>",
            "<|endoftext|>",
        ]
        convert_fn = getattr(tokenizer, "convert_tokens_to_ids", None)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if callable(convert_fn):
            for tok in special_tokens:
                try:
                    tid = convert_fn(tok)
                except Exception:
                    tid = None
                if tid is not None and (unk_id is None or tid != unk_id):
                    eos_ids.append(tid)

        # 3) Merge any EOS ids present in model.generation_config (if available)
        if hasattr(self.model, "generation_config"):
            conf_eos = self.model.generation_config.eos_token_id
            if isinstance(conf_eos, int):
                conf_eos = [conf_eos]
            if isinstance(conf_eos, (list, tuple)):
                eos_ids.extend([t for t in conf_eos if t is not None])

        # Return unique list with deterministic ordering
        return sorted(set(eos_ids))

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: ProcessingClassType,
        arguments: GRPOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        map_kwargs = {"writer_batch_size": 10}
        from datasets import Dataset

        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = arguments.dataset_num_proc

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
        if dataset_name == "train":
            self.train_is_conversational = is_conversational(dataset[0])
        else:
            self.eval_is_conversational = is_conversational(dataset[0])

        dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
        if not self.arguments.skip_apply_chat_template:
            dataset = dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class, "tools": arguments.tools},
                **map_kwargs,
            )

        def _tokenize(example):
            return processing_class(
                example["prompt"],
                return_tensors="np",
                padding="max_length",
                padding_side="left",
                max_length=arguments.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
                return_attention_mask=True,
            )

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"tokenizing {dataset_name} dataset"
        if self.data_tokenize_fn is not None:
            dataset = dataset.map(
                self.data_tokenize_fn,
                batched=True,
                fn_kwargs={
                    "tokenizer": processing_class,
                    "tools": arguments.tools,
                },
                **map_kwargs,
            )
        else:
            dataset = dataset.map(
                _tokenize,
                batched=True,
                **map_kwargs,
            )
        return dataset

    @property
    def step_sharding(self):
        return NamedSharding(
            mesh=self.model.mesh,
            spec=self.arguments.step_partition_spec,
        )
    
    def get_batch_shardings(self):
        """
        Create proper sharding specifications for each tensor in the batch.
        Handles mixed-rank tensor collections properly.
        """
        mesh = self.model.mesh
        base_spec = self.arguments.step_partition_spec
        
        # If no sharding needed
        if base_spec is None or base_spec == PartitionSpec():
            empty_spec = PartitionSpec()
            return {
                "prompt_ids": NamedSharding(mesh=mesh, spec=empty_spec),
                "prompt_mask": NamedSharding(mesh=mesh, spec=empty_spec),
                "completion_ids": NamedSharding(mesh=mesh, spec=empty_spec),
                "completion_mask": NamedSharding(mesh=mesh, spec=empty_spec),
                "ref_per_token_logps": NamedSharding(mesh=mesh, spec=empty_spec),
                "advantages": NamedSharding(mesh=mesh, spec=empty_spec),
            }
        
        # Extract batch dimension from base spec
        batch_dim = base_spec[0] if len(base_spec) > 0 else None
        seq_dim = base_spec[1] if len(base_spec) > 1 else None
        
        # For 1D tensors (advantages), only use batch dimension
        # If batch_dim is a tuple like ('dp', 'fsdp'), use just the first element
        if isinstance(batch_dim, tuple) and len(batch_dim) > 0:
            batch_dim_1d = batch_dim[0]
        else:
            batch_dim_1d = batch_dim
        
        spec_1d = PartitionSpec(batch_dim_1d) if batch_dim_1d else PartitionSpec()
        spec_2d = base_spec
        
        return {
            "prompt_ids": NamedSharding(mesh=mesh, spec=spec_2d),
            "prompt_mask": NamedSharding(mesh=mesh, spec=spec_2d),
            "completion_ids": NamedSharding(mesh=mesh, spec=spec_2d),
            "completion_mask": NamedSharding(mesh=mesh, spec=spec_2d),
            "ref_per_token_logps": NamedSharding(mesh=mesh, spec=spec_2d),
            "advantages": NamedSharding(mesh=mesh, spec=spec_1d),
        }

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configures and JIT-compiles the training and evaluation step functions.

        This method sets up the necessary functions for training and evaluation, including:
            - Initialization of the model state.
            - Sharding of the model parameters and optimizer state.
            - JIT-compilation of the training and evaluation step functions.

        Returns:
            TrainerConfigureFunctionOutput: An object containing the configured functions and other relevant information.
        """
        mesh = self.model.mesh

        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)
        
        # Derive rollouts_per_step from num_return_sequences if not provided
        mesh_shape = getattr(mesh, "shape", {})
        dp_size = mesh_shape.get("dp", 1) if hasattr(mesh_shape, "get") else 1
        total_dp = dp_size
        if getattr(self.arguments, 'rollouts_per_step', None) is None:
            derived_rps = int(total_dp) * int(self.arguments.total_batch_size) * int(self.arguments.num_return_sequences)
            self.arguments.rollouts_per_step = int(derived_rps)
            if jax.process_index() == 0:
                logger.info(
                    f"Rollout configuration: num_return_sequences={self.arguments.num_return_sequences}, "
                    f"derived rollouts_per_step={self.arguments.rollouts_per_step}, "
                    f"DP={total_dp, batch_size={self.arguments.total_batch_size}"
                )

        # Use adaptive sharding based on batch size and tensor parallelism
        from .adaptive_mesh import get_adaptive_sharding_spec
        _shard_sig = inspect.signature(get_adaptive_sharding_spec)
        _shard_kwargs = dict(
            total_batch_size=self.arguments.total_batch_size,
            force_tensor_parallel=self.arguments.force_tensor_parallel,
            mini_batch_size=self.arguments.mini_batch_size,
        )
        if 'force_data_parallel' in _shard_sig.parameters:
            _shard_kwargs['force_data_parallel'] = self.arguments.force_data_parallel
        if 'rollouts_per_step' in _shard_sig.parameters and getattr(self.arguments, 'rollouts_per_step', None):
            _shard_kwargs['rollouts_per_step'] = self.arguments.rollouts_per_step
        adaptive_spec = get_adaptive_sharding_spec(**_shard_kwargs)
        input_sharding = NamedSharding(
            mesh=mesh,
            spec=adaptive_spec
        )
        # Store input spec for debugging/inspection
        self.input_partition_spec = adaptive_spec
        step_sharding = NamedSharding(
            mesh=mesh,
            spec=self.arguments.step_partition_spec,
        )
        
        @ejit(
            in_shardings=(self.state_shardings, input_sharding, input_sharding, empty_sharding),
            out_shardings=(empty_sharding, input_sharding, input_sharding),
            static_argnums=(3,),
        )
        def generate(state: EasyDeLState, input_ids, attention_mask, num_return_sequences: int, prng_seed: int):
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
                # Proper generation config that relies on natural EOS stopping
                generation_config = GenerationConfig(
                    top_p=self.arguments.top_p,
                    top_k=self.arguments.top_k,
                    temperature=self.arguments.temperature,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,  # EasyDeL will stop naturally when these tokens are generated
                    max_new_tokens=self.arguments.max_completion_length,
                    max_length=self.arguments.max_completion_length + self.arguments.max_prompt_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    use_cache=False,
                )
                
                # Build PRNG key from provided seed to ensure per-chunk diversity
                prng_key = jax.random.PRNGKey(prng_seed)
                sequences = module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    prng_key=prng_key,
                ).sequences
                # Return inputs re-constrained to the input sharding spec to allow repeated calls
                input_ids = with_sharding_constraint(input_ids, adaptive_spec)
                attention_mask = with_sharding_constraint(attention_mask, adaptive_spec)
                return sequences, input_ids, attention_mask

        self.generate_function = generate

        self._train_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_train
            getattr(self.arguments, "log_logprobs_metrics", True),
        )

        static_argnames = (2, 3, 4, 5, 6, 7, 8, 9)
        
        sharded_training_step_function = ejit(
            grpo_step,
            in_shardings=(self.state_shardings, None),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
        )

        self._eval_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,  # is_train
        )

        sharded_evaluation_step_function = ejit(
            grpo_step,
            in_shardings=(self.state_shardings, None),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
        )

        def _compute_refmodel_logps(graphtree, graphother, ids, mask, graphdef):
            apply = flax.nnx.merge(graphdef, graphtree, graphother)
            with apply.mesh:
                # Ensure token arrays conform to the step partitioning spec before compute
                ids = with_sharding_constraint(ids, self.arguments.step_partition_spec)
                mask = with_sharding_constraint(mask, self.arguments.step_partition_spec)
                return get_per_token_logps(apply, ids, mask, self.arguments.max_prompt_length)

        # Allow input sharding of token ids and masks to pass through (we re-constrain inside the fn)
        # This avoids mismatches like: pjit expects replicated but arg is sharded as ('dp','tp')

        self.compute_refmodel_logps = ejit(
            partial(_compute_refmodel_logps, graphdef=self.model_state.graphdef),
            static_argnames=("graphdef",),
            in_shardings=(
                self.model_state.shardings.graphstate,
                self.model_state.shardings.graphother,
                None,
                None,
            ),
            out_shardings=empty_sharding,
        )

        sharded_training_step_function.static_argnums_ = static_argnames
        sharded_evaluation_step_function.static_argnums_ = static_argnames

        self.arguments.ensure_checkpoint_path()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def _make_attn_mask(self, arr):
        is_eos = jnp.isin(arr, jnp.asarray(self.eos_token_id).reshape(-1))
        return (
            (jnp.arange(is_eos.shape[1])[None, :].repeat(is_eos.shape[0], axis=0))
            <= jnp.where(
                is_eos.any(axis=1),
                jnp.argmax(is_eos.astype(jnp.int32), axis=1),
                jnp.full((is_eos.shape[0],), is_eos.shape[1] - 1),  # Fix off-by-one
            )[:, None]
        ).astype(jnp.int32)


    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]
            
            # Convert numpy arrays to JAX arrays using the working solution
            try:
                prompt_ids = jnp.asarray(prompt_ids)
                prompt_mask = jnp.asarray(prompt_mask)
            except Exception as e:
                logger.error(f"Failed to convert arrays to JAX on worker {jax.process_index()}: {e}")
                raise RuntimeError(f"Failed to convert numpy arrays to JAX arrays: {e}")
            
            # Chunked generation and reference log-prob computation to reduce peak memory
            rollout_chunk_size = getattr(self.arguments, "rollout_chunk_size", None)
            if rollout_chunk_size is None or rollout_chunk_size <= 0:
                # Auto: if TP>1, generate one completion per chunk (per TP group) to minimize memory
                try:
                    mesh_shape = getattr(self.model.mesh, "shape", {})
                    tp_size = mesh_shape.get("tp", 1) if hasattr(mesh_shape, "get") else 1
                except Exception:
                    tp_size = 1
                rollout_chunk_size = 1 if tp_size and tp_size > 1 else min(2, int(self.num_generations))

            sequences_chunks = []
            completion_ids_chunks = []
            completion_mask_chunks = []
            ref_logps_chunks = []
            comp_len_chunks = []
            cur_nrs_chunks = []  # Track chunk sizes to enable correct reordering later
            generation_time = 0.0
            token_logps_time = 0.0

            base_prompt_len = prompt_ids.shape[-1]
            nrs_remaining = int(self.num_generations)
            # Create a base PRNG key that varies by step and process to diversify chunked sampling
            try:
                cur_step_int = int(jax.device_get(state.step))
            except Exception:
                cur_step_int = 0
            base_key = jax.random.fold_in(jax.random.PRNGKey(cur_step_int), jax.process_index())
            chunk_idx = 0
            while nrs_remaining > 0:
                cur_nrs = int(min(rollout_chunk_size, nrs_remaining))
                with capture_time() as generation_time_fn:
                    # Use a simple per-chunk seed; avoids JIT recompiles and ensures diversity across chunks
                    per_chunk_seed = int(cur_step_int * 131071 + chunk_idx)
                    seq_chunk, prompt_ids, prompt_mask = jax.block_until_ready(
                        self.generate_function(state, prompt_ids, prompt_mask, cur_nrs, per_chunk_seed)
                    )
                # Debug output removed to prevent host divergence in compiled code
                generation_time += float(generation_time_fn())

                # Avoid explicit cross-host barriers here; rely on pjit collectives only

                # Extract completions for this chunk and build masks
                prompt_completion_ids_chunk = seq_chunk
                completion_ids_chunk = prompt_completion_ids_chunk[..., base_prompt_len:]
                completion_mask_chunk = self._make_attn_mask(completion_ids_chunk)
                ridmask_chunk = prompt_mask.repeat(cur_nrs, 0)

                with capture_time() as token_logps_time_fn:
                    full_mask_chunk = jnp.concatenate([ridmask_chunk, completion_mask_chunk], -1)
                    ref_logps_chunk = self.compute_refmodel_logps(
                        self.ref_state.graphstate,
                        self.ref_state.graphother,
                        prompt_completion_ids_chunk,
                        full_mask_chunk,
                    )
                token_logps_time += float(token_logps_time_fn())

                # Avoid explicit cross-host barriers here; rely on pjit collectives only

                # Accumulate
                sequences_chunks.append(prompt_completion_ids_chunk)
                completion_ids_chunks.append(completion_ids_chunk)
                completion_mask_chunks.append(completion_mask_chunk)
                ref_logps_chunks.append(ref_logps_chunk)
                comp_len_chunks.append(completion_mask_chunk.sum(-1))
                cur_nrs_chunks.append(cur_nrs)

                nrs_remaining -= cur_nrs
                chunk_idx += 1

            # Concatenate accumulated chunks — for memory-opt mode keep concat minimal if only one chunk
            if len(sequences_chunks) == 1:
                prompt_completion_ids = sequences_chunks[0]
                completion_ids = completion_ids_chunks[0]
                completion_mask = completion_mask_chunks[0]
                ref_per_token_logps = ref_logps_chunks[0]
                completion_lengths_per_seq = comp_len_chunks[0]
            else:
                # When generation is chunked across num_return_sequences, the natural concatenation
                # order is [for each chunk c: for each prompt b: for each gen in chunk] along axis 0.
                # Advantage computation expects [for each prompt b: all gens across chunks] ordering.
                # We therefore reorder to (prompt-major, then generation) before downstream use.
                num_prompts_local = prompt_ids.shape[0]

                def _reorder_from_chunks(chunks: list[jax.Array], is_scalar: bool = False):
                    # chunks: list of arrays with shape (cur_nrs_i * B, ...)
                    # Return: array with shape (B * sum(cur_nrs_i), ...), ordered by prompt-major
                    reshaped = []
                    for arr, k in zip(chunks, cur_nrs_chunks, strict=False):
                        # Reshape (k * B, ...) -> (B, k, ...)
                        new_shape = (num_prompts_local, k) + tuple(arr.shape[1:])
                        reshaped.append(jnp.reshape(arr, new_shape))
                    # Concat across generation axis -> (B, total_k, ...)
                    by_prompt = jnp.concatenate(reshaped, axis=1)
                    # Flatten back to (B * total_k, ...)
                    flat_shape = (by_prompt.shape[0] * by_prompt.shape[1],) + tuple(by_prompt.shape[2:])
                    return jnp.reshape(by_prompt, flat_shape)

                prompt_completion_ids = _reorder_from_chunks(sequences_chunks)
                completion_ids = _reorder_from_chunks(completion_ids_chunks)
                completion_mask = _reorder_from_chunks(completion_mask_chunks)
                ref_per_token_logps = _reorder_from_chunks(ref_logps_chunks)
                completion_lengths_per_seq = _reorder_from_chunks(comp_len_chunks)
            prompts = self.processing_class.batch_decode(batch["input_ids"], skip_special_tokens=True)
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            is_conversational = self.train_is_conversational if is_train else self.eval_is_conversational
            if is_conversational:
                completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
            else:
                completions = completions_text

            # Calculate completion lengths before rewards
            completion_lengths_per_seq = completion_mask.sum(-1)  # Length per sequence
            # Host-side diagnostic: print per-rollout completion token lengths
            if jax.process_index() == 0:
                try:
                    # Prefer fully addressable arrays
                    if isinstance(completion_lengths_per_seq, jax.Array) and getattr(
                        completion_lengths_per_seq, "is_fully_addressable", False
                    ):
                        _lens = jax.device_get(completion_lengths_per_seq)
                    else:
                        # Fallback to first local shard if available
                        try:
                            _shards = completion_lengths_per_seq.addressable_shards  # type: ignore[attr-defined]
                            if _shards and len(_shards) > 0:
                                _lens = jax.device_get(_shards[0].data)
                            else:
                                _lens = jax.device_get(completion_lengths_per_seq)
                        except Exception:
                            _lens = jax.device_get(completion_lengths_per_seq)
                    _lens = jnp.asarray(_lens).ast