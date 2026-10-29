import math
import types
import weakref
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch._dynamo.cache_size
import torch.distributed as dist
import torch.nn as nn
import torchair
from torchair import patch_for_hcom
import torch_npu
import os
from torch.distributed import ReduceOp
from vllm.attention import get_attn_backend
from vllm.config import CompilationLevel, VllmConfig, SpeculativeConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import get_dp_group, get_pp_group
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.inputs import INPUT_REGISTRY
from vllm.logger import logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType, LazyLoader, cdiv, DeviceMemoryProfiler)
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import is_spec_decode_supported
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.utils import (gather_mm_placeholders,
                                  sanity_check_mm_encoder_outputs,
                                  scatter_mm_placeholders)
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import (AscendAttentionMetadataBuilder,
                                                AscendAttentionState,
                                                AscendMetadata)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.distributed.utils import is_lmhead_tp
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.eplb_updator import EplbUpdator
from vllm_ascend.eplb.utils import model_register
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
from vllm.v1.sample.metadata import SamplingMetadata
from vllm_ascend.utils import ProfileExecuteDuration, TORCHAIR_CACHE_DIR
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.mtp_proposer_v1 import MtpProposer

from pangu_infer.pangu.adaptor_manager.adaptor_manager import register, get_config
from pangu_infer.pangu.utils.utils import check
from pangu_infer.patches.vllm_ascend.sample.rejection_sampler import set_mtp_logp_context, get_mtp_logp_context
from pangu_infer.pangu.accelerators.speculative_decoding.sam import SAM, SAMProposer, has_samfast_package
from pangu_infer.pangu.accelerators.speculative_decoding.suffix import SuffixDecodingProposer


if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

import vllm.envs as envs_vllm

import vllm_ascend.envs as envs_ascend

from pangu_infer.pangu.adaptor_manager.adaptor_manager import register

from pangu_infer.pangu.utils.utils import check, print_rank_0
from vllm_ascend.sample.rejection_sampler import rejection_greedy_sample_pytorch
from vllm_ascend.sample.rejection_sampler import rejection_greedy_sample_spec_len_1_pytorch

GREEDY_TEMPERATURE = -1
NZ_DIM = 16
import os
IS_SPEC_SAMPLE_GREEDY_COMPARE = int(os.environ.get('SPEC_SAMPLE_GREEDY_COMPARE', 0))


@register("vllm_ascend.worker.model_runner_v1.NPUModelRunner.execute_model")
@torch.inference_mode()
def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
) -> Union[ModelRunnerOutput, torch.Tensor]:
    with ProfileExecuteDuration().capture_async("prepare input and forward"):
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                logger.debug("skip this step for we receive the data from remote disaggregate prefill node")
                # Return empty ModelRunnerOuptut if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output)

        if self.dynamic_eplb:
            self.eplb_updator.forward_before()

        (
            attn_metadata,
            hidden_states,
            spec_decode_metadata,
            positions,
            num_scheduled_tokens,
            sample_indices,
            finished_sending,
            finished_recving,
        ) = self._process_reqs(scheduler_output, intermediate_tensors)
        
        if self.dynamic_eplb:
            self.eplb_updator.take_update_info_from_eplb_process()

    with ProfileExecuteDuration().capture_async("post process"):
        logits = self.model.compute_logits(hidden_states[sample_indices], None)

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            logits = self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata

        if spec_decode_metadata is None:
            if is_lmhead_tp():
                logits = logits[: self.input_batch.num_reqs]

            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            if is_lmhead_tp():
                logits = logits[: len(spec_decode_metadata.logits_indices)]

            if (IS_SPEC_SAMPLE_GREEDY_COMPARE == 1 and
                min(spec_decode_metadata.num_draft_tokens) == 1 and
                max(spec_decode_metadata.num_draft_tokens) == 1
            ):
                sample_logits = logits[spec_decode_metadata.logits_indices]
                temp_sampling_metadata = create_logit_sampling_metadata(
                    sampling_metadata,
                    spec_decode_metadata,
                    )
                sampler_output = self.sampler(
                    logits=sample_logits,
                    sampling_metadata=temp_sampling_metadata,
                )
                logits_token_ids = sampler_output.sampled_token_ids
                draft_token_ids = expand_draft_to_max(
                    spec_decode_metadata.draft_token_ids,
                    spec_decode_metadata.num_draft_tokens,
                    spec_decode_metadata.cu_num_draft_tokens,
                    spec_decode_metadata.max_spec_len,
                    )
                output_token_ids = norm_sample_then_rejection_greedy(
                    draft_token_ids,
                    spec_decode_metadata.num_draft_tokens,
                    spec_decode_metadata.max_spec_len,
                    logits_token_ids,
                    spec_decode_metadata.bonus_logits_indices,
                )
            else:
                # When indexing with a tensor (bonus_logits_indices), PyTorch
                # creates a new tensor with separate storage from the original
                # logits tensor. This means any in-place operations on bonus_logits
                # won't affect the original logits tensor.
                bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=sampling_metadata,
                    is_bonus_token=True,  # donot update sampling results for bonus tokens now
                )
                bonus_token_ids = sampler_output.sampled_token_ids
                # Just like `bonus_logits`, `target_logits` is a new tensor with
                # separate storage from the original `logits` tensor. Therefore,
                # it is safe to update `target_logits` in place.
                num_logprobs = sampling_metadata.max_num_logprobs
                if num_logprobs is not None:
                    set_mtp_logp_context(dict(
                        target_logits_indices=spec_decode_metadata.target_logits_indices,
                        bonus_logits_indices=spec_decode_metadata.bonus_logits_indices
                    ))
                else:
                    set_mtp_logp_context(None)
                target_logits = logits[spec_decode_metadata.target_logits_indices]
                output_token_ids = self.rejection_sampler(
                    spec_decode_metadata,
                    None,  # draft_probs
                    target_logits,
                    bonus_token_ids,
                    sampling_metadata,
                )

            output_token_ids = self.sampler.update_sampled_token_ids(output_token_ids, sampling_metadata)
            sampler_output.sampled_token_ids = output_token_ids

        # (woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        discard_sampled_tokens_req_indices = _handle_discard_sampled_tokens_req(self, scheduler_output)

        # NOTE: NPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() if logprobs_tensors is not None else None

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )
            # re-adjust logp
            num_logprobs = sampling_metadata.max_num_logprobs
            if num_logprobs is not None:
                from vllm.v1.outputs import LogprobsLists
                sampled_indices = get_mtp_logp_context()["output_indices"]
                sampled_logits = logits[sampled_indices.view(-1)]
                raw_logprobs = self.sampler.compute_logprobs(sampled_logits)
                logprobs_tensors = self.sampler.gather_logprobs(raw_logprobs, num_logprobs, 
                    token_ids=sampled_token_ids.long().view(-1))
                
                output_token_ids_np = sampled_token_ids.cpu().numpy()
                # Create mask for valid tokens.
                valid_mask = ((output_token_ids_np != -1) &
                            (output_token_ids_np < self.input_batch.vocab_size))
                spec_logprobslists = []
                logp_topklogp_shape = list(sampled_token_ids.shape) + [num_logprobs + 1]
                logprob_token_ids_s = logprobs_tensors.logprob_token_ids.reshape(logp_topklogp_shape)
                logprobs_s = logprobs_tensors.logprobs.reshape(logp_topklogp_shape)
                sampled_token_ranks_s = logprobs_tensors.selected_token_ranks.reshape(
                    sampled_token_ids.shape)
                for i, _ in enumerate(output_token_ids_np):
                    spec_logprobslists.append(LogprobsLists(
                        logprob_token_ids_s[i, valid_mask[i]].cpu().numpy().tolist(),
                        logprobs_s[i][valid_mask[i]].cpu().numpy().tolist(),
                        sampled_token_ranks_s[i][valid_mask[i]].cpu().numpy().tolist()
                    ))
                logprobs_lists = spec_logprobslists

        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        if self.speculative_config and not self.speculative_switch_on:
            self.use_spec_decode = True
        spec_token_ids = self._get_spec_token_ids(
            valid_sampled_token_ids,
            sampling_metadata,
            scheduler_output,
            spec_decode_metadata,
            positions,
            num_scheduled_tokens,
            hidden_states,
            attn_metadata,
        )
        if has_kv_transfer_group():
            get_kv_transfer_group().clear_connector_metadata()

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict={},
            finished_sending=finished_sending,
            finished_recving=finished_recving,
        )
    _handle_duration_and_eplb(self)

    return model_runner_output


def _handle_discard_sampled_tokens_req(self, scheduler_output):
    discard_sampled_tokens_req_indices = []
    for i, req_id in enumerate(self.input_batch.req_ids):
        req_state = self.requests[req_id]
        seq_len = req_state.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
        if seq_len < req_state.num_tokens:
            # Ignore the sampled token.
            # Rewind the generator state as if the token was not sampled.
            generator = self.input_batch.generators.get(i)
            if generator is not None:
                generator.set_offset(generator.get_offset() - 4)
            discard_sampled_tokens_req_indices.append(i)
    return discard_sampled_tokens_req_indices


def _handle_duration_and_eplb(self):
    durations = ProfileExecuteDuration().pop_captured_sync()
    if durations:
        dr_str = [f"[{tag}]:{duration:.2f}ms" for tag, duration in durations.items()]
        captured_name = "Decode" if self.attn_state == AscendAttentionState.DecodeOnly else "Prefill"
        logger.info("Profile execute duration [%s]:%s", captured_name, " ".join(dr_str))

    if self.dynamic_eplb:
        self.eplb_updator.forward_end()


@register("vllm_ascend.worker.model_runner_v1.NPUModelRunner._calc_spec_decode_metadata")
def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
) -> SpecDecodeMetadata:
    # Inputs:
    # Outputs:

    # Compute the logits indices.
    num_sampled_tokens: Any = num_draft_tokens + 1
    # Step 1. [4, 5, 8, 9, 11]
    cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
    total_num_sampled_tokens = cu_num_sampled_tokens[-1]
    # Step 2. [0, 0, 0, 0, 4, 5, 5, 5, 8, 9, 9]
    cumsums_offsets = np.repeat(cu_num_sampled_tokens - num_sampled_tokens, num_sampled_tokens)
    # Step 3. [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
    arange = self.arange_np[:total_num_sampled_tokens] - cumsums_offsets
    # Step 4. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
    logits_indices = np.repeat(cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
    # Step 5. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
    logits_indices += arange

    # Compute the bonus logits indices.
    bonus_logits_indices = cu_num_sampled_tokens - 1

    # Compute the draft logits indices.
    cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
    total_num_draft_tokens = cu_num_draft_tokens[-1]
    cumsums_offsets = np.repeat(cu_num_draft_tokens - num_draft_tokens, num_draft_tokens)
    arange = self.arange_np[:total_num_draft_tokens] - cumsums_offsets
    target_logits_indices = np.repeat(cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
    target_logits_indices += arange

    # Optimize the CPU -> NPU copy
    # patch: use pin memory to make not blocking
    cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).pin_memory().to(self.device, non_blocking=True)
    logits_indices = torch.from_numpy(logits_indices).pin_memory().to(self.device, non_blocking=True)
    target_logits_indices = torch.from_numpy(target_logits_indices).pin_memory().to(self.device, non_blocking=True)
    bonus_logits_indices = torch.from_numpy(bonus_logits_indices).pin_memory().to(self.device, non_blocking=True)

    # Compute the draft token ids.
    draft_token_ids = self.input_ids[logits_indices]
    draft_token_ids = draft_token_ids[target_logits_indices + 1]

    metadata = SpecDecodeMetadata(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens.tolist(),
        cu_num_draft_tokens=cu_num_draft_tokens,
        target_logits_indices=target_logits_indices,
        bonus_logits_indices=bonus_logits_indices,
        logits_indices=logits_indices,
    )
    return metadata



@register("vllm_ascend.worker.model_runner_v1.NPUModelRunner._get_forward_metadata_across_dp")
def _get_forward_metadata_across_dp(
            self, maybe_padded_num_tokens: int, num_tokens: int,
            with_prefill: bool, enable_dbo: bool, num_reqs: int = 0, cu_num_tokens: List[int] = 0,
            overlap: bool = False
) -> tuple[int, Optional[torch.Tensor], bool, bool]:
    def set_init_vals_for_overlap():
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens
        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                    non_blocking=True)
        self.slot_mapping[:num_tokens].copy_(
            self.slot_mapping_cpu[:num_tokens],
            non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.slot_mapping[num_tokens:].fill_(-1)
        self.seq_lens[num_reqs:].fill_(0)
        self.query_start_loc[num_reqs + 1:].fill_(-1)

    if self.dp_size == 1:
        if overlap:
            set_init_vals_for_overlap()
        return maybe_padded_num_tokens, None, with_prefill, enable_dbo
    if self.is_kv_producer and not envs_ascend.VLLM_ASCEND_ENABLE_CHUNK_MC2:
        if overlap:
            set_init_vals_for_overlap()
        num_tokens_across_dp = torch.tensor([num_tokens] * self.dp_size,
                                            device="cpu",
                                            dtype=torch.int32)
        return num_tokens, num_tokens_across_dp, True, enable_dbo
    is_kv_consumer_torchair_batchsize1 = self.is_kv_consumer and self.torchair_graph_enabled and \
        len(self.torchair_graph_batch_sizes) == 1 and not self.in_profile_run
    if is_kv_consumer_torchair_batchsize1:
        with_prefill_tensor = torch.tensor([with_prefill],
                                            device="npu",
                                            dtype=torch.bool)
        dist.all_reduce(with_prefill_tensor,
                        group=get_dp_group().device_group,
                        op=dist.ReduceOp.MAX)
        if not with_prefill_tensor.item():
            if overlap:
                set_init_vals_for_overlap()
            max_num_decode_tokens = self.torchair_graph_batch_sizes[0]
            num_tokens_across_dp = torch.tensor([max_num_decode_tokens] *
                                                self.dp_size,
                                                device="cpu",
                                                dtype=torch.int32)
            return max_num_decode_tokens, num_tokens_across_dp, False, enable_dbo

    num_tokens_across_dp = [0] * self.dp_size * 2
    num_tokens_across_dp[self.dp_rank] = maybe_padded_num_tokens
    num_tokens_across_dp[self.dp_size + self.dp_rank] = num_tokens
    forward_metadata = torch.tensor(num_tokens_across_dp +
                                    [with_prefill, not enable_dbo],
                                    device="npu",
                                    dtype=torch.int32)
    dist.all_reduce(forward_metadata, group=get_dp_group().device_group)
    if overlap:
        set_init_vals_for_overlap()
    with_prefill = bool(forward_metadata[-2].item())

    # NOTE: when with_prefill is false before all_reduce and true after all_reduce, we need to revert pad.
    if with_prefill:
        num_tokens_across_dp = forward_metadata[self.dp_size:self.dp_size *
                                                2]
        maybe_padded_num_tokens = num_tokens
    else:
        num_tokens_across_dp = forward_metadata[:self.dp_size]

    # NOTE: when in torchair_graph_mode, we need to pad local_num_tokens to
    # `max_num_tokens_across_dp`, in other situation it is not necessary.

    if self.torchair_graph_enabled and not with_prefill:
        maybe_padded_num_tokens = torch.max(num_tokens_across_dp).item()
        num_tokens_across_dp = torch.tensor([maybe_padded_num_tokens] *
                                            self.dp_size,
                                            device="cpu",
                                            dtype=torch.int32)
    else:
        num_tokens_across_dp = num_tokens_across_dp.to("cpu", non_blocking=True)
        
    return maybe_padded_num_tokens, num_tokens_across_dp, with_prefill, not bool(
        forward_metadata[-1])


@register("vllm_ascend.worker.model_runner_v1.NPUModelRunner._update_states")
def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
    """Update the cached states and the persistent batch with the scheduler
    output.

    The SamplingMetadata is updated and copied to the NPU if there is a
    new/resumed/paused/finished request in the batch.
    """
    # Remove finished requests from the cached states.
    for req_id in scheduler_output.finished_req_ids:
        self.requests.pop(req_id, None)
        self.encoder_cache.pop(req_id, None)
    # Remove the finished requests from the persistent batch.
    # NOTE(woosuk): There could be an edge case where finished_req_ids and
    # scheduled_req_ids overlap. This happens when a request is aborted and
    # then resubmitted with the same ID. In this case, we treat them as two
    # distinct requests - clearing the cached states for the first request
    # and handling the second as a new request.
    removed_req_indices: List[int] = []
    for req_id in scheduler_output.finished_req_ids:
        req_index = self.input_batch.remove_request(req_id)
        if req_index is not None:
            removed_req_indices.append(req_index)

    # Free the cached encoder outputs.
    for req_id, input_id in scheduler_output.free_encoder_input_ids:
        encoder_outputs = self.encoder_cache.get(req_id)
        if encoder_outputs is not None:
            encoder_outputs.pop(input_id, None)
            if not encoder_outputs:
                self.encoder_cache.pop(req_id, None)

    # Remove the unscheduled requests from the persistent batch.
    # NOTE(woosuk): The unscheduled requests are either preempted requests
    # or running requests that are not scheduled in this step. We remove
    # them from the persistent batch but keep their cached states since
    # they will be scheduled again sometime in the future.
    scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
    cached_req_ids = self.input_batch.req_id_to_index.keys()
    unscheduled_req_ids = cached_req_ids - scheduled_req_ids
    # NOTE(woosuk): The persistent batch optimization assumes that
    # consecutive batches contain mostly the same requests. If batches
    # have low request overlap (e.g., alternating between two distinct
    # sets of requests), this optimization becomes very inefficient.
    for req_id in unscheduled_req_ids:
        req_index = self.input_batch.remove_request(req_id)
        check(req_index is not None)
        removed_req_indices.append(req_index)

    req_ids_to_add: List[str] = []
    # Add new requests to the cached states.
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        sampling_params = new_req_data.sampling_params
        if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        self.requests[req_id] = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            mm_inputs=new_req_data.mm_inputs,
            mm_positions=new_req_data.mm_positions,
            sampling_params=sampling_params,
            generator=generator,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            output_token_ids=[],
            lora_request=new_req_data.lora_request,
        )

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            image_grid_thw = []
            video_grid_thw = []
            second_per_grid_ts = []
            audio_feature_lengths = []
            use_audio_in_video = False
            for mm_input in self.requests[req_id].mm_inputs:
                if mm_input.get("image_grid_thw") is not None:
                    image_grid_thw.extend(
                        mm_input["image_grid_thw"].tolist())
                if mm_input.get("video_grid_thw") is not None:
                    video_grid_thw.extend(
                        mm_input["video_grid_thw"].tolist())
                if mm_input.get("second_per_grid_ts") is not None:
                    second_per_grid_ts.extend(
                        mm_input["second_per_grid_ts"])
                if mm_input.get("audio_feature_lengths") is not None:
                    audio_feature_lengths.extend(
                        mm_input["audio_feature_lengths"])
                if mm_input.get("use_audio_in_video") is True:
                    use_audio_in_video = True

            hf_config = self.model_config.hf_config

            origin_prompt_token_ids = self.requests[req_id].prompt_token_ids
            
            self.requests[req_id].mrope_positions, \
            self.requests[req_id].mrope_position_delta = \
            MRotaryEmbedding.get_input_positions_tensor(
                origin_prompt_token_ids,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                audio_feature_lengths=audio_feature_lengths,
                use_audio_in_video=use_audio_in_video,
            )
            
        req_ids_to_add.append(req_id)

    # Update the states of the running/resumed requests.
    for req_data in scheduler_output.scheduled_cached_reqs:
        req_id = req_data.req_id
        req_state = self.requests[req_id]

        # Update the cached states.
        num_computed_tokens = req_data.num_computed_tokens
        req_state.num_computed_tokens = num_computed_tokens
        req_index = self.input_batch.req_id_to_index.get(req_id)
        # Add the sampled token(s) from the previous step (if any).
        # This doesn't include "unverified" tokens like spec decode tokens.
        num_new_tokens = (num_computed_tokens +
                            len(req_data.new_token_ids) -
                            req_state.num_tokens)
        if num_new_tokens == 1:
            # Avoid slicing list in most common case.
            req_state.output_token_ids.append(req_data.new_token_ids[-1])
        elif num_new_tokens > 0:
            req_state.output_token_ids.extend(req_data.new_token_ids[-num_new_tokens:])

        # Update the block IDs.
        if not req_data.resumed_from_preemption:
            # Append the new blocks to the existing block IDs.
            for block_ids, new_block_ids in zip(  # type: ignore[call-overload]
                    req_state.block_ids,
                    req_data.new_block_ids,
                    strict=True):
                block_ids.extend(new_block_ids)
        else:
            # The request is resumed from preemption.
            # Replace the existing block IDs with the new ones.
            req_state.block_ids = req_data.new_block_ids

        if req_index is None:
            # The request is not in the persistent batch.
            # The request was either preempted and resumed later, or was not
            # scheduled in the previous step and needs to be added again.
            req_ids_to_add.append(req_id)
            continue

        # Update the persistent batch.
        self.input_batch.num_computed_tokens_cpu[req_index] = (
            num_computed_tokens)

        start_index = (len(req_state.block_ids) -
                        len(req_data.new_block_ids))
        self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                req_index)
        # Add new_token_ids to token_ids_cpu.
        start_token_index = num_computed_tokens
        end_token_index = num_computed_tokens + len(req_data.new_token_ids)
        self.input_batch.token_ids_cpu[
            req_index,
            start_token_index:end_token_index] = req_data.new_token_ids
        self.input_batch.num_tokens_no_spec[req_index] = end_token_index
        # Add spec_token_ids to token_ids_cpu.
        spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
            req_id, ())
        if spec_token_ids:
            start_index = end_token_index
            end_token_index += len(spec_token_ids)
            self.input_batch.token_ids_cpu[
                req_index, start_index:end_token_index] = spec_token_ids
        # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
        self.input_batch.num_tokens[req_index] = end_token_index

    # Check if the batch has changed. If not, we can skip copying the
    # sampling metadata from CPU to GPU.
    batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

    # Add the new or resumed requests to the persistent batch.
    # The smaller empty indices are filled first.
    removed_req_indices = sorted(removed_req_indices, reverse=True)
    for req_id in req_ids_to_add:
        req_state = self.requests[req_id]
        if removed_req_indices:
            # Fill the empty index.
            req_index = removed_req_indices.pop()
        else:
            # Append to the end.
            req_index = None
        self.input_batch.add_request(req_state, req_index)
        spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
            req_id, ())
        if spec_token_ids:
            req_index = self.input_batch.num_reqs - 1
            start_index = len(req_state.prompt_token_ids) + len(
                req_state.output_token_ids)
            end_token_index = start_index + len(spec_token_ids)
            self.input_batch.token_ids_cpu[
                req_index, start_index:end_token_index] = spec_token_ids
            self.input_batch.num_tokens[req_index] = end_token_index

    # Condense the batched states if there are empty indices.
    if removed_req_indices:
        self.input_batch.condense(removed_req_indices)

    if batch_changed:
        self.input_batch.refresh_sampling_metadata()


@register("vllm_ascend.worker.model_runner_v1.NPUModelRunner._process_reqs")
def _process_reqs(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
) -> tuple[SpecDecodeMetadata, torch.Tensor, SpecDecodeMetadata,
torch.Tensor, int, torch.Tensor, Optional[set[str]],
Optional[set[str]]]:

    # Check input valid
    num_input_tokens, num_reqs, total_num_scheduled_tokens = check_input_valid(scheduler_output, self)

    modified_batch = self.attn_metadata_builder.reorder_batch(
        self.input_batch, scheduler_output)
    if modified_batch:
        self.input_batch.refresh_sampling_metadata()

    # OPTIMIZATION: Start copying the block table first.
    # This way, we can overlap the copy with the following CPU operations.
    self.input_batch.block_table.commit(num_reqs)

    # Get the number of scheduled tokens for each request.
    # The Python loop can be slow. Should Optimize.
    max_num_scheduled_tokens, num_scheduled_tokens, num_valid_tokens = get_scheduled_tokens(num_reqs, scheduler_output,
                                                                                            self)

    # PATCH NOTICE: 
    # The following 4 lines (calculation of self.seq_lens_np and seq_lens_cpu) are moved here.
    # The original code place is between `self.query_lens = torch.from_numpy(num_scheduled_tokens)` and 
    # `block_table_indices = (req_indices * self.max_num_blocks_per_req +`.
    # The motivation of this patch is to enable the calculation of attention state, 
    # so that we can decide `speculative_switch_on` based on with_prefill.
    self.seq_lens_np[:num_reqs] = (
        self.input_batch.num_computed_tokens_cpu[:num_reqs] +
        num_scheduled_tokens)
    seq_lens_cpu = self.seq_lens_cpu[:num_reqs]

    # PATCH NOTICE: 
    # The following 19 lines (`ascend_config = get_ascend_config() ... 
    # attn_state = AscendAttentionState.PrefillCacheHit`) are moved here.
    # The original code place is between `out=self.slot_mapping_np[:total_num_scheduled_tokens])` and 
    # `if not self.vllm_config.model_config.use_mla and \`.
    # The motivation of this patch is to enable the calculation of attention state, 
    # so that we can decide `speculative_switch_on` based on with_prefill.
    ascend_config = get_ascend_config()
    use_spec_decode = len(
        scheduler_output.scheduled_spec_decode_tokens) > 0
    if np.array_equal(self.seq_lens_np[:num_reqs], num_scheduled_tokens):
        attn_state = AscendAttentionState.PrefillNoCache
    # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
    elif np.all(num_scheduled_tokens == 1):
        attn_state = AscendAttentionState.DecodeOnly
        if self.speculative_config and self.speculative_config.method == 'deepseek_mtp':
            # support deepseek mtp spec decode in disaggregated-prefill scenario
            attn_state = AscendAttentionState.SpecDecoding
    # Speculative decoding.
    elif np.all(num_valid_tokens == 1):
        attn_state = AscendAttentionState.SpecDecoding
    # splitfuse
    elif not ascend_config.ascend_scheduler_config.enabled or self.chunked_prefill_enabled:
        attn_state = AscendAttentionState.ChunkedPrefill
    else:
        attn_state = AscendAttentionState.PrefillCacheHit
    
    # PATCH NOTICE: 
    # The following 3 lines (`with_prefill = attn_state not in [ ... ] are moved here.
    # The original code place is between `self.seq_lens_list = self.seq_lens_np.tolist()[:num_input_tokens]` and 
    # `if self.torchair_graph_enabled and not with_prefill:`.
    # The motivation of this patch is to enable the calculation of `with_prefill`, 
    # so that we can decide `speculative_switch_on` based on with_prefill.
    with_prefill = attn_state not in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]
    
    # PATCH NOTICE:
    # Speculative auto switch
    # if speculative_auto_switch is on, decide whether use speculation decoding or not based on batch size.
    if self.speculative_config:
        if self.speculative_auto_switch and num_reqs > self.speculative_auto_bz_thre:
            self.speculative_switch_on = False
        else:
            self.speculative_switch_on = True
    if with_prefill:
        self.speculative_switch_on = True

    if self.speculative_config and not self.speculative_switch_on:
        self.use_spec_decode = False
        num_scheduled_tokens = np.array([1 for _ in range(len(self.input_batch.req_ids))], dtype=np.int32)
        num_valid_tokens = np.array([1 for _ in range(len(self.input_batch.req_ids))], dtype=np.int32)
        max_num_scheduled_tokens = max(num_scheduled_tokens)
        attn_state = AscendAttentionState.DecodeOnly
        total_num_scheduled_tokens = num_reqs
        self.spec_attn_mask = None
        self.spec_token_num = 0
        self.decode_token_per_req = 1

    if (self.use_aclgraph and
            total_num_scheduled_tokens <= self.aclgraph_batch_sizes[-1]):
        # Add padding to the batch size.
        num_input_tokens = self.vllm_config.pad_for_cudagraph(
            total_num_scheduled_tokens)
    else:
        # Eager mode.
        num_input_tokens = total_num_scheduled_tokens


    # Hot-Swap lora model
    if self.lora_config:
        self.set_active_loras(self.input_batch, num_scheduled_tokens)

    # Prepare positions
    cu_num_tokens, positions_np, req_indices, sample_indices = prepare_positions(num_reqs, num_scheduled_tokens, self,
                                                                                total_num_scheduled_tokens)
    # Calculate M-RoPE positions.
    # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
    if self.uses_mrope:
        self._calc_mrope_positions(scheduler_output)

    if self.uses_mrope:
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        self.mrope_positions_cpu[:, total_num_scheduled_tokens:num_input_tokens].zero_()
        self.mrope_positions[:, :num_input_tokens].copy_(
            self.mrope_positions_cpu[:, :num_input_tokens],
            non_blocking=True)

    self.positions_cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
    self.positions[:num_input_tokens].copy_(
        self.positions_cpu[:num_input_tokens], non_blocking=True)
    positions_cpu = self.positions_cpu[:num_input_tokens]
    positions = self.positions[:num_input_tokens]
    self.query_lens = torch.from_numpy(num_scheduled_tokens)

    # PATCH NOTICE:
    # The codes of calculation of self.seq_lens_np and seq_lens_cpu are moved above. 
    # See PATCH NOTICE with `self.seq_lens_np[:num_reqs] = (...`.

    block_table_indices = (req_indices * self.max_num_blocks_per_req +
                           positions_np // self.block_size)

    block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
    block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
    block_offsets = positions_np % self.block_size
    np.add(block_numbers * self.block_size,
           block_offsets,
           out=self.slot_mapping_np[:total_num_scheduled_tokens])

    # PATCH NOTICE:
    # The original 19 lines here (`ascend_config = get_ascend_config() ... 
    # attn_state = AscendAttentionState.PrefillCacheHit`) are moved above. 
    # See PATCH NOTICE with `ascend_config = get_ascend_config()...`.

    # NOTE: when use ring_mla, attn_mask don't need to generate here.
    if not self.vllm_config.model_config.use_mla and \
        self.vllm_config.model_config.hf_config.model_type != 'PanguProMoE':
        attn_mask = self._make_attention_mask(seq_lens=seq_lens_cpu,
                                              position=positions_cpu,
                                              attn_state=attn_state)
        self.attn_mask = attn_mask

    self.attn_state = attn_state  # type: ignore

    extra_builder_kwargs = {}

    # Use host tensor, other wise error: tensor.hostData is null
    self.seq_lens_list = self.seq_lens_np.tolist()[:num_input_tokens]

    # PATCH NOTICE:
    # The original 3 lines here (`with_prefill = attn_state not in [ ... ]`) are moved above. 
    # See PATCH NOTICE with `with_prefill = attn_state not in...`.

    is_only_prefill = bool(np.all(num_valid_tokens != 1))

    enable_dbo = self._check_dbo_is_valid(self.query_lens.tolist(),
                                          attn_state,
                                          total_num_scheduled_tokens)

    maybe_padded_num_tokens = total_num_scheduled_tokens
    if self.torchair_graph_enabled and not with_prefill:
        maybe_padded_num_tokens = self.select_torchair_padded_batch_size(
            total_num_scheduled_tokens)
    
    (padded_num_tokens_across_dp, num_tokens_across_dp, with_prefill,
     enable_dbo) = self._get_forward_metadata_across_dp(
        maybe_padded_num_tokens, total_num_scheduled_tokens, with_prefill,
        enable_dbo, num_reqs, cu_num_tokens, True)

    if self.torchair_graph_enabled and not with_prefill:
        padded_batch_size = self.select_torchair_padded_batch_size(
            total_num_scheduled_tokens)
        graph_pad_size = padded_batch_size - num_reqs * self.decode_token_per_req 

        extra_builder_kwargs['graph_pad_size'] = graph_pad_size
    
    self.is_only_prefill = is_only_prefill
    self.enable_dbo = enable_dbo

    common_attn_metadata = AscendCommonAttentionMetadata(
        query_start_loc=self.query_start_loc[:num_reqs + 1],
        query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs + 1],
        seq_lens=self.seq_lens[:num_reqs],
        seq_lens_cpu=self.seq_lens_cpu[:num_reqs],
        num_reqs=num_reqs,
        num_actual_tokens=total_num_scheduled_tokens,
        max_query_len=max_num_scheduled_tokens,
        actual_seq_lengths_q=self.actual_seq_lengths_q,
        block_table_tensor=self.input_batch.block_table[0].
        get_device_tensor(),
        slot_mapping_cpu=self.
        slot_mapping_cpu[:total_num_scheduled_tokens],
        positions=self.mrope_positions[:, :num_input_tokens] if self.uses_mrope else self.positions[:num_input_tokens],
        attn_mask=self.attn_mask,
        spec_attn_mask=self.spec_attn_mask,
        attn_state=self.attn_state,  # type: ignore
        decode_token_per_req=self.decode_token_per_req,
        max_num_blocks_per_req=self.max_num_blocks_per_req,
        enable_dbo_across_dp=enable_dbo,
        is_only_prefill=is_only_prefill,
    )

    # TODO(zzzzwwjj): this code need to refactor afterwards.
    self.with_prefill = with_prefill
    # Add num_token_pad_size and num_reqs_pad_size here for torchair graph mode
    if self.torchair_graph_enabled and not self.with_prefill:
        num_token_pad_size = padded_num_tokens_across_dp - total_num_scheduled_tokens
        num_reqs_pad_size = (
                padded_num_tokens_across_dp // self.decode_token_per_req -
                num_reqs)

        extra_builder_kwargs['num_token_pad_size'] = num_token_pad_size
        extra_builder_kwargs['num_reqs_pad_size'] = num_reqs_pad_size
        self.num_reqs_pad_size = num_reqs_pad_size
        self.num_token_pad_size = num_token_pad_size
    self.extra_builder_kwargs = extra_builder_kwargs
    self.num_tokens_across_dp = num_tokens_across_dp

    # patch for sparse KV of SWA
    if self.kvcache_sparsify:

        req_idx = list(self.input_batch.req_id_to_index.values())
        attn_metadata = self.attn_metadata_builder.build(  # type: ignore
            common_attn_metadata=common_attn_metadata,
            req_idx=req_idx,
            **extra_builder_kwargs,
        )
    else:
        attn_metadata = self.attn_metadata_builder.build(  # type: ignore
            common_attn_metadata=common_attn_metadata,
            **extra_builder_kwargs,
        )
    attn_metadata.num_input_tokens = padded_num_tokens_across_dp

    # Prepare input_ids
    token_indices = (positions_np +
                     req_indices * self.input_batch.token_ids_cpu.shape[1])
    torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                       0,
                       torch.from_numpy(token_indices),
                       out=self.input_ids_cpu[:total_num_scheduled_tokens])
    # Copy the tensors to the NPU.
    self.input_ids[:total_num_scheduled_tokens].copy_(
        self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)

    mm_embeds = []

    input_ids = self.input_ids[:num_input_tokens]
    inputs_embeds = None
    if self.uses_mrope:
        positions = self.mrope_positions[:, :num_input_tokens]

    input_ids = self.input_ids[:padded_num_tokens_across_dp]
    positions = self.positions[:padded_num_tokens_across_dp]

    # Run forward pass
    # TODO(zzzzwwjj): check param `num_tokens_across_dp` later.
    with set_ascend_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=padded_num_tokens_across_dp,
            num_tokens_across_dp=num_tokens_across_dp,
            with_prefill=with_prefill,
            num_actual_tokens=total_num_scheduled_tokens):
        with ProfileExecuteDuration().capture_async("forward"):
            self.maybe_setup_kv_connector(scheduler_output)
            model_kwargs = {}
            if self.torchair_graph_enabled:
                model_kwargs["kv_caches"] = self.kv_caches
                model_kwargs["attn_metadata"] = attn_metadata
            if envs_ascend.VLLM_ASCEND_ENABLE_DBO:
                if with_prefill:
                    model_kwargs["graph_enable"] = False  # type: ignore
                else:
                    model_kwargs["graph_enable"] = True  # type: ignore
            hidden_states = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs)

    self.maybe_wait_for_kv_save()
    finished_sending, finished_recving = self.get_finished_kv_transfer(
        scheduler_output)
    use_spec_decode = len(
        scheduler_output.scheduled_spec_decode_tokens) > 0
    # auto_switch
    if self.speculative_config and not self.speculative_switch_on:
        use_spec_decode = False
    if not use_spec_decode:
        # NOTE(woosuk): Due to chunked prefills, the batch may contain
        # partial requests. While we should not sample any token
        # from these partial requests, we do so for simplicity.
        # We will ignore the sampled tokens from the partial requests.
        spec_decode_metadata = None
    else:
        # Get the number of draft tokens for each request.
        # Iterate over the dictionary rather than all requests since not all
        # requests have draft tokens.
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        for req_id, draft_token_ids in (
                scheduler_output.scheduled_spec_decode_tokens.items()):
            req_idx = self.input_batch.req_id_to_index[req_id]
            num_draft_tokens[req_idx] = len(draft_token_ids)

        spec_decode_metadata = self._calc_spec_decode_metadata(
            num_draft_tokens, cu_num_tokens)
        sample_indices = spec_decode_metadata.logits_indices

    if is_lmhead_tp():
        if not with_prefill:
            padded_num_indices = padded_num_tokens_across_dp
        else:
            padded_num_indices = self.max_num_reqs
        sample_indices = nn.functional.pad(
            sample_indices,
            (0, padded_num_indices - sample_indices.shape[0]))

    output = (
        attn_metadata,
        hidden_states,
        spec_decode_metadata,
        positions,
        total_num_scheduled_tokens,
        sample_indices,
        finished_sending,
        finished_recving,
    )
    return output


def prepare_positions(num_reqs, num_scheduled_tokens, self, total_num_scheduled_tokens):
    req_indices = np.repeat(self.arange_np[:num_reqs],
                            num_scheduled_tokens)
    cu_num_tokens = np.cumsum(num_scheduled_tokens)
    cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                num_scheduled_tokens)
    sample_indices = cu_num_tokens - 1
    sample_indices = torch.from_numpy(sample_indices).to(self.device,
                                                         non_blocking=True)
    arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets
    positions_np = self.positions_np[:total_num_scheduled_tokens]
    np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
           arange,
           out=positions_np)
    return cu_num_tokens, positions_np, req_indices, sample_indices


def get_scheduled_tokens(num_reqs, scheduler_output, self):
    num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
    num_valid_tokens = np.empty(num_reqs, dtype=np.int32)
    max_num_scheduled_tokens = 0
    for i, req_id in enumerate(self.input_batch.req_ids):
        num_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_scheduled_tokens[i] = num_tokens
        num_valid_tokens[i] = num_tokens - \
                              len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
        max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                       num_tokens)
    return max_num_scheduled_tokens, num_scheduled_tokens, num_valid_tokens


def check_input_valid(scheduler_output, self):
    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    num_reqs = self.input_batch.num_reqs
    if (self.use_aclgraph and
            total_num_scheduled_tokens <= self.aclgraph_batch_sizes[-1]):
        # Add padding to the batch size.
        num_input_tokens = self.vllm_config.pad_for_cudagraph(
            total_num_scheduled_tokens)
    else:
        # Eager mode.
        num_input_tokens = total_num_scheduled_tokens
    return num_input_tokens, num_reqs, total_num_scheduled_tokens


@register("vllm_ascend.worker.model_runner_v1.NPUModelRunner.__init__")
def NPUModelRunner__init__(self, vllm_config: VllmConfig, device: torch.device):
    self.pangu_config = get_config()
    self.vllm_config = vllm_config
    self.model_config = vllm_config.model_config
    self.cache_config = vllm_config.cache_config
    self.lora_config = vllm_config.lora_config
    self.parallel_config = vllm_config.parallel_config
    self.scheduler_config = vllm_config.scheduler_config
    self.speculative_config = vllm_config.speculative_config
    spec_decoding_config = self.pangu_config.get("additional_config", {}).get("speculative_decoding", None)
    self.speculative_auto_switch = False
    self.speculative_auto_bz_thre = 1024
    if spec_decoding_config:
        if spec_decoding_config.get("enabled", False):
            if not self.speculative_config:
                self.speculative_config = SpeculativeConfig(model="ngram", num_speculative_tokens=3)
            spec_method = spec_decoding_config.get("method", "sam")
            if spec_method not in ["sam", "suffix"]:
                raise NotImplementedError(f"Method {spec_method} is not supported. \
                                          You can only choose method from 'sam' and 'suffix'.")
            self.speculative_config.method = spec_method
            self.speculative_config.num_speculative_tokens = spec_decoding_config.get("num_speculative_tokens", 3)
            # SAM-related config
            if spec_method == "sam":
                self.asynchronous_sam = False
                if spec_decoding_config.get("asynchronous_sam", False):
                    self.asynchronous_sam = True
                    if not has_samfast_package():
                        raise ModuleNotFoundError("`samfast` package is required to \
                                                    use the asynchronous SAM construction function. \
                                                    Please install the package at \
                                                    Pangu_infer/pangu_infer/pangu/accelerators/\
                                                    speculative_decoding/samfast.")
            # suffix-tree-related config
            elif spec_method == "suffix":
                self.speculative_config.max_tree_depth = spec_decoding_config.get("suffix_max_tree_depth", 64)
                self.speculative_config.max_spec_factor = spec_decoding_config.get("suffix_max_spec_factor", 4)
                self.speculative_config.min_token_prob = spec_decoding_config.get("suffix_min_token_prob", 0.1)
                self.speculative_config.max_model_len = spec_decoding_config.get("suffix_max_model_len", 4000)
                self.speculative_config.max_cached_requests = \
                    spec_decoding_config.get("suffix_max_cached_requests", 10000)

            self.vllm_config.speculative_config = self.speculative_config
            self.speculative_auto_switch = spec_decoding_config.get("batch_size_auto_switch", False)
            self.speculative_auto_bz_thre = spec_decoding_config.get("batch_size_auto_switch_thre", 1024) 
    ascend_config = get_ascend_config()
    if ascend_config.ascend_scheduler_config.enabled:
        self.chunked_prefill_enabled = self.scheduler_config.chunked_prefill_enabled
    else:
        self.chunked_prefill_enabled = True
    self.device = device

    self.is_multimodal_model = False
    self.block_size = vllm_config.cache_config.block_size

    self.max_num_blocks_per_req = cdiv(self.model_config.max_model_len,
                                       self.block_size)
    self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
    self.max_num_reqs = self.scheduler_config.max_num_seqs

    # Model-related.
    self.num_attn_layers = self.model_config.get_num_layers_by_block_type(
        vllm_config.parallel_config, LayerBlockType.attention)
    self.hidden_size = self.model_config.get_hidden_size()
    self.dtype = self.model_config.dtype
    cache_config = vllm_config.cache_config
    if cache_config.cache_dtype == "auto":
        self.kv_cache_dtype = self.dtype
    else:
        self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
            cache_config.cache_dtype]

    self.head_size = self.model_config.get_head_size()
    self.attn_backend = get_attn_backend(
        self.head_size,
        self.dtype,
        self.kv_cache_dtype,
        self.block_size,
        self.model_config.is_attention_free,
        use_mla=self.model_config.use_mla,
    )
    if self.attn_backend is None:
        error_msg = (
            f"Error with get_att_backend: {self.head_size=}, "
            f"{self.dtype=}, {self.kv_cache_dtype=}, {self.block_size=}, "
            f"{self.model_config.is_attention_free=}, "
            f"{self.model_config.use_mla=}")
        logger.error(error_msg)
        raise NotImplementedError(
            "Non-Attention backend is not supported by V1 NPUModelRunner.")

    self.attn_metadata_builder = self.attn_backend.get_builder_cls()(
        vllm_config=self.vllm_config,
        device=self.device,
        runner=weakref.proxy(self))

    # Multi-modal data support
    self.input_registry = INPUT_REGISTRY
    self.mm_registry = MULTIMODAL_REGISTRY
    self.uses_mrope = self.model_config.uses_mrope

    self.max_num_encoder_input_tokens, self.encoder_cache_size = compute_encoder_budget(
        model_config=self.model_config,
        scheduler_config=self.scheduler_config,
        mm_registry=self.mm_registry)

    # Lazy initialization
    self.kv_caches: List[torch.Tensor] = []
    # req_id -> (input_id -> encoder_output)
    self.encoder_cache: Dict[str, Dict[int, torch.Tensor]] = {}

    # Set up speculative decoding.
    self.use_spec_decode = False
    self.spec_attn_mask = None
    self.actual_seq_lengths_q = []
    self.spec_token_num = 0
    self.decode_token_per_req = 1
    self.speculative_switch_on = False
    if self.speculative_config:
        self.use_spec_decode = True
        self.spec_token_num = self.speculative_config.num_speculative_tokens
        if self.spec_token_num <= 0:
            raise ValueError(f"self.spec_token_num is {self.spec_token_num}.")
        self.spec_attn_mask = torch.triu(torch.ones(2048,
                                                    2048,
                                                    dtype=torch.bool),
                                         diagonal=1).to("npu")
        if get_pp_group().is_last_rank:
            if self.speculative_config.method == "sam": # the speculative_config here is taken from pangu_config
                self.drafter = SAMProposer(self.speculative_config, asynchronous_sam=self.asynchronous_sam) 
            elif self.speculative_config.method == "suffix":
                self.drafter = SuffixDecodingProposer(self.speculative_config)
            elif self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.method == "eagle":
                self.drafter = EagleProposer(self.vllm_config,
                                             self.device)  # type: ignore
            elif self.speculative_config.method == 'deepseek_mtp':
                self.drafter = MtpProposer(self.vllm_config, self)
                self.decode_token_per_req = 1 + self.spec_token_num
                self.actual_seq_lengths_q = [
                    len for len in
                    range(self.decode_token_per_req, self.max_num_tokens +
                          1, self.decode_token_per_req)
                ]
            else:
                raise ValueError("Unknown speculative decoding method: "
                                 f"{self.speculative_config.method}")
            self.rejection_sampler = AscendRejectionSampler()

    # Request states.
    self.requests: Dict[str, CachedRequestState] = {}
    # Persistent batch.

    self.input_ids = torch.zeros(self.max_num_tokens,
                                 dtype=torch.int32,
                                 device=self.device)
    self.positions = torch.zeros(self.max_num_tokens,
                                 dtype=torch.int64,
                                 device=self.device)
    self.query_start_loc = torch.zeros(self.max_num_reqs + 1,
                                       dtype=torch.int32,
                                       device=self.device)
    self.seq_lens = torch.zeros(self.max_num_reqs,
                                dtype=torch.int32,
                                device=self.device)
    self.slot_mapping = torch.zeros(self.max_num_tokens,
                                    dtype=torch.int32,
                                    device=self.device)
    self.query_lens = torch.zeros(self.max_num_reqs,
                                  dtype=torch.int32,
                                  device=self.device)
    # None in the first PP rank. The rest are set after load_model.
    self.intermediate_tensors: Optional[IntermediateTensors] = None

    # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
    if self.uses_mrope:
        # NOTE: `mrope_positions` is implemented with one additional dummy
        # position on purpose to make it non-contiguous so that it can work with torch compile.
        # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

        # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
        # the modality of inputs. For text-only inputs, each dimension has
        # identical position IDs, making M-RoPE functionally equivalent to
        # 1D-RoPE.
        # See page 5 of https://arxiv.org/abs/2409.12191
        self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1),
                                           dtype=torch.int64,
                                           device=self.device)
        self.mrope_positions_cpu = torch.zeros(
            (3, self.max_num_tokens + 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True)

    # OPTIMIZATION: Cache the tensors rather than creating them every step.
    self.arange_np: npt.NDArray[np.int32] = np.arange(max(
        self.max_num_reqs + 1, self.model_config.max_model_len,
        self.max_num_tokens),
        dtype=np.int32)
    # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
    # a faster version of creating a new tensor every time. Thus, we should
    # not make any assumptions about the values in these tensors.
    self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device="cpu",
                                     pin_memory=True)
    self.positions_cpu = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device="cpu",
                                     pin_memory=True)
    self.positions_np = self.positions_cpu.numpy()

    self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=True)
    self.slot_mapping_np = self.slot_mapping_cpu.numpy()

    self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                           dtype=torch.int32,
                                           device="cpu",
                                           pin_memory=True)
    self.query_start_loc_np = self.query_start_loc_cpu.numpy()

    self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                    dtype=torch.int32,
                                    device="cpu",
                                    pin_memory=True)
    self.seq_lens_np = self.seq_lens_cpu.numpy()

    self.input_positions_cpu = torch.arange(0,
                                            self.max_num_tokens,
                                            device="cpu")
    self.attn_mask = None
    self.attn_state = None

    self.is_only_prefill = False
    self.enable_dbo = False

    self.use_aclgraph = (self.vllm_config.compilation_config.level
                         == CompilationLevel.PIECEWISE
                         and not self.model_config.enforce_eager)
    self.aclgraph_batch_sizes = list(
        reversed(
            self.vllm_config.compilation_config.cudagraph_capture_sizes))

    # patch code
    # NOTE: Pre-construct a mask matrix on cpu to improve the efficiency of
    # attention mask construction during inference.
    if not self.vllm_config.model_config.use_mla and \
        self.vllm_config.model_config.hf_config.model_type != 'PanguProMoE':
        self.attn_mask_builder = AttentionMaskBuilder.initialize_from_len(
                self.model_config.max_model_len, self.dtype)
    else:
        self.attn_mask_builder = None

    self.sampler = Sampler()
    self.new_kv_cache_bytes = -1
    self.torchair_compiled_model = None  # type: ignore
    self.torchair_compiled_models = {}  # type: ignore
    ascend_config = get_ascend_config()

    # torchair graph config
    self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled

    self.use_cached_npu_graph = ascend_config.torchair_graph_config.use_cached_graph
    self.use_cached_kv_cache_bytes = ascend_config.torchair_graph_config.use_cached_kv_cache_bytes
    self.torchair_graph_batch_sizes = ascend_config.torchair_graph_config.graph_batch_sizes

    # kv role
    self.is_kv_producer = False
    self.is_kv_consumer = False
    if vllm_config.kv_transfer_config is not None:
        self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

    if ascend_config.torchair_graph_config.graph_batch_sizes_init:
        self.init_torchair_graph_batch_sizes()

    self.check_torchair_graph_batch_sizes()
    ascend_config.torchair_graph_config.graph_batch_sizes = self.torchair_graph_batch_sizes

    # graph_block_tables shape: [num_request, cell(max_model_len / block_size)]
    self.graph_block_tables = np.zeros(
        (self.torchair_graph_batch_sizes[-1] // self.decode_token_per_req,
         (self.model_config.max_model_len + self.block_size - 1) //
         self.block_size),
        dtype=np.int32)

    torch._dynamo.cache_size.config.cache_size_limit += len(
        self.torchair_graph_batch_sizes)
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._logging.set_logs(
        recompiles=envs_ascend.VLLM_ASCEND_TRACE_RECOMPILES)

    self.dp_size = vllm_config.parallel_config.data_parallel_size
    self.dp_rank = vllm_config.parallel_config.data_parallel_rank

    # EPLB
    self.dynamic_eplb = ascend_config.dynamic_eplb
    if self.dynamic_eplb:
        self.eplb_adaptor: Optional[VllmEplbAdaptor] = None
        self.is_eplb_warmuped = False
        self.eplb_updator = EplbUpdator(ascend_config.expert_map_path)

    # NOTE: we need to use `in_profile_run` to determine whether `enable_force_load_balance` is True
    self.in_profile_run = False

    # patch for sparse KV of SWA
    self.kvcache_sparsify = vllm_config.additional_config.get("kvcache_sparsify", {}).get("enabled", False)

    self.image_token_prune_ratio = 0
    self.audio_token_prune_ratio = 0


@register("vllm_ascend.worker.model_runner_v1.NPUModelRunner._make_attention_mask")
def _make_attention_mask(self, seq_lens, position, attn_state) -> torch.Tensor:
    # patch code
    if self.attn_mask_builder is None:
        return None

    # Chunk Prefill situation.
    if attn_state == AscendAttentionState.ChunkedPrefill:
        return self.attn_mask_builder.get_splitfuse_attn_mask(
            seq_lens.clone(), position, self.dtype, self.device)
    # Prefill without cache situation.
    elif attn_state == AscendAttentionState.PrefillNoCache:
        max_seq_len = max(seq_lens, default=0)
        return self.attn_mask_builder.get_attn_mask(
            max_seq_len, self.dtype, self.device)
    # Prefill with cache hit.
    elif attn_state == AscendAttentionState.PrefillCacheHit:
        return self.attn_mask_builder.get_attn_mask(
            128, self.dtype, self.device)
        # Speculative decoding: same as chunked prefilling
    elif attn_state == AscendAttentionState.SpecDecoding:
        return self.attn_mask_builder.get_splitfuse_attn_mask(
            seq_lens, position, self.dtype, self.device)
    # Decode-only situation.
    else:
        return None


@register("vllm_ascend.worker.model_runner_v1.NPUModelRunner._get_spec_token_ids")
def _get_spec_token_ids(
    self,
    valid_sampled_token_ids: list[list[int]],
    sampling_metadata: SamplingMetadata,
    scheduler_output: "SchedulerOutput",
    spec_decode_metadata: SpecDecodeMetadata,
    positions: torch.Tensor,
    num_scheduled_tokens: int,
    hidden_states: torch.Tensor,
    attn_metadata: SpecDecodeMetadata,
) -> Optional[list[list[int]]]:
    if not self.use_spec_decode:
        # Speculative decoding is not enabled.
        spec_token_ids = None
    elif self.speculative_config.method == "sam":
        # PATCH NOTICE: According to the vllm-ascend and vllm code, 
        # there should be an assertion of self.drafter is a SAMProposer here.
        # The assertion is deleted due to codecheck.
        # Similar assertions corresponding to "ngram", "eagle" 
        # and "deepseek_mtp" are also removed.
        spec_token_ids = self._generate_sam_token_ids(
            valid_sampled_token_ids)
    elif self.speculative_config.method == "suffix":
        spec_token_ids = self._generate_sam_token_ids(
            valid_sampled_token_ids) 
    elif self.speculative_config.method == "ngram":
        spec_token_ids = self._generate_draft_token_ids(
            valid_sampled_token_ids, sampling_metadata)
    elif self.speculative_config.method == "eagle":
        raise NotImplementedError(
            "eagle method for spec decode doesn't work on vllm-ascend currently"
        )
    return spec_token_ids


@register("vllm_ascend.worker.model_runner_v1.NPUModelRunner._generate_sam_token_ids")
def _generate_sam_token_ids(
        self,
        sampled_token_ids: list[list[int]],
) -> list[list[int]]:
    # TODO(woosuk): Optimize.
    draft_token_ids: list[list[int]] = []
    for i, sampled_ids in enumerate(sampled_token_ids):
        num_sampled_ids = len(sampled_ids)
        if not num_sampled_ids:
            # Skip speculative decoding.
            draft_token_ids.append([])
            continue

        # Skip requests that require top-p, top-k, etc.
        req_id = self.input_batch.req_ids[i]
        if not is_spec_decode_supported(req_id, self.input_batch):
            draft_token_ids.append([])
            continue

        # Add sampled_token_ids to token_ids_cpu.
        start_idx = self.input_batch.num_tokens_no_spec[i]
        end_idx = start_idx + num_sampled_ids
        self.input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids

        drafter_output = self.drafter.propose(
            req_id,
            self.input_batch.token_ids_cpu[i, :start_idx],
            self.input_batch.token_ids_cpu[i, start_idx:end_idx],
            num_sampled_ids)

        if drafter_output is None or len(drafter_output) == 0:
            draft_token_ids.append([])

        else:
            draft_token_ids.append(drafter_output)

    return draft_token_ids


def norm_sample_then_rejection_greedy(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    all_sampled_tokens: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
):
    check(draft_token_ids.ndim == 1)
    batch_size = len(num_draft_tokens)
    device = all_sampled_tokens.device
    check(all_sampled_tokens.is_contiguous())
    valid_flag = torch.ones(batch_size, dtype=bool, device=device)
    accepted = draft_token_ids == (all_sampled_tokens.squeeze(1))
    accepted[bonus_logits_indices] = ~valid_flag
    _, accepted_num = accepted.view(batch_size, -1).min(-1)
    accepted_num = accepted_num.to(torch.int32)
    offset = torch.arange((max_spec_len + 1), device=device, dtype=torch.int32)
    output_token_ids = torch.where(offset[None, :] <= accepted_num[:, None],
        all_sampled_tokens.view(batch_size, -1), -1)
    return output_token_ids


def create_logit_sampling_metadata(
    sampling_metadata: SamplingMetadata,
    spec_metadata: SpecDecodeMetadata,
):
    if any(bias is not None and bias for bias in sampling_metadata.logit_bias):
        raise ValueError(f"sampling_metadata.logit_bias is {sampling_metadata.logit_bias},"
            f"IS_SPEC_SAMPLE_GREEDY_COMPARE don't support logit_bias")
    if any(bad_words for bad_words in sampling_metadata.bad_words_token_ids.values()):
        raise ValueError(f"sampling_metadata.bad_words_token_ids is {sampling_metadata.bad_words_token_ids},"
            f"IS_SPEC_SAMPLE_GREEDY_COMPARE don't support bad_words_token_ids")
    if any(min_token for min_token in sampling_metadata.min_tokens.values()):
        raise ValueError(f"sampling_metadata.min_tokens is {sampling_metadata.min_tokens},"
            f"IS_SPEC_SAMPLE_GREEDY_COMPARE don't support min_tokens")
    if len(sampling_metadata.generators) > 0:
        raise ValueError(f"sampling_metadata.generators is {sampling_metadata.generators},"
            f"IS_SPEC_SAMPLE_GREEDY_COMPARE don't support generators")
    num_total_tokens = sum([i + 1 for i in spec_metadata.num_draft_tokens])
    num_sample_tokens = spec_metadata.cu_num_draft_tokens.clone()
    num_sample_tokens[1:] -= spec_metadata.cu_num_draft_tokens[:-1]
    num_sample_tokens += 1
    repeat_parameters = {
        'repeats': num_sample_tokens,
        'dim': 0,
        'output_size': num_total_tokens,
    }
    temperature = sampling_metadata.temperature.repeat_interleave(**repeat_parameters)
    top_p = None if sampling_metadata.top_p is None else sampling_metadata.top_p.repeat_interleave(**repeat_parameters)
    top_k = None if sampling_metadata.top_k is None else sampling_metadata.top_k.repeat_interleave(**repeat_parameters)
    min_p = None if sampling_metadata.min_p is None else sampling_metadata.min_p.repeat_interleave(**repeat_parameters)
    top_n_sigma = None if sampling_metadata.top_n_sigma is None \
        else sampling_metadata.top_n_sigma.repeat_interleave(**repeat_parameters)
    allowed_token_ids_mask = None if sampling_metadata.allowed_token_ids_mask is None \
        else sampling_metadata.allowed_token_ids_mask.repeat_interleave(**repeat_parameters)

    prompt_token_ids = None if sampling_metadata.prompt_token_ids is None \
        else sampling_metadata.prompt_token_ids.repeat_interleave(**repeat_parameters)
    presence_penalties = None if sampling_metadata.presence_penalties is None \
        else sampling_metadata.presence_penalties.repeat_interleave(**repeat_parameters)
    frequency_penalties = None if sampling_metadata.frequency_penalties is None \
        else sampling_metadata.frequency_penalties.repeat_interleave(**repeat_parameters)
    repetition_penalties = None if sampling_metadata.repetition_penalties is None \
        else sampling_metadata.repetition_penalties.repeat_interleave(**repeat_parameters)

    return SamplingMetadata(
        temperature=temperature,
        all_greedy=sampling_metadata.all_greedy,
        all_random=sampling_metadata.all_random,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        generators=sampling_metadata.generators,
        max_num_logprobs=sampling_metadata.max_num_logprobs,

        prompt_token_ids=prompt_token_ids,
        frequency_penalties=frequency_penalties,
        presence_penalties=presence_penalties,
        repetition_penalties=repetition_penalties,

        output_token_ids=sampling_metadata.output_token_ids,
        min_tokens=sampling_metadata.min_tokens,
        no_penalties=sampling_metadata.no_penalties,
        logit_bias=sampling_metadata.logit_bias,
        allowed_token_ids_mask=allowed_token_ids_mask,
        bad_words_token_ids=sampling_metadata.bad_words_token_ids,

        top_n_sigma=top_n_sigma,
        no_top_n_sigma=sampling_metadata.no_top_n_sigma,
        seq_data=sampling_metadata.seq_data,
    )


def expand_draft_to_max(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    max_spec_len: int,
):
    if (max_spec_len == torch.tensor(num_draft_tokens)).all():
        num_total_tokens = sum([i + 1 for i in num_draft_tokens])
        num_sample_tokens = cu_num_draft_tokens.clone()
        num_sample_tokens[1:] -= cu_num_draft_tokens[:-1]
        num_sample_tokens += 1
        repeat_parameters = {
            'repeats': num_sample_tokens,
            'dim': 0,
            'output_size': num_total_tokens,
        }
        return draft_token_ids.repeat_interleave(**repeat_parameters)
    else:
        raise ValueError(f"max_spec_len is not torch.tensor(num_draft_tokens) ")

