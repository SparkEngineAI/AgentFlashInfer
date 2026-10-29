import torch
import os
from typing import Optional

from vllm.v1.sample.metadata import SamplingMetadata
from pangu_infer.pangu.adaptor_manager.adaptor_manager import register, get_config
from pangu_infer.pangu.utils.utils import check
import os
IS_SPEC_SAMPLE_GREEDY_COMPARE = int(os.environ.get('SPEC_SAMPLE_GREEDY_COMPARE', 0))


_mtp_logp_context = None


def set_mtp_logp_context(mtp_logp_context):
    global _mtp_logp_context
    _mtp_logp_context = mtp_logp_context


def get_mtp_logp_context():
    global _mtp_logp_context
    return _mtp_logp_context


@register("vllm_ascend.sample.rejection_sampler.expand_pytorch")
def expand_pytorch(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS,
):
    # no loop optimization version
    device = cu_num_tokens_ptr.device
    batch_size = input_ptr.shape[0]
    num_tokens = output_ptr.shape[0]

    if batch_size == 0 or num_tokens == 0:
        return

    cu_start = torch.cat([torch.tensor([0], pin_memory=True).to(device, non_blocking=True), cu_num_tokens_ptr[:-1]])
    cu_end = cu_num_tokens_ptr

    token_indices = torch.arange(num_tokens, device=device)[:, None]  # [num_tokens, 1]
    cu_start_exp = cu_start[None, :]  # [1, batch_size]
    cu_end_exp = cu_end[None, :]  # [1, batch_size]

    in_range = (token_indices >= cu_start_exp) & (token_indices < cu_end_exp)

    replaced_input = torch.where(input_ptr == replace_from, replace_to, input_ptr).float()

    token_values = torch.einsum("tb,b->t", in_range.float(), replaced_input)

    needs_update = in_range.any(dim=1)

    output_ptr[:] = torch.where(needs_update, token_values, output_ptr)


@register("vllm_ascend.sample.rejection_sampler.sample_recovered_tokens")
def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]

    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()

    num_draft_tensor = torch.tensor(num_draft_tokens, pin_memory=True).to(device, non_blocking=True)
    has_draft_mask = num_draft_tensor > 0

    for i, generator in sampling_metadata.generators.items():
        temp_q = torch.empty_like(q[i])
        temp_q.exponential_(generator=generator)
        q[i] = torch.where(has_draft_mask[i], temp_q, q[i])

    recovered_token_ids = torch.empty_like(draft_token_ids)
    sample_recovered_tokens_pytorch(
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        IS_NGRAM=draft_probs is None,
    )
    return recovered_token_ids


def sample_recovered_tokens_pytorch(
    output_token_ids,  # [num_tokens]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    q,  # [batch_size, vocab_size]
    vocab_size,
    IS_NGRAM=False,
):
    device = output_token_ids.device
    num_tokens = output_token_ids.shape[0]

    if num_tokens == 0:
        return

    cu_start = torch.cat(
        [
            torch.tensor([0], pin_memory=True).to(device, non_blocking=True),
            cu_num_draft_tokens[:-1],
        ]
    )
    cu_end = cu_num_draft_tokens

    token_indices = torch.arange(num_tokens, device=device)  # [num_tokens]

    token_indices_expanded = token_indices[:, None]  # [num_tokens, 1]
    cu_start_expanded = cu_start[None, :]  # [1, batch_size]
    cu_end_expanded = cu_end[None, :]  # [1, batch_size]

    in_range_mask = (token_indices_expanded >= cu_start_expanded) & (token_indices_expanded < cu_end_expanded)

    token_to_batch = torch.argmax(in_range_mask.int(), dim=1)

    has_match = in_range_mask.any(dim=1)
    token_to_batch = torch.where(has_match, token_to_batch, 0)

    if IS_NGRAM:
        token_indices = torch.arange(num_tokens, device=device)

        modified_target_probs = target_probs.clone()
        modified_target_probs[token_indices, draft_token_ids] = 0
        prob = modified_target_probs

    else:
        prob = torch.maximum(
            target_probs - draft_probs,
            torch.tensor(0.0, pin_memory=True).to(device, non_blocking=True),
        )

    q_values = q[token_to_batch]  # [num_tokens, vocab_size]

    epsilon = 1e-10
    q_values_safe = torch.where(q_values == 0, epsilon, q_values)
    q_values_safe = torch.where(torch.isinf(q_values), epsilon, q_values_safe)

    prob_over_q = prob / q_values_safe

    prob_over_q = torch.where((q_values == 0) | torch.isinf(q_values), -1e10, prob_over_q)

    recovered_ids = torch.argmax(prob_over_q, dim=1)

    output_token_ids[:] = recovered_ids


@register("vllm_ascend.sample.rejection_sampler.rejection_random_sample_pytorch")
def rejection_random_sample_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    bonus_token_ids,  # [batch_size]
    recovered_token_ids,  # [num_tokens]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    max_spec_len,
    vocab_size,
    IS_NGRAM=False,
):
    logp_ctx = get_mtp_logp_context()
    if logp_ctx is not None:
        bonus_logits_indices = logp_ctx["bonus_logits_indices"]
        target_logits_indices = logp_ctx["target_logits_indices"] 
        output_indices = torch.zeros_like(output_token_ids)
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device
    
    zero_cpu = torch.tensor([0], pin_memory=True)
    zero_device = zero_cpu.to(device, non_blocking=True)
    
    cu_start = torch.cat([zero_device, cu_num_draft_tokens[:-1]])
    cu_end = cu_num_draft_tokens
    num_draft_per_batch = cu_end - cu_start
    
    max_draft_len = max_spec_len
    pos_indices_cpu = torch.arange(max_draft_len, pin_memory=True)
    pos_indices = pos_indices_cpu.to(device, non_blocking=True)[None, :]
    
    valid_mask = pos_indices < num_draft_per_batch[:, None]
    global_token_indices = cu_start[:, None] + pos_indices
    global_token_indices = global_token_indices.clamp(0, draft_token_ids.shape[0]-1)
    draft_tokens = draft_token_ids[global_token_indices]  # [batch_size, max_draft_len]
    
    if IS_NGRAM:
        ones_cpu = torch.ones(1, pin_memory=True, dtype=torch.float32)
        draft_token_probs = ones_cpu.to(device, non_blocking=True).expand_as(draft_tokens)
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten()
        flat_draft_probs = draft_probs[flat_indices, flat_draft_tokens]
        draft_token_probs = flat_draft_probs.view(batch_size, max_draft_len)
    
    flat_indices = global_token_indices.flatten()
    flat_draft_tokens = draft_tokens.flatten()
    flat_target_probs = target_probs[flat_indices, flat_draft_tokens]
    target_token_probs = flat_target_probs.view(batch_size, max_draft_len)
    
    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]
    
    zero_threshold_cpu = torch.tensor([0.0], pin_memory=True, dtype=torch.float32)
    zero_threshold = zero_threshold_cpu.to(device, non_blocking=True)
    
    acceptance_condition = (draft_token_probs > zero_threshold) & (
        target_token_probs / draft_token_probs >= uniform_token_probs
    )
    
    first_rejection = (~acceptance_condition) & valid_mask
    
    default_pos_cpu = torch.full([batch_size, 1], max_draft_len, pin_memory=True)
    default_pos = default_pos_cpu.to(device, non_blocking=True)
    
    first_reject_pos = torch.where(
        first_rejection.any(dim=1, keepdim=True),
        first_rejection.float().argmax(dim=1, keepdim=True),
        default_pos
    )
    pos_mask = pos_indices >= first_reject_pos
    should_skip = pos_mask & valid_mask
    
    final_acceptance = acceptance_condition & (~should_skip)
    non_greedy_mask = ~is_greedy
    update_mask = non_greedy_mask[:, None] & valid_mask & (~should_skip)
    
    first_reject_mask = (pos_indices == first_reject_pos) & valid_mask & non_greedy_mask[:, None]
    final_update_mask = update_mask | first_reject_mask
    final_tokens = torch.where(
        first_reject_mask, 
        recovered_tokens,
        torch.where(final_acceptance, draft_tokens, output_token_ids[:, :max_draft_len])
    )
    
    output_token_ids[:, :max_draft_len] = torch.where(
        final_update_mask, 
        final_tokens, 
        output_token_ids[:, :max_draft_len]
    )
    
    if logp_ctx is not None:
        target_inds = target_logits_indices[global_token_indices]
        output_indices[:, :max_draft_len] = torch.where(
            final_update_mask, 
            target_inds, 
            output_indices[:, :max_draft_len]
        )
    
    no_rejection = first_reject_pos.squeeze(1) >= num_draft_per_batch
    should_add_bonus = non_greedy_mask & no_rejection
    
    bonus_positions = num_draft_per_batch  # [batch_size]
    
    seq_len = output_token_ids.shape[1]
    all_positions_cpu = torch.arange(seq_len, pin_memory=True)
    all_positions = all_positions_cpu.to(device, non_blocking=True)[None, :]  # [1, seq_len]
    
    batch_bonus_positions = bonus_positions[:, None]  # [batch_size, 1]
    
    max_spec_len_cpu = torch.tensor([max_spec_len], pin_memory=True)
    max_spec_len_device = max_spec_len_cpu.to(device, non_blocking=True)
    
    valid_bonus_pos = bonus_positions < (max_spec_len_device + 1)
    final_bonus_mask = should_add_bonus & valid_bonus_pos
    
    bonus_pos_match = (all_positions == batch_bonus_positions)
    bonus_pos_mask = bonus_pos_match & final_bonus_mask[:, None]
    
    bonus_values_expanded = bonus_token_ids.view(-1, 1).expand(-1, seq_len)
    output_token_ids[:] = torch.where(bonus_pos_mask, bonus_values_expanded, output_token_ids)

    if logp_ctx is not None:
        bonus_inds_expanded = bonus_logits_indices.view(-1, 1).expand(-1, seq_len)
        output_indices[:] = torch.where(bonus_pos_mask, bonus_inds_expanded, output_indices)
        set_mtp_logp_context(dict(output_indices=output_indices))



@register("vllm_ascend.sample.rejection_sampler.rejection_greedy_sample_spec_len_1_pytorch")
def rejection_greedy_sample_spec_len_1_pytorch(
        output_token_ids,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    check(batch_size == num_tokens)
    if IS_SPEC_SAMPLE_GREEDY_COMPARE == 1:
        target_argmax = target_argmax.squeeze(1)
    accept_req_mask = draft_token_ids == target_argmax
    output_token_ids[:, 0] = target_argmax
    bonus_token_ids = bonus_token_ids.squeeze(1)
    output_token_ids[:, 1] = torch.where(
        accept_req_mask,
        bonus_token_ids,
        output_token_ids[:, 1]
    )
    logp_ctx = get_mtp_logp_context()
    if logp_ctx is not None:
        bonus_logits_indices = logp_ctx["bonus_logits_indices"]
        target_logits_indices = logp_ctx["target_logits_indices"] 
        output_indices = torch.zeros_like(output_token_ids)
        output_indices[:, 0] = target_logits_indices
        output_indices[accept_req_mask, 1] = torch.where(
            accept_req_mask,
            bonus_logits_indices,
            output_indices[:, 1]
        )
        set_mtp_logp_context(dict(output_indices=output_indices))



@register("vllm_ascend.sample.rejection_sampler.rejection_greedy_sample_pytorch")
def rejection_greedy_sample_pytorch(
        output_token_ids,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens,  # [batch_size]
        draft_token_ids,  # [num_tokens]
        target_argmax,  # [num_tokens]
        bonus_token_ids,  # [batch_size]
        draft_tokens_per_req,  # [batch_size], list
        max_spec_len,
        is_greedy=None,  # [batch_size] or None
):
    logp_ctx = get_mtp_logp_context()
    if logp_ctx is not None:
        bonus_logits_indices = logp_ctx["bonus_logits_indices"]
        target_logits_indices = logp_ctx["target_logits_indices"] 
        output_indices = torch.zeros_like(output_token_ids)
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    device = output_token_ids.device
    draft_tokens_per_req = torch.tensor(draft_tokens_per_req).to(
        device, non_blocking=True)
    if is_greedy is None:
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)

    start_indices = cu_num_draft_tokens - draft_tokens_per_req
    req_ids = torch.arange(batch_size, device=device)
    token_req_ids = torch.repeat_interleave(req_ids, draft_tokens_per_req)
    token_positions = torch.arange(
        num_tokens, device=device) - start_indices[token_req_ids]

    # Find the first mismatch position of each request.
    mismatch_global = (draft_token_ids != target_argmax)
    if max_spec_len == 0:
        first_mismatch_pos_per_req = torch.zeros(batch_size,
                                                 dtype=torch.long,
                                                 device=device)
    else:
        pos_matrix = torch.full((batch_size, max_spec_len),
                                -1,
                                dtype=torch.long,
                                device=device)
        pos_matrix[token_req_ids, token_positions] = token_positions
        mismatch_matrix = torch.full((batch_size, max_spec_len),
                                     False,
                                     dtype=torch.bool,
                                     device=device)
        mismatch_matrix[token_req_ids, token_positions] = mismatch_global
        mismatch_positions = torch.where(mismatch_matrix, pos_matrix,
                                         max_spec_len * 2)
        first_mismatch_pos_per_req, _ = torch.min(mismatch_positions, dim=1)
        no_mismatch_mask = (first_mismatch_pos_per_req == max_spec_len * 2)
        first_mismatch_pos_per_req[no_mismatch_mask] = draft_tokens_per_req[
            no_mismatch_mask]

    # Copy matched target tokens into output.
    copy_len = torch.minimum(first_mismatch_pos_per_req + 1,
                             draft_tokens_per_req)
    copy_indices = torch.arange(max_spec_len + 1,
                                device=device).expand(batch_size, -1)
    copy_mask = copy_indices < copy_len.unsqueeze(1)
    greedy_mask = is_greedy.unsqueeze(1)
    final_copy_mask = copy_mask & greedy_mask
    global_idx = start_indices.unsqueeze(1) + copy_indices
    output_token_ids[final_copy_mask] = target_argmax[
        global_idx[final_copy_mask]]
    if logp_ctx is not None:
        output_indices[final_copy_mask] = target_logits_indices[global_idx[final_copy_mask]]

    # Fill bonus token.
    needs_bonus = is_greedy & (first_mismatch_pos_per_req >=
                               draft_tokens_per_req)
    if torch.any(needs_bonus):
        bonus_rows = torch.where(needs_bonus)[0]
        bonus_cols = draft_tokens_per_req[bonus_rows]
        bonus_token_ids = bonus_token_ids.squeeze(1)
        output_token_ids[bonus_rows, bonus_cols] = bonus_token_ids[bonus_rows]
        if logp_ctx is not None:
            output_indices[bonus_rows, bonus_cols] = bonus_logits_indices[bonus_rows]
    if logp_ctx is not None:
        set_mtp_logp_context(dict(output_indices=output_indices))