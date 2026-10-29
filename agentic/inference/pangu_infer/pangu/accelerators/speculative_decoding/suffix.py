# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import VllmConfig, SpeculativeConfig
import numpy as np


class SuffixDecodingProposer:
    """
    The class is adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/suffix_decoding.py.
    Speculative decoding proposer for Suffix Decoding (https://arxiv.org/pdf/2411.04975).
    This class imports and uses the official implementation from Arctic Inference
    (https://github.com/snowflakedb/ArcticInference).
    """

    def __init__(self, speculative_config: SpeculativeConfig):
        self.num_speculative_tokens = speculative_config.num_speculative_tokens
        self.max_tree_depth = speculative_config.max_tree_depth
        self.max_spec_factor = speculative_config.max_spec_factor
        self.min_token_prob = speculative_config.min_token_prob
        self.max_model_len = speculative_config.max_model_len
        self.max_cached_requests = speculative_config.max_cached_requests

        # Lazy import to avoid error when Suffix Decoding is not used.
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        # Initialize and empty cache. This object will take care of caching request
        # outputs, evicting old requests, and manages the per-prompt suffix trees.
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=self.max_tree_depth,
            max_cached_requests=self.max_cached_requests,
        )

    def propose(
        self,
        req_id,
        old_ids: list[int],
        sampled_ids: list[int],
        k: int
    ) -> list[list[int]]:
        """
        Propose speculative tokens for each request in the input batch. Suffix Decoding
        will speculate a dynamic number of tokens for each request every decoding step,
        so each entry in the returned list may have different lengths.
        """

        if req_id not in self.suffix_cache.active_requests:
            print("req_id not in self.suffix_cache.active_requests")
            if req_id in self.suffix_cache.cached_requests:
                # Reset the suffix cache for this request.
                self.suffix_cache.evict_cached_response(req_id)
            # Start a new request, this will build the suffix tree for that prompt.
            self.suffix_cache.start_request(req_id, old_ids)

        # Append the newly sampled ids to the suffix cache for this request.
        self.suffix_cache.add_active_response(req_id, sampled_ids)

        # Suffix decoding only uses the most recent tokens up to max_tree_depth, so
        # we extract the pattern from the end of the input.
        token_ids_cpu = np.concatenate((old_ids, sampled_ids))
        num_tokens = len(token_ids_cpu)
        start = max(0, num_tokens - self.max_tree_depth)
        pattern = token_ids_cpu[start:num_tokens]
        draft = self.suffix_cache.speculate(
            req_id,
            pattern,
            max_spec_tokens=self.num_speculative_tokens,
            max_spec_factor=self.max_spec_factor,
            min_token_prob=self.min_token_prob,
        )
        return draft.token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass