from enum import Enum
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import RequestStatus


logger = init_logger(__name__)


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""
    FCFS = "fcfs"
    PRIORITY = "priority"
    SJF = "sjf"
    AGENT = "agent"  # AGENT scheduling policy


def _record_loras(scheduled_running_reqs, lora_config) -> set:
    scheduled_loras: set[int] = set()
    if lora_config:
        scheduled_loras = set(
            req.lora_request.lora_int_id for req in scheduled_running_reqs
            if req.lora_request and req.lora_request.lora_int_id > 0)
        if len(scheduled_loras) > lora_config.max_loras:
            logger.error(
                f"Length of scheduled loras: {len(scheduled_loras)}" 
                + f" is greater than the max length of loras: {lora_config.max_loras}"
            )
    
    return scheduled_loras


def _allocate_kv_slots(
        self,
        request, 
        num_new_tokens,
        num_draft_tokens,
        scheduled_timestamp,
        preempted_reqs,
        ):
    while True:
        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            num_draft_tokens=num_draft_tokens,
            num_lookahead_tokens=self.num_lookahead_tokens)
        if new_blocks is None:
            # The request cannot be scheduled.
            # Preempt the lowest-priority request.
            if self.additional_policy_enabled:
                # Patched for scheduler acceleration
                if self.policy == SchedulingPolicy.PRIORITY:
                    preempted_req = max(
                        self.running,
                        key=lambda r: (r.priority, r.arrival_time),
                    )
                    self.running.remove(preempted_req)
                else:
                    preempted_req = self.running.pop()
            else:
                # Original vLLM implementation
                preempted_req = self.running.pop()
            self.kv_cache_manager.free(preempted_req)
            preempted_req.status = RequestStatus.PREEMPTED
            preempted_req.num_computed_tokens = 0
            if self.log_stats:
                preempted_req.record_event(
                    EngineCoreEventType.PREEMPTED, scheduled_timestamp)

            if self.additional_policy_enabled:
                # Patched for scheduler acceleration
                self.waiting.prepend_request(preempted_req)
            else:
                # Original vLLM implementation
                self.waiting.appendleft(preempted_req)
            preempted_reqs.append(preempted_req)
            if preempted_req == request:
                # No more request to preempt.
                can_schedule = False
                break
        else:
            # The request can be scheduled.
            can_schedule = True
            break
    
    return can_schedule, new_blocks, preempted_reqs


def _handle_remote_kv_waiting(self, request, skipped_waiting_requests):
    if request.status != RequestStatus.WAITING_FOR_REMOTE_KVS:
        return False

    is_ready = self._update_waiting_for_remote_kv(request)
    if is_ready:
        request.status = RequestStatus.WAITING
        return False
    else:
        logger.debug(f"{request.request_id} is still in WAITING_FOR_REMOTE_KVS state.")
        if self.additional_policy_enabled:
            # Patched for scheduler acceleration
            self.waiting.pop_request()
            skipped_waiting_requests.prepend_request(request)
        else:
            # Original vLLM implementation
            self.waiting.popleft()
            skipped_waiting_requests.appendleft(request)
        return True


def _handle_fsm_waiting(self, request, skipped_waiting_requests):
    if request.status != RequestStatus.WAITING_FOR_FSM:
        return False

    structured_output_req = request.structured_output_request
    if structured_output_req and structured_output_req.grammar:
        request.status = RequestStatus.WAITING
        return False
    else:
        if self.additional_policy_enabled:
            # Patched for scheduler acceleration
            self.waiting.pop_request()
            skipped_waiting_requests.prepend_request(request)
        else:
            # Original vLLM implementation
            self.waiting.popleft()
            skipped_waiting_requests.appendleft(request)
        return True


def _check_lora_constraint(self, request, scheduled_loras, skipped_waiting_requests):
    if not (self.lora_config and request.lora_request):
        return False

    lora_id = request.lora_request.lora_int_id
    if len(scheduled_loras) == self.lora_config.max_loras and lora_id not in scheduled_loras:
        # Scheduling would exceed max_loras, skip.
        if self.additional_policy_enabled:
            # Patched for scheduler acceleration
            self.waiting.pop_request()
            skipped_waiting_requests.prepend_request(request)
        else:
            # Original vLLM implementation
            self.waiting.popleft()
            skipped_waiting_requests.appendleft(request)
        return True
    return False


def _calculate_computed_tokens(self, request):
    """compute cached tokens"""
    num_external_computed_tokens = 0
    load_kv_async = False

    # Get already-cached tokens.
    if request.num_computed_tokens == 0:
        # Get locally-cached tokens.
        new_computed_blocks, num_new_local_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)
        
        # Get externally-cached tokens if using a KVConnector.
        if self.connector is not None:
            num_external_computed_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
                request, num_new_local_computed_tokens
            )
        
        # Total computed tokens (local + external).
        num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
    # KVTransfer: WAITING reqs have num_computed_tokens > 0
    # after async KV recvs are completed.
    else:
        new_computed_blocks = self.kv_cache_manager.create_empty_block_list()
        num_new_local_computed_tokens = 0
        num_computed_tokens = request.num_computed_tokens

    return (new_computed_blocks, num_new_local_computed_tokens,
            num_external_computed_tokens, load_kv_async, num_computed_tokens)


def _calculate_new_tokens(self, request, num_computed_tokens, num_external_computed_tokens, 
                          load_kv_async, token_budget, encoder_budget):
    # KVTransfer: loading remote KV, do not allocate for new work.  
    encoder_inputs_to_schedule = None
    new_encoder_budget = encoder_budget
    skip = False

    # KVTransfer: loading remote KV, do not allocate for new work.
    if load_kv_async:
        if num_external_computed_tokens <= 0:
            logger.error(f"Number of external computed tokens {num_external_computed_tokens} \
                should be > 0 when loading KV async")
        num_new_tokens = 0
    # Number of tokens to be scheduled.
    else:
        # We use `request.num_tokens` instead of
        # `request.num_prompt_tokens` to consider the resumed
        # requests, which have output tokens.
        num_new_tokens = request.num_tokens - num_computed_tokens
        if self.long_prefill_token_threshold_enabled:
            # Patched for scheduler acceleration: long prefill optimize
            if (0 < 2 * self.scheduler_config.long_prefill_token_threshold
                <= num_new_tokens and self.chunked_prefill_enabled):
                logger.debug(
                    f"using self.chunked_prefill_enabled is {self.chunked_prefill_enabled}" 
                    + " and current num_new_tokens is {num_new_tokens}"
                )
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        else:
            # Original vLLM implementation
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold

        num_new_tokens = min(num_new_tokens, token_budget)
        if num_new_tokens <= 0:
            logger.error(f"Invalid new tokens count: {num_new_tokens}, it should be > 0")

        # Schedule encoder inputs.
        if request.has_encoder_inputs:
            encoder_inputs_to_schedule, num_new_tokens, new_encoder_budget = self._try_schedule_encoder_inputs(
                request, num_computed_tokens, num_new_tokens, encoder_budget
            )
            if num_new_tokens == 0:
                # The request cannot be scheduled.
                skip = True

    return num_new_tokens, encoder_inputs_to_schedule, new_encoder_budget, skip


def _remove_from_waiting(self, request, load_kv_async, skipped_waiting_requests,
                         structured_output_request_ids, req_index):
    # pop from waiting queue
    if self.additional_policy_enabled:
        # Patched for scheduler acceleration
        # Request was already popped from self.waiting
        # unless it was re-added above due to new_blocks being None.
        request = self.waiting.pop_request()
    else:
        # Original vLLM implementation
        self.waiting.popleft()

    if load_kv_async:
        # If loading async, allocate memory and put request
        # into the WAITING_FOR_REMOTE_KV state.
        skipped_waiting_requests.prepend_request(request) \
            if self.additional_policy_enabled else skipped_waiting_requests.appendleft(request)
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        return True, structured_output_request_ids, req_index

    if request.use_structured_output:
        structured_output_request_ids[request.request_id] = req_index

    return False, structured_output_request_ids, req_index + 1


def _validate_scheduling_constraints(
        self, 
        num_scheduled_tokens, 
        token_budget, 
        scheduled_new_reqs,
        scheduled_resumed_reqs,
        scheduled_running_reqs
        ) -> bool:
    valid = True
    # Check if the scheduling constraints are satisfied.
    total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
    if total_num_scheduled_tokens > self.max_num_scheduled_tokens:
        logger.error(
            f"Total number of scheduled tokens: {total_num_scheduled_tokens}" 
            + f" is greater than the maximum: {self.max_num_scheduled_tokens}."
        )
        valid = False
    if token_budget < 0:
        logger.error(f"Tokens budget: {token_budget} is not enough.")
        valid = False
    if len(self.running) > self.max_num_running_reqs:
        logger.error(
            f"NUmber of running requests: {len(self.running)}" 
            + f"is greater than the maximum: {self.max_num_running_reqs}."
        )
        valid = False
    # Since some requests in the RUNNING queue may not be scheduled in
    # this step, the total number of scheduled requests can be smaller than
    # len(self.running).
    if len(scheduled_new_reqs) \
        + len(scheduled_resumed_reqs) \
        + len(scheduled_running_reqs) > len(self.running):
        logger.error(
            f"Number of scheduled new requests: {len(scheduled_new_reqs)} +" 
            + f"resumed requests: {len(scheduled_resumed_reqs)} + " 
            + f"running requests: {len(scheduled_running_reqs)}" 
            + f"is greater than length of running queue: {len(self.running)}."
        )
        valid = False
    
    return valid, total_num_scheduled_tokens
