from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any, Optional, Union

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from pangu_infer.pangu.accelerators.schedule_policy.request_queue import create_request_queue
from pangu_infer.pangu.accelerators.schedule_policy.scheduler import PrefillDecodeScheduler
from pangu_infer.pangu.accelerators.schedule_policy.schedule_policies import (SchedulingPolicy, 
                                                                              _record_loras, 
                                                                              _allocate_kv_slots,
                                                                              _handle_remote_kv_waiting,
                                                                              _handle_fsm_waiting,
                                                                              _check_lora_constraint,
                                                                              _calculate_computed_tokens,
                                                                              _calculate_new_tokens,
                                                                              _remove_from_waiting,
                                                                              _validate_scheduling_constraints
                                                                              )
from pangu_infer.pangu.adaptor_manager.adaptor_manager import register, get_config
from pangu_infer.pangu.utils.utils import check
from pangu_infer.pangu.accelerators.schedule_policy.agent_scheduler import (
    _log_scheduler_metrics,
    _start_tracking_request,
    _stop_tracking_request,
)


logger = init_logger(__name__)


@register("vllm.v1.core.sched.scheduler.Scheduler.__init__")
def __init__(
    self,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    structured_output_manager: StructuredOutputManager,
    mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    include_finished_set: bool = False,
    log_stats: bool = False,
) -> None:
    self.vllm_config = vllm_config
    self.scheduler_config = vllm_config.scheduler_config
    self.cache_config = vllm_config.cache_config
    self.lora_config = vllm_config.lora_config
    self.kv_cache_config = kv_cache_config
    self.kv_events_config = vllm_config.kv_events_config
    self.log_stats = log_stats
    self.structured_output_manager = structured_output_manager

    # include_finished_set controls whether a separate set of finished
    # request ids should be included in the EngineCoreOutputs returned
    # by update_from_outputs(). This is currently used in the multi-engine
    # case to track request lifetimes efficiently.
    self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
        defaultdict(set) if include_finished_set else None)

    # Scheduling constraints.
    self.max_num_running_reqs = self.scheduler_config.max_num_seqs
    self.max_num_scheduled_tokens = \
        self.scheduler_config.max_num_batched_tokens
    self.max_model_len = self.scheduler_config.max_model_len
    self.enable_kv_cache_events = (
        self.kv_events_config is not None
        and self.kv_events_config.enable_kv_cache_events)

    # Create KVConnector for the Scheduler. Note that each Worker
    # will have a corresponding KVConnector with Role=WORKER.
    # KV Connector pushes/pull of remote KVs for P/D and offloading.
    self.connector = None
    if self.vllm_config.kv_transfer_config is not None:
        if len(self.kv_cache_config.kv_cache_groups) != 1:
            logger.error(
                f"Multiple KV cache groups: {len(self.kv_cache_config.kv_cache_groups)}" 
                + "are not currently supported with KV connectors")
        self.connector = KVConnectorFactory.create_connector_v1(
            config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

    self.kv_event_publisher = EventPublisherFactory.create(
        self.kv_events_config,
        vllm_config.parallel_config.data_parallel_rank,
    )

    num_gpu_blocks = self.cache_config.num_gpu_blocks
    if num_gpu_blocks is None or num_gpu_blocks <= 0:
        logger.error(f"Number of GPU blocks error: {num_gpu_blocks=}")

    self.block_size = self.cache_config.block_size

    # Patched for scheduler acceleration
    # Read from config and set hyperparameters
    self.pangu_config = get_config()
    self.additional_scheduling_policy = self.pangu_config.additional_config.get(
        "agent_scheduler", {}
        ).get("scheduling_policy", None)
    self.additional_policy_enabled = self.additional_scheduling_policy is not None
    print(f"!! scheduler init {self.additional_scheduling_policy=}", flush=True)
    long_prefill_token_threshold = self.pangu_config.additional_config.get(
        "agent_scheduler", {}
        ).get("long_prefill_token_threshold", None)
    if long_prefill_token_threshold is not None:
        self.scheduler_config.long_prefill_token_threshold = long_prefill_token_threshold
        print(f"!! scheduler long_prefill_token_threshold {long_prefill_token_threshold=}", flush=True)
    self.long_prefill_token_threshold_enabled = long_prefill_token_threshold is not None

    # req_id -> Request
    self.requests: dict[str, Request] = {}

    # Patched for scheduler acceleration
    if self.additional_policy_enabled:
        # Scheduling policy
        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.additional_scheduling_policy == "sjf":
            self.policy = SchedulingPolicy.SJF
        elif self.additional_scheduling_policy == "agent":
            self.policy = SchedulingPolicy.AGENT
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}")
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
    else:
        # Priority queues for requests.
        self.waiting: deque[Request] = deque()

    self.running: list[Request] = []

    # The request IDs that are finished in between the previous and the
    # current steps. This is used to notify the workers about the finished
    # requests so that they can free the cached states for those requests.
    # This is flushed at the end of each scheduling step.
    self.finished_req_ids: set[str] = set()

    # KV Connector: requests in process of async KV loading or recving
    self.finished_recving_kv_req_ids: set[str] = set()

    # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
    # them at each scheduling step.
    # Request id -> deque of CachedRequestData
    self._cached_reqs_data: dict[str, deque[CachedRequestData]] = defaultdict(deque)

    # Encoder-related.
    # Calculate encoder cache size if applicable
    # NOTE: For now we use the same budget for both compute and space.
    # This can be changed when we make encoder cache for embedding caching
    # across requests.
    encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
        model_config=vllm_config.model_config,
        scheduler_config=vllm_config.scheduler_config,
        mm_registry=mm_registry,
    )

    # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
    # projector if needed). Currently, we assume that the encoder also
    # has the Transformer architecture (e.g., ViT).
    self.max_num_encoder_input_tokens = encoder_compute_budget
    # NOTE: For the models without encoder (e.g., text-only models),
    # the encoder cache will not be initialized because cache size is 0
    # for these models.
    self.encoder_cache_manager = EncoderCacheManager(
        cache_size=encoder_cache_size)

    speculative_config = vllm_config.speculative_config

    self.use_eagle = False
    self.num_spec_tokens = self.num_lookahead_tokens = 0
    if speculative_config:
        self.num_spec_tokens = speculative_config.num_speculative_tokens
        if speculative_config.use_eagle():
            self.use_eagle = True
            self.num_lookahead_tokens = self.num_spec_tokens

    # Create the KV cache manager.
    self.kv_cache_manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=self.max_model_len,
        enable_caching=self.cache_config.enable_prefix_caching,
        caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
        use_eagle=self.use_eagle,
        log_stats=self.log_stats,
        enable_kv_cache_events=self.enable_kv_cache_events,
    )

    # Patched for scheduler acceleration: peak split
    self.peak_split = self.pangu_config.additional_config.get("agent_scheduler", {}).get("peak_split", False)
    if self.peak_split:
        self.peak_split_factor = self.pangu_config.additional_config.agent_scheduler.get("peak_split_factor", 0.5)
        self.chunked_prefill_enabled = self.scheduler_config.chunked_prefill_enabled
    logger.info(f"peak_split: {self.peak_split}")

    try:
        if self.policy == SchedulingPolicy.AGENT:
            # Add kv cache manager to the request queue
            self.waiting.set_kv_cache_manager(self.kv_cache_manager)
    except Exception as e:
        logger.error(f"Failed to add kv cache manager to the request queue: {e}")

    # Scheduler metrics logging configuration
    self._scheduler_metrics_log_file = self.pangu_config.additional_config.get(
        "agent_scheduler", {}
    ).get("scheduler_metrics_log_file", "scheduler_metrics.jsonl")
    self._metrics_log_enabled = self.pangu_config.additional_config.get(
        "agent_scheduler", {}
    ).get("metrics_log_enabled", True)

    # Make sure scheduler metrics log file path is valid
    import os
    log_dir = os.path.dirname(self._scheduler_metrics_log_file)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"Created log directory: {log_dir}")
        except Exception as e:
            logger.error(f"Failed to create log directory {log_dir}: {e}")
            self._metrics_log_enabled = False
    logger.info(f"Initialized scheduler metrics logging to {self._scheduler_metrics_log_file}")

    # Initialize wait time tracking for AGENT scheduling
    self._wait_time_tracking = {
        'request_start_times': {},  # request_id -> arrival_time (monotonic)
        'request_finish_times': {},  # request_id -> finish_time (monotonic)
    }

    # Get the threshold for long requests (if defined)
    self.long_request_threshold = self.pangu_config.additional_config.get(
        "agent_scheduler", {}
    ).get("long_request_threshold", 1000)  # Default: 1000 tokens


@register("vllm.v1.core.sched.scheduler.Scheduler.schedule")
def schedule(self) -> SchedulerOutput:
    self.agent_schedule_enabled = False
    pangu_config = get_config()
    logger.debug(f"{pangu_config=}")
    self.prefill_schedule_enabled = pangu_config.additional_config.get(
        "agent_scheduler", {}).get("prefill_schedule", {}).get("enabled", False)
    if self.prefill_schedule_enabled:
        if not hasattr(self, 'pd_scheduler'):
            self.pd_scheduler = PrefillDecodeScheduler()

        skip_schedule = self.pd_scheduler.should_skip_scheduling(self.waiting, self.running)
        if skip_schedule:
            # Return an empty SchedulerOutput
            return SchedulerOutput(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=[],
                num_scheduled_tokens={},
                total_num_scheduled_tokens=0,
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                num_common_prefix_blocks=[0] * len(self.kv_cache_config.kv_cache_groups),
                finished_req_ids=self.finished_req_ids,
                free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
                structured_output_request_ids={},
                grammar_bitmask=None,
            )

    # NOTE(woosuk) on the scheduling algorithm:
    # There's no "decoding phase" nor "prefill phase" in the scheduler.
    # Each request just has the num_computed_tokens and num_tokens_with_spec.num_tokens_with_spec =
    # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
    # At each step, the scheduler tries to assign tokens to the requests
    # so that each request's num_computed_tokens can catch up its
    # num_tokens_with_spec. This is general enough to cover
    # chunked prefills, prefix caching, speculative decoding,
    # and the "jump decoding" optimization in the future.

    running_count = len(self.running)
    waiting_count = len(self.waiting)
    if hasattr(self, '_metrics_log_enabled') and self._metrics_log_enabled:
        _log_scheduler_metrics(self, running_count, waiting_count)

    try:
        if hasattr(self.waiting, "reset_plan") and callable(getattr(self.waiting, "reset_plan")):
            self.waiting.reset_plan()
    except Exception as e:
        logger.error(f"Failed to reset plan of waiting queue: {e}")

    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []

    # structured_output_request_ids maps
    # a request's (request that uses structured output)
    # request_id to the running request index.
    # This will helps us determine to slice the grammar bitmask
    # and only applies valid mask for requests that
    # uses structured decoding.
    structured_output_request_ids: dict[str, int] = {}

    req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens
    # Encoder-related.
    scheduled_encoder_inputs: dict[str, list[int]] = {}
    encoder_budget = self.max_num_encoder_input_tokens
    # Spec decode-related.
    scheduled_spec_decode_tokens: dict[str, list[int]] = {}

    # Patched for scheduler acceleration: peak split
    if self.peak_split:
        if len(self.running) < self.peak_split_factor * self.max_num_running_reqs:
            self.chunked_prefill_enabled = False
        else:
            self.chunked_prefill_enabled = True
        logger.debug(
            f"peak_split is {self.peak_split} and chunked_prefill_enabled is {self.chunked_prefill_enabled}")

    # For logging.
    scheduled_timestamp = time.monotonic()

    # First, schedule the RUNNING requests.
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]
        num_new_tokens = (request.num_tokens_with_spec -
                            request.num_computed_tokens)
        if self.long_prefill_token_threshold_enabled:
            # Patched for scheduler acceleration: long prefill optimize
            # When the remaining tokens are not too many
            # Chunk splitting is ignored to speed up processing
            if (0 < 2 * self.scheduler_config.long_prefill_token_threshold <=
                num_new_tokens and self.chunked_prefill_enabled):
                logger.debug(
                    f"using self.chunked_prefill_enabled is {self.chunked_prefill_enabled}" 
                    + " and current num_new_tokens is {num_new_tokens}"
                )
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
        else:
            # Original vLLM implementation
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
        num_new_tokens = min(num_new_tokens, token_budget)

        # Make sure the input position does not exceed the max model len.
        # This is necessary when using spec decoding.
        num_new_tokens = min(
            num_new_tokens,
            self.max_model_len - request.num_computed_tokens)

        # Schedule encoder inputs.
        encoder_inputs_to_schedule = None
        new_encoder_budget = encoder_budget
        if request.has_encoder_inputs:
            (encoder_inputs_to_schedule, num_new_tokens,
                new_encoder_budget) = self._try_schedule_encoder_inputs(
                    request, request.num_computed_tokens, num_new_tokens,
                    encoder_budget)

        if num_new_tokens == 0:
            # The request cannot be scheduled because one of the following
            # reasons:
            # 1. No new tokens to schedule. This may happen when PP>1 and
            #    we have already scheduled all prompt tokens but they are
            #    not finished yet.
            # 2. The encoder budget is exhausted.
            # 3. The encoder cache is exhausted.
            # Here, by doing `continue` instead of `break`,
            # we do not strictly follow the FCFS scheduling policy and
            # allow the lower-priority requests to be scheduled.
            req_index += 1
            continue

        num_draft_tokens = max(
            num_new_tokens + request.num_computed_tokens -
            request.num_tokens, 0)

        # Patched inside for scheduler acceleration
        # allocate KV slots
        can_schedule, new_blocks, preempted_reqs = _allocate_kv_slots(
            self,
            request, 
            num_new_tokens,
            num_draft_tokens,
            scheduled_timestamp,
            preempted_reqs
        )

        if not can_schedule:
            break
        if new_blocks is None:
            logger.error(f"New blocks error: {new_blocks=}")

        # Schedule the request.
        scheduled_running_reqs.append(request)
        if request.use_structured_output:
            # PERF: in case of chunked prefill,
            # request might not include any new tokens.
            # Therefore, we might introduce some additional
            # cycle to fill in the bitmask, which could be a big no-op.
            structured_output_request_ids[request.request_id] = req_index
        req_to_new_block_ids[request.request_id] = (
            new_blocks.get_block_ids())
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
        req_index += 1

        # Speculative decode related.
        if request.spec_token_ids:
            num_scheduled_spec_tokens = (num_new_tokens +
                                            request.num_computed_tokens -
                                            request.num_tokens)
            if num_scheduled_spec_tokens > 0:
                # Trim spec_token_ids list to num_scheduled_spec_tokens.
                del request.spec_token_ids[num_scheduled_spec_tokens:]
                scheduled_spec_decode_tokens[request.request_id] = (
                    request.spec_token_ids)

        # Encoder-related.
        if encoder_inputs_to_schedule:
            scheduled_encoder_inputs[request.request_id] = (
                encoder_inputs_to_schedule)
            # Allocate the encoder cache.
            for i in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, i)
            encoder_budget = new_encoder_budget

    # Record the LoRAs in scheduled_running_reqs
    scheduled_loras = _record_loras(scheduled_running_reqs, self.lora_config)
    
    if self.additional_policy_enabled:
        # Patched for scheduler acceleration
        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)
    else:
        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        skipped_waiting_requests: deque[Request] = deque()

    # Next, schedule the WAITING requests.
    if not preempted_reqs:
        while self.waiting and token_budget > 0:
            if len(self.running) == self.max_num_running_reqs:
                break
            
            if self.additional_policy_enabled:
                # Patched for scheduler acceleration
                request = self.waiting.peek_request()
            else:
                request = self.waiting[0]

            if not request:
                break

            # Patched inside for scheduler acceleration
            # KVTransfer: skip request if still waiting for remote kvs.
            if _handle_remote_kv_waiting(self, request, skipped_waiting_requests):
                continue

            # Patched inside for scheduler acceleration
            # Skip request if the structured output request is still waiting
            # for FSM compilation.
            if _handle_fsm_waiting(self, request, skipped_waiting_requests):
                continue

            # Patched inside for scheduler acceleration
            # Check that adding the request still respects the max_loras
            # constraint.
            if _check_lora_constraint(self, request, scheduled_loras, skipped_waiting_requests):
                continue

            # Patched inside for scheduler acceleration
            # Calculate cached tokens
            (new_computed_blocks, num_new_local_computed_tokens,
             num_external_computed_tokens, load_kv_async, 
             num_computed_tokens) = _calculate_computed_tokens(self, request)

            # Calculate new token count and encoder inputs
            (num_new_tokens, encoder_inputs_to_schedule,
             new_encoder_budget, skip) = _calculate_new_tokens(
                self, request, num_computed_tokens, num_external_computed_tokens, 
                load_kv_async, token_budget, encoder_budget
            )
            if skip:
                break
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens + num_external_computed_tokens,
                num_new_local_computed_tokens,
                new_computed_blocks,
                num_lookahead_tokens=self.num_lookahead_tokens,
                delay_cache_blocks=load_kv_async,
            )
            if not new_blocks:
                # The request cannot be scheduled.
                break

            # KVTransfer: the connector uses this info to determine
            # if a load is needed. Note that
            # This information is used to determine if a load is
            # needed for this request.
            if self.connector is not None:
                self.connector.update_state_after_alloc(
                    request,
                    new_computed_blocks + new_blocks,
                    num_external_computed_tokens,
                )

            # Patched inside for scheduler acceleration
            skip, structured_output_request_ids, req_index = _remove_from_waiting(
                self, request, load_kv_async, skipped_waiting_requests,
                structured_output_request_ids, req_index
            )
            if skip:
                continue
            
            self.running.append(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED,
                                        scheduled_timestamp)
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(
                    f"Invalid request status: {request.status}")

            if self.lora_config and request.lora_request:
                scheduled_loras.add(request.lora_request.lora_int_id)
            req_to_new_block_ids[request.request_id] = (
                self.kv_cache_manager.get_block_ids(request.request_id))
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens
            # Count the number of prefix cached tokens.
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = num_computed_tokens
            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

    # Put back any skipped requests at the head of the waiting queue
    if skipped_waiting_requests:
        if self.additional_policy_enabled:
            # Patched for scheduler acceleration
            self.waiting.prepend_requests(skipped_waiting_requests)
        else:
            # Original vLLM implementation
            self.waiting.extendleft(skipped_waiting_requests)

    # Check if the scheduling constraints are satisfied.
    _, total_num_scheduled_tokens = _validate_scheduling_constraints(
        self, 
        num_scheduled_tokens, 
        token_budget, 
        scheduled_new_reqs,
        scheduled_resumed_reqs,
        scheduled_running_reqs
    )

    # Get the longest common prefix among all requests in the running queue.
    # This can be potentially used for cascade attention.
    num_common_prefix_blocks = [0] * len(
        self.kv_cache_config.kv_cache_groups)
    if self.running:
        any_request = self.running[0]
        num_common_prefix_blocks = (
            self.kv_cache_manager.get_num_common_prefix_blocks(
                any_request, len(self.running)))

    grammar_bitmask = self.structured_output_manager.grammar_bitmask(
        self.requests,
        structured_output_request_ids,
        scheduled_spec_decode_tokens,
    )
    # Construct the scheduler output.
    new_reqs_data = [
        NewRequestData.from_request(req,
                                    req_to_new_block_ids[req.request_id])
        for req in scheduled_new_reqs
    ]
    resumed_reqs_data = [
        self._make_cached_request_data(
            req,
            num_scheduled_tokens[req.request_id],
            len(scheduled_spec_decode_tokens.get(req.request_id, ())),
            req_to_new_block_ids[req.request_id],
            resumed_from_preemption=True,
        ) for req in scheduled_resumed_reqs
    ]
    running_reqs_data = [
        self._make_cached_request_data(
            req,
            num_scheduled_tokens[req.request_id],
            len(scheduled_spec_decode_tokens.get(req.request_id, ())),
            req_to_new_block_ids[req.request_id],
            resumed_from_preemption=False,
        ) for req in scheduled_running_reqs
    ]
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=new_reqs_data,
        scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        scheduled_encoder_inputs=scheduled_encoder_inputs,
        num_common_prefix_blocks=num_common_prefix_blocks,
        # finished_req_ids is an existing state in the scheduler,
        # instead of being newly scheduled in this step.
        # It contains the request IDs that are finished in between
        # the previous and the current steps.
        finished_req_ids=self.finished_req_ids,
        free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
        structured_output_request_ids=structured_output_request_ids,
        grammar_bitmask=grammar_bitmask
    )

    # NOTE(Kuntai): this function is designed for multiple purposes:
    # 1. Plan the KV cache store
    # 2. Wrap up all the KV cache load / save ops into an opaque object
    # 3. Clear the internal states of the connector
    if self.connector is not None:
        meta = self.connector.build_connector_meta(scheduler_output)
        scheduler_output.kv_connector_metadata = meta

    events = self.kv_cache_manager.take_events()
    if events:
        batch = KVEventBatch(ts=time.time(), events=events)
        self.kv_event_publisher.publish(batch)

    # Advance the number of computed tokens for the request AFTER
    # the request is scheduled.
    # 1. The scheduler_output of the current step has to include the
    #    original number of scheduled tokens to determine input IDs.
    # 2. Advance the number of computed tokens here allowing us to
    #    schedule the prefill request again immediately in the next
    #    scheduling step.
    # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
    #    computed tokens will be adjusted in update_from_output.
    for req_id, num_scheduled_token in num_scheduled_tokens.items():
        self.requests[req_id].num_computed_tokens += num_scheduled_token

    self.finished_req_ids = set()
    return scheduler_output


@register("vllm.v1.core.sched.scheduler.Scheduler.add_request")
def add_request(self, request: Request) -> None:
    if self.additional_policy_enabled:
        # Patched for scheduler acceleration
        self.waiting.add_request(request)
    else:
        self.waiting.append(request)
    self.requests[request.request_id] = request
    _start_tracking_request(self, request)
    if self.log_stats:
        request.record_event(EngineCoreEventType.QUEUED)


@register("vllm.v1.core.sched.scheduler.Scheduler.finish_requests")
def finish_requests(
    self,
    request_ids: Union[str, Iterable[str]],
    finished_status: RequestStatus,
) -> None:
    """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
    """
    if not RequestStatus.is_finished(finished_status):
        logger.error(f"Request status is not finished.")
    if isinstance(request_ids, str):
        request_ids = (request_ids, )
    else:
        request_ids = set(request_ids)

    if self.additional_policy_enabled:
        # Patched for scheduler acceleration
        running_requests_to_remove = []
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.append(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        for request in running_requests_to_remove:
            self.running.remove(request)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            _stop_tracking_request(self, request)
            self._free_request(request)
    else:
        # Original vLLM implementation
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
            else:
                self.waiting.remove(request)
            request.status = finished_status
            self._free_request(request)


@register("vllm.v1.core.sched.scheduler.Scheduler._try_schedule_encoder_inputs")
def _try_schedule_encoder_inputs(
    self,
    request: Request,
    num_computed_tokens: int,
    num_new_tokens: int,
    encoder_budget: int,
) -> tuple[list[int], int, int]:
    """
    Determine which encoder inputs need to be scheduled in the current step,
    and update `num_new_tokens` and encoder token budget accordingly.

    An encoder input will be scheduled if:
    - Its output tokens overlap with the range of tokens being computed
    in this step, i.e.,
    [num_computed_tokens, num_computed_tokens + num_new_tokens).
    - It is not already computed and stored in the encoder cache.
    - There is sufficient encoder token budget to process it.
    - The encoder cache has space to store it.

    If an encoder input cannot be scheduled due to cache or budget
    limitations, the method adjusts `num_new_tokens` to schedule only the
    decoder tokens up to just before the unschedulable encoder input.

    Note that num_computed_tokens includes both locally cached
    blocks and externally cached blocks (via KVConnector).
    """
    if num_new_tokens == 0 or not request.has_encoder_inputs:
        return [], num_new_tokens, encoder_budget
    encoder_inputs_to_schedule: list[int] = []
    mm_positions = request.mm_positions
    check(mm_positions is not None)
    check(len(mm_positions) > 0)
    pre_start_pos = -1
    for i, pos_info in enumerate(mm_positions):
        start_pos = pos_info.offset
        if start_pos == pre_start_pos:
            encoder_inputs_to_schedule.append(i)
            continue
        num_encoder_tokens = pos_info.length

        # The encoder output is needed if the two ranges overlap:
        # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
        # [start_pos, start_pos + num_encoder_tokens)
        if start_pos >= num_computed_tokens + num_new_tokens:
            # The encoder input is not needed in this step.
            break
        if start_pos + num_encoder_tokens <= num_computed_tokens:
            # The encoder input is already computed and stored
            # in the decoder's KV cache.
            continue

        if self.encoder_cache_manager.has_cache(request, i):
            # The encoder input is already computed and cached.
            continue

        # If no encoder input chunking is allowed, we do not want to
        # partially schedule a multimodal item. If the scheduled range would
        # only cover part of the mm input, roll back to before the mm item.
        if (self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)):
            num_new_tokens = start_pos - num_computed_tokens
            break

        if (not self.encoder_cache_manager.can_allocate(request, i)
                or num_encoder_tokens > encoder_budget):
            # The encoder cache is full or the encoder budget is exhausted.
            # NOTE(woosuk): We assume that the encoder input tokens should
            # be processed altogether, as the encoder usually uses
            # bidirectional attention.
            if num_computed_tokens < start_pos:
                # We only schedule the decoder tokens just before the
                # encoder input.
                num_new_tokens = start_pos - num_computed_tokens
            else:
                # Because of prefix caching, num_computed_tokens is greater
                # than start_pos even though its encoder input is not
                # available. In this case, we can't schedule any token for
                # the request in this step.
                num_new_tokens = 0
            break

        encoder_budget -= num_encoder_tokens
        encoder_inputs_to_schedule.append(i)
        pre_start_pos = start_pos
    return encoder_inputs_to_schedule, num_new_tokens, encoder_budget


@register("vllm.v1.core.sched.scheduler.Scheduler.update_from_output")
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
    sampled_token_ids = model_runner_output.sampled_token_ids
    spec_token_ids = model_runner_output.spec_token_ids
    logprobs = model_runner_output.logprobs
    prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens

    new_running: list[Request] = []
    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
    spec_decoding_stats: Optional[SpecDecodingStats] = None

    # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
    # loop can be a performance bottleneck. We should do our best to avoid
    # expensive operations inside the loop.
    for request in self.running:
        req_id = request.request_id
        num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
        if num_tokens_scheduled == 0:
            # The request was not scheduled in this step.
            new_running.append(request)
            continue

        req_index = model_runner_output.req_id_to_index[req_id]
        generated_token_ids = sampled_token_ids[req_index]

        scheduled_spec_token_ids = (
            scheduler_output.scheduled_spec_decode_tokens.get(req_id))
        if scheduled_spec_token_ids:
            # num_computed_tokens represents the number of tokens
            # processed in the current step, considering scheduled
            # tokens and rejections. If some tokens are rejected,
            # num_computed_tokens is decreased by the number of rejected
            # tokens, where is given by:
            # len(scheduled_spec_token_ids) + 1 - len(generated_token_ids).
            num_tokens_rejected = (len(scheduled_spec_token_ids) + 1 -
                                    len(generated_token_ids))
            request.num_computed_tokens -= num_tokens_rejected
            spec_decoding_stats = self.make_spec_decoding_stats(
                spec_decoding_stats,
                num_draft_tokens=len(scheduled_spec_token_ids),
                num_accepted_tokens=len(generated_token_ids) - 1)

        cached_encoder_input_ids = (
            self.encoder_cache_manager.get_cached_input_ids(request))
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if cached_encoder_input_ids:
            for input_id in list(cached_encoder_input_ids):
                mm_positions = request.mm_positions[input_id]
                start_pos = mm_positions.offset
                num_tokens = mm_positions.length
                if start_pos + num_tokens <= request.num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    self.encoder_cache_manager.free_encoder_input(
                        request, input_id)

        stopped = False
        new_logprobs = None
        new_token_ids = generated_token_ids
        kv_transfer_params = None

        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                kv_transfer_params = self._free_request(request)
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break

        # Extract sample logprobs if needed.
        if request.sampling_params.logprobs is not None and logprobs:
            # NOTE: once we support N tokens per step (spec decode),
            # the outer lists can be of length > 1.
            if isinstance(logprobs, list):
                new_logprobs = logprobs[req_index]
            else:
                new_logprobs = logprobs.slice(req_index, req_index + 1)

        if new_token_ids and self.structured_output_manager.should_advance(
                request):
            # NOTE: structured output request
            # should not be None if use_structured_output, we have
            # check above, so safe to ignore type warning
            request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                req_id, new_token_ids)

        # Add newly generated spec token ids to the request.
        if spec_token_ids is not None:
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                # Needs to happen after new_token_ids are accepted.
                request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                    spec_token_ids[req_index])
            else:
                request.spec_token_ids = spec_token_ids[req_index]

        # Get prompt logprobs for this request.
        prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
        if new_token_ids or kv_transfer_params:

            # Add EngineCoreOutput for this Request.
            outputs[request.client_index].append(
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=new_token_ids,
                    finish_reason=request.get_finished_reason(),
                    new_logprobs=new_logprobs,
                    new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                    stop_reason=request.stop_reason,
                    events=request.take_events(),
                    kv_transfer_params=kv_transfer_params,
                    num_cached_tokens=request.num_cached_tokens,
                ))

        else:
            # Invariant: EngineCore returns no partial prefill outputs.
            pass

        if not stopped:
            new_running.append(request)

    # KV Connector: update state for finished KV Transfers.
    self._update_from_kv_xfer_finished(model_runner_output)

    # Return the cached request data to the queue so they can be reused.
    for req_data in scheduler_output.scheduled_cached_reqs:
        # NOTE(rob): since we free stopped reqs above, adding stopped reqs
        # to _cached_reqs_data will cause a memory leak.
        if req_data.req_id not in self.finished_req_ids:
            self._cached_reqs_data[req_data.req_id].append(req_data)

    self.running = new_running

    # Create EngineCoreOutputs for all clients that have requests with
    # outputs in this step.
    engine_core_outputs = {
        client_index: EngineCoreOutputs(outputs=outs)
        for client_index, outs in outputs.items()
    }

    finished_req_ids = self.finished_req_ids_dict
    if finished_req_ids:
        # Include ids of requests that finished since last outputs
        # were sent.
        for client_index, finished_set in finished_req_ids.items():
            # Set finished request set in EngineCoreOutputs for this client.
            if (eco := engine_core_outputs.get(client_index)) is not None:
                eco.finished_requests = finished_set
            else:
                engine_core_outputs[client_index] = EngineCoreOutputs(
                    finished_requests=finished_set)
        finished_req_ids.clear()

    if engine_core_outputs:
        # Return stats to only one of the front-ends.
        next(iter(engine_core_outputs.values())).scheduler_stats = (
            self.make_stats(spec_decoding_stats))

    return engine_core_outputs
