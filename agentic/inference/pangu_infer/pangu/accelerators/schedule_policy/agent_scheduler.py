import math
import time
from collections import deque
from typing import Iterable, Iterator, Optional, List

from sortedcontainers import SortedList
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import Request
from .request_queue import RequestQueue

logger = init_logger(__name__)


def _start_tracking_request(self, request: Request) -> None:
    """Start tracking wait time for a new request."""
    request_id = request.request_id
    current_time = time.monotonic()
    
    # Record arrival time
    setattr(request, '_arrival_time', current_time)
    self._wait_time_tracking['request_start_times'][request_id] = current_time
    
    logger.debug(f"Started tracking request {request_id}")


def _stop_tracking_request(self, request: Request) -> float:
    """Stop tracking and return wait time for a completed request.
    
    Returns:
        float: Wait time in seconds, or 0 if not tracked
    """
    request_id = request.request_id
    start_time = self._wait_time_tracking['request_start_times'].pop(request_id, None)
    
    if start_time is None:
        return 0.0
    
    stop_time = time.monotonic()
    wait_time = stop_time - start_time
    
    # Store the finish time
    self._wait_time_tracking['request_finish_times'][request_id] = stop_time
    
    logger.debug(f"Request {request_id} completed with wait time: {wait_time:.2f}s")
    return wait_time


def _log_scheduler_metrics(self, running_count: int, waiting_count: int) -> None:
    """Log scheduler metrics including running and waiting queue information."""
    try:
        timestamp = time.time()
        
        # Get lambda value (if using AgentMixRequestQueue)
        lambda_val = 0.0
        mode = "Unknown"
        if hasattr(self.waiting, '_lambda'):
            lambda_val = getattr(self.waiting, '_lambda', 0.0)
            mode = "KV-priority" if lambda_val > 0.5 else "SJF-priority"
        
        # Get GPU memory information
        free_blocks = 0
        total_blocks = 0
        try:
            free_blocks = self.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
            total_blocks = int(
                getattr(
                    self.kv_cache_manager.block_pool, 
                    "num_gpu_blocks", 
                    len(self.kv_cache_manager.block_pool.blocks)
                    )
                )
        except Exception as e:
            logger.warn(f"Error getting GPU block info: {e}")
        
        # Collect running requests details with timing information
        running_request_details = []
        try:
            for request in self.running:
                prompt_length = len(getattr(request, "prompt_token_ids", []))
                
                # Get timing information for running requests
                arrival_time = getattr(request, '_arrival_time', timestamp)
                start_time = getattr(request, '_start_time', timestamp)
                wait_time = timestamp - arrival_time
                running_time = timestamp - start_time
                
                request_info = {
                    "request_id": str(request.request_id),
                    "length": int(prompt_length),
                    "status": "running",
                    "arrival_time": arrival_time,  # When request entered waiting queue
                    "start_time": start_time,      # When request started running
                    "wait_time": wait_time,  # Total wait time
                    "running_time": running_time   # Time spent running
                }
                running_request_details.append(request_info)
        except Exception as e:
            logger.debug(f"Error collecting running request details: {e}")
        
        # Collect waiting requests details with timing information
        waiting_request_details = []
        try:
            for request in self.waiting:
                prompt_length = len(getattr(request, "prompt_token_ids", []))
                
                # Get timing information for waiting requests
                arrival_time = getattr(request, '_arrival_time', timestamp)
                wait_time = timestamp - arrival_time
                
                request_info = {
                    "request_id": str(request.request_id),
                    "length": int(prompt_length),
                    "status": "waiting",
                    "arrival_time": arrival_time,  # When request entered waiting queue
                    "wait_time": wait_time  # Current wait time
                }
                waiting_request_details.append(request_info)
        except Exception as e:
            logger.debug(f"Error collecting waiting request details: {e}")
        
        # Create a complete JSON object for the log entry
        import json
        try:
            log_dict = {
                'timestamp': timestamp,
                'lambda': lambda_val,
                'free_blocks': free_blocks,
                'total_blocks': total_blocks,
                'running_count': running_count,
                'waiting_count': waiting_count,
                'running_details': running_request_details,
                'waiting_details': waiting_request_details,
                'mode': mode
            }
            
            # Convert the entire dictionary to JSON
            log_line = json.dumps(log_dict) + '\n'
            
            logger.debug(f"JSON log entry created: running={len(running_request_details)}, \
                waiting={len(waiting_request_details)}")
        except Exception as e:
            logger.warn(f"Error creating JSON log entry: {e}")
            # Fallback: create a minimal JSON object
            log_dict = {
                'timestamp': timestamp,
                'lambda': lambda_val,
                'free_blocks': free_blocks,
                'total_blocks': total_blocks,
                'running_count': running_count,
                'waiting_count': waiting_count,
                'running_details': [],
                'waiting_details': [],
                'mode': mode
            }
            log_line = json.dumps(log_dict) + '\n'
        logger.debug(f"{log_line=}")
        # Write to log file
        log_file = getattr(self, '_scheduler_metrics_log_file', 'scheduler_metrics.jsonl')
        try:
            with open(log_file, 'a') as f:
                f.write(log_line)
        except Exception as e:
            logger.warn(f"Error writing to metrics file: {e}")
            
    except Exception as e:
        logger.warn(f"Error in _log_scheduler_metrics: {e}")


class _SortedValueDict:
    """Maintain a descending-sorted (value, key) view for quick max retrieval."""

    def __init__(self) -> None:
        # Sort by (-value, key)
        self._sorted = SortedList(key=lambda x: (-x[0], x[1]))
        self._key2entry: dict[str, tuple[int, str]] = {}

    def push(self, key: str, value: int) -> None:
        """Insert or update (key, value) pair."""
        if key in self._key2entry:
            self._remove_key(key)
        entry = (value, key)
        self._sorted.add(entry)
        self._key2entry[key] = entry

    def delete(self, key: str) -> None:
        """Remove entry by key."""
        if key in self._key2entry:
            self._remove_key(key)

    def _remove_key(self, key: str) -> None:
        entry = self._key2entry[key]
        self._sorted.remove(entry)
        del self._key2entry[key]

    def get_max_key(self) -> Optional[str]:
        """Return the key with the highest value."""
        if self._sorted:
            _, key = self._sorted[0]
            return key
        return None

    def clear(self) -> None:
        """Clear all entries."""
        self._sorted.clear()
        self._key2entry.clear()

    @property
    def keys(self) -> set[str]:
        return set(self._key2entry.keys())

    def __len__(self) -> int:
        return len(self._key2entry)


class AgentRequestQueue(deque[Request], RequestQueue):
    """
    A vLLM-style request queue that prioritizes requests with more reusable
    KV cache blocks once eviction has occurred.

    Before the first KV eviction, behaves like a FIFO queue.
    After eviction, requests are re-scored and re-sorted periodically
    based on their cached token counts.

    Args:
        kv_cache_manager: Instance of KVCacheManager used to query cached blocks.
        update_interval_sec: Interval (in seconds) for full score refresh.
    """

    def __init__(
        self,
        kv_cache_manager: KVCacheManager = None,
        update_interval_sec: float = 2.0,
    ) -> None:
        deque.__init__(self)
        self._kvc: KVCacheManager = kv_cache_manager
        self._update_interval: float = float(update_interval_sec)

        # State flags
        self._eviction_happened: bool = True
        self._last_update_ts: float = 0.0

        # Cached scores
        self._score_view = _SortedValueDict()

    # --------------------------------------------------------------------------
    # Public RequestQueue interface
    # --------------------------------------------------------------------------

    def set_kv_cache_manager(self, kv_cache_manager: KVCacheManager):
        self._kvc: KVCacheManager = kv_cache_manager

    def add_request(self, request: Request) -> None:
        """Add a request to the queue and apply scheduling policy.

        Args:
            request: The Request object to add.
        """
        print(f"Adding request: {request=}", flush=True)
        self.append(request)
        self._maybe_mark_eviction()
        self._rescore_and_resort_if_needed(incremental_add=[request])

    def pop_request(self) -> Request:
        """Pop the next scheduled request.

        Returns:
            The next Request object according to scheduling policy.
        """
        self._maybe_mark_eviction()
        self._rescore_and_resort_if_needed()
        res = self.popleft()
        print(f"Popping request: {res=}", flush=True)
        return res

    def peek_request(self) -> Request:
        """Peek at the next scheduled request without removing it.

        Returns:
            The next Request object at the head of the queue.
        """
        if not self:
            raise IndexError("peek from an empty queue")
        self._maybe_mark_eviction()
        self._rescore_and_resort_if_needed()
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a single request to the front of the queue.

        Args:
            request: The Request object to prepend.
        """
        self.appendleft(request)
        self._maybe_mark_eviction()
        self._rescore_and_resort_if_needed(incremental_add=[request])

    def prepend_requests(self, requests: "AgentRequestQueue") -> None:
        """Prepend multiple requests at once.

        Args:
            requests: Another AgentRequestQueue instance.
        """
        self.extendleft(reversed(requests))
        self._maybe_mark_eviction()
        self._rescore_and_resort_if_needed(incremental_add=list(requests))

    def remove_request(self, request: Request) -> None:
        """Remove a specific request.

        Args:
            request: The Request object to remove.
        """
        super().remove(request)
        self._score_view.delete(request.request_id)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests.

        Args:
            requests: Iterable of Request objects to remove.
        """
        to_remove = set(r.request_id for r in requests)
        kept = [req for req in self if req.request_id not in to_remove]
        self.clear()
        self.extend(kept)
        for rid in to_remove:
            self._score_view.delete(rid)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        return super().__reversed__()

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    def _maybe_mark_eviction(self) -> None:
        """Detect the first KV eviction event."""
        if self._eviction_happened:
            return
        try:
            head = self._kvc.block_pool.free_block_queue.free_list_head
            if head is not None and getattr(head, "_block_hash", None) is not None:
                self._eviction_happened = True
        except Exception as e:
            # Defensive: structure differences won't crash here
            logger.error(f"Failed to detect the first KV eviction event: {e}") 

    def _now(self) -> float:
        """Return monotonic timestamp."""
        return time.monotonic()

    def _need_full_refresh(self) -> bool:
        """Check if a full rescore is required."""
        return (self._now() - self._last_update_ts) > self._update_interval

    def _get_num_cached_tokens(self, req: Request) -> int:
        """Get cached token (or block) count for a given request.

        Args:
            req: The Request object to query.

        Returns:
            int: Cached token count. Returns 0 on error.
        """
        try:
            _, num_cached_tokens = self._kvc.get_computed_blocks(req)
            return int(num_cached_tokens)
        except Exception:
            return 0

    def _refresh_full_scores(self) -> None:
        """Perform a full rescore and re-sort of all requests."""
        self._score_view.clear()
        for req in self:
            score = self._get_num_cached_tokens(req)
            self._score_view.push(req.request_id, score)

        id2score = {rid: val for val, rid in self._score_view._sorted}
        self._stable_sort_by_score_desc(id2score)
        self._last_update_ts = self._now()

    def _incremental_update_scores(
        self,
        incremental_add: Optional[List[Request]] = None,
    ) -> None:
        """Incrementally update cached scores.

        Args:
            incremental_add: Optional list of newly added requests.
        """
        curr_ids = {req.request_id for req in self}
        last_ids = self._score_view.keys

        # Remove vanished requests
        for rid in (last_ids - curr_ids):
            self._score_view.delete(rid)

        # Add missing or new requests
        if incremental_add:
            for req in incremental_add:
                score = self._get_num_cached_tokens(req)
                self._score_view.push(req.request_id, score)
        else:
            for req in self:
                if req.request_id not in self._score_view.keys:
                    score = self._get_num_cached_tokens(req)
                    self._score_view.push(req.request_id, score)

    def _rescore_and_resort_if_needed(
        self,
        incremental_add: Optional[List[Request]] = None,
    ) -> None:
        """Re-score and re-sort queue if policy requires.

        Args:
            incremental_add: Optional list of newly added requests.
        """
        if not self:
            return
        if not self._eviction_happened:
            return  # Keep FIFO until eviction occurs

        if self._need_full_refresh():
            self._refresh_full_scores()
            return

        self._incremental_update_scores(incremental_add=incremental_add)
        id2score = {rid: val for val, rid in self._score_view._sorted}
        self._stable_sort_by_score_desc(id2score)

    def _stable_sort_by_score_desc(self, id2score: dict[str, int]) -> None:
        """Stable sort by descending score and ascending request_id.

        Args:
            id2score: Mapping from request_id to score.
        """
        buf = list(self)
        buf.sort(key=lambda r: (-id2score.get(r.request_id, 0), r.request_id))
        self.clear()
        self.extend(buf)
        print(f"{list(self)=} {id2score=}", flush=True)



import math
import time
import threading
from collections import deque, defaultdict
from typing import Dict, List, Iterable, Iterator, Optional, Tuple

# Optional: FastAPI intake server for virtual requests.
try:
    from fastapi import FastAPI
    import uvicorn
    _FASTAPI_AVAILABLE = True
except Exception:
    _FASTAPI_AVAILABLE = False


class AgentMixRequestQueue(deque[Request], RequestQueue):
    """
    Mixed SJF × KV-aware queue with a shadow-price (lambda) controller,
    periodic hit/need refresh, and deferral when near-future high-hit
    virtual requests are expected.

    Args:
        kv_cache_manager: KVCacheManager instance.
        hit_refresh_interval_s: Seconds between full hit/need refreshes.
        lambda_step: Step size for lambda updates (small positive float).
        ema_beta: EMA factor for smoothing capacity gap (0..1).
        deferral_window_s: Look-ahead window (seconds) for virtuals in deferral.
        max_defer_times: Max deferrals per request to avoid starvation.
        soft_wait_s: Soft aging threshold (seconds) to slightly boost fairness.
        hard_wait_s: Hard aging threshold (seconds) to force admission.
        long_job_quantile: Quantile used to mark "long jobs" by need (e.g., 0.75).
        start_virtual_server: Whether to start a FastAPI server to receive virtuals.
        virtual_host: Host for the FastAPI intake server.
        virtual_port: Port for the FastAPI intake server.

    Returns:
        None
    """

    # ------------------------------ construction ------------------------------

    def __init__(self,
                 kv_cache_manager: KVCacheManager = None,
                 hit_refresh_interval_s: float = 2.0,
                 lambda_step: float = 1e-3,
                 ema_beta: float = 0.9,
                 deferral_window_s: float = 3.0,
                 max_defer_times: int = 2,
                 soft_wait_s: float = 5.0,
                 hard_wait_s: float = 20.0,
                 long_job_quantile: float = 0.75,
                 start_virtual_server: bool = False,
                 virtual_host: str = "0.0.0.0",
                 virtual_port: int = 18080,
                 metrics_log_file: str = "scheduler_metrics.log"
                 ) -> None:
        deque.__init__(self)
        self._debug: bool = False  # set True to enable verbose prints
        
        self._kvc: KVCacheManager = kv_cache_manager

        # Hit/need refresh policy
        self._hit_refresh_interval_s: float = float(hit_refresh_interval_s)
        self._last_hit_refresh_ts: float = 0.0  # monotonic clock

        # Shadow price lambda & EMA(gap)
        self._lambda: float = 0.0
        self._lambda_step: float = float(lambda_step)
        self._ema_beta: float = float(ema_beta)
        self._gap_ema: float = 0.0

        # Scoring weights (SJF × KV × aging)
        self._a: float = 0.01      # weight for hit
        self._b: float = 0.35      # base weight for need
        self._c: float = 0.002     # weight for wait (aging)

        # Deferral & fairness
        self._deferral_window_s: float = float(deferral_window_s)
        self._max_defer_times: int = int(max_defer_times)
        self._soft_wait_s: float = float(soft_wait_s)
        self._hard_wait_s: float = float(hard_wait_s)
        self._long_job_quantile: float = float(long_job_quantile)

        # Per-request runtime stats
        self._req_hit: Dict[str, int] = {}          # request_id -> hit blocks
        self._req_need: Dict[str, int] = {}         # request_id -> new blocks needed
        self._req_arrival: Dict[str, float] = {}    # request_id -> arrival ts (monotonic)
        self._req_defer_cnt: Dict[str, int] = defaultdict(int)  # request_id -> defers

        # Samples to approximate quantile for "long job"
        self._need_samples: List[int] = []

        # Virtual (fake) request pool: request_id -> (eta_ts_wallclock, prompt_token_ids)
        self._virtual_lock = threading.Lock()
        self._virtual_pool: Dict[str, Tuple[float, List[int]]] = {}

        # Try to fetch tokens_per_block from block_pool if available.
        tpb = getattr(getattr(self._kvc, "block_pool", None), "tokens_per_block", 128)
        if not isinstance(tpb, int) or tpb <= 0:
            tpb = 128
        self.block_size = tpb

        # --- Shadow-price mapping (soft) + PI trims (minimal intrusive) ---
        # Soft mapping
        self._lam_max: float = 2.0    # upper bound of lambda
        self._lam_k: float = 8.0      # slope around z=1
        self._lam_eps: float = 1.0    # avoid div-by-zero in z = D/(U+eps)

        # PI trims on gap: small gains for stability
        self._lam_kp: float = 5e-3    # proportional gain
        self._lam_ki: float = 1e-3    # integral gain
        self._lam_beta_int: float = 0.9  # EMA integrator (anti-noise)
        self._lam_int: float = 0.0    # integral state
        self._lam_max = 1.2     # ↓ upper bound
        self._lam_k = 3.0     # ↓ slope
        self._lam_eps = 4.0     # ↑ eps
        self._lam_kp = 2e-3    # ↓ PI P
        self._lam_ki = 5e-4    # ↓ PI I

        # Optional FastAPI server
        self._virtual_server_thread: Optional[threading.Thread] = None
        if start_virtual_server:
            if _FASTAPI_AVAILABLE:
                self._start_virtual_intake_server(virtual_host, virtual_port)
            else:
                self._dbg(lambda: "[AgentMixQueue] FastAPI not available; virtual intake disabled.")

        self._dbg(lambda: "[AgentMixQueue] Initialized. "
              f"hit_refresh_interval={self._hit_refresh_interval_s}s, "
              f"lambda_step={self._lambda_step}, ema_beta={self._ema_beta}, "
              f"deferral_window={self._deferral_window_s}s, max_defer={self._max_defer_times}")

        # ------------------------------ NEW: sticky-peek state ------------------------------
        self._rev: int = 0                                   # content revision
        self._planned: Optional[Tuple[str, Optional[str], int]] = None
        self._plan_made_ts: Optional[float] = None           # monotonic clock when plan made
        self._plan_ttl_s: float = 1.2                        # small TTL for sticky plans

    def _dbg(self, msg_fn) -> None:
        """Lazy-format logging: pass a zero-arg lambda that returns the string.
        Only evaluated when debug is enabled."""
        if self._debug:
            try:
                print(msg_fn(), flush=True)
            except Exception as e:
                logger.error(f"Failed while lazy-format logging: {e}")

    # ------------------------------ public API ------------------------------

    def set_kv_cache_manager(self, kv_cache_manager: KVCacheManager):
        self._kvc: KVCacheManager = kv_cache_manager

    def add_request(self, request: Request) -> None:
        """Add a real request into the queue.

        Args:
            request: A vLLM Request instance.

        Returns:
            None
        """
        self.append(request)
        self._register_request_meta(request)
        self._maybe_refresh_hits(full_or_incremental="incremental")
        self._dbg(lambda: f"[add] Added req={request.request_id} queue_len={len(self)}")
        self._bump_rev()  # NEW: content changed

    def pop_request(self) -> Request:
        """Schedule and pop the next request with detailed logging.

        Returns:
            The selected Request instance.
        """
        _t0 = time.perf_counter()  # unconditional timing start

        # ------------------------------ NEW (S3): honor sticky plan if valid & fresh ------------------------------
        if self._planned is not None:
            sel_id, orig_id, plan_rev = self._planned
            ttl_ok = (self._plan_made_ts is not None) and ((time.monotonic() - self._plan_made_ts) <= self._plan_ttl_s)
            if plan_rev == self._rev and ttl_ok:
                r_sel = self._find_request(sel_id)
                if r_sel is not None:
                    if orig_id and orig_id != sel_id:
                        r_orig = self._find_request(orig_id)
                        if r_orig is not None:
                            self._mark_deferred(r_orig)
                    self._dbg(lambda: f"[plan] Honoring planned pop id={sel_id}")
                    super().remove(r_sel)
                    self._cleanup_request_meta(r_sel)
                    self._planned = None
                    self._plan_made_ts = None
                    self._bump_rev()  # content changed
                    self._update_lambda()
                    # unconditional performance log
                    _t1 = time.perf_counter()
                    print(f"[perf] pop_request total={(_t1 - _t0)*1000:.3f}ms (honor plan)", flush=True)
                    return r_sel
            self._planned = None
            self._plan_made_ts = None
        # ----------------------------------------------------------------------------------------------------------

        self._dbg(lambda: "=" * 72)
        self._dbg(lambda: f"[schedule] Cycle @ {time.strftime('%H:%M:%S')} queue_len={len(self)}")

        # Print detailed metrics at scheduling cycle
        if self._debug:
            self._print_real_time_metrics()
        
        # Unified planning path (same as peek)
        sel, orig = self._plan_next()  # no side effects

        if sel is None:
            # unconditional performance log before raising
            _t1 = time.perf_counter()
            print(f"[perf] pop_request total={(_t1 - _t0)*1000:.3f}ms (empty)", flush=True)
            raise IndexError("No requests available to schedule.")

        if orig and orig != sel.request_id:
            r_orig = self._find_request(orig)
            if r_orig is not None:
                self._mark_deferred(r_orig)

        self._dbg(lambda: f"[schedule] Execute picked={sel.request_id}")

        super().remove(sel)
        self._cleanup_request_meta(sel)
        self._dbg(lambda: f"[execute] Dispatched req={sel.request_id} λ={self._lambda:.4f} gap_ema={self._gap_ema:.2f}")
        self._bump_rev()  # content changed

        self._update_lambda()

        # unconditional performance log at normal return
        _t1 = time.perf_counter()
        print(f"[perf] pop_request total={(_t1 - _t0)*1000:.3f}ms", flush=True)
        return sel

    def peek_request(self) -> Request:
        """Peek the next scheduled request without removing.

        Returns:
            The next Request instance.
        """
        _t0 = time.perf_counter()  # unconditional timing start

        sel, orig = self._plan_next()
        if sel is None:
            _t1 = time.perf_counter()
            print(f"[perf] peek_request total={(_t1 - _t0)*1000:.3f}ms (empty)", flush=True)
            raise IndexError("peek from an empty queue")

        self._dbg(lambda: f"[peek] Next req={sel.request_id} (no side effects)")
        self._planned = (sel.request_id, orig, self._rev)
        self._plan_made_ts = time.monotonic()

        _t1 = time.perf_counter()
        print(f"[perf] peek_request total={(_t1 - _t0)*1000:.3f}ms", flush=True)
        return sel

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front.

        Args:
            request: The Request to prepend.

        Returns:
            None
        """
        self.appendleft(request)
        self._register_request_meta(request)
        self._dbg(lambda: f"[prepend] Prepend req={request.request_id}")
        self._bump_rev()  # NEW: content changed

    def prepend_requests(self, requests: "AgentMixRequestQueue") -> None:
        """Prepend multiple requests in order.

        Args:
            requests: Another queue of requests.

        Returns:
            None
        """
        self.extendleft(reversed(requests))
        for r in requests:
            self._register_request_meta(r)
        self._dbg(lambda: f"[prepend_many] count={len(requests)} new_len={len(self)}")
        self._bump_rev()  # NEW: content changed

    def remove_request(self, request: Request) -> None:
        """Remove a specific request.

        Args:
            request: The Request to remove.

        Returns:
            None
        """
        super().remove(request)
        self._cleanup_request_meta(request)
        self._dbg(lambda: f"[remove] Removed req={request.request_id} new_len={len(self)}")
        self._bump_rev()  # NEW: content changed

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests.

        Args:
            requests: Requests to remove.

        Returns:
            None
        """
        to_remove = set(r.request_id for r in requests)
        kept = [r for r in self if r.request_id not in to_remove]
        self.clear()
        self.extend(kept)
        for rid in to_remove:
            self._cleanup_request_meta_by_id(rid)
        self._dbg(lambda: f"[remove_many] count={len(to_remove)} new_len={len(self)}")
        self._bump_rev()  # NEW: content changed

    # ------------------------------ virtual intake ------------------------------

    def _start_virtual_intake_server(self, host: str, port: int) -> None:
        """Start a FastAPI server in a background thread to intake virtual requests.

        Exposes one endpoint:
            POST /virtual_request
            Body: {
                "request_id": str,
                "eta_ts": float,            # wall-clock timestamp (seconds)
                "prompt_token_ids": List[int]
            }

        Args:
            host: Bind host of the FastAPI server.
            port: Bind port of the FastAPI server.

        Returns:
            None
        """
        app = FastAPI()

        @app.post("/virtual_request")
        def receive_virtual(payload: Dict):
            rid = str(payload["request_id"])
            eta_ts = float(payload["eta_ts"])
            prompt_ids = list(map(int, payload["prompt_token_ids"]))
            with self._virtual_lock:
                self._virtual_pool[rid] = (eta_ts, prompt_ids)
            self._dbg(lambda: f"[virtual] Received rid={rid} eta={eta_ts}")
            return {"ok": True, "stored": rid}

        def _run():
            self._dbg(lambda: f"[virtual] FastAPI listening on {host}:{port}")
            uvicorn.run(app, host=host, port=port, log_level="warning")

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        self._virtual_server_thread = th

    def _materialize_due_virtuals(self) -> None:
        """Create real Requests from virtual ones when eta_ts is reached.

        Returns:
            None
        """
        now_wall = time.time()
        to_create: List[Tuple[str, List[int]]] = []
        with self._virtual_lock:
            for rid, (eta, prompt_ids) in list(self._virtual_pool.items()):
                if eta <= now_wall:
                    to_create.append((rid, prompt_ids))
                    del self._virtual_pool[rid]

        for rid, prompt_ids in to_create:
            try:
                req = Request(
                    request_id=rid,
                    prompt_token_ids=prompt_ids,
                    arrival_time=time.monotonic(),
                )
                self.add_request(req)  # add_request will bump rev
                self._dbg(lambda: f"[virtual] Materialized rid={rid} -> real request")
            except Exception as e:
                self._dbg(lambda: f"[virtual] Materialize rid={rid} failed: {e}")

    # ------------------------------ scoring / sorting ------------------------------

    def _sorted_candidates(self, topk: int = 5,
                           hit_map: Optional[Dict[str, int]] = None,
                           need_map: Optional[Dict[str, int]] = None,
                           wait_map: Optional[Dict[str, float]] = None
                           ) -> list[tuple[float, float, Request]]:
        """Compute request scores and return a sorted candidate list, while
        printing the composition of scores for the top-K candidates.

        Args:
            topk: Number of top candidates to print for debugging.
            hit_map/need_map/wait_map: Optional per-request snapshot to avoid repeated scans.  # (S2)

        Returns:
            A list of tuples (score, arrival_time, request), sorted by score
            descending then FIFO by arrival_time.
        """
        now = time.monotonic()
        scored: list[tuple[float, float, Request]] = []
        details: list[tuple[float, Request, int, int, float, float, float, float]] = []

        # Optional: show current scoring params once per call (S7 lazy)
        self._dbg(lambda: f"[score-params] a={self._a} b={self._b} c={self._c} λ={self._lambda:.4f}")

        for r in self:
            rid = r.request_id
            hit = int((hit_map or self._req_hit).get(rid, 0))
            need = int((need_map or self._req_need).get(rid, 0))
            wait = float((wait_map or {}).get(rid, max(0.0, now - self._req_arrival.get(rid, now))))

            tpb = self.block_size
            prompt_tokens = len(getattr(r, "prompt_token_ids", []) or [])

            term_hit = self._a * hit
            term_need = -(self._b + self._lambda) * ((prompt_tokens / tpb) - hit)
            term_wait = self._c * wait
            score = term_hit + term_need + term_wait

            scored.append((score, r.arrival_time, r))
            details.append((score, r, hit, need, wait, term_hit, term_need, term_wait, (prompt_tokens / tpb) - hit))

        scored.sort(key=lambda x: (-x[0], x[1]))
        details.sort(key=lambda x: -x[0])

        def _topk_dump() -> str:
            lines = ["[score-sort] Top candidates:"]
            for i, (s, r, hit, need, wait, t_hit, t_need, t_wait, need_tok) in enumerate(details[:topk], 1):
                lines.append(
                    f"  {i:>2d} id={r.request_id} "
                    f"hit={hit} need={need} wait={wait:.2f}s | "
                    f"a*hit={t_hit:.3f}  -(b+λ)*{need_tok=}={t_need:.3f}  c*wait={t_wait:.3f}  "
                    f"=> score={s:.3f}"
                )
            return "\n".join(lines)

        self._dbg(_topk_dump)
        return scored

    # ------------------------------ feasibility pick ------------------------------

    def _pick_first_feasible(self,
                             scored: List[Tuple[float, float, Request]],
                             fallback_use_free: bool = False,
                             cap_override: Optional[int] = None   # NEW (S2)
                             ) -> Optional[Request]:
        """Pick the first candidate that fits capacity constraint.

        Args:
            scored: Sorted (score, arrival_time, request) list.
            fallback_use_free: If True, use free_blocks instead of F_eff.
            cap_override: If provided, use this capacity snapshot without re-query.  # (S2)

        Returns:
            The first feasible Request or None.
        """
        free_blocks = self._get_free_blocks() if cap_override is None else cap_override
        F_eff = free_blocks
        cap = free_blocks if fallback_use_free else F_eff

        for _, _, r in scored:
            need = self._req_need.get(r.request_id, 0)
            if need <= cap:
                self._dbg(lambda: f"[pick] Feasible req={r.request_id} need={need} cap={cap}")
                return r

        self._dbg(lambda: f"[pick] No feasible under cap={cap}")
        return None

    def _pick_next_feasible(self,
                            scored: List[Tuple[float, float, Request]],
                            exclude: Request,
                            free_blocks_override: Optional[int] = None  # NEW (S2)
                            ) -> Optional[Request]:
        """Pick the next feasible candidate excluding a specific request.

        Args:
            scored: Sorted (score, arrival_time, request) list.
            exclude: Request to exclude.
            free_blocks_override: If provided, use this free snapshot.  # (S2)

        Returns:
            Alternative feasible Request or None.
        """
        free_blocks = self._get_free_blocks() if free_blocks_override is None else free_blocks_override
        for _, _, r in scored:
            if r.request_id == exclude.request_id:
                continue
            need = self._req_need.get(r.request_id, 0)
            if need <= free_blocks:
                self._dbg(lambda: f"[pick-alt] alt={r.request_id} need={need} cap={free_blocks}")
                return r
        self._dbg(lambda: "[pick-alt] No alternative feasible found.")
        return None

    def _pick_min_need(self,
                       scored: List[Tuple[float, float, Request]]) -> Optional[Request]:
        """Pick the minimal-need request (SJF fallback) if no feasible one found.

        Args:
            scored: Sorted (score, arrival_time, request) list.

        Returns:
            Request with minimal 'need' or None if queue is empty.
        """
        if not scored:
            return None
        cand = min(scored, key=lambda t: self._req_need.get(t[2].request_id, 0))
        r = cand[2]
        self._dbg(lambda: f"[pick-min] No feasible; fallback choose minimal-need id={r.request_id} "
                          f"need={self._req_need.get(r.request_id, 0)}")
        return r

    # ------------------------------ deferral logic ------------------------------

    def _kv_tight(self, free_blocks: int, total_blocks: int,
                  H_snapshot: Optional[int] = None, D_snapshot: Optional[int] = None  # NEW (S2)
                  ) -> bool:
        """KV tight if lambda is high OR pressure ratio z > 1 (optional).

        Args:
            free_blocks: Current free blocks (N).
            total_blocks: Total blocks.
            H_snapshot/D_snapshot: Optional snapshots to avoid recomputation.  # (S2)

        Returns:
            True if KV is tight.
        """
        used_ratio = 1.0 - (free_blocks / max(1, total_blocks))
        lambda_tight = self._lambda > 0.75
        ratio_tight = used_ratio >= 0.85

        if H_snapshot is None or D_snapshot is None:
            H = sum(int(self._req_hit.get(r.request_id, 0)) for r in self)
            D = sum(int(self._req_need.get(r.request_id, 0)) for r in self)
        else:
            H = H_snapshot
            D = D_snapshot
        U = max(0, free_blocks - H)
        z = float(D) / float(U + self._lam_eps)
        z_tight = z > 1.0

        tight = lambda_tight or ratio_tight or z_tight
        self._dbg(lambda: f"[kv-tight] used_ratio={used_ratio:.3f} λ={self._lambda:.4f} z={z:.3f} -> tight={tight}")
        return tight

    def _should_defer(self, picked: Request) -> bool:
        """Check whether to defer the picked request due to future high-hit virtuals."""
        rid = picked.request_id

        if self._is_wait_hard_exceeded(picked):
            self._dbg(lambda: f"[defer-check] req={rid} exceeded hard wait; no defer.")
            return False

        if self._req_defer_cnt[rid] >= self._max_defer_times:
            self._dbg(lambda: f"[defer-check] req={rid} reached max defers; no defer.")
            return False

        if not self._is_long_job(picked):
            self._dbg(lambda: f"[defer-check] req={rid} not a long job; no defer.")
            return False

        picked_hit = self._req_hit.get(rid, 0)
        now_wall = time.time()
        with self._virtual_lock:
            for vrid, (eta, prompt_ids) in self._virtual_pool.items():
                if eta - now_wall <= self._deferral_window_s:
                    est_hit = self._estimate_future_hit(prompt_ids)
                    if est_hit > picked_hit:
                        self._dbg(lambda: f"[defer-check] future={vrid} est_hit={est_hit} "
                                          f"> picked_hit={picked_hit} -> defer")
                        return True

        self._dbg(lambda: f"[defer-check] No future higher-hit found; no defer.")
        return False

    def _mark_deferred(self, req: Request) -> None:
        """Increase deferral counter for a request."""
        self._req_defer_cnt[req.request_id] += 1
        self._dbg(lambda: f"[defer] Marked req={req.request_id} defers={self._req_defer_cnt[req.request_id]}")

    # ------------------------------ refresh & helpers ------------------------------

    def _maybe_refresh_hits(self, full_or_incremental: str = "periodic") -> None:
        """Refresh hit/need with controlled overhead."""
        now = time.monotonic()

        def _refresh_one(req: Request) -> None:
            rid = req.request_id
            try:
                _, num_cached_tokens = self._kvc.get_computed_blocks(req)
            except Exception:
                num_cached_tokens = 0

            hit_blocks = num_cached_tokens // self.block_size
            self._dbg(lambda: f"[DEBUG] rid={rid} num_cached_tokens={num_cached_tokens} hit_blocks={hit_blocks}")
            need_blocks = max(self._estimate_total_blocks(req) - hit_blocks, 0)
            self._req_hit[rid] = int(hit_blocks)
            self._req_need[rid] = int(need_blocks)
            self._need_samples.append(need_blocks)

        if full_or_incremental == "incremental":
            added = 0
            for req in self:
                if req.request_id not in self._req_hit:
                    _refresh_one(req)
                    added += 1
            if added > 0:
                self._dbg(lambda: f"[hit-refresh] Incremental init {added} entries.")
            return

        if (now - self._last_hit_refresh_ts) >= self._hit_refresh_interval_s:
            for req in self:
                _refresh_one(req)
            self._last_hit_refresh_ts = now
            self._dbg(lambda: f"[hit-refresh] Full refresh at {now:.2f}, qsize={len(self)}")
        else:
            added = 0
            for req in self:
                if req.request_id not in self._req_hit:
                    _refresh_one(req)
                    added += 1
            if added > 0:
                self._dbg(lambda: f"[hit-refresh] Patched {added} new entries.")

    def _update_lambda(self) -> None:
        """Update lambda using capacity after subtracting hit occupancy."""
        N = self._get_free_blocks()
        H = 0
        D = 0
        for r in self:
            rid = r.request_id
            H += int(self._req_hit.get(rid, 0))
            D += int(self._req_need.get(rid, 0))

        U = max(0, N - H)
        e = float(D - U)
        z = float(D) / float(U + self._lam_eps)

        s = math.log((D + self._lam_eps) / (U + self._lam_eps))
        lam_soft = 0.5 * self._lam_max * (1.0 + math.tanh(self._lam_k * s))

        self._lam_int = self._lam_beta_int * self._lam_int + (1.0 - self._lam_beta_int) * e

        lam_raw = lam_soft + self._lam_kp * e + self._lam_ki * self._lam_int
        lam_new = max(0.0, min(self._lam_max, lam_raw))

        old_lambda = self._lambda
        max_step = 0.25
        lam_clamped = max(old_lambda - max_step, min(old_lambda + max_step, lam_new))
        self._lambda = lam_clamped

        mode = "KV-priority" if self._lambda > 0.5 else "SJF-priority"
        used_ratio = 1.0 - (N / max(1, self._get_total_blocks()))
        self._dbg(lambda:
            "[λ-update] "
            f"N={N} H={H} U={U} D={D} gap*={e:.2f} z={z:.3f} "
            f"lam_soft={lam_soft:.4f} PI=({self._lam_kp*e:+.4f},{self._lam_ki*self._lam_int:+.4f}) "
            f"λ={self._lambda:.4f}({old_lambda:.4f}→) mode={mode} used_ratio={used_ratio:.3f}"
        )

    def _is_long_job(self, req: Request) -> bool:
        """Classify long jobs by need quantile."""
        need = self._req_need.get(req.request_id, 0)
        if not self._need_samples:
            return need > 0
        samples = sorted(self._need_samples)
        qidx = max(0, int(self._long_job_quantile * (len(samples) - 1)))
        qv = samples[qidx]
        is_long = need >= qv
        self._dbg(lambda: f"[long-job] req={req.request_id} need={need} \
            q{int(self._long_job_quantile*100)}={qv} -> {is_long}")
        return is_long

    def _is_wait_hard_exceeded(self, req: Request) -> bool:
        """Check if hard waiting threshold is exceeded."""
        now = time.monotonic()
        wait = max(0.0, now - self._req_arrival.get(req.request_id, now))
        exceeded = wait >= self._hard_wait_s
        if exceeded:
            self._dbg(lambda: f"[wait] req={req.request_id} wait={wait:.2f}s >= hard={self._hard_wait_s}s")
        return exceeded

    def _register_request_meta(self, req: Request) -> None:
        """Initialize per-request metadata."""
        rid = req.request_id
        if rid not in self._req_arrival:
            self._req_arrival[rid] = time.monotonic()
        self._dbg(lambda: f"[meta] register req={rid}")

    def _cleanup_request_meta(self, req: Request) -> None:
        """Cleanup per-request metadata on removal."""
        self._cleanup_request_meta_by_id(req.request_id)

    def _cleanup_request_meta_by_id(self, rid: str) -> None:
        """Cleanup per-request metadata by request id."""
        self._req_hit.pop(rid, None)
        self._req_need.pop(rid, None)
        self._req_arrival.pop(rid, None)
        self._req_defer_cnt.pop(rid, None)
        self._dbg(lambda: f"[meta] cleanup req={rid}")

    # ------------------------------ capacity & estimators ------------------------------

    def _get_free_blocks(self) -> int:
        """Get current number of free KV blocks."""
        try:
            val = int(self._kvc.block_pool.free_block_queue.num_free_blocks)
            return val
        except Exception as e:
            logger.warn(f"Error getting GPU block info: {e}")
            return 0

    def _get_total_blocks(self) -> int:
        """Get total number of KV blocks."""
        try:
            return int(getattr(self._kvc.block_pool, "num_blocks",
                               len(self._kvc.block_pool.blocks)))
        except Exception as e:
            logger.warn(f"Error getting GPU block info: {e}")
            return -1

    def _estimate_total_blocks(self, req: Request) -> int:
        """Estimate total prompt blocks for a request."""
        toks = len(getattr(req, "prompt_token_ids", []) or [])
        tpb = self.block_size
        blocks = (toks + tpb - 1) // tpb
        return blocks

    def _as_int_blocks(self, hit_blocks_obj) -> int:
        """Best-effort conversion of KVCacheBlocks-like objects to an int count."""
        if isinstance(hit_blocks_obj, int):
            return max(0, hit_blocks_obj)
        for attr in ("num_blocks", "num_kv_blocks", "count", "size"):
            try:
                val = getattr(hit_blocks_obj, attr)
                if isinstance(val, int):
                    return max(0, val)
            except Exception as e:
                logger.error(f"Failed while converting of KVCacheBlocks-like objects: {e}")
        for attr in ("blocks", "items", "data"):
            try:
                container = getattr(hit_blocks_obj, attr)
                return max(0, len(container))
            except Exception as e:
                logger.error(f"Failed while converting of KVCacheBlocks-like objects: {e}")
        try:
            return max(0, len(hit_blocks_obj))  # type: ignore
        except Exception as e:
            logger.error(f"Failed while converting of KVCacheBlocks-like objects: {e}")
        self._dbg(lambda: f"[warn] Could not parse hit blocks from type={type(hit_blocks_obj)}; assume 0")
        return 0

    def _estimate_future_hit(self, prompt_token_ids: List[int]) -> int:
        """Estimate future hit blocks for a virtual request."""
        return 0

    # ------------------------------ deque interface ------------------------------

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        return super().__reversed__()

    # ------------------------------ NEW: external reset API ------------------------------

    def reset_plan(self) -> None:
        """Externally reset the sticky-peek plan."""
        self._planned = None
        self._plan_made_ts = None
        self._bump_rev()

    # ------------------------------ NEW: planning helper (no side effects) ------------------------------

    def _plan_next(self) -> Tuple[Optional[Request], Optional[str]]:
        """Plan the next request to execute using the exact same selection
        pipeline as pop(), but without side effects."""
        self._maybe_refresh_hits(full_or_incremental="periodic")

        # ------------------------------ S2 snapshot ------------------------------
        now_mono = time.monotonic()
        hit_map: Dict[str, int] = {}
        need_map: Dict[str, int] = {}
        wait_map: Dict[str, float] = {}
        H_sum = 0
        D_sum = 0
        for r in self:
            rid = r.request_id
            h = int(self._req_hit.get(rid, 0))
            n = int(self._req_need.get(rid, 0))
            w = max(0.0, now_mono - self._req_arrival.get(rid, now_mono))
            hit_map[rid] = h
            need_map[rid] = n
            wait_map[rid] = w
            H_sum += h
            D_sum += n
        free_blocks = self._get_free_blocks()
        total_blocks = self._get_total_blocks()
        # ------------------------------------------------------------------------

        candidates = self._sorted_candidates(hit_map=hit_map, need_map=need_map, wait_map=wait_map)

        picked = self._pick_first_feasible(candidates, cap_override=free_blocks)
        if picked is None:
            picked = self._pick_min_need(candidates)
            if picked is None:
                return None, None

        if self._kv_tight(free_blocks, total_blocks, H_snapshot=H_sum, D_snapshot=D_sum) and self._should_defer(picked):
            alt = self._pick_next_feasible(candidates, exclude=picked, free_blocks_override=free_blocks)
            if alt is not None:
                self._dbg(lambda: f"[schedule(plan)] Defer req={picked.request_id} -> alt={alt.request_id}")
                return alt, picked.request_id

        return picked, None

    # ------------------------------ NEW: tiny helpers ------------------------------

    def _find_request(self, rid: str) -> Optional[Request]:
        """Find a request object by id in the deque."""
        for r in self:
            if r.request_id == rid:
                return r
        return None

    def _bump_rev(self) -> None:
        """Bump the content revision to invalidate any cached plan."""
        self._rev = (self._rev + 1) % 1_000_000_000

    def _print_real_time_metrics(self) -> None:
        """Real-time monitoring: 
            Print lambda, 
            GPU free blocks, 
            running/waiting requests, 
            and waiting request lengths.
        """
        
        # Get current metrics
        lambda_val = self._lambda
        free_blocks = self._get_free_blocks()
        total_blocks = self._get_total_blocks()
        waiting_requests_count = len(self)
        
        # Get running requests count (approximation - requests that are not in waiting queue)
        # Note: This might need adjustment based on your specific vLLM setup
        running_requests_count = 0  # Placeholder - you may need to get this from vLLM engine
        
        print("\n" + "="*60)
        print("REAL-TIME QUEUE METRICS:")
        print(f"Lambda (λ): {lambda_val:.4f}")
        print(f"GPU Free Blocks: {free_blocks}/{total_blocks} ({free_blocks/total_blocks*100:.1f}% free)")
        print(f"Running Requests: {running_requests_count}")
        print(f"Waiting Requests: {waiting_requests_count}")
        
        # Print details for each waiting request
        if waiting_requests_count > 0:
            print("\nWAITING REQUESTS DETAILS:")
            for i, request in enumerate(self, 1):
                prompt_length = len(getattr(request, "prompt_token_ids", []))
                hit_blocks = self._req_hit.get(request.request_id, 0)
                need_blocks = self._req_need.get(request.request_id, 0)
                wait_time = time.monotonic() - self._req_arrival.get(request.request_id, time.monotonic())
                
                print(f"  {i}. Request ID: {request.request_id}")
                print(f"     Prompt Length: {prompt_length} tokens")
                print(f"     Hit Blocks: {hit_blocks}")
                print(f"     Need Blocks: {need_blocks}")
                print(f"     Wait Time: {wait_time:.2f}s")
                print(f"     Defer Count: {self._req_defer_cnt.get(request.request_id, 0)}")
        else:
            print("\nNo waiting requests.")
        
        print("="*60 + "\n", flush=True)
