import time

from vllm.logger import init_logger


logger = init_logger(__name__)


class PrefillDecodeScheduler:
    def __init__(self):
        self.last_accessed = 0
        self.last_force_schedule = 0

    def should_skip_scheduling(self, waiting, running):
        # Dynamic scheduling threshold adjustment logic
        skip_schedule = False
        # Base threshold parameters
        BASE_MIN_SEQS = 2
        BASE_MAX_WAIT = 6
        
        # Get current waiting queue length
        current_waiting = len(waiting)
        
        # Dynamically adjust the minimum number of sequences to schedule (range: 2-6)
        min_num_seqs_to_schedule = max(
            BASE_MIN_SEQS, 
            min(6, int(BASE_MIN_SEQS * (1 + current_waiting/256.0))))
        
        # Dynamically adjust the maximum wait count (range: 6-10)
        max_num_schedule_to_wait = max(
            6, 
            min(10, int(BASE_MAX_WAIT * (1 - current_waiting/256.0))))
        
        num_running_tasks = len(running)
        
        DEBUG_SCHEDULER = True
        # Debug info output
        if DEBUG_SCHEDULER:
            logger.debug(
                f"min_num_seqs_to_schedule={min_num_seqs_to_schedule} | \
                max_num_schedule_to_wait={max_num_schedule_to_wait} | \
                current_waiting={current_waiting}"
            )
        
        if current_waiting > 0 and num_running_tasks >= min_num_seqs_to_schedule:            
            # ========== Request queue aging detection ========== #
            # Check the wait time of the oldest request in the queue
            current_time = time.monotonic()
            oldest_wait = current_time - min(req.arrival_time for req in waiting)
            
            # Force schedule if any request has been waiting for more than 0.5s
            if oldest_wait > 0.5:
                if DEBUG_SCHEDULER:
                    logger.debug(f"Force schedule due to old request: {oldest_wait*1000:.1f}ms")
                skip_schedule = False
                self.last_accessed = 0
                self.last_force_schedule = current_time
            else:
                # Decide whether to skip this scheduling round
                if current_waiting < min_num_seqs_to_schedule:
                    if self.last_accessed < max_num_schedule_to_wait:
                        skip_schedule = True
                
                # Smart counter reset (step-down decrease instead of resetting to zero)
                if self.last_accessed >= max_num_schedule_to_wait:
                    self.last_accessed = max(0, self.last_accessed - 5)
                    if DEBUG_SCHEDULER:
                        logger.debug(f"Reset last_accessed to {self.last_accessed}")
                
                if DEBUG_SCHEDULER:
                    logger.debug(
                        f"current_waiting={current_waiting} | \
                        last_accessed={self.last_accessed} | \
                        skip_schedule={skip_schedule} | \
                        num_running_tasks={num_running_tasks} | \
                        oldest_wait={oldest_wait*1000:.1f}ms"
                    )
                
                # Update access counter
                if skip_schedule:
                    self.last_accessed += 1

        return skip_schedule
