import hashlib
import json
import redis
import time
import threading
import zlib
import copy
import inspect
import logging
import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, DefaultDict, List as TypingList
from collections import defaultdict
from functools import wraps
import config.config as config

def init_Model_logger(log_file_path =  "tool_cache_log.log"):
    """日志器"""
    tools_logger = logging.getLogger("tool.tool_cache")
    tools_logger.setLevel(logging.INFO)
    tools_logger.propagate = False
    tools_logger.handlers = []
    file_handler_tool = logging.FileHandler(log_file_path, mode = 'w', encoding="utf-8")
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler_tool.setFormatter(log_formatter)
    file_handler_tool.setLevel(logging.DEBUG)
    tools_logger.addHandler(file_handler_tool)
    return tools_logger

MODEL_LOGGER = init_Model_logger()
_PENDING_TOOLS: List[Tuple[Callable, Dict[str, Any]]] = []
_ACTIVE_TOOL_CACHE: Optional['ToolCache'] = None
_DECORATOR_LOCK = threading.Lock()
_TOOL_CACHE_SWITCH = config.tool_cache_switch  # 工具缓存总开关

def tool(metadata: Dict[str, Any]) -> Callable:
    """工具装饰器（仅支持异步函数，保留原有代码结构，最小化修改）"""
    def decorator(func: Callable) -> Callable:
        
        if not _TOOL_CACHE_SWITCH:
            print(f"[Tool装饰器] START_TOOLCACHE=False，禁用{func.__name__}装饰器")
            return func
        
        print(f"[Tool装饰器-注册] 开始处理工具函数：{func.__name__}")

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            """异步包装器：批量I/O异步化（保留原有业务逻辑，仅补全await）"""
            print(f"[Tool装饰器-调用] 开始执行{func.__name__}的异步缓存逻辑（批量I/O并行）")
            
            if _ACTIVE_TOOL_CACHE is None:
                raise RuntimeError("ToolCache has not been initialized yet.")

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            call_params = bound_args.arguments 
            
            if func.__name__ == "url_crawler":
                documents = call_params.get("documents", [])
                if isinstance(documents, list) and len(documents) > 0:
                    async def _async_process_single_doc(single_doc):
                        single_params = call_params.copy()
                        single_params["documents"] = [single_doc]
                        single_result, _ = await _ACTIVE_TOOL_CACHE.run(func.__name__, single_params)
                        return single_result
                    
                    all_single_results = await asyncio.gather(
                        *[_async_process_single_doc(doc) for doc in documents]
                    )
                    
                    combined_data = []
                    for single_result in all_single_results:
                        if "data" in single_result:
                            combined_data.extend(single_result["data"])
                        else:
                            combined_data.append(single_result)
                    
                    return {
                        "success": True,
                        "data": combined_data
                    }
            
            primary_args = metadata.get("primary_args", "")
            if primary_args:
                param_name = primary_args[0]
                if param_name in call_params:
                    param_value = call_params[param_name]
                    
                    if isinstance(param_value, list) and len(param_value) > 0:

                        async def _async_process_single_value(single_val):
                            single_params = call_params.copy()
                            single_params[param_name] = [single_val]
                            print(single_params)
                            single_result, _ = await _ACTIVE_TOOL_CACHE.run(func.__name__, single_params)
                            return single_result
                        
                        all_single_results = await asyncio.gather(
                            *[_async_process_single_value(val) for val in param_value]
                        )
                        
                        list_results = []
                        for single_result in all_single_results:
                            list_results.extend(single_result)
                            
                        return list_results

            result, _ = await _ACTIVE_TOOL_CACHE.run(func.__name__, call_params)
            return result
        
        wrapper = async_wrapper
        wrapper._original_func = func
        wrapper._metadata = metadata
        with _DECORATOR_LOCK:
            _PENDING_TOOLS.append((wrapper, metadata))

        return wrapper

    return decorator

# ------------------------------ 1. 核心数据结构定义 ------------------------------
@dataclass
class CachePlan:
    """缓存计划：记录工具的缓存策略"""
    tool_name: str
    kind: str
    cacheability: str = 'NONE'
    expiration_time: int = -1
    primary_args: List[str] = field(default_factory=list)
    invalidates: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class HistoricalMetrics:
    """历史指标：记录工具的调用、命中、驱逐等统计信息"""
    total_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    write_operations: int = 0
    hit_rates: List[float] = field(default_factory=list)
    window_hits: int = 0
    window_misses: int = 0
    window_evictions: int = 0


@dataclass
class ToolState:
    """工具状态：用于Replanning的EWMA和冷却时间管理"""
    ewma_hit: float = 0.0
    ewma_evict: float = 0.0
    base_hit: float = 0.0
    base_evict: float = 0.0
    last_replan: float = 0.0


# ------------------------------ 2. 历史数据库 ------------------------------
class HistoricalDatabase:
    def __init__(self):
        self.metrics: Dict[str, HistoricalMetrics] = defaultdict(HistoricalMetrics)
        self.cache_plans: Dict[str, CachePlan] = {}
        self.tool_states: Dict[str, ToolState] = defaultdict(ToolState)
        self.used_memory_initial: int = 0
        self.used_memory_final: int = 0
        self._lock = threading.RLock()

    def set_cache_plan(self, tool_name: str, plan: CachePlan) -> None:
        with self._lock:
            self.cache_plans[tool_name] = plan

    def get_cache_plan(self, tool_name: str) -> Optional[CachePlan]:
        with self._lock:
            return self.cache_plans.get(tool_name)

    def set_tool_state(self, tool_name: str, state: ToolState) -> None:
        with self._lock:
            self.tool_states[tool_name] = state

    def get_tool_state(self, tool_name: str) -> ToolState:
        with self._lock:
            return self.tool_states[tool_name]

    def set_used_memory_initial(self, memory: int) -> None:
        with self._lock:
            self.used_memory_initial = memory
            self.used_memory_final = memory

    def update_used_memory_final(self, memory: int) -> None:
        with self._lock:
            self.used_memory_final = memory

    def get_metrics(self, tool_name: str) -> HistoricalMetrics:
        with self._lock:
            return self.metrics[tool_name]

    def update_call_metrics(self, tool_name: str, is_hit: bool) -> None:
        with self._lock:
            metrics = self.metrics[tool_name]
            metrics.total_calls += 1
            if is_hit:
                metrics.cache_hits += 1
                metrics.window_hits += 1
            else:
                metrics.cache_misses += 1
                metrics.window_misses += 1

    def update_eviction_metric(self, tool_name: str, count: int = 1) -> None:
        with self._lock:
            self.metrics[tool_name].evictions += count
            self.metrics[tool_name].window_evictions += count

    def update_write_metric(self, tool_name: str) -> None:
        with self._lock:
            self.metrics[tool_name].write_operations += 1

    def _get_and_reset_window_metrics(self, tool_name: str) -> Tuple[float, int]:
        """获取当前窗口的命中率和驱逐次数，并重置窗口指标，不可单独使用"""
        metrics = self.metrics[tool_name]
        total = metrics.window_hits + metrics.window_misses
        hit_ratio = metrics.window_hits / total if total > 0 else 0.0
        metrics.hit_rates.append(hit_ratio)
        evictions = metrics.window_evictions

        # 重置窗口指标
        metrics.window_hits = 0
        metrics.window_misses = 0
        metrics.window_evictions = 0

        return hit_ratio, evictions

    def get_snapshot(self) -> Dict[str, Any]:
        snapshot = {}
        with self._lock:
            for tool_name in self.cache_plans.keys():
                snapshot[tool_name] = self._get_and_reset_window_metrics(tool_name)
            snapshot['used_memory'] = self.used_memory_final - self.used_memory_initial
        return snapshot

    def initialize_tool_states(self, snapshot):
        with self._lock:
            for tool_name, cache_plan in self.cache_plans.items():
                hit_ratio, evictions = snapshot[tool_name]
                tool_state = ToolState(
                    ewma_hit=hit_ratio,
                    ewma_evict=evictions,
                    base_hit=hit_ratio,
                    base_evict=evictions,
                    last_replan=cache_plan.created_at,
                )
                self.tool_states[tool_name] = tool_state


# ------------------------------ 3. Redis缓存管理器 ------------------------------
class RedisCacheManager:
    def __init__(
            self,
            max_memory: int = config.REDIS_MAX_MEMORY,
            redis_host: str = config.REDIS_HOST,
            redis_port: int = config.REDIS_PORT,
            redis_db: int = config.REDIS_DB,
            redis_password: Optional[str] = None,
            redis_client: Optional[redis.Redis] = None
    ):
        self.redis = redis_client if redis_client else redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True
        )
        self.redis.config_set("maxmemory", f"{max_memory}mb")
        self.redis.config_set("maxmemory-policy", "allkeys-lru")
        self.param_index: DefaultDict[str, DefaultDict[str, DefaultDict[Any, set]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set))
        )
        self._lock = threading.RLock()

    def _generate_cache_key(self, tool_name: str, params: Dict[str, Any], primary_args: List[str]) -> str:
        filtered_params = {k: params[k] for k in primary_args if k in params} 
        sorted_params = sorted(filtered_params.items(), key=lambda x: x[0])
        param_str = json.dumps(sorted_params, sort_keys=True, ensure_ascii=False)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        return f"{tool_name}:{param_hash}"

    def _build_param_index(self, cache_key: str, tool_name: str, params: Dict[str, Any], primary_args: List[str]) -> None:
        #TODO: param_index没有驱逐机制，redis驱逐缓存后param_index不更新
        with self._lock:
            for param_name in primary_args:
                if param_name in params:
                    if tool_name == "url_crawler":
                        param_value = params[param_name][0]["url"]
                    else:
                        param_value = str(params[param_name][0])
                    
                    if isinstance(param_value, str) and len(param_value) > 8:
                        param_value = format(zlib.adler32(param_value.encode()), '08x')
                        
                    self.param_index[tool_name][param_name][param_value].add(cache_key)

    def get_cache(self, plan: CachePlan, params: Dict[str, Any]) -> Optional[Any]:
        if plan.kind != "READ" or plan.cacheability == "NONE":
            return None

        try:
            cached_value = self.redis.get(self._generate_cache_key(plan.tool_name, params, plan.primary_args))
            return json.loads(cached_value) if cached_value else None
        except json.JSONDecodeError:
            self.redis.delete(self._generate_cache_key(plan.tool_name, params, plan.primary_args))
            return None

    def set_cache(self, plan: CachePlan, params: Dict[str, Any], result: Any) -> None:
        if plan.kind != "READ" or plan.cacheability == "NONE":
            return

        cache_key = self._generate_cache_key(plan.tool_name, params, plan.primary_args)
        serialized_result = json.dumps(result)

        if plan.cacheability == "STATIC":
            ttl = 86400 if plan.expiration_time == -1 else plan.expiration_time
        else:  # TRANSIENT
            ttl = plan.expiration_time if plan.expiration_time > 0 else 60

        if ttl == -1:
            self.redis.set(cache_key, serialized_result)
        else:
            self.redis.setex(cache_key, ttl, serialized_result)

        if plan.primary_args:
            self._build_param_index(cache_key, plan.tool_name, params, plan.primary_args)

    def invalidate_cache(self, plan: CachePlan, params: Dict[str, Any]) -> int:
        if plan.kind != "WRITE" or not plan.invalidates:
            return 0

        invalidated_count = 0

        with self._lock:
            for rule in plan.invalidates:
                target_tool = rule["target_tool"]
                match_args = rule["match_args"]

                cache_keys_to_delete = set()

                for target_arg, current_arg in match_args.items():
                    if current_arg in params:
                        param_value = params[current_arg]
                        if isinstance(param_value, str) and len(param_value) > 8:
                            param_value = format(zlib.adler32(param_value.encode()), '08x')
                        cache_keys = self.param_index[target_tool].get(target_arg, {}).get(param_value, set())
                        cache_keys_to_delete.update(cache_keys)

                        if target_arg in self.param_index[target_tool]:
                            if param_value in self.param_index[target_tool][target_arg]:
                                del self.param_index[target_tool][target_arg][param_value]
                            if not self.param_index[target_tool][target_arg]:
                                del self.param_index[target_tool][target_arg]
                        if not self.param_index[target_tool]:
                            del self.param_index[target_tool]

                if cache_keys_to_delete:
                    deleted_count = self.redis.delete(*cache_keys_to_delete)
                    invalidated_count += deleted_count

        return invalidated_count

    def get_used_memory(self) -> int:
        try:
            return int(self.redis.info("memory").get("used_memory", 0)) // (1024 * 1024)
        except Exception as e:
            print(f"获取Redis内存占用失败：{str(e)}")
            MODEL_LOGGER.error(f"获取Redis内存占用失败：{str(e)}")
            return 0


# ------------------------------ 4. 缓存计划器 ------------------------------
class CachePlanner:
    def __init__(self, historical_db: HistoricalDatabase, max_memory: int):
        self.historical_db = historical_db
        self.max_memory = max_memory
        self.replan_params = {
            "alpha": 0.5,
            "beta": 2,
            "theta_E": 50,
            "theta_H": 0.1,
            "cooldown": 60
        }

    def _infer_primary_args(self, tool_metadata: Dict[str, Any]) -> List[str]:
        return tool_metadata.get("primary_args", [])

    def _generate_invalidates_rules(self, tool_metadata: Dict[str, Any]) -> List[Dict]:
        return tool_metadata.get("invalidates", [])

    def generate_initial_plan(self, tool_name: str, tool_metadata: Dict[str, Any]) -> CachePlan:
        is_write = any(k in tool_metadata.get("description", "").lower() for k in ["update", "delete", "insert"])
        kind = "WRITE" if is_write else "READ"

        if kind == "READ":
            return CachePlan(
                tool_name=tool_name,
                kind=kind,
                cacheability=tool_metadata.get("cacheability", "TRANSIENT"),
                expiration_time=tool_metadata.get("expiration_time", 60),
                primary_args=self._infer_primary_args(tool_metadata)
            )
        else:
            return CachePlan(
                tool_name=tool_name,
                kind=kind,
                invalidates=self._generate_invalidates_rules(tool_metadata)
            )

    def generate_replan(self, tool_name: str, tool_metadata: Dict[str, Any], snapshot: Dict[str, Any],
                        replan_cond: Tuple) -> CachePlan:
        #TODO：只有READ工具进行replan
        low_hit_ratio, hit_drop, evict_spike = replan_cond
        cache_plan = copy.deepcopy(self.historical_db.get_cache_plan(tool_name))
        base_ttl = cache_plan.expiration_time

        used_memory = snapshot['used_memory']
        u = used_memory / self.max_memory

        if u > 0.9 or evict_spike:
            new_ttl = max(10, int(base_ttl * 0.8))
        elif u < 0.5 and (low_hit_ratio or hit_drop):
            new_ttl = min(3600, int(base_ttl * 1.2))
        else:
            new_ttl = base_ttl

        cache_plan.expiration_time = new_ttl
        cache_plan.created_at = time.time()

        return cache_plan

    def need_replan(self, tool_name: str, snapshot: Dict[str, Any]) -> Tuple[bool, Tuple]:
        #TODO: 暂时不考虑WRITE工具
        cache_plan = self.historical_db.get_cache_plan(tool_name)
        if cache_plan.kind != "READ":
            return False, ()

        replan_flag = False
        params = self.replan_params
        hit_ratio, delta_evict = snapshot[tool_name]

        used_memory = snapshot['used_memory']
        u = used_memory / self.max_memory

        state = copy.deepcopy(self.historical_db.get_tool_state(tool_name))
        state.ewma_hit = params["alpha"] * hit_ratio + (1 - params["alpha"]) * state.ewma_hit
        state.ewma_evict = params["alpha"] * delta_evict + (1 - params["alpha"]) * state.ewma_evict

        low_hit_ratio = hit_ratio < 0.2
        hit_drop = state.ewma_hit < (1 - params["theta_H"]) * state.base_hit
        evict_spike = state.ewma_evict > state.base_evict + params["theta_E"] * (1 - params["beta"] * u)

        state.base_hit = state.ewma_hit if state.ewma_hit > state.base_hit else state.base_hit
        state.base_evict = state.ewma_evict if state.ewma_evict < state.base_evict else state.base_evict

        if time.time() - state.last_replan > params["cooldown"]:
            if low_hit_ratio or hit_drop or evict_spike:
                state.base_hit = state.ewma_hit
                state.base_evict = state.ewma_evict
                state.last_replan = time.time()
                replan_flag = True

        self.historical_db.set_tool_state(tool_name, state)
        return replan_flag, (low_hit_ratio, hit_drop, evict_spike)


# ------------------------------ 5. 协调器 ------------------------------
class Orchestrator:
    def __init__(
            self,
            cache_manager: RedisCacheManager,
            cache_planner: CachePlanner,
            historical_db: HistoricalDatabase,
            replan_interval: int = 100
    ):
        self.cache_manager = cache_manager
        self.cache_planner = cache_planner
        self.historical_db = historical_db
        self.tool_registry: Dict[str, Tuple[Callable, Dict[str, Any]]] = {}
        self.replan_interval = replan_interval
        self.total_calls = 0
        self._lock = threading.RLock()
        self._replanning_thread: Optional[threading.Thread] = None

    def register_tool(self, tool_name: str, tool_func: Callable, tool_metadata: Dict[str, Any]) -> None:
        if tool_name in self.tool_registry:
            print(f"工具 {tool_name} 已注册，跳过重复注册")
            MODEL_LOGGER.info(f"工具 {tool_name} 已注册，跳过重复注册")
            return

        self.tool_registry[tool_name] = (tool_func, tool_metadata)
        initial_plan = self.cache_planner.generate_initial_plan(tool_name, tool_metadata)
        self.historical_db.set_cache_plan(tool_name, initial_plan)
        print(f"工具 {tool_name} 注册成功，初始Plan：{initial_plan}")
        MODEL_LOGGER.info(f"工具 {tool_name} 注册成功，初始Plan：{initial_plan}")
    
    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        tool_func = self.tool_registry[tool_name][0]
        
        if inspect.iscoroutinefunction(tool_func):
            return await tool_func(** params)
        else:
            return tool_func(** params)

    async def run(self, tool_name: str, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if tool_name not in self.tool_registry:
            raise ValueError(f"工具 '{tool_name}' 未注册。")

        current_plan = self.historical_db.get_cache_plan(tool_name)

        cache_result = self.cache_manager.get_cache(current_plan, params)
        is_hit = cache_result is not None

        self.historical_db.update_call_metrics(tool_name, is_hit)

        with self._lock:
            self.total_calls += 1

            if self.total_calls % self.replan_interval == 0:
                current_window = self.total_calls // self.replan_interval
                self.historical_db.update_used_memory_final(self.cache_manager.get_used_memory())
                snapshot = self.historical_db.get_snapshot()

                if current_window == 1:
                    print(f"\n完成第 {current_window} 个统计窗口（累计调用 {self.total_calls} 次）。")
                    MODEL_LOGGER.info(f"\n完成第 {current_window} 个统计窗口（累计调用 {self.total_calls} 次）。")
                    print("开始初始化所有工具的 ToolState...")
                    MODEL_LOGGER.info("开始初始化所有工具的 ToolState...")
                    self.historical_db.initialize_tool_states(snapshot)
                    print("ToolState 初始化完成。")
                    MODEL_LOGGER.info("ToolState 初始化完成。")

                elif current_window >= 2:
                    print(f"\n完成第 {current_window} 个统计窗口（累计调用 {self.total_calls} 次）。")
                    MODEL_LOGGER.info(f"\n完成第 {current_window} 个统计窗口（累计调用 {self.total_calls} 次）。")
                    print("准备启动自适应重新规划线程...")
                    MODEL_LOGGER.info("准备启动自适应重新规划线程...")
                    self._start_adaptive_replanning_thread(snapshot)

        print(f'缓存命中：{is_hit}')
        MODEL_LOGGER.info(f'缓存命中：{is_hit}')
        if is_hit:
            return cache_result, {
                "status": "cache_hit",
                "tool": tool_name,
                "plan_created_at": current_plan.created_at
            }

        tool_result = await self._execute_tool(tool_name, params)

        if current_plan.kind == "READ":
            self.cache_manager.set_cache(current_plan, params, tool_result)
            return tool_result, {
                "status": "cache_miss",
                "tool": tool_name,
                "action": "cache_set",
                "plan_created_at": current_plan.created_at
            }
        else:  # WRITE
            self.historical_db.update_write_metric(tool_name)
            invalidated_count = self.cache_manager.invalidate_cache(current_plan, params)
            self.historical_db.update_eviction_metric(tool_name, invalidated_count)
            return tool_result, {
                "status": "cache_miss",
                "tool": tool_name,
                "action": "cache_invalidate",
                "invalidated_count": invalidated_count
            }

    def _start_adaptive_replanning_thread(self, snapshot):
        """启动一个独立线程来执行自适应重新规划，避免阻塞主流程"""
        if self._replanning_thread and self._replanning_thread.is_alive():
            print("上一个重新规划线程仍在运行中，跳过本次触发。")
            MODEL_LOGGER.info("上一个重新规划线程仍在运行中，跳过本次触发。")
            return

        self._replanning_thread = threading.Thread(
            target=self._adaptive_replanning_task,
            name="AdaptiveReplanningThread",
            args=(snapshot,),
            daemon=True
        )
        self._replanning_thread.start()
        print(f"自适应重新规划线程已启动 (Thread ID: {self._replanning_thread.ident})。")
        MODEL_LOGGER.info(f"自适应重新规划线程已启动 (Thread ID: {self._replanning_thread.ident})。")

    def _adaptive_replanning_task(self, snapshot):
        """在独立线程中执行的重新规划任务"""
        print("自适应重新规划线程开始工作...")
        MODEL_LOGGER.info("自适应重新规划线程开始工作...")

        replanned_tools = []
        for tool_name, (_, tool_metadata) in self.tool_registry.items():
            replan_flag, replan_cond = self.cache_planner.need_replan(tool_name, snapshot)
            if replan_flag:
                print(f"-> 工具 '{tool_name}' 需要重新规划...")
                MODEL_LOGGER.info(f"-> 工具 '{tool_name}' 需要重新规划...")
                new_plan = self.cache_planner.generate_replan(tool_name, tool_metadata, snapshot, replan_cond)
                self.historical_db.set_cache_plan(tool_name, new_plan)
                replanned_tools.append(tool_name)
                print(f"工具 '{tool_name}' 计划更新完成。新 TTL: {new_plan.expiration_time}s。")
                MODEL_LOGGER.info(f"工具 '{tool_name}' 计划更新完成。新 TTL: {new_plan.expiration_time}s。")

        if not replanned_tools:
            print("所有工具性能稳定，无需调整计划。")
            MODEL_LOGGER.info("所有工具性能稳定，无需调整计划。")

        print("自适应重新规划线程工作完成。")
        MODEL_LOGGER.info("自适应重新规划线程工作完成。")


# ------------------------------ 6. 工具缓存入口 ------------------------------
class ToolCache:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
            self,
            replan_interval: int = 100,
            max_memory: int = 100,  # 默认100MB
            **redis_kwargs
    ):
        if ToolCache._initialized:
            return
        global _ACTIVE_TOOL_CACHE, _PENDING_TOOLS

        print("正在初始化 ToolCache...")

        self.historical_db = HistoricalDatabase()
        self.cache_manager = RedisCacheManager(max_memory=max_memory, **redis_kwargs)
        self.cache_planner = CachePlanner(self.historical_db, max_memory)
        self.orchestrator = Orchestrator(
            self.cache_manager,
            self.cache_planner,
            self.historical_db,
            replan_interval
        )
        
        if _PENDING_TOOLS:
            print("\n发现待注册的工具，开始自动注册...")
            MODEL_LOGGER.info("\n发现待注册的工具，开始自动注册...")
            for wrapped_func, metadata in _PENDING_TOOLS:
                tool_name = wrapped_func._original_func.__name__
                self.orchestrator.register_tool(tool_name, wrapped_func._original_func, metadata)
            _PENDING_TOOLS.clear()  # 清空待注册列表

        self.historical_db.set_used_memory_initial(self.cache_manager.get_used_memory())

        _ACTIVE_TOOL_CACHE = self

        print("\nToolCache 初始化完成，已准备就绪。")
        MODEL_LOGGER.info("\nToolCache 初始化完成，已准备就绪。")
        ToolCache._initialized = True
        
    async def run(self, tool_name: str, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        return await self.orchestrator.run(tool_name, params)
    
    def __del__(self):
        try:
            self.cache_manager.redis.close()
            print("Redis连接已关闭")
        except Exception as e:
            print(f"关闭Redis连接失败: {e}")

    def manual_save_metrics(self):
        """手动保存historical_db的metrics到新文件（不覆盖）"""
        try:
            save_dir = Path(config.TOOL_CACHE_METRICS_PATH)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = save_dir / f"toolcache_metrics_{timestamp}.json"
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "metrics": self.historical_db.metrics
                }, f, ensure_ascii=False, indent=2)
            
            print(f"Metrics保存成功：{file_path}")
            
        except Exception as e:
            print(f"Metrics保存失败：{str(e)}")


def init_tool_cache(replan_interval: int = config.REPLAN_INTERVAL, max_memory: int = config.REDIS_MAX_MEMORY) -> ToolCache:
    """初始化ToolCache"""
    if not _TOOL_CACHE_SWITCH:
        print("\n---TOOL_CACHE_SWITCH=false，跳过ToolCache初始化 ---")
        
        MODEL_LOGGER.info("\n---TOOL_CACHE_SWITCH=false，跳过ToolCache初始化 ---")
        class _NullToolCache:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
            
        return _NullToolCache()
    
    print("\n---加载工具函数 ---")
    MODEL_LOGGER.info("\n---加载工具函数 ---")
    try:
        from agent_infer.mcp_factory.tools.searcher.batch_web_search import batch_web_search
        from agent_infer.mcp_factory.tools.searcher.jina_reader import jina_reader
        from agent_infer.mcp_factory.tools.searcher.url_crawler import url_crawler
        
    except ImportError as e:
        raise RuntimeError(f"加载工具函数失败: {str(e)}")

    print("\n---实例化ToolCache ---")
    MODEL_LOGGER.info("\n---实例化ToolCache ---")
    
    return ToolCache(replan_interval=replan_interval, max_memory=max_memory)

TOOL_CACHE_INSTANCE = init_tool_cache()