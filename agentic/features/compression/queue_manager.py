from __future__ import annotations

import asyncio
import logging
import threading
import time
from copy import deepcopy
from typing import Any, Callable, List, NamedTuple

logger = logging.getLogger("compression")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_handler)


class AsyncTaskResult(NamedTuple):
    """异步任务结果 — 替代脆弱的 5-tuple 隐式契约"""

    result: Any | None  # 任务返回值（压缩 LLM 的输出）
    error: Exception | None  # 异常信息
    timestamps: tuple[float, float]  # (submit_ts, complete_ts)
    cached_prefix: list  # kv_warmup 回调返回值


class ThreadSafeDict:
    """线程安全的字典，用于 AsyncQueueManager 存储任务结果"""

    def __init__(self, lock: threading.Lock):
        self._dict: dict = {}
        self._lock = lock

    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]

    def __contains__(self, key):
        with self._lock:
            return key in self._dict

    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)

    def keys(self) -> list:
        with self._lock:
            return list(self._dict.keys())

    def values(self) -> list:
        with self._lock:
            return list(self._dict.values())

    def items(self) -> list:
        with self._lock:
            return list(self._dict.items())


class AsyncQueueManager:
    """异步任务调度器 — 后台 daemon 线程运行事件循环

    设计决策：
    - max_pending_tasks 并发控制：构造方法接收，submit 时内部判断，
      超限返回 False，由调用方决定如何处理
    - callback 保留：为 KV 预热设计，CompressionManager 提交时
      传入 config.kv_warmup_callback
    - get_result 返回 AsyncTaskResult 命名元组：替代 5-tuple，
      语义清晰，解包方便，后续加字段不破坏兼容性
    - 不暴露 shutdown：daemon 线程自动清理
    """

    def __init__(self, max_pending_tasks: int = 1):
        self._max_pending = max_pending_tasks
        self._pending_count = 0
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._results = ThreadSafeDict(self._lock)

    def _run_loop(self):
        """在后台线程中运行事件循环"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit_async_task(
        self,
        async_func: Callable,
        task_id: str,
        callback: Callable | None,
        *args,
        **kwargs,
    ) -> bool:
        """提交异步任务

        Args:
            async_func: 异步函数（如 async_multi_round_compress）
            task_id: 任务唯一标识
            callback: 可选回调，签名 async (result) -> list，
                      用于 KV 预热（config.kv_warmup_callback）
            *args, **kwargs: 传递给 async_func 的参数

        Returns:
            True 提交成功，False 超过 max_pending_tasks 拒绝提交
        """
        with self._lock:
            if self._pending_count >= self._max_pending:
                logger.info(
                    f"submit_async_task rejected: task_id={task_id}, pending={self._pending_count}, max={self._max_pending}"
                )
                return False
            self._pending_count += 1
            logger.info(
                f"submit_async_task accepted: task_id={task_id}, pending={self._pending_count}/{self._max_pending}"
            )

        async def run_async():
            return await async_func(*args, **kwargs)

        async def _handle_completion(future, tid, cb):
            try:
                result = future.result()
                complete_ts = time.time()
                logger.info(
                    f"[async_compress] task completed: task_id={tid}, result_len={len(result) if result else 0}"
                )
                prefix = []
                if cb:
                    prefix = await cb(result)
                self._results[tid] = AsyncTaskResult(
                    result=result,
                    error=None,
                    timestamps=(complete_ts, complete_ts),
                    cached_prefix=prefix,
                )
            except Exception as e:
                err_ts = time.time()
                logger.error(f"Async task {tid} failed: {e}")
                self._results[tid] = AsyncTaskResult(
                    result=None,
                    error=e,
                    timestamps=(err_ts, err_ts),
                    cached_prefix=[],
                )
            finally:
                with self._lock:
                    self._pending_count -= 1

        def on_complete_sync(future):
            asyncio.run_coroutine_threadsafe(
                _handle_completion(future, task_id, callback),
                self._loop,
            )

        asyncio_future = asyncio.run_coroutine_threadsafe(run_async(), self._loop)
        asyncio_future.add_done_callback(on_complete_sync)
        return True

    def get_result(self, key) -> AsyncTaskResult:
        """获取并移除结果，不存在返回全 None 的 AsyncTaskResult"""
        try:
            if key in self._results:
                ret = deepcopy(self._results[key])
                del self._results[key]
                return ret
            else:
                return AsyncTaskResult(
                    result=None, error=None, timestamps=(0.0, 0.0), cached_prefix=[]
                )
        except Exception as e:
            return AsyncTaskResult(
                result=None, error=e, timestamps=(0.0, 0.0), cached_prefix=[]
            )

    def get_finished_task_ids(self) -> list[str]:
        """返回已完成任务的 ID 列表"""
        return list(self._results.keys())