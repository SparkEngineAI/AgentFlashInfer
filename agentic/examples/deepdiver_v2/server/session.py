import asyncio
import uuid
from asyncio import Queue, Task
from dataclasses import field, dataclass
from typing import Optional, Dict, Set, Callable, Any
import time
import threading
import copy

from examples.deepdiver_v2.tracing.processor import send_sse_stop_signal
import config.config as config

# todo 删除任务的stop的那个函数修复一下 ， 当前任务结束后有时候会显示done(),但是

# 配置存储：按 userToken 存储每个窗口的配置覆盖
SESSION_CONFIG_STORE: Dict[str, Dict[str, Any]] = {}
config_lock = threading.Lock()
_original_config_store: Dict[str, Any] = {}  # 存储原始配置，用于恢复（保留用于其他地方的兼容性）


def get_window_config(user_token: str) -> Dict[str, Any]:
    """获取指定窗口的配置覆盖"""
    with config_lock:
        return SESSION_CONFIG_STORE.get(user_token, {}).copy()


def set_window_config(user_token: str, config_overrides: Dict[str, Any]):
    """设置指定窗口的配置覆盖"""
    with config_lock:
        if user_token not in SESSION_CONFIG_STORE:
            SESSION_CONFIG_STORE[user_token] = {}
        SESSION_CONFIG_STORE[user_token].update(config_overrides)


def apply_config_overrides(config_overrides: Dict[str, Any], backup_original: bool = True) -> Dict[str, Any]:
    """
    应用配置覆盖到全局 config 模块

    Args:
        config_overrides: 要覆盖的配置项
        backup_original: 是否备份原始值（用于后续恢复）

    Returns:
        备份的原始值（仅当 backup_original=True 时），否则返回空字典
    """
    if not config_overrides:
        return {}

    backed_up_values = {}

    with config_lock:
        for key, value in config_overrides.items():
            # 检查模块是否具有该属性
            has_attr = hasattr(config, key)

            if has_attr:
                # 备份原始值（如果需要）
                if backup_original and key not in _original_config_store:
                    original_value = getattr(config, key, None)
                    if original_value is not None:
                        _original_config_store[key] = copy.deepcopy(original_value)
                        backed_up_values[key] = copy.deepcopy(original_value)

                # 应用新值
                original_value = getattr(config, key)

                # 对于布尔值，确保类型正确
                if isinstance(value, str) and value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'

                if isinstance(value, dict) and isinstance(original_value, dict):
                    # 对于字典类型，需要特殊处理
                    original_dict = getattr(config, key)
                    original_dict.update(value)
                else:
                    # 尝试设置到模块（模块级变量）
                    setattr(config, key, value)
                    print(f"  [apply_config] Set config.{key} = {value} (type: {type(value).__name__})")
            else:
                print(f"  [apply_config] WARNING: config.{key} not found in config module")

    return backed_up_values


def restore_config(original_values: Dict[str, Any]):
    """恢复原始配置"""
    if not original_values:
        return

    with config_lock:
        for key, value in original_values.items():
            if hasattr(config, key):
                if isinstance(value, dict) and isinstance(getattr(config, key), dict):
                    # 恢复字典
                    current_dict = getattr(config, key)
                    current_dict.clear()
                    current_dict.update(value)
                else:
                    setattr(config.__class__ if hasattr(config.__class__, key) else config, key, value)

class Session:
    def __init__(self, session_id: str, user_token: str, concurrency: int, running_tasks:int):
        self.session_id=session_id
        self.user_token=user_token
        self.concurrency=concurrency
        self.trace_count: int=0
        # completed_traces: int = 0
        # method_stats: Dict[str, MethodStats] = field(default_factory=dict)
        # self.tasks: Dict[str, asyncio.Task] = {}  # task_id → Task
        self.task_meta: Dict[str, dict] = {}  # task_id → metadata
        self.tasks: Dict[str, asyncio.Task] ={} # task_id = trace_id
        # sse_queue: Queue = field(default_factory=lambda: Queue(maxsize=100))
        # sse_consumer_task: Optional[Task] = None   # 👈 新增：SSE 消费者协程
        self.running_tasks=running_tasks                     # 当前还在跑的任务数
        self.created_at: float = field(default_factory=time.time)
        self.completed_at: Optional[float] = None
    def stop_session(self):
        # 1. 断开sse流
        for trace in self.task_meta.keys():
            send_sse_stop_signal(self.session_id)
        # 2. 停止所有并发的任务,
        task_ids_to_stop = list(self.tasks.keys())

        for task_id in task_ids_to_stop:
            self._stop_task(task_id)

        self.completed_at = time.time()

    def _stop_task(self,task_id: str):
        """程序手动执行结束. 单个协程关闭， 用于在stop_session方法中关闭所有任务"""
        task = self.tasks.get(task_id)
        self.running_tasks -= 1
        self.task_meta.pop(task_id, None)
        if task and not task.done():
            task.cancel()
        # 清理任务引用（即使没完成也移除）
        self.tasks.pop(task_id, None)


    def end_task(self, task_id: str):
        """程序正常执行结束."""
        task = self.tasks.get(task_id)
        self.running_tasks -= 1
        self.task_meta[task_id].update({'end_time': time.time()})
        if self.running_tasks <= 0:
            self.completed_at = time.time()
            # 单session的sse流同时传输多个trace数据流，仅仅当该session所有的trace都执行完毕时，断开sse流
            send_sse_stop_signal(self.session_id)

    # 更新状态完成
    def clean_up(self):
        pass
    def create_task(self, work: callable, request, index, task_id:str):
        coro = work()
        task = asyncio.create_task(
            coro
        )
        self.tasks[task_id] = task
        self.task_meta[task_id] = {
            "session_id": self.session_id,
            "user_token": self.user_token,
            "start_time": time.time(),
            "index": index,
            "request": request
        }


class SessionManager:
    def __init__(self, sse_close_delay: int = 5, user_ttl_seconds: int = 7200):
        # Session & Task Storage
        self.sessions: Dict[str, Session] = {}
        # self.tasks: Dict[str, asyncio.Task] = {}      # task_id → Task
        # self.task_meta: Dict[str, dict] = {}          # task_id → metadata
        self.user_sessions: Dict[str, Set[str]] = {}  # user_token → {session_id}
        # self.user_last_seen: Dict[str, float] = {}    # user_token → timestamp

        # Config
        # self.sse_close_delay = sse_close_delay
        # self.user_ttl_seconds = user_ttl_seconds


    # ──────────────── Submit & Task Management ────────────────
    def cleanup_user_sessions(self, user_id: str):
        """
        立即清理指定 user_id 的所有会话和相关资源
        """
        # 1. 获取该用户所有 session_id
        session_ids = self.user_sessions.pop(user_id, set())

        # 2. 逐个清理每个 session
        for sid in session_ids:
            session = self.sessions.pop(sid, None)
            session.stop_session()
            # 清除  SSE 相关资源
        # # 3. 清理心跳记录
        # user_last_seen.pop(user_id, None)


    def create_session(self, user_token, concurrency) -> Session:
        session_id = f"sess_{uuid.uuid4().hex}"
        session = Session(
            session_id=session_id,
            user_token=user_token,
            concurrency=concurrency,
            running_tasks=concurrency
        )
        self.sessions[session_id] = session

        # 用户 ↔ Session 映射
        if session.user_token not in self.user_sessions:
            self.user_sessions[session.user_token] = set()
        self.user_sessions[session.user_token].add(session_id)
        return session

    def get_session_by_session_id(self, session_id: str) -> Session:
        if session_id in self.sessions:
            return self.sessions[session_id]

    # ──────────────── Stats for Monitoring ────────────────
    def stop_session(self, session_id: str):
        to_del_session_ids = list(self.sessions.keys())
        if session_id in to_del_session_ids:
            self.sessions[session_id].stop_session()
            del self.sessions[session_id]


    def get_system_stats(self) -> dict:
        """供 /stats 接口调用，返回全局统计信息"""
        now = time.time()

        #
        # # 活跃用户（最近 TTL 内有心跳）
        # active_users = [
        #     user for user, ts in self.user_last_seen.items()
        #     if (now - ts) < self.user_ttl_seconds
        # ]

        # 按用户聚合任务数
        user_task_count = {}
        total_running = 0
        for session in self.sessions.values():
            user = session.user_token
            user_task_count[user] = user_task_count.get(user, 0) + session.running_tasks
            total_running += session.running_tasks

        warning_threshold = 100
        warning = total_running > warning_threshold

        return {
            "total_running_tasks": total_running,
            # "active_users_count": len(active_users),
            "tasks_per_user": user_task_count,
            "warning": warning,
            "warning_threshold": warning_threshold,
            "timestamp": now
        }




sessionManager = SessionManager()