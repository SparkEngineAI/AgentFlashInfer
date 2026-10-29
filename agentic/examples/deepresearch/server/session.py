import asyncio
import uuid
import time
import threading
from typing import Optional, Dict, Set

from examples.deepresearch.tracing.processor import send_sse_stop_signal


class Session:
    def __init__(self, session_id: str, user_token: str, concurrency: int = 1):
        self.session_id = session_id
        self.user_token = user_token
        self.concurrency = concurrency
        self.running_tasks = concurrency
        self.tasks: Dict[str, asyncio.Task] = {}
        self.task_meta: Dict[str, dict] = {}
        self.created_at: float = time.time()
        self.completed_at: Optional[float] = None

    def stop_session(self):
        for trace in self.task_meta.keys():
            send_sse_stop_signal(self.session_id)
        task_ids_to_stop = list(self.tasks.keys())
        for task_id in task_ids_to_stop:
            self._stop_task(task_id)
        self.completed_at = time.time()

    def _stop_task(self, task_id: str):
        task = self.tasks.get(task_id)
        self.running_tasks -= 1
        self.task_meta.pop(task_id, None)
        if task and not task.done():
            task.cancel()
        self.tasks.pop(task_id, None)

    def end_task(self, task_id: str):
        task = self.tasks.get(task_id)
        self.running_tasks -= 1
        self.task_meta[task_id].update({'end_time': time.time()})
        if self.running_tasks <= 0:
            self.completed_at = time.time()
            send_sse_stop_signal(self.session_id)

    def create_task(self, work: callable, request, index, task_id: str):
        coro = work()
        task = asyncio.create_task(coro)
        self.tasks[task_id] = task
        self.task_meta[task_id] = {
            "session_id": self.session_id,
            "user_token": self.user_token,
            "start_time": time.time(),
            "index": index,
            "request": request
        }


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.user_sessions: Dict[str, Set[str]] = {}

    def cleanup_user_sessions(self, user_id: str):
        session_ids = self.user_sessions.pop(user_id, set())
        for sid in session_ids:
            session = self.sessions.pop(sid, None)
            if session:
                session.stop_session()

    def create_session(self, user_token: str, concurrency: int = 1) -> Session:
        session_id = f"sess_{uuid.uuid4().hex}"
        session = Session(
            session_id=session_id,
            user_token=user_token,
            concurrency=concurrency
        )
        self.sessions[session_id] = session
        if session.user_token not in self.user_sessions:
            self.user_sessions[session.user_token] = set()
        self.user_sessions[session.user_token].add(session_id)
        return session

    def get_session_by_session_id(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def stop_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].stop_session()
            del self.sessions[session_id]

    def get_system_stats(self) -> dict:
        now = time.time()
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
            "tasks_per_user": user_task_count,
            "warning": warning,
            "warning_threshold": warning_threshold,
            "timestamp": now
        }


sessionManager = SessionManager()