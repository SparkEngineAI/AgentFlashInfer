import copy
import threading
import time
import json
from copy import deepcopy
from typing import Any, Dict, Optional

from fusion_mem.memory.dram_cache_manager import DramCacheManager
from fusion_mem.memory.long_term_memory import Memory
from fusion_mem.memory.short_mem import ShortTermMemory
from fusion_mem.memory.middle_term_mem import MiddleTermMemory
from collections import deque, defaultdict
from fusion_mem.configs.base import MemoryManagerConfig
from fusion_mem.exceptions import ValidationError as MemValidationError
from fusion_mem.utils.factory import (
    EmbedderFactory,
    GraphStoreFactory,
    LlmFactory,
    VectorStoreFactory,
)
from fusion_mem.memory.storage import SQLiteManager
from datetime import datetime
import asyncio
import pytz
import logging

logger = logging.getLogger(__name__)


def preprocess_metadata(agent_id, metadata, run_id, user_id, session_id):
    if user_id:
        metadata["user_id"] = user_id
    if session_id:
        metadata["session_id"] = session_id
    if agent_id:
        metadata["agent_id"] = agent_id
    if run_id:
        metadata["run_id"] = run_id
    metadata["time_stamp"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()
    # metadata["time_stamp"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

    return metadata


def _build_filters_and_metadata(
        *,  # Enforce keyword-only arguments
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        actor_id: Optional[str] = None,  # For query-time filtering
        input_metadata: Optional[Dict[str, Any]] = None,
        input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` arg → `filters["actor_id"]`
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id (Optional[str]): User identifier, for session scoping.
        agent_id (Optional[str]): Agent identifier, for session scoping.
        run_id (Optional[str]): Run identifier, for session scoping.
        actor_id (Optional[str]): Explicit actor identifier, used as a potential source for
            actor-specific filtering. See actor resolution precedence in the main description.
        input_metadata (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session identifiers for the storage metadata template. Defaults to an empty dict.
        input_filters (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session and actor identifiers for query filters. Defaults to an empty dict.

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - base_metadata_template (Dict[str, Any]): Metadata template for storing memories,
              scoped to the provided session(s).
            - effective_query_filters (Dict[str, Any]): Filters for querying memories,
              scoped to the provided session(s) and potentially a resolved actor.
    """

    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- add all provided session ids ----------
    session_ids_provided = []

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if session_id:
        base_metadata_template["session_id"] = session_id
        effective_query_filters["session_id"] = session_id
        session_ids_provided.append("session_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise MemValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation."
        )

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


# 支持向量数据库
# 作为memory的管理中心，承担记忆分层调度管理、外部记忆激活、agent记忆流转、格式转化等功能
# 所有memory的外部链接的生命周期统一在此管理,TODO 向量数据库抽象有问题,根据collection_name划分实例，需要看怎么抽象
# 特性：
#    1. 分层管理： 使能短中长，不使能退化为纯长期记忆
class MemoryManager:
    def __init__(
            self,
            config_dict: Dict[str, Any],
    ):
        self.config = MemoryManagerConfig(**config_dict)
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )

        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.sql_db_path)
        self.vector_client = VectorStoreFactory.get_client(self.config.vector_store.provider,
                                                           self.config.vector_store.config.model_dump())

        if self.config.graph_store.config:
            provider = self.config.graph_store.provider
            self.graph = GraphStoreFactory.create(provider, self.config)
            self.enable_graph = True
        else:
            self.graph = None
        self.enable_hierarchical_memory = self.config.enable_hierarchical_memory
        self.short_term_mem: ShortTermMemory = None
        self.middle_term_mem: MiddleTermMemory = None
        self.long_term_mem: Memory = None
        self.shared_mem: Memory = None
        self.tmp_memory = DramCacheManager()
        self.init_memory()

    def start(self):
        if self.enable_hierarchical_memory:
            self._stop_event = threading.Event()
            self.processing_thread = threading.Thread(
                target=self.middle_mem_to_long_mem,
                daemon=True  # 设置为守护线程，如果主线程退出，它也会随之退出
            )
            self.processing_thread.start()
            self.shared_mem_thread = threading.Thread(
                target=self.working_mem_to_shared_mem,
                daemon=True  # 设置为守护线程，如果主线程退出，它也会随之退出
            )
            self.shared_mem_thread.start()

    def init_memory(self):
        if self.enable_hierarchical_memory:
            short_mem_vector_store = self.create_vector_store(self.config.short_memory_config.collection_name)
            self.short_term_mem = ShortTermMemory(self.config.short_memory_config, short_mem_vector_store,
                                                  self.embedding_model.config.embedding_dims)

            middle_mem_vector_store = self.create_vector_store(self.config.middle_memory_config.collection_name)
            self.middle_term_mem = MiddleTermMemory(self.config.middle_memory_config, middle_mem_vector_store,
                                                    self.embedding_model)

        long_mem_vector_store = self.create_vector_store(self.config.long_memory_config.collection_name)
        self.long_term_mem = Memory(self.config.long_memory_config, self.embedding_model, long_mem_vector_store,
                                    self.llm, self.db,
                                    self.graph)

        shared_mem_vector_store = self.create_vector_store(self.config.shared_memory_config.collection_name)
        self.shared_mem = Memory(self.config.shared_memory_config, self.embedding_model, shared_mem_vector_store,
                                    self.llm, self.db,
                                    self.graph)

    def create_vector_store(self, collection_name):
        vector_store_config = copy.deepcopy(self.config.vector_store.config)
        vector_store_config.collection_name = collection_name
        vector_store_config.client = self.vector_client
        short_mem_vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, vector_store_config
        )
        return short_mem_vector_store

    def add(
            self,
            messages,
            *,
            user_id: str = None,
            session_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            run_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            infer: bool = True,
            memory_type: Optional[str] = None,
            prompt: Optional[str] = None,
    ):
        messages, metadata = self.preprocess_input(messages, metadata)
        metadata = preprocess_metadata(agent_id, metadata, run_id, user_id, session_id)
        if self.enable_hierarchical_memory:
            if self.short_term_mem.is_full(messages, user_id):
                self.short_mem_to_middle_mem(user_id)
            memory_items = self.short_term_mem.add(messages, user_id=user_id, session_id=session_id, metadata=metadata)
            self.tmp_memory.add_data({"memories":memory_items, "trans_to_long_done": False, "trans_to_shared_done": False, "user_id": user_id, "session_id": session_id})
        else:
            self.long_term_mem.add(messages=messages, user_id=user_id, metadata=metadata)

    def add_scenario_memory(
            self,
            messages,
            *,
            user_id: str = None,
            agent_id: Optional[str] = None,
            session_id: Optional[str] = None,
            run_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            infer: bool = True,
            memory_type: Optional[str] = None,
            prompt: Optional[str] = None,
    ):
        messages, metadata = self.preprocess_input(messages, metadata)
        metadata = preprocess_metadata(agent_id, metadata, run_id, user_id, session_id)
        self.middle_term_mem.add(messages=messages, user_id=user_id, metadata=metadata)

    def preprocess_input(self, messages, metadata):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            raise MemValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )
        if not metadata:
            metadata = {}
        return messages, metadata

    def search_scenario_memory(self, query: str,
                               *,
                               user_id: Optional[str] = None,
                               session_id: Optional[str] = None,
                               agent_id: Optional[str] = None,
                               run_id: Optional[str] = None,
                               limit: int = 100,
                               filters: Optional[Dict[str, Any]] = None,
                               threshold: Optional[float] = None,
                               ):
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, session_id=session_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )
        memories = []
        middle_memories = self.middle_term_mem.search(query, user_id=user_id, session_id=session_id,
                                                      filters=effective_filters,
                                                      threshold=threshold)
        memories.extend(middle_memories["results"])
        return {"results": memories}

    def add_shared_memory(
            self,
            messages,
            *,
            user_id: str = None,
            session_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            run_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            infer: bool = True,
            memory_type: Optional[str] = None,
            prompt: Optional[str] = None,
    ):
        messages, metadata = self.preprocess_input(messages, metadata)
        metadata = preprocess_metadata(agent_id, metadata, run_id, user_id, session_id)
        self.long_term_mem.add_incremental(messages=messages, 
                                           user_id=user_id, session_id=session_id, metadata=metadata)
            
    def search_shared_memory(self, query: str,
                             *,
                             user_id: Optional[str] = None,
                             session_id: Optional[str] = None,
                             agent_id: Optional[str] = None,
                             run_id: Optional[str] = None,
                             limit: int = 100,
                             filters: Optional[Dict[str, Any]] = None,
                             threshold: Optional[float] = None,
                             ):
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, session_id=session_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )
        memories = []
        default_memories = self.shared_mem.search(query, user_id=user_id, session_id=session_id, agent_id=agent_id,
                                                     run_id=run_id,
                                                     filters=effective_filters, limit=limit, threshold=threshold)
        memories.extend(default_memories["results"])
        return {"results": memories}

    def search(self, query: str,
               *,
               user_id: Optional[str] = None,
               session_id: Optional[str] = None,
               agent_id: Optional[str] = None,
               run_id: Optional[str] = None,
               limit: int = 100,
               filters: Optional[Dict[str, Any]] = None,
               threshold: Optional[float] = None,
               ):
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, session_id=session_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )
        memories = []
        if self.enable_hierarchical_memory:
            short_memories = self.short_term_mem.search(query, user_id=user_id, session_id=session_id)
            middle_memories = self.middle_term_mem.search(query, user_id=user_id, session_id=session_id,
                                                          filters=effective_filters,
                                                          threshold=threshold)
            memories.extend(short_memories["results"])
            memories.extend(middle_memories["results"])
        default_memories = self.long_term_mem.search(query, user_id=user_id, session_id=session_id, agent_id=agent_id,
                                                     run_id=run_id,
                                                     filters=effective_filters, limit=limit, threshold=threshold)
        memories.extend(default_memories["results"])
        return {"results": memories}

    def short_mem_to_middle_mem(self, user_id: str):
        msgs = self.short_term_mem.pop_oldest(user_id)
        if not msgs:
            logger.warning(
                "No message to transfer, maybe short_mem_transfer_num %d is lager than short_mem_max_capacity %d" % (
                    self.config.short_memory_config.short_mem_transfer_num,
                    self.config.short_memory_config.short_mem_max_capacity_per_user))
            return
        self.middle_term_mem.add(messages=msgs, user_id=user_id)
        logger.info("transfer %d messages from short term to middle term" % len(msgs))

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        if self.enable_hierarchical_memory:
            self.short_term_mem.delete_all(user_id, agent_id, run_id)
            self.middle_term_mem.delete_all(user_id, agent_id, run_id)

        self.long_term_mem.delete_all(user_id, agent_id, run_id)

        return {"message": "Memories deleted successfully!"}

    def middle_mem_to_long_mem(self):
        while not self._stop_event.is_set():
            now = datetime.now()
            # if 15 <= now.hour or now.hour < 6:
            # 搜索中期记忆，按时间排序

            logger.debug(f"Checking memory at {now}")

            # middle_mem = self.middle_term_mem.list_oldest_records(limit=100)
            memory_cache = self.tmp_memory.query_data({"trans_to_long_done": False}, limit=100)
            middle_mem = [mem["memories"] for mem in memory_cache ]
            if not middle_mem:
                logger.debug("no middle memory found %s" % str(len(middle_mem)))
                time.sleep(10)
                continue
            for mem, m_cache in zip(middle_mem, memory_cache):
                # 按照时间降序逐条添加中期记忆到长期记忆中
                metadata = mem.payload
                user_id = metadata["user_id"]
                metadata["memory_type"] = "long"
                if "data" in metadata and metadata["data"]:
                    messages = json.loads(metadata["data"])
                else:
                    logger.warning("no data found in middle term memory")
                    continue
                logger.debug("messages middle to long", messages)

                res = self.long_term_mem.add(messages, user_id=user_id, metadata=metadata)
                if res["success"]:
                    self.tmp_memory.update_by_id(m_cache[self.tmp_memory.ID_COL],{"trans_to_shared_done": True})
                # 每分钟检查1次，避免占用过多资源
            self.tmp_memory.delete_data({"trans_to_long_done": True,"trans_to_shared_done": True})
            logger.info("transfer middle term to long term %s" % str(len(middle_mem)))
            time.sleep(60)

    def working_mem_to_shared_mem(self):
        while not self._stop_event.is_set():
            now = datetime.now()
            # if 15 <= now.hour or now.hour < 6:
            # 搜索中期记忆，按时间排序

            logger.debug(f"transfer working memory to shared memory at {now}")

            # middle_mem = self.middle_term_mem.list_oldest_records(limit=100)
            memory_cache = self.tmp_memory.query_data({"trans_to_shared_done": False}, limit=100)
            working_mem = [mem["memories"] for mem in memory_cache ]
            if not working_mem:
                logger.debug("no working memory found")
                time.sleep(10)
                continue
            for mem, m_cache in zip(working_mem, memory_cache):
                # 按照时间降序逐条添加中期记忆到长期记忆中
                metadata = mem.payload
                user_id = metadata["user_id"]
                metadata["memory_type"] = "shared"
                if "data" in metadata and metadata["data"]:
                    messages = json.loads(metadata.pop("data"))
                else:
                    logger.warning("no data found in middle term memory")
                    continue
                logger.debug("messages working to shared", messages)

                res = self.shared_mem.add_incremental(messages, user_id=user_id,session_id=metadata["session_id"],agent_id=metadata["agent_id"], metadata=metadata)
                if res["success"]:
                    self.tmp_memory.update_by_id(m_cache[self.tmp_memory.ID_COL],{"trans_to_shared_done": True})
            self.tmp_memory.delete_data({"trans_to_long_done": True,"trans_to_shared_done": True})
            logger.info("transfer working term to shared term %s" % str(len(working_mem)))
            time.sleep(60)

    def remaining_middle_mem_to_long_mem(self):
        middle_mem = self.tmp_memory.query_data({"trans_to_shared_done": False})
        logger.debug("memory nums %d" %len(middle_mem))
        if not middle_mem:
            logger.debug("no middle memory found" )
            return
        for mem in middle_mem:
            # 按照时间降序逐条添加中期记忆到长期记忆中
            metadata = mem.payload
            user_id = metadata["user_id"]
            metadata["memory_type"] = "long"
            if metadata["data"]:
                messages = json.loads(metadata["data"])
            else:
                messages = []
            self.long_term_mem.add(messages, user_id=user_id, metadata=metadata)
            # 每分钟检查1次，避免占用过多资源
        logger.info("transfer middle term to long term %s" % str(len(middle_mem)))

    def stop(self, enable_remaining_middle_mem=False):
        """
        优雅地停止后台线程。
        """
        if not self.enable_hierarchical_memory:
            return
        logger.info("[Main Thread] 🛑 发送停止信号到后台线程...")
        self._stop_event.set()

        # 等待后台线程处理完所有剩余任务并完全退出
        self.processing_thread.join()
        logger.info("[Main Thread] ✅ 后台线程已成功停止。")
        if enable_remaining_middle_mem:
            self.remaining_middle_mem_to_long_mem()

    def get(self, user_id: str, memory_id: str):
        memories = []
        if self.enable_hierarchical_memory:
            short_mem = self.short_term_mem.get(user_id, memory_id)
            if short_mem:
                memories.append(short_mem)
            middle_mem = self.middle_term_mem.get(user_id, memory_id)
            if middle_mem:
                memories.append(middle_mem)
        memories.append(self.long_term_mem.get(user_id, memory_id))
        return {"results": memories}

    def list(self, user_id: str):
        memories = []
        if self.enable_hierarchical_memory:
            short_mem = self.short_term_mem.get_all(user_id)
            if short_mem:
                memories.extend(short_mem["results"])
            middle_mem = self.middle_term_mem.get_all(user_id)
            if middle_mem:
                memories.extend(middle_mem["results"])
        long_mem = self.long_term_mem.get_all(user_id=user_id)
        if long_mem:
            memories.extend(long_mem["results"])
        return {"results": memories}

    def update(self, user_id: str, memory_id: str, update_content: Dict[str, Any]):
        # TODO 接口预留暂未实现o
        if self.enable_hierarchical_memory:
            self.short_term_mem.update(user_id, memory_id, update_content)
            self.middle_term_mem.update(user_id, memory_id, update_content)
        self.long_term_mem.update(user_id, memory_id, update_content)

    def delete(self, user_id: str, memory_id: str):
        # TODO 接口预留暂未实现o
        if self.enable_hierarchical_memory:
            self.short_term_mem.delete(user_id, memory_id)
            self.middle_term_mem.delete(user_id, memory_id)
        self.long_term_mem.delete(user_id, memory_id)

    def create(self, user_id: str, memory_content: Dict[str, Any]):
        # TODO 接口预留暂未实现o
        memory_type = memory_content.get("memory_type")
        if self.enable_hierarchical_memory:
            if memory_type == "middle":
                self.middle_term_mem.create(user_id, memory_content)
            elif memory_type == "long":
                self.long_term_mem.create(user_id, memory_content)
            else:
                self.short_term_mem.create(user_id, memory_content)
        else:
            self.long_term_mem.create(user_id, memory_content)
