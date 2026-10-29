from copy import deepcopy
from datetime import datetime

import pytz

from fusion_mem.memory.base import MemoryBase
from fusion_mem.configs.base import MiddleMemoryConfig, MemoryItem
from typing import Any, Dict, Optional
import uuid
import hashlib
import logging
import json
from fusion_mem.memory.utils import parse_messages

logger = logging.getLogger(__name__)


class MiddleTermMemory(MemoryBase):
    def get(self, user_id, memory_id):
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory or memory.payload.get("user_id", "") != user_id:
            return None
        return self.prepare_memory_item(memory, memory.score)

    def get_all(self, user_id):
        return self.prepare_memory_items(self.vector_store.list(filters={"user_id": user_id})[0], 0)

    def update(self, user_id, memory_id, data):
        pass

    def create(self, user_id, data):
        pass

    def delete(self, user_id, memory_id):
        # TODO 先根据memoryid查询确认信息后，再删除
        self.vector_store.delete(memory_id)

    def history(self, memory_id):
        pass

    def __init__(self, config: MiddleMemoryConfig, vector_store_con, embedding_model):
        self.vector_store = vector_store_con
        self.embedding_model = embedding_model
        self.config = config
        self.search_limit = config.search_limit

    def add(
            self,
            messages,
            *,
            user_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            run_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            infer: bool = True,
            memory_type: Optional[str] = None,
            prompt: Optional[str] = None,
    ):
        for msg in messages:
            if metadata:
                tmp_metadata = deepcopy(metadata)
            elif "metadata" in msg:
                tmp_metadata = msg.pop('metadata')
            else:
                tmp_metadata = {}
            data = ""
            if "memory" in msg:
                data = msg["memory"]
            elif "content" in msg:
                data = msg["content"]
            embeddings = self.embedding_model.embed(data, "add")
            memory_id = str(uuid.uuid4())
            tmp_metadata["data"] = data
            tmp_metadata["memory_type"] = "middle"
            tmp_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
            tmp_metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()
            self.vector_store.insert(
                vectors=[embeddings],
                ids=[memory_id],
                payloads=[tmp_metadata],
            )

    def list_oldest_records(self, limit: int = None) -> list:
        """
        获取指定用户历史最久的N条记录。

        该函数首先使用 ChromaDB 的元数据过滤获取该用户的所有记录，
        然后在客户端按时间戳升序排序，最后返回前 N 条。

        Args:
            limit (int): 要返回的记录数量上限。

        Returns:
            list: 包含记录信息的字典列表，按时间戳从旧到新排序。
                  如果找不到记录，则返回空列表。
        """

        results = self.vector_store.list(limit=None)[0]

        # 在客户端按时间戳（升序）排序
        # a['metadata']['timestamp'] 是关键
        sorted_results = sorted(
            results,
            key=lambda r: r.payload['time_stamp']
        )

        # 返回排序后的前 limit 条记录
        return sorted_results[:limit]

    def calc_score(self, memory):
        # TODO 拓展分数计算公式
        return memory.score

    def search(
            self,
            query: str,
            *,
            user_id: str,
            session_id: str,
            filters: Optional[Dict[str, Any]] = None,
            threshold: Optional[float] = None,
    ):
        embeddings = self.embedding_model.embed(query, "search")
        memories = self.vector_store.search(query=query, vectors=embeddings, limit=self.search_limit, filters=filters)
        original_memories = self.prepare_memory_items(memories, threshold)
        middle_memories_final = []
        for middle_mem in original_memories:
            dialogs = []
            try:
                dialog_data = json.loads(middle_mem["memory"])
                for each in dialog_data:
                    content = each.get("content", "").strip()
                    if content:
                        dialogs.append(content)
            except Exception as e:
                dialogs.append(middle_mem["memory"])

            full_dialog = "\n".join(dialogs)
            middle_mem["memory"] = full_dialog
            # print("full_dialog", full_dialog)
            middle_memories_final.append(middle_mem)
        return {"results": middle_memories_final}

    def delete_all(self, user_id: Optional[str] = None, session_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if session_id:
            filters["session_id"] = session_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        memories = self.vector_store.list(filters=filters)[0]
        #print("delete_all memory", memories)
        for memory in memories:
            self.vector_store.delete(memory.id)
        logger.info(f"Deleted {len(memories)} memories")
        return
