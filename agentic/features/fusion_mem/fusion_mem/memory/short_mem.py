import copy
import json
from collections import deque, defaultdict
from typing import Any, Dict, Optional

from fusion_mem.memory.base import MemoryBase
from fusion_mem.configs.base import ShortMemoryConfig
import uuid
from fusion_mem.memory.base import MemoryBase
from fusion_mem.vector_stores.base import VectorStoreMemory
import logging
from fusion_mem.memory.utils import parse_messages

logger = logging.getLogger(__name__)
class ShortTermMemory(MemoryBase):
    def get(self, user_id, session_id, memory_id):
        if user_id not in self.memory or int(memory_id) >= len(self.memory[user_id]):
            return None
        return self.memory[user_id][session_id][memory_id]

    def get_all(self, user_id, session_id):
        return self.retrieve_all(user_id, session_id)

    def update(self, user_id, session_id, memory_id, data):
        if user_id not in self.memory or int(memory_id) >= len(self.memory[user_id]):
            return None
        self.memory[user_id][memory_id] = data

    def delete(self, user_id, session_id, memory_id):
        if user_id not in self.memory or int(memory_id) >= len(self.memory[user_id]):
            return None
        self.memory[user_id][session_id].remove(memory_id)

    def create(self, user_id, session_id, data):
        if user_id not in self.memory:
            self.memory[user_id][session_id] = deque()
        self.memory[user_id][session_id].append(data)

    def history(self, memory_id):
        pass

    def calc_score(self, memory):
        return 1

    def __init__(self, config: ShortMemoryConfig, vector_store_con, embedding_dims):
        self.max_capacity = config.short_mem_max_capacity_per_user
        print("self.max_capacity", self.max_capacity)
        #self.memory = {str:{str: deque}}
        self.memory = defaultdict(lambda: defaultdict(deque))
        self.save_freq = max(self.max_capacity // 2, 5)
        self._record_number = 0
        self.vector_store = vector_store_con
        self.config = config
        self.default_embedding = [0.0] * 512
        self.short_mem_transfer_num = self.config.short_mem_transfer_num if self.config.short_mem_transfer_num else 1
        # self.init_memory_from_vector_store()

    def init_memory_from_vector_store(self):
        memories = self.vector_store.list(limit=None)[0]
        for memory in memories:
            payload = memory.payload
            user_id = memory.id
            #self.memory[user_id]=deque(json.loads(payload.get("memories")))
            print("deque(json.loads(payload.get(memories)))", deque(json.loads(payload.get("memories"))))
            mem_deque = deque(json.loads(payload.get("memories")))
            print("mem_deque", mem_deque)
            session_group = {}
            # 将各个session的
            for item in mem_deque:
                print("item", item)
                # 提取 session_id（优先顶层，缺失则取 metadata 内的）
                session_id = item.get("session_id") or item.get("metadata", {}).get("session_id")
                # 确保 session_id 为字符串（避免类型混乱）
                session_id = str(session_id) if session_id is not None else "unknown"
                # 将当前数据添加到对应 session_id 的分组中
                if session_id not in session_group:
                    session_group[session_id] = [item]
                else:
                    session_group[session_id].append(item)
            for i in range(len(session_group)):
                self.memory[user_id][session_id] = session_group[str(i)]
            print("self.memory[user_id]", self.memory[user_id])

    def add(
            self,
            messages,
            *,
            user_id: Optional[str] = None,
            session_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            run_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            infer: bool = True,
            memory_type: Optional[str] = None,
            prompt: Optional[str] = None,
    ):
        memories = []

        tmp_medata = copy.deepcopy(metadata)
        tmp_medata["data"] = json.dumps(messages)
        tmp_medata["memory_type"] = "short"
        memory = VectorStoreMemory(id=user_id, payload=tmp_medata, score=1)
        memories.append(memory)    

        memory_items = self.prepare_memory_items(memories,  0, [])

        if user_id not in self.memory:
            self.memory[user_id] = {}  # 第一步：初始化用户层为字典
        if session_id not in self.memory[user_id]:
            self.memory[user_id][session_id] = deque()  # 第二步：初始化会话层为 deque
        self.memory[user_id][session_id].extend(memory_items)
        self._record_number += len(messages)
        if self._record_number >= self.save_freq:
            self.save(user_id, session_id)
            self._record_number = 0
        return memory

    def search(
            self,
            query: str,
            *,
            user_id: str,
            session_id: str,
            filters: Optional[Dict[str, Any]] = None,
            threshold: Optional[float] = None,
    ):
        short_memories = self.retrieve_all(user_id, session_id)
        short_memories_final = []
        for each in short_memories.values():
            for short_mem in each:
                dialogs = []
                dialog_data = json.loads(short_mem["metadata"]["data"])
                for each in dialog_data:
                    content = each.get("content", "").strip()
                    if content:  
                        dialogs.append(content)
                full_dialog = " ".join(dialogs)
                short_mem["memory"] = full_dialog
                short_memories_final.append(short_mem)

        #print("short_memories_final", short_memories_final)
        return {"results": short_memories_final}

    def is_full(self, add_messages: list[dict], user_id: str):
        total_size = self.get_current_size(user_id) + len(add_messages)
        if total_size > self.max_capacity:
            return True
        else:
            return False

    def pop_oldest(self, user_id: str, message_number: int = 10):
        if not message_number:
            message_number = self.short_mem_transfer_num
        if user_id not in self.memory:
            return None
        msgs = []
        pop_count = 0
        for session_id, session_queue in self.memory[user_id].items():
            if pop_count >= message_number:
                break
            if len(session_queue) == 0:
                continue
            msg = session_queue.popleft()
            msgs.append(msg)
            pop_count += 1
        return msgs

    def save(self, user_id: str, session_id: str):
        # 使用零向量构造一个虚拟集，仅通过查询id查询到文档信息，将向量数据库当成关系数据库使用
        embeddings = self.default_embedding
        memory_id = user_id
        memory_records = self.retrieve_all(user_id, session_id)
        memory_payloads =  {"memories": json.dumps(list(memory_records))}
        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[memory_payloads],
        )
        return

    def get_current_size(self, user_id: str):
        if user_id not in self.memory:
            return 0
        return len(self.memory[user_id])

    def retrieve_all(self, user_id: str, session_id: str):
        if user_id not in self.memory:
            return []
        if not session_id:
            return self.memory[user_id]
        else:
            return self.memory[user_id][session_id]

    def clear_all(self, user_id: str):
        if user_id not in self.memory:
            return
        self.memory[user_id].clear()
        # self.vector_store.delete(ids=[user_id])

    def delete_all(self,  user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        if agent_id or run_id:
            raise Exception("Short memory not implemented")

        if user_id and user_id in self.memory:
            self.memory[user_id].clear()
            # self.vector_store.delete(ids=[user_id])
        else:
            for user_id in self.memory.keys():
                self.clear_all(user_id)