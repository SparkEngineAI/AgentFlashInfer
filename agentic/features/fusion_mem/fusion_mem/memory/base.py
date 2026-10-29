from abc import ABC, abstractmethod
from fusion_mem.configs.base import MemoryItem
from fusion_mem.vector_stores.base import VectorStoreMemory
from typing import Any, Dict
promoted_payload_keys = [
    "user_id",
    "agent_id",
    "run_id",
    "actor_id",
    "role",
]
DEFAULT_EXCLUDE_KEYS = {"hash", "created_at", "updated_at", "id", *promoted_payload_keys}

class MemoryBase(ABC):
    @abstractmethod
    def get(self, user_id, memory_id) -> MemoryItem:
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        pass

    @abstractmethod
    def get_all(self, user_id) -> list[MemoryItem]:
        """
        List all memories.

        Returns:
            list: List of all memories.
        """
        pass

    @abstractmethod
    def update(self, user_id, memory_id, data:Dict[str,Any]) -> MemoryItem:
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.

        Returns:
            dict: Success message indicating the memory was updated.
        """
        pass

    @abstractmethod
    def create(self, user_id, data:Dict[str,Any]) -> MemoryItem:
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.

        Returns:
            dict: Success message indicating the memory was updated.
        """
        pass

    @abstractmethod
    def delete(self, user_id, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        pass

    @abstractmethod
    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        pass

    @abstractmethod
    def calc_score(self, memory):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        pass

    def prepare_memory_items(self, memories:list[VectorStoreMemory], threshold, exclude_keys=None):

        if exclude_keys is not None:
            core_and_promoted_keys = exclude_keys
        else:
            core_and_promoted_keys = DEFAULT_EXCLUDE_KEYS
        original_memories = []
        for mem in memories:
            score = self.calc_score(mem)
            if threshold and score < threshold:
                continue
            memory_item_dict = self.prepare_memory_item(mem, score, core_and_promoted_keys)
            original_memories.append(memory_item_dict)
        return original_memories

    def prepare_memory_item(self,  mem, score, core_and_promoted_keys=DEFAULT_EXCLUDE_KEYS):
        memory_item_dict = MemoryItem(
            id=mem.id,
            session_id = mem.payload.get("session_id", ""),
            memory=mem.payload.get("data", ""),
            hash=mem.payload.get("hash", ""),
            created_at=mem.payload.get("created_at", ""),
            updated_at=mem.payload.get("updated_at", ""),
            score=score,
        ).model_dump()
        for key in promoted_payload_keys:
            if key in mem.payload:
                memory_item_dict[key] = mem.payload[key]
        self.filter_keys(mem.payload, memory_item_dict, core_and_promoted_keys)
        return memory_item_dict

    def filter_keys(self, payload, memory_item_dict, core_and_promoted_keys=DEFAULT_EXCLUDE_KEYS):
        additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            memory_item_dict["metadata"] = additional_metadata