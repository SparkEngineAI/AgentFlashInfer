import asyncio
import concurrent
import gc
import hashlib
import json
import logging
import os
import uuid
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional
import openai
import time

import pytz
from pydantic import ValidationError

from fusion_mem.configs.base import MemoryConfig, MemoryItem
from fusion_mem.configs.enums import MemoryType
from fusion_mem.configs.prompts import (
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    get_update_memory_messages,
    get_shared_memory_topic_summary_messages,
    get_shared_memory_update_messages,
)
from fusion_mem.exceptions import ValidationError as MemValidationError
from fusion_mem.memory.base import MemoryBase
from fusion_mem.memory.dram_cache_manager import DramCacheManager
from fusion_mem.memory.setup import mem_dir, setup_config
from fusion_mem.memory.storage import SQLiteManager
from fusion_mem.memory.utils import (
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    remove_code_blocks,
)
from fusion_mem.utils.factory import (
    EmbedderFactory,
    GraphStoreFactory,
    LlmFactory,
    VectorStoreFactory,
)

# Suppress SWIG deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

# Initialize logger early for util functions
logger = logging.getLogger(__name__)


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


setup_config()
logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    def create(self, user_id, data: Dict[str, Any]):
        pass

    def calc_score(self, memory):
        return memory.score

    def __init__(self, config: MemoryConfig, embedding_model, vector_store, llm, db=None, graph=None):
        self.config = config
        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.db = db
        self.api_version = self.config.version
        self.enable_graph = False
        if graph:
            self.graph = graph
            self.enable_graph = True
        else:
            self.graph = None
        self.session_topic = DramCacheManager()

    # @classmethod
    # def from_config(cls, config_dict: Dict[str, Any]):
    #     try:
    #         config = cls._process_config(config_dict)
    #         config = MemoryConfig(**config_dict)
    #     except ValidationError as e:
    #         logger.error(f"Configuration validation error: {e}")
    #         raise
    #     return cls(config)

    # @staticmethod
    # def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    #     if "graph_store" in config_dict:
    #         if "vector_store" not in config_dict and "embedder" in config_dict:
    #             config_dict["vector_store"] = {}
    #             config_dict["vector_store"]["config"] = {}
    #             config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
    #                 "embedding_dims"
    #             ]
    #     try:
    #         return config_dict
    #     except ValidationError as e:
    #         logger.error(f"Configuration validation error: {e}")
    #         raise

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
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update, or delete related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.memory_type (str, optional): Type of memory to create. Defaults to None. By default, it creates the short term memories and long term (semantic and episodic) memories. Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`

        Raises:
            MemValidationError: If input validation fails (invalid memory_type, messages format, etc.).
            VectorStoreError: If vector store operations fail.
            GraphStoreError: If graph store operations fail.
            EmbeddingError: If embedding generation fails.
            LLMError: If LLM operations fail.
            DatabaseError: If database operations fail.
        """

        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )
        processed_metadata["memory_type"] = "long"
        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise MemValidationError(
                message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories.",
                error_code="VALIDATION_002",
                details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
                suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories."
            )

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            results["success"] = True
            return results

        # if self.config.llm.config.get("enable_vision"):
        #     messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        # else:
        messages = parse_vision_messages(messages)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self._add_to_vector_store, messages, processed_metadata, effective_filters, infer)
            future2 = executor.submit(self._add_to_graph, messages, effective_filters)

            concurrent.futures.wait([future1, future2])

            vector_store_result = future1.result()
            vector_store_result_mem = vector_store_result[0]
            vector_store_result_status = vector_store_result[1]
            graph_result = future2.result()

        if self.enable_graph:
            return {
                "results": vector_store_result_mem,
                "relations": graph_result,
            }

        return {"results": vector_store_result_mem, "success": vector_store_result_status}

    def _add_to_vector_store(self, messages, metadata, filters, infer):
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                        not isinstance(message_dict, dict)
                        or message_dict.get("role") is None
                        or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = self.embedding_model.embed(msg_content, "add")
                mem_id = self._create_memory(msg_content, msg_embeddings, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories, True

        parsed_messages = parse_messages(messages)

        if self.config.custom_fact_extraction_prompt:
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)

        # for attempt in range(3):
        #     try:
        #         response = self.llm.generate_response(
        #             messages=[
        #                 {"role": "system", "content": system_prompt},
        #                 {"role": "user", "content": user_prompt},
        #             ],
        #             response_format={"type": "json_object"},
        #         )
        #     except openai.RateLimitError:
        #         wait_time = (attempt + 1) * 2  # 2, 4, 6秒
        #         print(f"等待 {wait_time} 秒后重试...")
        #         time.sleep(wait_time)
        # raise Exception("多次重试后仍然失败")

        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        logger.info(f"llm response is: \n {response}")
        print("llm response is:", response)
        try:
            response = remove_code_blocks(response)
            new_retrieved_facts = json.loads(response)["facts"]
        except Exception as e:
            logger.error(f"Error in new_retrieved_facts: {e}")
            return [], False

        if not new_retrieved_facts:
            logger.info("No new facts retrieved from input. Skipping memory update LLM call.")
            return [], True
        retrieved_old_memory = []
        new_message_embeddings = {}
        print("new_retrieved_facts", new_retrieved_facts)
        for new_mem in new_retrieved_facts:
            print("new mem", new_mem)
            if isinstance(new_mem, list):
                continue
            messages_embeddings = self.embedding_model.embed(new_mem, "add")
            new_message_embeddings[new_mem] = messages_embeddings
            existing_memories = self.vector_store.search(
                query=new_mem,
                vectors=messages_embeddings,
                limit=5,
                filters=filters,
            )
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload.get("data", "")})

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logger.info(f"Total existing memories: {len(retrieved_old_memory)}")

        # mapping UUIDs with integers for handling UUID hallucinations
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)
        print("new_retrieved_facts", new_retrieved_facts)
        if new_retrieved_facts:
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
            )

            try:
                response: str = self.llm.generate_response(
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.error(f"Error in new memory actions response: {e}")
                return [], False

            try:
                if not response or not response.strip():
                    logger.warning("Empty response from LLM, no memories to extract")
                    new_memories_with_actions = {}
                else:
                    response = remove_code_blocks(response)
                    new_memories_with_actions = json.loads(response)
            except Exception as e:
                logger.error("response", response)
                logger.error("retrieved_old_memory", retrieved_old_memory)
                logger.error(f"Invalid JSON response: {e}")
                new_memories_with_actions = {}
        else:
            new_memories_with_actions = {}

        returned_memories = []
        print("new_memories_with_actions", new_memories_with_actions)
        try:
            for resp in new_memories_with_actions.get("fusion_mem", []):
                print("resp", resp)
                logger.info(resp)
                try:
                    action_text = resp.get("text")
                    print("action_text", action_text)
                    if not action_text:
                        logger.info("Skipping memory entry because of empty `text` field.")
                        continue

                    event_type = resp.get("event")
                    if event_type == "ADD":
                        memory_id = self._create_memory(
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=deepcopy(metadata),
                        )
                        returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
                    elif event_type == "UPDATE":
                        self._update_memory(
                            memory_id=temp_uuid_mapping[resp.get("id")],
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=deepcopy(metadata),
                        )
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": action_text,
                                "event": event_type,
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif event_type == "DELETE":
                        self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")])
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": action_text,
                                "event": event_type,
                            }
                        )
                    elif event_type == "NONE":
                        # print("NOOP for Memory.")
                        logger.info("NOOP for Memory.")
                except Exception as e:
                    logger.error(f"Error processing memory action: {resp}, Error: {e}")
        except Exception as e:
            logger.error(f"Error iterating new_memories_with_actions: {e}")

        logger.info(f"长期记忆添加完毕{returned_memories}")
        print("长期记忆添加完毕", returned_memories)
        return returned_memories, True

    def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = self.graph.add(data, filters)

        return added_entities

    def add_incremental(
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
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )
        processed_metadata["memory_type"] = "shared"

        # if self.config.llm.config.get("enable_vision"):
        #     messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        # else:

        messages = [m for m in messages if m["role"] != "system"]
        messages = parse_vision_messages(messages)

        # TODO: may need a new function to deal with trajectory logs
        parsed_messages = parse_messages(messages)

        # check whether is the first tool/sub-agent after user input query
        research_topic, research_category = self.get_session_topic(messages, session_id, user_id, agent_id)
        if research_category is None:
            research_category = "N/A"
        processed_metadata['category'] = research_category
        # Update the trajectory to the shared memory
        existing_memories = \
            self.get_all(user_id=user_id, agent_id=agent_id,
                         filters={"session_id": session_id, "category": research_category})["results"]
        if len(existing_memories) > 1:
            logger.warning(f"There are more than one shared memory in a single query.")
        if not existing_memories:
            existing_memory_string = ""
            existing_memory_id = None
        else:
            existing_memory_string = existing_memories[0]["memory"]
            existing_memory_id = existing_memories[0].get("id")

        new_shared_memory = self.summarize_working_mem(existing_memory_string, parsed_messages, 
                                                       research_topic, research_category)

        if not new_shared_memory:
            logger.info("No new shared memory retrieved from input. Skipping memory update.")
            return{"results": [], "success": True}

        try:
            # TODO: this original update function should work, need to double check
            if existing_memory_id:
                self._update_memory(
                    memory_id=existing_memory_id,
                    data=new_shared_memory,
                    existing_embeddings={},
                    metadata=processed_metadata,
                )
            else:
                self._create_memory(new_shared_memory,{}, metadata=processed_metadata)
        except Exception as e:
            logger.error(f"Error processing shared memory update: {new_shared_memory}, Error: {e}")
            return {"results": new_shared_memory, "success": False}
        logger.info(f"共享记忆更新完毕{new_shared_memory}")

        return {"results": new_shared_memory, "success": True}

    def summarize_working_mem(self, existing_memory, parsed_messages, research_topic, research_category):
        function_calling_prompt = get_shared_memory_update_messages(parsed_messages, research_topic, 
                                                                    research_category,
                                                                    existing_memory)
        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": function_calling_prompt},
            ],
            response_format={"type": "json_object"},
        )
        logger.info(f"llm response is: \n {response}")
        response = remove_code_blocks(response)
        try:
            new_shared_memory = json.loads(response)["shared_memory"]
        except Exception as e:
            logger.warning(f"Error parsing JSON response: {e}, response {response}")
            new_shared_memory = ""
        return new_shared_memory

    def get_session_topic(self, messages, session_id: str, user_id: str, agent_id: str):
        user_query = ""
        for msg in messages:
            if msg["role"] == "user":
                user_query = msg["content"]
                break
        research_topic = None
        topics = self.session_topic.query_data({"session_id": session_id, "user_id": user_id, "agent_id": agent_id})
        research_category = None
        if len(topics) >= 1:
            research_topic = topics[0]["topic"]
            research_category = topics[0]["category"]
        if len(topics) > 1:
            logger.warning(f"There are multiple topics in a single query.")
        if user_query:
            research_category = self.extract_query_topic(user_query)
            if topics:
                self.session_topic.update_by_id(topics[0][self.session_topic.ID_COL], {"topic": research_topic})
            else:
                self.session_topic.add_data(
                    {"topic": research_topic, "session_id": session_id, "user_id": user_id, "agent_id": agent_id})

        return research_topic, research_category


    def extract_query_topic(self, user_query):
        # Use LLM to summarise the research topic/domain of the current query
        function_calling_prompt = get_shared_memory_topic_summary_messages(user_query)
        response: str = self.llm.generate_response(
            messages=[{"role": "system", "content": function_calling_prompt}],
            response_format={"type": "json_object"},
        )
        response = remove_code_blocks(response)
        research_category = json.loads(response)["category"]
        return research_category

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    def get_all(
            self,
            *,
            user_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            run_id: Optional[str] = None,
            filters: Optional[Dict[str, Any]] = None,
            limit: int = 100,
    ):
        """
        List all memories.

        Args:
            user_id (str, optional): user id
            agent_id (str, optional): agent id
            run_id (str, optional): run id
            filters (dict, optional): Additional custom key-value filters to apply to the search.
                These are merged with the ID-based scoping filters. For example,
                `filters={"actor_id": "some_user"}`.
            limit (int, optional): The maximum number of memories to return. Defaults to 100.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key,
                  and potentially "relations" if graph store is enabled. For API v1.0,
                  it might return a direct list (see deprecation warning).
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, effective_filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.get_all, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories_result = future_memories.result()
            graph_entities_result = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": all_memories_result, "relations": graph_entities_result}
        return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit):
        memories_result = self.vector_store.list(filters=filters, limit=limit)
        actual_memories = (
            memories_result[0]
            if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0
            else memories_result
        )

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                session_id = mem.payload.get("session_id", ""),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    def search(
            self,
            query: str,
            *,
            user_id: Optional[str] = None,
            session_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            run_id: Optional[str] = None,
            limit: int = 100,
            filters: Optional[Dict[str, Any]] = None,
            threshold: Optional[float] = None,
    ):
        """
        Searches for memories based on a query
        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Filters to apply to the search. Defaults to None..
            threshold (float, optional): Minimum score for a memory to be included in the results. Defaults to None.

        Returns:
            dict: A dictionary containing the search results, typically under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
        """
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, session_id=session_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._search_vector_store, query, effective_filters, limit, threshold)
            future_graph_entities = (
                executor.submit(self.graph.search, query, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            original_memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}

        return {"results": original_memories}

    def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
        embeddings = self.embedding_model.embed(query, "search")
        memories = self.vector_store.search(query=query, vectors=embeddings, limit=limit, filters=filters)
        original_memories = self.prepare_memory_items(memories, threshold)

        # promoted_payload_keys = [
        #     "user_id",
        #     "agent_id",
        #     "run_id",
        #     "actor_id",
        #     "role",
        # ]
        #
        # core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        return original_memories

    def update(self, memory_id, data):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """

        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        # delete all vector memories and reset the collections
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)
        self.vector_store.reset()

        logger.info(f"Deleted {len(memories)} memories")

        if self.enable_graph:
            self.graph.delete_all(filters)

        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        return self.db.get_history(memory_id)

    def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
            procedural_memory = remove_code_blocks(procedural_memory)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        if "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")

        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    def _delete_memory(self, memory_id):
        logger.info(f"Deleting memory with {memory_id=}")
        existing_memory = self.vector_store.get(vector_id=memory_id)
        prev_value = existing_memory.payload.get("data", "")
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )
        return memory_id

    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")

        if hasattr(self.db, "connection") and self.db.connection:
            self.db.connection.execute("DROP TABLE IF EXISTS history")
            self.db.connection.close()

        self.db = SQLiteManager(self.config.sql_db_path)

        if hasattr(self.vector_store, "reset"):
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        else:
            logger.warning("Vector store does not support reset. Skipping.")
            self.vector_store.delete_col()
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")
