import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from fusion_mem.embeddings.configs import EmbedderConfig
from fusion_mem.graphs.configs import GraphStoreConfig
from fusion_mem.llms.configs import LlmConfig
from fusion_mem.vector_stores.configs import VectorStoreConfig

# Set up the directory path
home_dir = os.path.expanduser("~")
mem_dir = os.path.join(home_dir, ".fusion_mem")


class MemoryItem(BaseModel):
    id: str = Field(..., description="The unique identifier for the text data")
    session_id: str = Field(..., description="The unique identifier for the session")
    memory: str = Field(
        ..., description="The fusion_mem deduced from the text data"
    )  # TODO After prompt changes from platform, update this
    hash: Optional[str] = Field(None, description="The hash of the fusion_mem")
    # The metadata value can be anything and not just string. Fix it
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the text data")
    score: Optional[float] = Field(None, description="The score associated with the text data")
    created_at: Optional[str] = Field(None, description="The timestamp when the fusion_mem was created")
    updated_at: Optional[str] = Field(None, description="The timestamp when the fusion_mem was updated")


class MemoryConfig(BaseModel):

    version: str = Field(
        description="The version of the API",
        default="v1.1",
    )
    custom_fact_extraction_prompt: Optional[str] = Field(
        description="Custom prompt for the fact extraction",
        default=None,
    )
    custom_update_memory_prompt: Optional[str] = Field(
        description="Custom prompt for the update fusion_mem",
        default=None,
    )

class ShortMemoryConfig(BaseModel):
    short_mem_transfer_num: int = Field(
        description="The number of short fusion_mem records transfer to middle",
        default=5,
    )
    short_mem_max_capacity_per_user: int = Field(
        description="The capacity of short fusion_mem per user",
        default=10,
    )
    collection_name: Optional[str] = Field(
        description="The vectordb collection name of short fusion_mem",
        default="short_memory",
    )

class MiddleMemoryConfig(BaseModel):
    collection_name: Optional[str] = Field(
        description="The vectordb collection name of middle fusion_mem",
        default="middle_memory",
    )
    search_limit: int = Field(
        description="The search limit number of vectordb",
        default=10,
    )

class LongMemoryConfig(MemoryConfig):
    collection_name: Optional[str] = Field(
        description="The vectordb collection name of middle fusion_mem",
        default="long_memory",
    )
    search_limit: int = Field(
        description="The search limit number of vectordb",
        default=10,
    )

class SharedMemoryConfig(MemoryConfig):
    collection_name: Optional[str] = Field(
        description="The vectordb collection name of middle fusion_mem",
        default="shared_memory",
    )
    search_limit: int = Field(
        description="The search limit number of vectordb",
        default=10,
    )

class MemoryManagerConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(
        description="Configuration for the vector store",
        default_factory=VectorStoreConfig,
    )
    llm: LlmConfig = Field(
        description="Configuration for the language model",
        default_factory=LlmConfig,
    )
    embedder: EmbedderConfig = Field(
        description="Configuration for the embedding model",
        default_factory=EmbedderConfig,
    )
    sql_db_path: str = Field(
        description="Path to the history database",
        default=os.path.join(mem_dir, "history.db"),
    )
    graph_store: GraphStoreConfig = Field(
        description="Configuration for the graph",
        default_factory=GraphStoreConfig,
    )
    enable_hierarchical_memory: Optional[bool] = Field(
        description="Custom prompt for the update fusion_mem",
        default=True,
    )
    short_memory_config: ShortMemoryConfig = Field(
        description="short_memory_config",
        default=ShortMemoryConfig(),
    )
    middle_memory_config: MiddleMemoryConfig = Field(
        description="middle_memory_config",
        default=MiddleMemoryConfig(),
    )
    long_memory_config: MemoryConfig = Field(
        description="long_memory_config",
        default=LongMemoryConfig(),
    )
    shared_memory_config: MemoryConfig = Field(
        description="shared_memory_config",
        default=SharedMemoryConfig(),
    )


class AzureConfig(BaseModel):
    """
    Configuration settings for Azure.

    Args:
        api_key (str): The API key used for authenticating with the Azure service.
        azure_deployment (str): The name of the Azure deployment.
        azure_endpoint (str): The endpoint URL for the Azure service.
        api_version (str): The version of the Azure API being used.
        default_headers (Dict[str, str]): Headers to include in requests to the Azure API.
    """

    api_key: str = Field(
        description="The API key used for authenticating with the Azure service.",
        default=None,
    )
    azure_deployment: str = Field(description="The name of the Azure deployment.", default=None)
    azure_endpoint: str = Field(description="The endpoint URL for the Azure service.", default=None)
    api_version: str = Field(description="The version of the Azure API being used.", default=None)
    default_headers: Optional[Dict[str, str]] = Field(
        description="Headers to include in requests to the Azure API.", default=None
    )
