import importlib
from typing import Dict, Optional, Union

from fusion_mem.configs.embeddings.base import BaseEmbedderConfig
from fusion_mem.configs.llms.base import BaseLlmConfig
from fusion_mem.configs.llms.openai import OpenAIConfig
from fusion_mem.configs.llms.vllm import VllmConfig
from fusion_mem.vector_stores.chroma import get_chroma_client

def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    """
    Factory for creating LLM instances with appropriate configurations.
    Supports both old-style BaseLlmConfig and new provider-specific configs.
    """

    # Provider mappings with their config classes
    provider_to_class = {
        "openai": ("fusion_mem.llms.openai.OpenAILLM", OpenAIConfig),
        "vllm": ("fusion_mem.llms.vllm.VllmLLM", VllmConfig),
    }

    @classmethod
    def create(cls, provider_name: str, config: Optional[Union[BaseLlmConfig, Dict]] = None, **kwargs):
        """
        Create an LLM instance with the appropriate configuration.

        Args:
            provider_name (str): The provider name (e.g., 'openai', 'anthropic')
            config: Configuration object or dict. If None, will create default config
            **kwargs: Additional configuration parameters

        Returns:
            Configured LLM instance

        Raises:
            ValueError: If provider is not supported
        """
        if provider_name not in cls.provider_to_class:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")

        class_type, config_class = cls.provider_to_class[provider_name]
        llm_class = load_class(class_type)

        # Handle configuration
        if config is None:
            # Create default config with kwargs
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            # Merge dict config with kwargs
            config.update(kwargs)
            config = config_class(**config)
        elif isinstance(config, BaseLlmConfig):
            # Convert base config to provider-specific config if needed
            if config_class != BaseLlmConfig:
                # Convert to provider-specific config
                config_dict = {
                    "model": config.model,
                    "temperature": config.temperature,
                    "api_key": config.api_key,
                    "max_tokens": config.max_tokens,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "enable_vision": config.enable_vision,
                    "vision_details": config.vision_details,
                    "http_client_proxies": config.http_client,
                }
                config_dict.update(kwargs)
                config = config_class(**config_dict)
            else:
                # Use base config as-is
                pass
        else:
            # Assume it's already the correct config type
            pass

        return llm_class(config)

    @classmethod
    def register_provider(cls, name: str, class_path: str, config_class=None):
        """
        Register a new provider.

        Args:
            name (str): Provider name
            class_path (str): Full path to LLM class
            config_class: Configuration class for the provider (defaults to BaseLlmConfig)
        """
        if config_class is None:
            config_class = BaseLlmConfig
        cls.provider_to_class[name] = (class_path, config_class)

    @classmethod
    def get_supported_providers(cls) -> list:
        """
        Get list of supported providers.

        Returns:
            list: List of supported provider names
        """
        return list(cls.provider_to_class.keys())


class EmbedderFactory:
    provider_to_class = {
        "openai": "fusion_mem.embeddings.openai.OpenAIEmbedding",
        "ollama": "fusion_mem.embeddings.ollama.OllamaEmbedding",
        "huggingface": "fusion_mem.embeddings.huggingface.HuggingFaceEmbedding",
        "azure_openai": "fusion_mem.embeddings.azure_openai.AzureOpenAIEmbedding",
        "gemini": "fusion_mem.embeddings.gemini.GoogleGenAIEmbedding",
        "vertexai": "fusion_mem.embeddings.vertexai.VertexAIEmbedding",
        "together": "fusion_mem.embeddings.together.TogetherEmbedding",
        "lmstudio": "fusion_mem.embeddings.lmstudio.LMStudioEmbedding",
        "langchain": "fusion_mem.embeddings.langchain.LangchainEmbedding",
        "aws_bedrock": "fusion_mem.embeddings.aws_bedrock.AWSBedrockEmbedding",
    }

    @classmethod
    def create(cls, provider_name, config, vector_config: Optional[dict]):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            embedder_instance = load_class(class_type)
            base_config = BaseEmbedderConfig(**config)
            return embedder_instance(base_config)
        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")


class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "fusion_mem.vector_stores.qdrant.Qdrant",
        "chroma": "fusion_mem.vector_stores.chroma.ChromaDB",
        "pgvector": "fusion_mem.vector_stores.pgvector.PGVector",
        "milvus": "fusion_mem.vector_stores.milvus.MilvusDB",
        "upstash_vector": "fusion_mem.vector_stores.upstash_vector.UpstashVector",
        "azure_ai_search": "fusion_mem.vector_stores.azure_ai_search.AzureAISearch",
        "azure_mysql": "fusion_mem.vector_stores.azure_mysql.AzureMySQL",
        "pinecone": "fusion_mem.vector_stores.pinecone.PineconeDB",
        "mongodb": "fusion_mem.vector_stores.mongodb.MongoDB",
        "redis": "fusion_mem.vector_stores.redis.RedisDB",
        "valkey": "fusion_mem.vector_stores.valkey.ValkeyDB",
        "databricks": "fusion_mem.vector_stores.databricks.Databricks",
        "elasticsearch": "fusion_mem.vector_stores.elasticsearch.ElasticsearchDB",
        "vertex_ai_vector_search": "fusion_mem.vector_stores.vertex_ai_vector_search.GoogleMatchingEngine",
        "opensearch": "fusion_mem.vector_stores.opensearch.OpenSearchDB",
        "supabase": "fusion_mem.vector_stores.supabase.Supabase",
        "weaviate": "fusion_mem.vector_stores.weaviate.Weaviate",
        "faiss": "fusion_mem.vector_stores.faiss.FAISS",
        "langchain": "fusion_mem.vector_stores.langchain.Langchain",
        "s3_vectors": "fusion_mem.vector_stores.s3_vectors.S3Vectors",
        "baidu": "fusion_mem.vector_stores.baidu.BaiduDB",
        "neptune": "fusion_mem.vector_stores.neptune_analytics.NeptuneAnalyticsVector",
    }

    @classmethod
    def get_client(cls,provider_name, config):
        kw_args = {}
        if "host" in config and config["host"] and "port" in config and config["port"]:
            kw_args["host"] = config.get("host")
            kw_args["port"] = config.get("port")
        if "path" in config:
            kw_args["path"] = config.get("path")
        client = None
        if provider_name == "chroma":
            client = get_chroma_client(**kw_args)
        return client

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            vector_store_instance = load_class(class_type)
            return vector_store_instance(**config)
        else:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")

    @classmethod
    def reset(cls, instance):
        instance.reset()
        return instance


class GraphStoreFactory:
    """
    Factory for creating MemoryGraph instances for different graph store providers.
    Usage: GraphStoreFactory.create(provider_name, config)
    """

    provider_to_class = {
        "default": "fusion_mem.fusion_mem.graph_memory.MemoryGraph",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name, cls.provider_to_class["default"])
        try:
            GraphClass = load_class(class_type)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import MemoryGraph for provider '{provider_name}': {e}")
        return GraphClass(config)
