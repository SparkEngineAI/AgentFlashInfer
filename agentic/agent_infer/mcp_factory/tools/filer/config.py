import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
import config.config as real_config
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration class for API keys and settings"""

    # Unified LLM Service Provider Configuration
    # Single API key and base URL for all models (OpenAI, Claude, Gemini, etc.)
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None

    # Legacy OpenAI Configuration (for compatibility)
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 4000

    # Legacy Claude Configuration (for compatibility)
    claude_api_key: Optional[str] = None
    claude_base_url: Optional[str] = None
    # Note: openai/ prefix for unified service
    claude_model: str = "openai/claude-3-sonnet-20240229"
    claude_temperature: float = 0.7
    claude_max_tokens: int = 4000

    # Google Search Configuration
    google_search_api_key: Optional[str] = None
    google_search_engine_id: Optional[str] = None
    google_search_results_count: int = 10

    # Jina Reader Configuration
    jina_reader_api_key: Optional[str] = None
    jina_reader_base_url: str = "https://r.jina.ai"

    # Serp API Configuration (alternative search)
    serp_api_key: Optional[str] = None

    # Tavily Search API Configuration
    tavily_api_key: Optional[str] = None

    # HuggingFace Configuration
    huggingface_api_key: Optional[str] = None

    # Pinecone Configuration
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None

    # Redis Configuration (for caching)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None

    # MCP Server Configuration
    mcp_server_url: Optional[str] = None
    mcp_auth_token: Optional[str] = None
    mcp_use_stdio: bool = True  # Default to stdio for backward compatibility

    # General Settings
    debug_mode: bool = False
    max_retries: int = 3
    timeout: int = 30

    def __post_init__(self):
        """Load configuration from environment variables"""
        self.load_from_env()
        self.validate_required_keys()
        self.configure_litellm()

    def load_from_env(self):
        """Load API keys and settings from environment variables"""
        # Unified LLM Service Provider
        self.llm_api_key = os.getenv("LLM_API_KEY")
        self.llm_base_url = os.getenv("LLM_BASE_URL")

        # Legacy OpenAI (fallback to unified if not set)
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or self.llm_api_key
        self.openai_base_url = os.getenv(
            "OPENAI_BASE_URL") or self.llm_base_url
        self.openai_model = os.getenv("OPENAI_MODEL", self.openai_model)
        self.openai_temperature = float(
            os.getenv("OPENAI_TEMPERATURE", self.openai_temperature))
        self.openai_max_tokens = int(
            os.getenv("OPENAI_MAX_TOKENS", self.openai_max_tokens))

        # Legacy Claude (fallback to unified if not set)
        self.claude_api_key = os.getenv("CLAUDE_API_KEY") or os.getenv(
            "ANTHROPIC_API_KEY") or self.llm_api_key
        self.claude_base_url = os.getenv("CLAUDE_BASE_URL") or os.getenv(
            "ANTHROPIC_API_BASE") or self.llm_base_url
        self.claude_model = os.getenv("CLAUDE_MODEL", self.claude_model)
        self.claude_temperature = float(
            os.getenv("CLAUDE_TEMPERATURE", self.claude_temperature))
        self.claude_max_tokens = int(
            os.getenv("CLAUDE_MAX_TOKENS", self.claude_max_tokens))

        # Google Search
        self.google_search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        self.google_search_results_count = int(
            os.getenv("GOOGLE_SEARCH_RESULTS_COUNT", self.google_search_results_count))

        # Jina Reader
        self.jina_reader_api_key = os.getenv("JINA_READER_API_KEY")
        self.jina_reader_base_url = os.getenv(
            "JINA_READER_BASE_URL", self.jina_reader_base_url)

        # Other APIs
        self.serp_api_key = os.getenv("SERP_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

        # Pinecone
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

        # Redis
        self.redis_host = os.getenv("REDIS_HOST", self.redis_host)
        self.redis_port = int(os.getenv("REDIS_PORT", self.redis_port))
        self.redis_password = os.getenv("REDIS_PASSWORD")

        # MCP Server
        host = real_config.MCP_HOST
        port = real_config.MCP_PORT
        self.mcp_server_url = f"http://{host}:{port}/mcp"
        self.mcp_auth_token = os.getenv("MCP_AUTH_TOKEN")
        self.mcp_use_stdio = os.getenv(
            "MCP_USE_STDIO", "true").lower() == "true"

        # General Settings
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.max_retries = int(os.getenv("MAX_RETRIES", self.max_retries))
        self.timeout = int(os.getenv("TIMEOUT", self.timeout))

    def validate_required_keys(self):
        """Validate that required API keys are present"""
        missing_keys = []

        # Check for unified LLM API key or legacy keys
        if not self.llm_api_key and not self.openai_api_key and not self.claude_api_key:
            missing_keys.append(
                "At least one LLM API key (LLM_API_KEY, OPENAI_API_KEY, or CLAUDE_API_KEY)")

        if missing_keys and not self.debug_mode:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
            logger.info("Set DEBUG_MODE=true to skip validation")

    def is_unified_service_provider(self) -> bool:
        """Check if using a unified service provider setup"""
        return bool(self.llm_api_key and self.llm_base_url)

    def get_effective_api_key(self) -> Optional[str]:
        """Get the effective API key (unified or fallback to OpenAI)"""
        return self.llm_api_key or self.openai_api_key

    def get_effective_base_url(self) -> Optional[str]:
        """Get the effective base URL (unified or fallback to OpenAI)"""
        return self.llm_base_url or self.openai_base_url

    def get_available_llm_providers(self) -> list:
        """Get list of available LLM providers based on API keys"""
        providers = []
        if self.openai_api_key:
            providers.append("openai")
        if self.claude_api_key:
            providers.append("claude")
        return providers

    def get_available_search_providers(self) -> list:
        """Get list of available search providers based on API keys"""
        providers = []
        if self.google_search_api_key and self.google_search_engine_id:
            providers.append("google")
        if self.serp_api_key:
            providers.append("serp")
        if self.tavily_api_key:
            providers.append("tavily")
        return providers

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excluding sensitive data)"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if "api_key" in key.lower() or "password" in key.lower():
                config_dict[key] = "***" if value else None
            else:
                config_dict[key] = value
        return config_dict

    def configure_litellm(self):
        """Configure LiteLLM with unified service provider or legacy settings"""
        try:
            import os

            if self.is_unified_service_provider():
                # Unified service provider setup
                effective_key = self.get_effective_api_key()
                effective_url = self.get_effective_base_url()

                # Set unified keys for all model providers
                os.environ["OPENAI_API_KEY"] = effective_key
                os.environ["OPENAI_API_BASE"] = effective_url
                os.environ["ANTHROPIC_API_KEY"] = effective_key
                os.environ["ANTHROPIC_API_BASE"] = effective_url

                logger.info(
                    f"Configured LiteLLM with unified service provider: {effective_url}")
                logger.info(
                    "All models (OpenAI, Claude, etc.) will route through unified endpoint")
                logger.info(
                    "Remember to use 'openai/' prefix for non-OpenAI models in YAML config")

            else:
                # Legacy individual provider setup
                if self.openai_api_key:
                    os.environ["OPENAI_API_KEY"] = self.openai_api_key
                    if self.openai_base_url:
                        os.environ["OPENAI_API_BASE"] = self.openai_base_url
                        logger.info(
                            f"Configured LiteLLM with OpenAI base URL: {self.openai_base_url}")

                if self.claude_api_key:
                    os.environ["ANTHROPIC_API_KEY"] = self.claude_api_key
                    if self.claude_base_url:
                        os.environ["ANTHROPIC_API_BASE"] = self.claude_base_url
                        logger.info(
                            f"Configured LiteLLM with Claude base URL: {self.claude_base_url}")

            # Set verbosity for debugging if needed
            # if self.debug_mode:
            #     litellm.set_verbose = True

            logger.info("LiteLLM configuration completed")

        except ImportError:
            logger.error(
                "LiteLLM package not installed. Install with: pip install litellm")

    def get_openai_client(self):
        """Get configured OpenAI client"""
        if not self.openai_api_key:
            return None

        try:
            from openai import OpenAI

            # Configure LiteLLM for other uses
            self.configure_litellm()

            # Return actual OpenAI client for embeddings and other direct API calls
            client_kwargs = {
                "api_key": self.openai_api_key,
                "timeout": self.timeout
            }

            if self.openai_base_url:
                client_kwargs["base_url"] = self.openai_base_url

            client = OpenAI(**client_kwargs)
            logger.debug(
                f"OpenAI client created successfully with base_url: {client_kwargs.get('base_url', 'default')}")
            return client

        except ImportError:
            logger.error(
                "OpenAI package not installed. Install with: pip install openai")
            return None
        except Exception as e:
            logger.error(f"Failed to create OpenAI client: {e}")
            logger.error(f"Client kwargs: {client_kwargs}")
            return None

    def get_claude_client(self):
        """Get configured Claude client (for backward compatibility)"""
        # This method is kept for backward compatibility but now configures LiteLLM
        self.configure_litellm()
        return True if self.claude_api_key else None


# Global configuration instance
config = APIConfig()


def get_config() -> APIConfig:
    """Get the global configuration instance"""
    return config


def reload_config():
    """Reload configuration from environment variables"""
    global config
    config = APIConfig()
    logger.info("Configuration reloaded")


def validate_api_key(api_key: Optional[str], service_name: str) -> bool:
    """Validate that an API key is present and not empty"""
    if not api_key or api_key.strip() == "":
        logger.error(f"Missing or empty API key for {service_name}")
        return False
    return True


def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI specific configuration"""
    if not validate_api_key(config.openai_api_key, "OpenAI"):
        return {}

    return {
        "api_key": config.openai_api_key,
        "base_url": config.openai_base_url,
        "model": config.openai_model,
        "temperature": config.openai_temperature,
        "max_tokens": config.openai_max_tokens,
        "timeout": config.timeout
    }


def get_claude_config() -> Dict[str, Any]:
    """Get Claude specific configuration"""
    if not validate_api_key(config.claude_api_key, "Claude"):
        return {}

    return {
        "api_key": config.claude_api_key,
        "base_url": config.claude_base_url,
        "model": config.claude_model,
        "temperature": config.claude_temperature,
        "max_tokens": config.claude_max_tokens,
        "timeout": config.timeout
    }


def get_google_search_config() -> Dict[str, Any]:
    """Get Google Search specific configuration"""
    if not validate_api_key(config.google_search_api_key, "Google Search"):
        return {}

    if not config.google_search_engine_id:
        logger.error("Missing Google Search Engine ID")
        return {}

    return {
        "api_key": config.google_search_api_key,
        "engine_id": config.google_search_engine_id,
        "results_count": config.google_search_results_count
    }


def get_jina_reader_config() -> Dict[str, Any]:
    """Get Jina Reader specific configuration"""
    if not validate_api_key(config.jina_reader_api_key, "Jina Reader"):
        return {}

    return {
        "api_key": config.jina_reader_api_key,
        "base_url": config.jina_reader_base_url,
        "timeout": config.timeout
    }


def get_serper_config() -> Dict[str, Any]:
    """Get Serper Search specific configuration"""
    if not validate_api_key(config.serp_api_key, "Serper"):
        return {}

    return {
        "api_key": config.serp_api_key,
        "timeout": config.timeout
    }


def get_mcp_config() -> Dict[str, Any]:
    """Get MCP server specific configuration"""
    return {
        "server_url": config.mcp_server_url,
        "auth_token": config.mcp_auth_token,
        "use_stdio": config.mcp_use_stdio,
        "timeout": config.timeout
    }


# Example usage and testing
if __name__ == "__main__":
    import logging
    logger = logging.getLogger(__name__)
    logger.info("=== Multi Agent System Configuration ===")
    logger.info(f"Debug Mode: {config.debug_mode}")
    logger.info(
        f"Available LLM Providers: {config.get_available_llm_providers()}")
    logger.info(
        f"Available Search Providers: {config.get_available_search_providers()}")
    logger.info("\nConfiguration Summary:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")
