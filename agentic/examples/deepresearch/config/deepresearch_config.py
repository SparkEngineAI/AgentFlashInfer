import os

# VLLM
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "")
VLLM_MODEL_PATH = os.getenv("VLLM_MODEL_PATH", "")
VLLM_PORTS = [int(p) for p in os.getenv("VLLM_PORTS", "6001").split(",") if p.strip()]

# Agent limits
MAX_LLM_CALL_PER_RUN = int(os.getenv("MAX_LLM_CALL_PER_RUN", "100"))
MAX_TIMEOUT_MINUTES = int(os.getenv("MAX_TIMEOUT_MINUTES", "150"))
MAX_TOKEN_LIMIT = int(os.getenv("MAX_TOKEN_LIMIT", str(110 * 1024)))

# Generation
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.6"))
GENERATION_TOP_P = float(os.getenv("GENERATION_TOP_P", "0.95"))
GENERATION_PRESENCE_PENALTY = float(os.getenv("GENERATION_PRESENCE_PENALTY", "1.1"))
GENERATION_MAX_TOKENS = int(os.getenv("GENERATION_MAX_TOKENS", "10000"))

# Tool API keys
SERPER_KEY = os.getenv("SEARCH_API_KEY", os.getenv("SERPER_KEY", ""))
JINA_API_KEYS = os.getenv("URL_CRAWLER_API_KEYS", os.getenv("JINA_API_KEYS", ""))
SUMMARY_API_KEY = os.getenv("API_KEY", "")
SUMMARY_API_BASE = os.getenv("API_BASE", "")
SUMMARY_MODEL_NAME = os.getenv("SUMMARY_MODEL_NAME", "")
SANDBOX_FUSION_ENDPOINTS = [e for e in os.getenv("SANDBOX_FUSION_ENDPOINT", "").split(",") if e.strip()]

# File parsing
USE_IDP = os.getenv("USE_IDP", "True").lower() in ("true", "1", "yes", "on")
IDP_KEY_ID = os.getenv("IDP_KEY_ID", "")
IDP_KEY_SECRET = os.getenv("IDP_KEY_SECRET", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_API_BASE = os.getenv("DASHSCOPE_API_BASE", "")

# Visit tool
VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", "200"))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", "150000"))
VISIT_SERVER_MAX_RETRIES = int(os.getenv("VISIT_SERVER_MAX_RETRIES", "1"))

# Service
DR_SERVER_PORT = int(os.getenv("DR_SERVER_PORT", "8290"))
DR_WORKSPACE_PATH = os.getenv("DR_WORKSPACE_PATH", "/tmp/deepresearch_workspace")

# Compression
COMPRESSION_ENABLED = os.getenv("COMPRESSION_ENABLED", "false").lower() in ("true", "1", "yes", "on")
COMPRESSION_TOKEN_THRESHOLD = int(os.getenv("COMPRESSION_TOKEN_THRESHOLD", str(80 * 1024)))
COMPRESSION_TRIGGER_TOOLS = [t for t in os.getenv("COMPRESSION_TRIGGER_TOOLS", "search").split(",") if t.strip()]
COMPRESSION_LLM_URL = os.getenv("COMPRESSION_LLM_URL", "")
COMPRESSION_LLM_API_KEY = os.getenv("COMPRESSION_LLM_API_KEY", "")
COMPRESSION_LLM_MODEL_NAME = os.getenv("COMPRESSION_LLM_MODEL_NAME", "pangu_auto")
COMPRESSION_STALENESS_THRESHOLD = int(os.getenv("COMPRESSION_STALENESS_THRESHOLD", "2"))

# Proxy
PROXY_HOST = os.getenv("PROXY_HOST", "")
PROXY_PORT = os.getenv("PROXY_PORT", "")
PROXY_USER = os.getenv("PROXY_USER", "")
PROXY_PASS = os.getenv("PROXY_PASS", "")

# Model routing
MODEL_ROUTING_SWITCH = os.getenv("MODEL_ROUTING_SWITCH", "false").lower() == "true"
SMALL_MODEL_URL = os.getenv("SMALL_MODEL_URL", "")
LARGE_MODEL_URL = os.getenv("LARGE_MODEL_URL", "")
SMALL_MODEL_NAME = os.getenv("SMALL_MODEL_NAME", "")
LARGE_MODEL_NAME = os.getenv("LARGE_MODEL_NAME", "")
SMALL_MODEL_API_KEY = os.getenv("SMALL_MODEL_API_KEY", "EMPTY")
LARGE_MODEL_API_KEY = os.getenv("LARGE_MODEL_API_KEY", "EMPTY")
PROGRESS_CHECK_MIN_ITER = int(os.getenv("PROGRESS_CHECK_MIN_ITER", "3"))
LARGE_ROUNDS = int(os.getenv("LARGE_ROUNDS", "2"))

# early stop
EARLY_STOP_ENABLED = os.getenv("EARLY_STOP_ENABLED", "false").lower() == "true"
EARLY_STOP_MIN_ITER = int(os.getenv("EARLY_STOP_MIN_ITER", "2"))
