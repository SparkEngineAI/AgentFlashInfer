import os
from dotenv import load_dotenv
from typing import Any, Dict


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# --- Non-sensitive, global application constants ---
# MCP config
MCP_HOST = "localhost" # mcp服务ip
# MCP_PORT = 6279 # mcp端口
MCP_PORT = int(os.getenv("MCP_PORT", "6279"))
CLIENT_SESSION_TIMEOUT_SECONDS = 3600

# path
WORKSPACE_PATH = "./workspace"
SWANLAB_LOG_PATH = "./logs"
NLTK_DATA_PATH = os.path.join(WORKSPACE_PATH, "nltk_data")
NLTK_DATA_PATH = os.path.join(WORKSPACE_PATH, "nltk_data")

# tool cache config
REDIS_BIN_PATH = "/usr/local/redis/bin" # redis路径（当前为默认路径）
REDIS_HOST = "localhost" # redis服务ip
REDIS_PORT = 6379 # redis服务端口
REDIS_MAX_MEMORY = 100 # redis可用最大内存（MB）
REDIS_CONF_DIR = os.path.join(os.getcwd(), "redis_file") # redis配置/日志/PID路径
REDIS_DB = 0 # redis默认
TOOL_CACHE_METRICS_PATH = "./toolcache_metrics" # metrics保存路径
REPLAN_INTERVAL = 100 # 重规划检查步数
EXPIRATION_TIME = 3600 # 工具缓存时限（s）

# LLM
model_temperature = 0.3
# Maximum tokens in the response.
model_max_tokens: int = 5*1024
model_request_timeout: int = 180
model_max_tokens: int = 5 * 1024

# conversation_history max_tokens,达到阈值会触发task_done
TOKEN_THRESHOLD = 30*1024

# max_turns
PLANNER_MAX_TURNS = 10
SEEKER_MAX_TURNS = 15
WRITER_MAX_TURNS = 20

# fusion_mem switch
# --- Load sensitive or environment-specific configurations from .env file ---
load_dotenv()
ENABLE_MEMORY: bool = _env_bool("ENABLE_MEMORY")
ENABLE_SHARED_MEMORY: bool = _env_bool("ENABLE_SHARED_MEMORY")

# 在 auto、writing 或 qa 模式间切换
PLANNER_MODE = "auto"

# Available tools for different agents
PLANNER_ALLOWED_TOOL_NAMES = [
    "download_files",
    "document_qa",

    "search_shared_memories",
    "semantic_search",

    "file_read",
    "file_write",
    "str_replace_based_edit_tool",

    "list_workspace",
    "file_find_by_name",

    "think",
    "reflect",
] + (["semantic_search", "search_shared_memories"]
    if ENABLE_MEMORY
    else ["document_qa"])

PLANNER_AUTO_TOOL_NAMES = [
    "assign_multi_subjective_tasks_to_info_seeker",
    "assign_multi_objective_tasks_to_info_seeker",
    "assign_subjective_task_to_writer",
    "writer_subjective_task_done",
    "planner_subjective_task_done",
    "planner_objective_task_done"
]

PLANNER_WRITING_TOOL_NAMES = [
    "assign_multi_subjective_tasks_to_info_seeker",
    "assign_subjective_task_to_writer",
    "writer_subjective_task_done",
    "planner_subjective_task_done"
]

PLANNER_QA_TOOL_NAMES = [
    "assign_multi_objective_tasks_to_info_seeker",
    "planner_objective_task_done"
]

SEEKER_ALLOWED_TOOL_NAMES = (
    [
        "batch_web_search",
        "document_extract",
        "download_files",
        "file_read",
        "file_write",
        "str_replace_based_edit_tool",
        "list_workspace",
        "file_find_by_name",
        "think",
        "reflect",
    ] +
    (
        ["url_crawler","semantic_search", "search_shared_memories"]
        if ENABLE_MEMORY
        else ["url_crawler","document_qa"]
    )
)

SUBJECTIVE_SEEKER_ALLOWED_TOOL_NAMES = SEEKER_ALLOWED_TOOL_NAMES + ["info_seeker_subjective_task_done"]
OBJECTIVE_SEEKER_ALLOWED_TOOL_NAMES = SEEKER_ALLOWED_TOOL_NAMES + ["info_seeker_objective_task_done"]

WRITER_ALLOWED_TOOL_NAMES = [
    "file_read",
    "list_workspace",
    "file_find_by_name",

    "search_result_classifier",
    "section_writer",
    "concat_section_files",
    "think",
    "reflect",
    "writer_subjective_task_done",
]

MODE_TOOL_NAMES = {
    "auto" : PLANNER_AUTO_TOOL_NAMES,
    "writing" : PLANNER_WRITING_TOOL_NAMES,
    "qa" : PLANNER_QA_TOOL_NAMES
}

# model config


# --- Load sensitive or environment-specific configurations from .env file ---
load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXAMPLE_MODEL_NAME = os.getenv("EXAMPLE_MODEL_NAME")
PROXY_CACHE = os.getenv("PROXY_CACHE")

# In routing mode the two endpoint/name pairs are the complete model
# configuration. The bootstrap client uses the small model until the router
# resolves the model for the current turn.
MODEL_ROUTING_SWITCH: bool = _env_bool("MODEL_ROUTING_SWITCH", default=True)
model_routing_switch: bool = MODEL_ROUTING_SWITCH
SMALL_MODEL_URL: str = os.getenv("SMALL_MODEL_URL", "")
LARGE_MODEL_URL: str = os.getenv("LARGE_MODEL_URL", "")
SMALL_MODEL_NAME: str = os.getenv("SMALL_MODEL_NAME", "")
LARGE_MODEL_NAME: str = os.getenv("LARGE_MODEL_NAME", "")
SMALL_MODEL_API_KEY: str = os.getenv("SMALL_MODEL_API_KEY", "EMPTY")
LARGE_MODEL_API_KEY: str = os.getenv("LARGE_MODEL_API_KEY", "EMPTY")

if MODEL_ROUTING_SWITCH:
    DEFAULT_MODEL_URL = SMALL_MODEL_URL
    DEFAULT_MODEL_NAME = SMALL_MODEL_NAME
    DEFAULT_MODEL_API_KEY = SMALL_MODEL_API_KEY
else:
    DEFAULT_MODEL_URL = OPENAI_BASE_URL
    DEFAULT_MODEL_NAME = EXAMPLE_MODEL_NAME
    DEFAULT_MODEL_API_KEY = OPENAI_API_KEY

MEMORY_BASE_URL = os.getenv("MEMORY_BASE_URL", "http://127.0.0.1:15002")

# crawler_config
# URL_CRAWLER_BASE_URL= os.getenv("URL_CRAWLER_BASE_URL")                       # Crawler API endpoint
URL_CRAWLER_API_KEYS = os.getenv("URL_CRAWLER_API_KEYS")  # Crawler API keys
URL_CRAWLER_MAX_TOKENS = 100000
CRAWLER_TIMEOUT = 30

# web_search_config
SEARCH_URL = os.getenv("SEARCH_URL")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
timeout = 30
search_compress_sync_switch: bool = _env_bool("SEARCH_COMPRESS_SYNC_SWITCH")
mcp_hub_switch: bool = False
tool_predict_switch: bool = _env_bool("TOOL_PREDICT_SWITCH")
# compress config
resum_compress_async: bool = True # 特性 异步压缩
early_stop_min_iter: int = 2
early_stop_enabled: bool = True # 特性 早停
search_compress_async_switch: bool = False  # compress search results asynchronously
tool_cache_switch: bool = False
SEARCH_COMPRESS_URL = os.getenv(
    "SEARCH_COMPRESS_URL",
    f"{DEFAULT_MODEL_URL.rstrip('/')}/chat/completions" if DEFAULT_MODEL_URL else None,
)
SEARCH_COMPRESS_API_KEY = os.getenv("SEARCH_COMPRESS_API_KEY", DEFAULT_MODEL_API_KEY)
SEARCH_COMPRESS_MODEL_NAME = os.getenv("SEARCH_COMPRESS_MODEL_NAME", DEFAULT_MODEL_NAME)
SEARCH_COMPRESS_TEMPERATURE = 0.6
SEARCH_COMPRESS_MAX_TOKENS = 8192

chat_templates = {
    "small_model": "{% for message in messages %}"
                   "{% if loop.first and messages[0]['role'] != 'system' %}{{ '<s>[unused9]系统：[unused10]' }}{% endif %}"
                   "{% if message['role'] == 'system' %}{{'<s>[unused9]系统：' + message['content'] + '[unused10]'}}{% endif %}"
                   "{% if message['role'] == 'assistant' %}{{'[unused9]助手：' + message['content'] + '[unused10]'}}{% endif %}"
                   "{% if message['role'] == 'tool' %}{{'[unused9]工具：' + message['content'] + '[unused10]'}}{% endif %}"
                   "{% if message['role'] == 'function' %}{{'[unused9]方法：' + message['content'] + '[unused10]'}}{% endif %}"
                   "{% if message['role'] == 'progress_check' %}"
                   "{{'[unused9]助手：[unused16][unused17][unused11][{\"name\": \"think\", \"arguments\": {\"thought\": \"' + message['content']}}"
                   "{% endif %}"
                   "{% if message['role'] == 'user' %}{{'[unused9]用户：' + message['content'] + '[unused10]'}}{% endif %}"
                   "{% endfor %}{% if add_generation_prompt %}{{ '[unused9]助手：' }}{% endif %}",

    "large_model": "{% for message in messages %}"
                   "{% if loop.first and messages[0]['role'] != 'system' %}{{ '[unused9]系统：[unused10]' }}{% endif %}"
                   "{% if message['role'] == 'system' %}{{ '[unused9]系统：' + message['content'] + '[unused10]' }}{% endif %}"
                   "{% if message['role'] == 'assistant' %}{{'[unused9]助手：' + message['content'] + '[unused10]'}}{% endif %}"
                   "{% if message['role'] == 'tool' %}{{'[unused9]工具：' + message['content'] + '[unused10]'}}{% endif %}"
                   "{% if message['role'] == 'function' %}{{'[unused9]方法：' + message['content'] + '[unused10]'}}{% endif %}"
                   "{% if message['role'] == 'progress_check' %}"
                   "{{'[unused9]助手：[unused16][unused17][unused11][{\"name\": \"think\", \"arguments\": {\"thought\": \"' + message['content']}}"
                   "{% endif %}"
                   "{% if message['role'] == 'user' %}{{'[unused9]用户：' + message['content'] + '[unused10]'}}{% endif %}"
                   "{% endfor %}{% if add_generation_prompt %}{{ '[unused9]助手：' }}{% endif %}"
}

# model routing config
SMALL_MODEL_TEMPLATE: str = chat_templates["small_model"]
LARGE_MODEL_TEMPLATE: str = chat_templates["large_model"]

# --- Validate that all required configurations are present ---
required_configs = (
    {
        "SMALL_MODEL_URL": SMALL_MODEL_URL,
        "SMALL_MODEL_NAME": SMALL_MODEL_NAME,
        "LARGE_MODEL_URL": LARGE_MODEL_URL,
        "LARGE_MODEL_NAME": LARGE_MODEL_NAME,
    }
    if MODEL_ROUTING_SWITCH
    else {
        "OPENAI_BASE_URL": OPENAI_BASE_URL,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "EXAMPLE_MODEL_NAME": EXAMPLE_MODEL_NAME,
    }
)
required_configs.update({
    "SEARCH_URL": SEARCH_URL,
    "SEARCH_API_KEY": SEARCH_API_KEY,
    "URL_CRAWLER_API_KEYS": URL_CRAWLER_API_KEYS,
})

missing_configs = [name for name, value in required_configs.items() if not value]

if missing_configs:
    raise ValueError(
        f"The following required configurations are missing or empty, "
        f"please check your .env file: {', '.join(missing_configs)}"
    )

def download_nltk_data():
    if os.path.exists(NLTK_DATA_PATH):
        return
    os.makedirs(NLTK_DATA_PATH, exist_ok=True)
    import nltk
    import ssl
    nltk.data.path.insert(0, NLTK_DATA_PATH)
    print(f"正在下载NLTK数据到: {NLTK_DATA_PATH} ...")
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    try:
        # 下载分词模型 (推理最常用)
        nltk.download('punkt', download_dir=NLTK_DATA_PATH, quiet=False)
        print("✅ 下载完成")
    except Exception as e:
        print(f"❌ 下载失败: {e}")

if ENABLE_MEMORY:
    download_nltk_data()
    try:
        from fusion_mem.client import MemoryClient
    except ImportError:
        raise ImportError("fusion_mem is not installed. Please install it with `pip install -e .`")


def get_search_compress_config() -> Dict[str, Any]:
    return {
        "search_compress_sync_switch": search_compress_sync_switch,
    }


def get_search_compress_model_config() -> Dict[str, Any]:
    return {
        "url": SEARCH_COMPRESS_URL,
        "api_key": SEARCH_COMPRESS_API_KEY,
        "model_name": SEARCH_COMPRESS_MODEL_NAME,
        "temperature": SEARCH_COMPRESS_TEMPERATURE,
        "max_tokens": SEARCH_COMPRESS_MAX_TOKENS
    }


def get_search_engine_config() -> Dict[str, Any]:
    """Get generic search engine configuration"""
    api_keys = SEARCH_API_KEY
    base_url = SEARCH_URL

    if not api_keys:
        return {}

    # Parse comma-separated API keys for rotation
    api_key_list = [key.strip() for key in api_keys.split(",")] if isinstance(api_keys, str) else [api_keys]

    return {
        "api_keys": api_key_list,
        "base_url": base_url,
        "timeout": timeout
    }



def get_custom_llm_config(self) -> Dict[str, Any]:
    """Get configuration for custom LLM service"""
    return {
        "url": self.model_request_url,
        "aux_url": self.aux_model_request_url,
        "token": self.model_request_token,
        "model": self.model_name,
        "temperature": self.model_temperature,
        "max_tokens": self.model_max_tokens,
        "timeout": self.model_request_timeout,
        "base_url": self.model_request_url  # For backward compatibility with model_config.get('base_url')
    }

trajectory_storage_path: str = "./workspace"
report_output_path: str = "./report"
document_analysis_path: str = "./doc_analysis"
def get_storage_config() -> Dict[str, Any]:
    """Get storage and trajectory configuration"""
    return {
        "trajectory_storage_path": trajectory_storage_path,
        "report_output_path": report_output_path,
        "document_analysis_path": document_analysis_path
    }
