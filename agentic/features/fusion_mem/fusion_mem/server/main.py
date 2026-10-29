import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from scipy.io.tests.test_fortran import DATA_PATH

from fusion_mem import MemoryManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
#
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_URL = os.environ.get("OPENAI_API_URL")
OPENAI_API_MODEL = os.environ.get("OPENAI_API_MODEL")

EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY")
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL")
EMBEDDING_API_MODEL = os.environ.get("EMBEDDING_API_MODEL")
EMBEDDING_MODEL_DIM = int(os.environ.get("EMBEDDING_MODEL_DIM", 1024))

MEMORY_DATA_PATH = os.environ.get("DATA_PATH", "/tmp/fusion_mem")
# HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")

DEFAULT_CONFIG = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "path": f"{MEMORY_DATA_PATH}/working_memory/db",
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": OPENAI_API_MODEL,
            # "model": "qwen3-32b",
            "api_key": OPENAI_API_KEY,
            "openai_base_url": OPENAI_API_URL
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": EMBEDDING_API_MODEL,
            "api_key": EMBEDDING_API_KEY,
            "embedding_dims": EMBEDDING_MODEL_DIM,
            "openai_base_url": EMBEDDING_API_URL,
        }
    },
    "sql_db_path": f"{MEMORY_DATA_PATH}/working_memory/history.db",
}
print(f"Server data save on {MEMORY_DATA_PATH}")
# 确保 MEMORY_DATA_PATH 目录存在
os.makedirs(MEMORY_DATA_PATH, exist_ok=True)

# 创建 working_memory 和 scenario_memory 子目录
working_memory_path = os.path.join(MEMORY_DATA_PATH, "working_memory")
scenario_memory_path = os.path.join(MEMORY_DATA_PATH, "scenario_memory")

os.makedirs(working_memory_path, exist_ok=True)
os.makedirs(scenario_memory_path, exist_ok=True)
# TODO 受限于chroma向量数据库的问题，拆分两个实例通过文件隔离分别存放不同类型记忆，后续可统一
WORKING_MEMORY_INSTANCE = MemoryManager(DEFAULT_CONFIG)
SCENARIO_MEMORY_CONFIG = deepcopy(DEFAULT_CONFIG)
SCENARIO_MEMORY_CONFIG["vector_store"]["config"]["path"] = f"{MEMORY_DATA_PATH}/scenario_memory/db"
SCENARIO_MEMORY_CONFIG["sql_db_path"] = f"{MEMORY_DATA_PATH}/scenario_memory/history.db"
SCENARIO_MEMORY_INSTANCE = MemoryManager(SCENARIO_MEMORY_CONFIG)

app = FastAPI(
    title="FusionMem REST APIs",
    description="A REST API for managing and searching memories for your AI Agents and Apps.",
    version="1.0.0",
)


class Metadata(BaseModel):
    user_id: Optional[str] = Field(None, description="用户ID", example="user123")
    session_id: Optional[str] = Field(None, description="会话ID", example="session789")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    agent_id: Optional[str] = Field(None, description="Agent ID")
    run_id: Optional[str] = None


class Message(BaseModel):
    role: Optional[str] = Field(None, description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(Metadata):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: str = Field(..., description="User ID.")


class SearchRequest(Metadata):
    query: str = Field(..., description="Search query.")
    filters: Optional[Dict[str, Any]] = None


@app.post("/configure", summary="Configure FusionMem")
def set_config(config: Dict[str, Any]):
    """Set fusion_mem configuration."""
    global WORKING_MEMORY_INSTANCE
    WORKING_MEMORY_INSTANCE = MemoryManager(config)
    return {"message": "Configuration set successfully"}


@app.post("/memories/working_memories", summary="Create working memories")
def add_working_memory(memory_create: MemoryCreate):
    """Store new memories."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
    params["user_id"] = memory_create.user_id
    params["session_id"] = memory_create.session_id
    try:
        response = WORKING_MEMORY_INSTANCE.add(messages=[m.model_dump() for m in memory_create.messages], **params)
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")  # This will log the full traceback
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/scenario_memory", summary="Create scenario memories")
def add_scenario_memory(memory_create: MemoryCreate):
    """Store new memories."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
    params["user_id"] = memory_create.user_id
    params["session_id"] = memory_create.session_id
    try:
        response = SCENARIO_MEMORY_INSTANCE.add_scenario_memory(messages=[m.model_dump() for m in memory_create.messages], **params)
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")  # This will log the full traceback
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/working_memories", summary="Search working memories")
def search_working_memories(search_req: SearchRequest):
    """Search for memories based on a query."""
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return WORKING_MEMORY_INSTANCE.search(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/shared_memories", summary="Search shared memories")
def search_shared_memories(search_req: SearchRequest):
    """Search for memories based on a query."""
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return WORKING_MEMORY_INSTANCE.search_shared_memory(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/scenario_memory", summary="Search scenario memories")
def search_scenario_memories(search_req: SearchRequest):
    """Search for memories based on a query."""
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return SCENARIO_MEMORY_INSTANCE.search_scenario_memory(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories/working_memories", summary="Delete all working memories")
def delete_all_memories(
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
):
    """Delete all memories for a given identifier."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        WORKING_MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories/scenario_memory", summary="Delete all scenario memories")
def delete_all_memories(
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
):
    """Delete all memories for a given identifier."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        SCENARIO_MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")


def start_server():
    # 使用 uvicorn 启动服务
    # 默认以 HTTP 方式运行在 8000 端口
    host = os.environ.get("FUSION_MEM_HOST", "127.0.0.1")
    port = int(os.environ.get("FUSION_MEM_PORT", 15002))
    print(f"Starting server on http://{host}:{port}")

    WORKING_MEMORY_INSTANCE.start()
    # SCENARIO_MEMORY_INSTANCE.start() # 情景记忆不需要转化，不用开启
    # HTTP 启动命令:
    uvicorn.run(app,
                host=host,
                port=port,
                #     ssl_keyfile="./key.pem",
                #     ssl_certfile="./cert.pem"
                )


if __name__ == "__main__":
    start_server()
