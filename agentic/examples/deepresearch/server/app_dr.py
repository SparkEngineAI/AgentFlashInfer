import asyncio
import json
import os
import uuid
import copy
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from examples.deepresearch.config.deepresearch_config import (
    VLLM_MODEL_NAME, VLLM_PORTS, DR_WORKSPACE_PATH,
    GENERATION_TEMPERATURE, GENERATION_TOP_P, GENERATION_PRESENCE_PENALTY, GENERATION_MAX_TOKENS,
    COMPRESSION_ENABLED, COMPRESSION_TOKEN_THRESHOLD, COMPRESSION_TRIGGER_TOOLS,
    COMPRESSION_LLM_URL, COMPRESSION_LLM_API_KEY, COMPRESSION_LLM_MODEL_NAME,
    COMPRESSION_STALENESS_THRESHOLD,
)
from features.compression.config import CompressionConfig
from examples.deepresearch.tracing.processor import (
    register_session, unregister_session, send_sse_stop_signal, emit_react_event, event_generator,
)
from examples.deepresearch.server.session import SessionManager, sessionManager
from examples.deepresearch.react_agent import MultiTurnReactAgent, _get_next_port


# Workspace setup
workspace_path = Path(DR_WORKSPACE_PATH)
workspace_path.mkdir(exist_ok=True)

# FastAPI app
app_dr = FastAPI(
    title="DeepResearch Agent API",
    version="1.0",
    description="DeepResearch Agent service: submit research questions and stream ReAct progress via SSE"
)
app_dr.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conversation history
CONVERSATION_HISTORY: List[Dict[str, Any]] = []


class ResearchRequest(BaseModel):
    user_input: str
    user_token: str = "default"
    concurrency: int = 1


class HistoryItem(BaseModel):
    task_id: str
    user_input: str
    agent_response: str
    timestamp: float
    completed: bool


async def start_agent(request: ResearchRequest, session_id: str, task_id: str):
    """Run the MultiTurnReactAgent and emit SSE events."""
    try:
        planning_port = _get_next_port()

        data = {
            "item": {
                "question": request.user_input,
                "answer": "",
            },
            "planning_port": planning_port,
        }

        def sse_emitter(event_type: str, event_data: dict):
            emit_react_event(session_id, event_type, event_data)
        
        llm_config = {
            "model": os.getenv("VLLM_MODEL_PATH", ""),
            "generate_cfg": {
                "temperature": GENERATION_TEMPERATURE,
                "top_p": GENERATION_TOP_P,
                "presence_penalty": GENERATION_PRESENCE_PENALTY,
            }
        }
 
        compression_config = None
        if COMPRESSION_ENABLED:
            compression_config = CompressionConfig(
                enabled=True,
                token_threshold=COMPRESSION_TOKEN_THRESHOLD,
                trigger_tool_names=COMPRESSION_TRIGGER_TOOLS,
                compress_llm_url=COMPRESSION_LLM_URL,
                compress_llm_api_key=COMPRESSION_LLM_API_KEY,
                compress_llm_model_name=COMPRESSION_LLM_MODEL_NAME,
                staleness_threshold=COMPRESSION_STALENESS_THRESHOLD,
            )
            print(f"Compression enabled: threshold={COMPRESSION_TOKEN_THRESHOLD}, "
                f"trigger_tools={COMPRESSION_TRIGGER_TOOLS}")
    
        agent = MultiTurnReactAgent(llm=llm_config, compression_config=compression_config)

        result = await agent.async_run(data, VLLM_MODEL_NAME, sse_emitter=sse_emitter)

        # Emit trace end event
        emit_react_event(session_id, "on_trace_end", {
            "termination": result.get("termination", "unknown"),
            "prediction": result.get("prediction", ""),
        })

        # Store in history
        CONVERSATION_HISTORY.append({
            "task_id": session_id,
            "user_input": request.user_input,
            "agent_response": result.get("prediction", ""),
            "timestamp": datetime.utcnow().timestamp(),
            "completed": True
        })

        return result

    except Exception as e:
        emit_react_event(session_id, "on_trace_end", {
            "termination": "error",
            "error": str(e),
        })
        raise e

    finally:
        session = sessionManager.get_session_by_session_id(session_id)
        if session is not None:
            session.end_task(task_id)


@app_dr.post("/api/submit")
async def submit_job(request: ResearchRequest):
    user_token = request.user_token
    input_data = request.user_input

    if not input_data:
        return {"error": "user_input is required"}

    session = sessionManager.create_session(user_token, concurrency=1)
    register_session(session.session_id)

    task_id = f"task_{uuid.uuid4().hex}"

    async def run_in_background():
        await start_agent(request, session.session_id, task_id)

    bound_func = lambda: run_in_background()
    session.create_task(bound_func, request, 0, task_id)

    return {"session_id": session.session_id, "task_id": task_id, "status": "started"}


@app_dr.get("/tracestream/{session_id}")
async def tracestream(session_id: str):
    register_session(session_id)
    return EventSourceResponse(event_generator(session_id))


@app_dr.get("/api/stop/{session_id}")
async def stop(session_id: str):
    sessionManager.stop_session(session_id)
    return {"status": "stopped", "session_id": session_id}


@app_dr.get("/api/stats")
async def stats():
    return sessionManager.get_system_stats()


@app_dr.get("/history", response_model=List[HistoryItem])
async def get_conversation_history(task_id: Optional[str] = None):
    if task_id:
        filtered_history = [
            item for item in CONVERSATION_HISTORY if item["task_id"] == task_id
        ]
        return filtered_history
    return sorted(CONVERSATION_HISTORY, key=lambda x: x["timestamp"], reverse=True)


if __name__ == "__main__":
    import uvicorn
    from examples.deepresearch.config.deepresearch_config import DR_SERVER_PORT
    uvicorn.run(
        "app_dr:app_dr",
        host="0.0.0.0",
        port=DR_SERVER_PORT,
        reload=True
    )
