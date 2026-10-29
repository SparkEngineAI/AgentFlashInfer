import uvicorn
from examples.deepresearch.config.deepresearch_config import DR_SERVER_PORT


if __name__ == "__main__":
    uvicorn.run(
        "examples.deepresearch.server.app_dr:app_dr",
        host="0.0.0.0",
        port=DR_SERVER_PORT,
        reload=True,
    )