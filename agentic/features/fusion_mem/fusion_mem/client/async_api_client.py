import logging
import warnings
import asyncio
from typing import Any, Dict, List, Optional

import httpx
# 注意：确保原本的 api_error_handler 装饰器支持 async 函数
# 如果它是同步装饰器，你需要修改它以支持 await func()
from fusion_mem.client.utils import api_error_handler

logger = logging.getLogger(__name__)

warnings.filterwarnings("default", category=DeprecationWarning)


class AsyncMemoryClient:
    """Async Client for interacting with the FusionMem API.

    This class provides asynchronous methods to create, retrieve, search, and delete
    memories using the FusionMem API.

    Attributes:
        host (str): The base URL for the FusionMem API.
        client (httpx.AsyncClient): The HTTP client used for making API requests.
    """

    def __init__(
            self,
            host: Optional[str] = None,
            client: Optional[httpx.AsyncClient] = None,
            user_id: Optional[str] = "anonymous_user",
    ):
        """Initialize the Async MemoryClient.

        Args:
            host: The base URL for the FusionMem API.
            client: A custom httpx.AsyncClient instance. If provided, it will be
                    used instead of creating a new one.
        """
        self.host = host
        self.user_id = user_id

        if client is not None:
            self.client = client
            # Ensure the client has the correct base_url and headers
            # Note: Modifying base_url on an existing client is generally safe but check httpx version
            self.client.base_url = httpx.URL(self.host)
            self.client.headers.update(
                {
                    "Mem-User-ID": self.user_id,
                }
            )
        else:
            self.client = httpx.AsyncClient(
                base_url=self.host,
                headers={
                    "Mem-User-ID": self.user_id,
                },
                timeout=300,
            )

    async def __aenter__(self):
        """Support for async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Close the client when exiting context."""
        await self.close()

    async def close(self):
        """Close the underlying httpx client."""
        await self.client.aclose()

    @api_error_handler
    async def add(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Add new working memories asynchronously."""
        kwargs = self._prepare_params(kwargs)
        payload = self._prepare_payload(messages, kwargs)
        payload["user_id"] = self.user_id

        response = await self.client.post("/memories/working_memories", json=payload)
        response.raise_for_status()

        if "metadata" in kwargs:
            del kwargs["metadata"]
        return response.json()

    @api_error_handler
    async def add_scenario_memories(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Add new scenario memories asynchronously."""
        kwargs = self._prepare_params(kwargs)
        payload = self._prepare_payload(messages, kwargs)
        payload["user_id"] = self.user_id

        response = await self.client.post("/memories/scenario_memory", json=payload)
        response.raise_for_status()

        if "metadata" in kwargs:
            del kwargs["metadata"]
        return response.json()

    @api_error_handler
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search working memories based on a query asynchronously."""
        payload = {"query": query, "user_id": self.user_id}
        params = self._prepare_params(kwargs)
        payload.update(params)

        response = await self.client.post("/search/working_memories", json=payload)
        response.raise_for_status()

        if "metadata" in kwargs:
            del kwargs["metadata"]
        return response.json()

    @api_error_handler
    async def search_shared_memories(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search shared memories based on a query asynchronously."""
        payload = {"query": query, "user_id": self.user_id}
        params = self._prepare_params(kwargs)
        payload.update(params)

        response = await self.client.post("/search/shared_memories", json=payload)
        response.raise_for_status()

        if "metadata" in kwargs:
            del kwargs["metadata"]
        return response.json()

    @api_error_handler
    async def search_scenario_memories(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search scenario memories based on a query asynchronously."""
        payload = {"query": query, "user_id": self.user_id}
        params = self._prepare_params(kwargs)
        payload.update(params)

        response = await self.client.post("/search/scenario_memory", json=payload)
        response.raise_for_status()

        if "metadata" in kwargs:
            del kwargs["metadata"]
        return response.json()

    @api_error_handler
    async def get(self, memory_id: str) -> Dict[str, Any]:
        """Retrieve a specific memory by ID asynchronously."""
        params = self._prepare_params()
        response = await self.client.get(f"/memories/{memory_id}/", params=params)
        response.raise_for_status()
        return response.json()

    @api_error_handler
    async def get_all(self, version: str = "v1", **kwargs) -> List[Dict[str, Any]]:
        """Retrieve all memories asynchronously, with optional filtering."""
        params = self._prepare_params(kwargs)

        if "page" in params and "page_size" in params:
            query_params = {
                "page": params.pop("page"),
                "page_size": params.pop("page_size"),
            }
            response = await self.client.post(f"/{version}/memories/", json=params, params=query_params)
        else:
            response = await self.client.post(f"/{version}/memories/", json=params)

        response.raise_for_status()

        if "metadata" in kwargs:
            del kwargs["metadata"]
        return response.json()

    @api_error_handler
    async def update(
            self,
            memory_id: str,
            text: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update a memory by ID asynchronously."""
        if text is None and metadata is None:
            raise ValueError("Either text or metadata must be provided for update.")

        payload = {}
        if text is not None:
            payload["text"] = text
        if metadata is not None:
            payload["metadata"] = metadata

        params = self._prepare_params()
        response = await self.client.put(f"/memories/{memory_id}/", json=payload, params=params)
        response.raise_for_status()
        return response.json()

    @api_error_handler
    async def delete(self, memory_id: str) -> Dict[str, Any]:
        """Delete a specific memory by ID asynchronously."""
        params = self._prepare_params()
        response = await self.client.delete(f"/memories/{memory_id}/", params=params)
        response.raise_for_status()
        return response.json()

    @api_error_handler
    async def delete_all(self, **kwargs) -> Dict[str, str]:
        """Delete all memories asynchronously."""
        params = self._prepare_params(kwargs)
        response = await self.client.delete("/memories/", params=params)
        response.raise_for_status()
        return response.json()

    @api_error_handler
    async def history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Retrieve the history of a specific memory asynchronously."""
        params = self._prepare_params()
        response = await self.client.get(f"/memories/{memory_id}/history/", params=params)
        response.raise_for_status()
        return response.json()

    @api_error_handler
    async def users(self) -> Dict[str, Any]:
        """Get all users, agents, and sessions asynchronously."""
        params = self._prepare_params()
        response = await self.client.get("/entities/", params=params)
        response.raise_for_status()
        return response.json()

    @api_error_handler
    async def delete_users(
            self,
            user_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            app_id: Optional[str] = None,
            run_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Delete specific entities asynchronously."""

        if user_id:
            to_delete = [{"type": "user", "name": user_id}]
        elif agent_id:
            to_delete = [{"type": "agent", "name": agent_id}]
        elif app_id:
            to_delete = [{"type": "app", "name": app_id}]
        elif run_id:
            to_delete = [{"type": "run", "name": run_id}]
        else:
            # Note: This awaits the users() call
            entities = await self.users()
            to_delete = [{"type": entity["type"], "name": entity["name"]} for entity in entities["results"]]

        params = self._prepare_params()

        if not to_delete:
            raise ValueError("No entities to delete")

        # Sequentially await deletion to ensure order and error handling as per original logic.
        # Ideally, one could use asyncio.gather for parallel deletion if the API supports high concurrency.
        for entity in to_delete:
            response = await self.client.delete(f"/v2/entities/{entity['type']}/{entity['name']}/", params=params)
            response.raise_for_status()

        return {
            "message": "Entity deleted successfully."
            if (user_id or agent_id or app_id or run_id)
            else "All users, agents, apps and runs deleted."
        }

    @api_error_handler
    async def reset(self) -> Dict[str, str]:
        """Reset the client by deleting all users and memories asynchronously."""
        await self.delete_users()
        return {"message": "Client reset successful. All users and memories deleted."}

    @api_error_handler
    async def batch_update(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch update memories asynchronously."""
        response = await self.client.put("/batch/", json={"memories": memories})
        response.raise_for_status()
        return response.json()

    @api_error_handler
    async def batch_delete(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch delete memories asynchronously."""
        # httpx.AsyncClient uses .request as well, works the same
        response = await self.client.request("DELETE", "/batch/", json={"memories": memories})
        response.raise_for_status()
        return response.json()

    @api_error_handler
    async def create_memory_export(self, schema: str, **kwargs) -> Dict[str, Any]:
        """Create a memory export asynchronously."""
        response = await self.client.post(
            "/exports/",
            json={"schema": schema, **self._prepare_params(kwargs)},
        )
        response.raise_for_status()
        return response.json()

    @api_error_handler
    async def get_memory_export(self, **kwargs) -> Dict[str, Any]:
        """Get a memory export asynchronously."""
        response = await self.client.post("/exports/get/", json=self._prepare_params(kwargs))
        response.raise_for_status()
        return response.json()

    # --- Helper methods remain synchronous as they are CPU-bound ---

    def _prepare_payload(self, messages: List[Dict[str, str]], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the payload for API requests."""
        payload = {}
        payload["messages"] = messages
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        return payload

    def _prepare_params(self, kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare query parameters for API requests."""
        if kwargs is None:
            kwargs = {}
        return {k: v for k, v in kwargs.items() if v is not None}