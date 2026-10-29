import logging
import warnings
from typing import Any, Dict, List, Optional

import httpx
from fusion_mem.client.utils import api_error_handler

logger = logging.getLogger(__name__)

warnings.filterwarnings("default", category=DeprecationWarning)



class MemoryClient:
    """Client for interacting with the FusionMem API.

    This class provides methods to create, retrieve, search, and delete
    memories using the FusionMem API.

    Attributes:
        host (str): The base URL for the FusionMem API.
        client (httpx.Client): The HTTP client used for making API requests.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        user_id: Optional[str] = "anonymous_user",
    ):
        """Initialize the MemoryClient.

        Args:
            host: The base URL for the FusionMem API.
            client: A custom httpx.Client instance. If provided, it will be
                    used instead of creating a new one. Note that base_url and
                    headers will be set/overridden as needed.

        Raises:
            ValueError: If no API key is provided or found in the environment.
        """
        self.host = host
        self.user_id = user_id

        # Create MD5 hash of API key for user_id
        # self.user_id = hashlib.md5(self.api_key.encode()).hexdigest()

        if client is not None:
            self.client = client
            # Ensure the client has the correct base_url and headers
            self.client.base_url = httpx.URL(self.host)
            self.client.headers.update(
                {
                    "Mem-User-ID": self.user_id,
                }
            )
        else:
            self.client = httpx.Client(
                base_url=self.host,
                headers={
                    "Mem-User-ID": self.user_id,
                },
                timeout=300,
            )



    @api_error_handler
    def add(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Add new working memories.

        Args:
            messages: A list of message dictionaries.
            **kwargs: Additional parameters such as user_id, agent_id, app_id,
                      metadata, filters.

        Returns:
            A dictionary containing the API response.

        Raises:
            ValidationError: If the input data is invalid.
            AuthenticationError: If authentication fails.
            RateLimitError: If rate limits are exceeded.
            MemoryQuotaExceededError: If fusion_mem quota is exceeded.
            NetworkError: If network connectivity issues occur.
            MemoryNotFoundError: If the fusion_mem doesn't exist (for updates/deletes).
        """
        kwargs = self._prepare_params(kwargs)
        payload = self._prepare_payload(messages, kwargs)
        payload["user_id"] = self.user_id
        response = self.client.post(f"/memories/working_memories", json=payload)
        response.raise_for_status()
        if "metadata" in kwargs:
            del kwargs["metadata"]
        return response.json()

    @api_error_handler
    def add_scenario_memories(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Add new scenario memories.

        Args:
            messages: A list of message dictionaries.
            **kwargs: Additional parameters such as user_id, agent_id, app_id,
                      metadata, filters.

        Returns:
            A dictionary containing the API response.

        Raises:
            ValidationError: If the input data is invalid.
            AuthenticationError: If authentication fails.
            RateLimitError: If rate limits are exceeded.
            MemoryQuotaExceededError: If fusion_mem quota is exceeded.
            NetworkError: If network connectivity issues occur.
            MemoryNotFoundError: If the fusion_mem doesn't exist (for updates/deletes).
        """
        kwargs = self._prepare_params(kwargs)
        payload = self._prepare_payload(messages, kwargs)
        payload["user_id"] = self.user_id
        response = self.client.post(f"/memories/scenario_memory", json=payload)
        response.raise_for_status()
        if "metadata" in kwargs:
            del kwargs["metadata"]
        return response.json()

    @api_error_handler
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search working memories based on a query.

        Args:
            query: The search query string.
            version: The API version to use for the search endpoint.
            **kwargs: Additional parameters such as user_id, agent_id, app_id,
                      top_k, filters.

        Returns:
            A list of dictionaries containing search results.

        Raises:
            ValidationError: If the input data is invalid.
            AuthenticationError: If authentication fails.
            RateLimitError: If rate limits are exceeded.
            MemoryQuotaExceededError: If fusion_mem quota is exceeded.
            NetworkError: If network connectivity issues occur.
            MemoryNotFoundError: If the fusion_mem doesn't exist (for updates/deletes).
        """
        payload = {"query": query, "user_id": self.user_id}
        params = self._prepare_params(kwargs)
        payload.update(params)
        response = self.client.post(f"/search/working_memories", json=payload)
        response.raise_for_status()
        if "metadata" in kwargs:
            del kwargs["metadata"]
        return response.json()

    @api_error_handler
    def search_shared_memories(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search shared memories based on a query.

        Args:
            query: The search query string.
            version: The API version to use for the search endpoint.
            **kwargs: Additional parameters such as user_id, agent_id, app_id,
                      top_k, filters.

        Returns:
            A list of dictionaries containing search results.

        Raises:
            ValidationError: If the input data is invalid.
            AuthenticationError: If authentication fails.
            RateLimitError: If rate limits are exceeded.
            MemoryQuotaExceededError: If fusion_mem quota is exceeded.
            NetworkError: If network connectivity issues occur.
            MemoryNotFoundError: If the fusion_mem doesn't exist (for updates/deletes).
        """
        payload = {"query": query, "user_id": self.user_id}
        params = self._prepare_params(kwargs)
        payload.update(params)
        response = self.client.post(f"/search/shared_memories", json=payload)
        response.raise_for_status()
        if "metadata" in kwargs:
            del kwargs["metadata"]
        return response.json()

    @api_error_handler
    def search_scenario_memories(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search scenario memories based on a query.

        Args:
            query: The search query string.
            version: The API version to use for the search endpoint.
            **kwargs: Additional parameters such as user_id, agent_id, app_id,
                      top_k, filters.

        Returns:
            A list of dictionaries containing search results.

        Raises:
            ValidationError: If the input data is invalid.
            AuthenticationError: If authentication fails.
            RateLimitError: If rate limits are exceeded.
            MemoryQuotaExceededError: If fusion_mem quota is exceeded.
            NetworkError: If network connectivity issues occur.
            MemoryNotFoundError: If the fusion_mem doesn't exist (for updates/deletes).
        """
        payload = {"query": query, "user_id": self.user_id}
        params = self._prepare_params(kwargs)
        payload.update(params)
        response = self.client.post(f"/search/scenario_memory", json=payload)
        response.raise_for_status()
        if "metadata" in kwargs:
            del kwargs["metadata"]
        return response.json()

    @api_error_handler
    def get(self, memory_id: str) -> Dict[str, Any]:
        """Retrieve a specific fusion_mem by ID.

        Args:
            memory_id: The ID of the fusion_mem to retrieve.

        Returns:
            A dictionary containing the fusion_mem data.

        Raises:
            ValidationError: If the input data is invalid.
            AuthenticationError: If authentication fails.
            RateLimitError: If rate limits are exceeded.
            MemoryQuotaExceededError: If fusion_mem quota is exceeded.
            NetworkError: If network connectivity issues occur.
            MemoryNotFoundError: If the fusion_mem doesn't exist (for updates/deletes).
        """
        params = self._prepare_params()
        response = self.client.get(f"/memories/{memory_id}/", params=params)
        response.raise_for_status()
        return response.json()

    @api_error_handler
    def get_all(self, version: str = "v1", **kwargs) -> List[Dict[str, Any]]:
        """Retrieve all memories, with optional filtering.

        Args:
            version: The API version to use for the search endpoint.
            **kwargs: Optional parameters for filtering (user_id, agent_id,
                      app_id, top_k).

        Returns:
            A list of dictionaries containing memories.

        Raises:
            ValidationError: If the input data is invalid.
            AuthenticationError: If authentication fails.
            RateLimitError: If rate limits are exceeded.
            MemoryQuotaExceededError: If fusion_mem quota is exceeded.
            NetworkError: If network connectivity issues occur.
            MemoryNotFoundError: If the fusion_mem doesn't exist (for updates/deletes).
        """
        params = self._prepare_params(kwargs)
        if "page" in params and "page_size" in params:
            query_params = {
                "page": params.pop("page"),
                "page_size": params.pop("page_size"),
            }
            response = self.client.post(f"/{version}/memories/", json=params, params=query_params)
        else:
            response = self.client.post(f"/{version}/memories/", json=params)
        response.raise_for_status()
        if "metadata" in kwargs:
            del kwargs["metadata"]
        return response.json()

    @api_error_handler
    def update(
        self,
        memory_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update a fusion_mem by ID.
        
        Args:
            memory_id (str): Memory ID.
            text (str, optional): New content to update the fusion_mem with.
            metadata (dict, optional): Metadata to update in the fusion_mem.
            
        Returns:
            Dict[str, Any]: The response from the server.
            
        Example:
            >>> client.update(memory_id="mem_123", text="Likes to play tennis on weekends")
        """
        if text is None and metadata is None:
            raise ValueError("Either text or metadata must be provided for update.")

        payload = {}
        if text is not None:
            payload["text"] = text
        if metadata is not None:
            payload["metadata"] = metadata

        params = self._prepare_params()
        response = self.client.put(f"/memories/{memory_id}/", json=payload, params=params)
        response.raise_for_status()
        return response.json()

    @api_error_handler
    def delete(self, memory_id: str) -> Dict[str, Any]:
        """Delete a specific fusion_mem by ID.

        Args:
            memory_id: The ID of the fusion_mem to delete.

        Returns:
            A dictionary containing the API response.

        Raises:
            ValidationError: If the input data is invalid.
            AuthenticationError: If authentication fails.
            RateLimitError: If rate limits are exceeded.
            MemoryQuotaExceededError: If fusion_mem quota is exceeded.
            NetworkError: If network connectivity issues occur.
            MemoryNotFoundError: If the fusion_mem doesn't exist (for updates/deletes).
        """
        params = self._prepare_params()
        response = self.client.delete(f"/memories/{memory_id}/", params=params)
        response.raise_for_status()
        return response.json()

    @api_error_handler
    def delete_all(self, **kwargs) -> Dict[str, str]:
        """Delete all memories, with optional filtering.

        Args:
            **kwargs: Optional parameters for filtering (user_id, agent_id,
                      app_id).

        Returns:
            A dictionary containing the API response.

        Raises:
            ValidationError: If the input data is invalid.
            AuthenticationError: If authentication fails.
            RateLimitError: If rate limits are exceeded.
            MemoryQuotaExceededError: If fusion_mem quota is exceeded.
            NetworkError: If network connectivity issues occur.
            MemoryNotFoundError: If the fusion_mem doesn't exist (for updates/deletes).
        """
        params = self._prepare_params(kwargs)
        response = self.client.delete("/memories/", params=params)
        response.raise_for_status()
        return response.json()

    @api_error_handler
    def history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Retrieve the history of a specific fusion_mem.

        Args:
            memory_id: The ID of the fusion_mem to retrieve history for.

        Returns:
            A list of dictionaries containing the fusion_mem history.

        Raises:
            ValidationError: If the input data is invalid.
            AuthenticationError: If authentication fails.
            RateLimitError: If rate limits are exceeded.
            MemoryQuotaExceededError: If fusion_mem quota is exceeded.
            NetworkError: If network connectivity issues occur.
            MemoryNotFoundError: If the fusion_mem doesn't exist (for updates/deletes).
        """
        params = self._prepare_params()
        response = self.client.get(f"/memories/{memory_id}/history/", params=params)
        response.raise_for_status()
        return response.json()

    @api_error_handler
    def users(self) -> Dict[str, Any]:
        """Get all users, agents, and sessions for which memories exist."""
        params = self._prepare_params()
        response = self.client.get("/entities/", params=params)
        response.raise_for_status()
        return response.json()

    @api_error_handler
    def delete_users(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Delete specific entities or all entities if no filters provided.

        Args:
            user_id: Optional user ID to delete specific user
            agent_id: Optional agent ID to delete specific agent
            app_id: Optional app ID to delete specific app
            run_id: Optional run ID to delete specific run

        Returns:
            Dict with success message

        Raises:
            ValueError: If specified entity not found
            ValidationError: If the input data is invalid.
            AuthenticationError: If authentication fails.
            MemoryNotFoundError: If the entity doesn't exist.
            NetworkError: If network connectivity issues occur.
        """

        if user_id:
            to_delete = [{"type": "user", "name": user_id}]
        elif agent_id:
            to_delete = [{"type": "agent", "name": agent_id}]
        elif app_id:
            to_delete = [{"type": "app", "name": app_id}]
        elif run_id:
            to_delete = [{"type": "run", "name": run_id}]
        else:
            entities = self.users()
            # Filter entities based on provided IDs using list comprehension
            to_delete = [{"type": entity["type"], "name": entity["name"]} for entity in entities["results"]]

        params = self._prepare_params()

        if not to_delete:
            raise ValueError("No entities to delete")

        # Delete entities and check response immediately
        for entity in to_delete:
            response = self.client.delete(f"/v2/entities/{entity['type']}/{entity['name']}/", params=params)
            response.raise_for_status()

        return {
            "message": "Entity deleted successfully."
            if (user_id or agent_id or app_id or run_id)
            else "All users, agents, apps and runs deleted."
        }

    @api_error_handler
    def reset(self) -> Dict[str, str]:
        """Reset the client by deleting all users and memories.

        This method deletes all users, agents, sessions, and memories
        associated with the client.

        Returns:
            Dict[str, str]: Message client reset successful.

        Raises:
            ValidationError: If the input data is invalid.
            AuthenticationError: If authentication fails.
            RateLimitError: If rate limits are exceeded.
            MemoryQuotaExceededError: If fusion_mem quota is exceeded.
            NetworkError: If network connectivity issues occur.
            MemoryNotFoundError: If the fusion_mem doesn't exist (for updates/deletes).
        """
        self.delete_users()

        return {"message": "Client reset successful. All users and memories deleted."}

    @api_error_handler
    def batch_update(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch update memories.

        Args:
            memories: List of fusion_mem dictionaries to update. Each dictionary must contain:
                - memory_id (str): ID of the fusion_mem to update
                - text (str, optional): New text content for the fusion_mem
                - metadata (dict, optional): New metadata for the fusion_mem

        Returns:
            Dict[str, Any]: The response from the server.

        Raises:
            ValidationError: If the input data is invalid.
            AuthenticationError: If authentication fails.
            RateLimitError: If rate limits are exceeded.
            MemoryQuotaExceededError: If fusion_mem quota is exceeded.
            NetworkError: If network connectivity issues occur.
            MemoryNotFoundError: If the fusion_mem doesn't exist (for updates/deletes).
        """
        response = self.client.put("/batch/", json={"memories": memories})
        response.raise_for_status()

        return response.json()

    @api_error_handler
    def batch_delete(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch delete memories.

        Args:
            memories: List of fusion_mem dictionaries to delete. Each dictionary
                      must contain:
                - memory_id (str): ID of the fusion_mem to delete

        Returns:
            str: Message indicating the success of the batch deletion.

        Raises:
            ValidationError: If the input data is invalid.
            AuthenticationError: If authentication fails.
            RateLimitError: If rate limits are exceeded.
            MemoryQuotaExceededError: If fusion_mem quota is exceeded.
            NetworkError: If network connectivity issues occur.
            MemoryNotFoundError: If the fusion_mem doesn't exist (for updates/deletes).
        """
        response = self.client.request("DELETE", "/batch/", json={"memories": memories})
        response.raise_for_status()

        return response.json()

    @api_error_handler
    def create_memory_export(self, schema: str, **kwargs) -> Dict[str, Any]:
        """Create a fusion_mem export with the provided schema.

        Args:
            schema: JSON schema defining the export structure
            **kwargs: Optional filters like user_id, run_id, etc.

        Returns:
            Dict containing export request ID and status message
        """
        response = self.client.post(
            "/exports/",
            json={"schema": schema, **self._prepare_params(kwargs)},
        )
        response.raise_for_status()
        return response.json()

    @api_error_handler
    def get_memory_export(self, **kwargs) -> Dict[str, Any]:
        """Get a fusion_mem export.

        Args:
            **kwargs: Filters like user_id to get specific export

        Returns:
            Dict containing the exported data
        """
        response = self.client.post("/exports/get/", json=self._prepare_params(kwargs))
        response.raise_for_status()
        return response.json()



    def _prepare_payload(self, messages: List[Dict[str, str]], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the payload for API requests.

        Args:
            messages: The messages to include in the payload.
            kwargs: Additional keyword arguments to include in the payload.

        Returns:
            A dictionary containing the prepared payload.
        """
        payload = {}
        payload["messages"] = messages

        payload.update({k: v for k, v in kwargs.items() if v is not None})
        return payload

    def _prepare_params(self, kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare query parameters for API requests.

        Args:
            kwargs: Keyword arguments to include in the parameters.

        Returns:
            A dictionary containing the prepared parameters.

        Raises:
            ValueError: If either org_id or project_id is provided but not both.
        """

        if kwargs is None:
            kwargs = {}

        return {k: v for k, v in kwargs.items() if v is not None}


