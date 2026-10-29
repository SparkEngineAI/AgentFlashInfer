import logging
from typing import Callable, TypeVar, ParamSpec
from fastapi import APIRouter
from fastapi.routing import APIRoute

P = ParamSpec("P")
R = TypeVar("R")

_logger = logging.getLogger(__name__)


def replace_route(router: APIRouter, path: str, method: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator factory that replaces the first registered route on `router`
    matching `path` and HTTP `method` with the decorated endpoint function.

    This preserves all original route metadata (methods, name, response model,
    status code, tags, dependencies, summaries, responses, etc.) and only swaps
    the endpoint implementation. The replacement is done in-place to keep the
    route order unchanged.

    Args:
        router: The APIRouter whose routes will be mutated.
        path: The route path to match (e.g., "/v1/completions").
        method: The HTTP method to match (e.g., "POST").

    Returns:
        A decorator that mutates `router` by replacing the matching route with
        a clone that uses the decorated endpoint. The original callable is
        returned unchanged so it can be stacked with other decorators.
    """

    def _decorator(new_endpoint: Callable[P, R]) -> Callable[P, R]:
        _logger.debug(
            "replace_route: attempting to replace route path=%r method=%r on router=%s (routes=%d)",
            path,
            method,
            hex(id(router)),
            len(router.routes),
        )

        replaced = False
        for i, r in enumerate(list(router.routes)):
            if isinstance(r, APIRoute):
                _logger.debug(
                    "replace_route: inspecting route[%d]: path=%r methods=%s name=%r",
                    i,
                    r.path,
                    sorted(r.methods or []),
                    r.name,
                )
            # Only consider FastAPI APIRoute instances and match both path and method.
            if isinstance(r, APIRoute) and r.path == path and method in r.methods:
                _logger.info(
                    "replace_route: match found at index=%d for path=%r method=%r; " "old_endpoint=%r",
                    i,
                    path,
                    method,
                    getattr(r.endpoint, "__name__", repr(r.endpoint)),
                )
                # Build a new APIRoute that clones all metadata but swaps the endpoint.
                new_route = APIRoute(
                    path=r.path,
                    endpoint=new_endpoint,  # <-- swap to the patched endpoint
                    methods=r.methods,
                    name=r.name,
                    response_model=r.response_model,
                    status_code=r.status_code,
                    tags=r.tags,
                    dependencies=r.dependencies,
                    summary=r.summary,
                    description=r.description,
                    response_description=r.response_description,
                    responses=r.responses,
                    deprecated=r.deprecated,
                    include_in_schema=r.include_in_schema,
                    response_class=r.response_class,
                    callbacks=r.callbacks,
                    openapi_extra=r.openapi_extra,
                )
                router.routes[i] = new_route  # Replace in place to preserve order.
                _logger.info(
                    "replace_route: route replaced successfully at index=%d; new_endpoint=%r",
                    i,
                    getattr(new_endpoint, "__name__", repr(new_endpoint)),
                )
                replaced = True
                break

        if not replaced:
            _logger.warning(
                "replace_route: no matching route found for path=%r method=%r on router=%s",
                path,
                method,
                hex(id(router)),
            )

        # Return the endpoint unchanged so it composes with other decorators.
        return new_endpoint

    return _decorator
