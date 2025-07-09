from __future__ import annotations

from typing import Any, Callable, Coroutine, TypeVar

__all__ = [
    "FastAPI",
    "APIRouter",
    "Request",
    "HTTPException",
    "Query",
    "Depends",
    "status",
]

F = TypeVar("F", bound=Callable[..., Any])


def _decorator(func: F) -> F:
    return func


class Request:
    app: Any
    headers: dict[str, str]
    client: Any
    url: Any


class FastAPI:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from types import SimpleNamespace

        self.state = SimpleNamespace()
        self.dependency_overrides: dict[Any, Any] = {}

    def add_middleware(self, *args: Any, **kwargs: Any) -> None:
        ...

    def on_event(self, event: str) -> Callable[[F], F]:
        return lambda func: func

    def get(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        return _decorator

    def post(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        return _decorator

    def include_router(self, router: Any) -> None:
        ...


class APIRouter:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def get(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        return _decorator

    def post(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        return _decorator


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail


def Query(default: Any = None, **_: Any) -> Any:
    return default


class Depends:
    def __init__(self, dependency: Callable[..., Any]) -> None:
        self.dependency = dependency


class _Status:
    HTTP_201_CREATED = 201
    HTTP_422_UNPROCESSABLE_ENTITY = 422
    HTTP_500_INTERNAL_SERVER_ERROR = 500
    HTTP_503_SERVICE_UNAVAILABLE = 503
    HTTP_204_NO_CONTENT = 204


status = _Status()
