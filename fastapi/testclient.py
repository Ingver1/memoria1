import asyncio
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional

from starlette.responses import JSONResponse, Response

from . import FastAPI, Request


class _TestResponse:
    def __init__(self, result: Response | Any) -> None:
        if isinstance(result, Response):
            self._resp = result
        else:
            self._resp = JSONResponse(result)

    @property
    def status_code(self) -> int:
        return self._resp.status_code

    def json(self) -> Any:
        return self._resp.json()

    @property
    def text(self) -> str:
        if hasattr(self._resp, "content"):
            return str(self._resp.content)
        return ""

    @property
    def headers(self) -> Dict[str, str]:
        return getattr(self._resp, "headers", {})

    def raise_for_status(self) -> None:
        if 400 <= self.status_code:
            raise RuntimeError(f"HTTP {self.status_code}")


class TestClient:
    def __init__(self, app: FastAPI, base_url: str = "http://test") -> None:
        self.app = app
        self.base_url = base_url.rstrip("/")
        self._loop = asyncio.new_event_loop()
        self._lifespan_cm = None

    # Context manager -------------------------------------------------
    def __enter__(self) -> "TestClient":
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._startup())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._loop.run_until_complete(self._shutdown())
        self._loop.close()
        asyncio.set_event_loop(None)

    async def _startup(self) -> None:
        if self.app.lifespan is not None:
            self._lifespan_cm = self.app.lifespan(self.app)
            await self._lifespan_cm.__aenter__()
        for func in self.app.events.get("startup", []):
            result = func()
            if asyncio.iscoroutine(result):
                await result

    async def _shutdown(self) -> None:
        for func in self.app.events.get("shutdown", []):
            result = func()
            if asyncio.iscoroutine(result):
                await result
        if self._lifespan_cm is not None:
            await self._lifespan_cm.__aexit__(None, None, None)

    # Request dispatch ------------------------------------------------
    def _resolve_handler(self, method: str, url: str) -> Optional[Callable]:
        path = url.split("?")[0]
        if path.startswith(self.base_url):
            path = path[len(self.base_url) :]
        for m, p, func in self.app.routes:
            if m == method and p == path:
                return func
        return None

    def _call(self, method: str, url: str, *, json: Any = None, params: Dict[str, Any] | None = None) -> _TestResponse:
        handler = self._resolve_handler(method, url)
        if handler is None:
            return _TestResponse(Response(status_code=404))

        kwargs: Dict[str, Any] = {}
        if params:
            kwargs.update(params)
        body_payload = json
        sig = handler.__signature__  # type: ignore[attr-defined]
        for name, param in sig.parameters.items():
            if name == "request":
                req = Request()
                req.app = self.app
                kwargs[name] = req
            elif name in {"body", "payload", "memories", "query"} and body_payload is not None:
                kwargs[name] = body_payload
            elif name in kwargs:
                continue
            elif param.default is not param.empty:
                kwargs[name] = param.default
            else:
                kwargs[name] = None
        result = handler(**kwargs)
        if asyncio.iscoroutine(result):
            result = self._loop.run_until_complete(result)
        return _TestResponse(result)

    # Public methods --------------------------------------------------
    def get(self, url: str, *, params: Dict[str, Any] | None = None) -> _TestResponse:
        return self._call("GET", url, params=params)

    def post(self, url: str, *, json: Any = None, params: Dict[str, Any] | None = None) -> _TestResponse:
        return self._call("POST", url, json=json, params=params)

    def delete(self, url: str, *, params: Dict[str, Any] | None = None) -> _TestResponse:
        return self._call("DELETE", url, params=params)
