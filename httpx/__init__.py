from typing import Any, Optional

from fastapi.testclient import TestClient


class Response:
    def __init__(self, resp: Any) -> None:
        self._resp = resp

    @property
    def status_code(self) -> int:
        return self._resp.status_code

    def json(self) -> Any:
        return self._resp.json()

    @property
    def text(self) -> str:
        return self._resp.text

    @property
    def headers(self) -> Any:
        return self._resp.headers

class AsyncClient:
    def __init__(self, app: Any | None = None, base_url: str = "http://test", timeout: float | None = None) -> None:
        self._app = app
        self._base_url = base_url
        self._client: Optional[TestClient] = None
        self._timeout = timeout

    async def __aenter__(self) -> "AsyncClient":
        self._client = TestClient(self._app, base_url=self._base_url)
        self._client.__enter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client:
            self._client.__exit__(exc_type, exc, tb)

    async def get(self, url: str, **kwargs: Any) -> Response:
        resp = self._client.get(url, **kwargs)
        return Response(resp)

    async def post(self, url: str, **kwargs: Any) -> Response:
        resp = self._client.post(url, **kwargs)
        return Response(resp)
