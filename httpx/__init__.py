from fastapi.testclient import TestClient


class Response:
    def __init__(self, resp):
        self._resp = resp

    @property
    def status_code(self):
        return self._resp.status_code

    def json(self):
        return self._resp.json()

    @property
    def text(self):
        return self._resp.text

    @property
    def headers(self):
        return self._resp.headers

class AsyncClient:
    def __init__(self, app=None, base_url="http://test"):
        self._app = app
        self._base_url = base_url
        self._client = None

    async def __aenter__(self):
        self._client = TestClient(self._app, base_url=self._base_url)
        self._client.__enter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client:
            self._client.__exit__(exc_type, exc, tb)

    async def get(self, url, **kwargs):
        resp = self._client.get(url, **kwargs)
        return Response(resp)

    async def post(self, url, **kwargs):
        resp = self._client.post(url, **kwargs)
        return Response(resp)
