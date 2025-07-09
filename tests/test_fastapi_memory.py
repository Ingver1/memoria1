import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from memory_system.api.routes.memory import router as memory_router
from memory_system.core.store import SQLiteMemoryStore, lifespan_context


@pytest.fixture
def app(tmp_path):
    app = FastAPI(lifespan=lifespan_context)
    app.include_router(memory_router, prefix="/api/v1/memory")
    return app


def test_startup_injects_store(app):
    with TestClient(app) as client:
        assert isinstance(client.app.state.memory_store, SQLiteMemoryStore)


def test_add_get_search_cycle(app):
    with TestClient(app) as client:
        payload = {"text": "hello world"}
        resp = client.post("/api/v1/memory/", json=payload)
        assert resp.status_code == 201
        mem_id = resp.json()["id"]

        resp = client.get("/api/v1/memory", params={"user_id": None})
        assert resp.status_code == 200

        resp = client.post("/api/v1/memory/search", json={"query": "hello", "top_k": 5})
        assert resp.status_code == 200
        assert any(r["id"] == mem_id for r in resp.json())

        bad = client.post("/api/v1/memory/", json={"text": ""})
        assert bad.status_code == 422
