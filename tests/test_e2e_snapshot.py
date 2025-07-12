"""
Adds a memory through REST, retrieves it, asserts round-trip integrity.
Uses TestClient instead of HTTPX to stay in-process and fast.
"""
import numpy as np
from fastapi.testclient import TestClient
from memory_system.api.app import create_app
from memory_system.config.settings import UnifiedSettings

cfg = UnifiedSettings.for_testing()
app = create_app(cfg)
client = TestClient(app)

DIM = cfg.model.vector_dim
VEC = np.random.rand(DIM).astype("float32").tolist()


def test_add_and_get_memory():
    # 1. add
    r = client.post("/memory", json={"text": "e2e-snap", "embedding": VEC})
    assert r.status_code == 200
    memory_id = r.json()["id"]

    # 2. search
    s = client.post("/search", json={"vector": VEC, "k": 1})
    assert s.status_code == 200
    assert s.json()[0]["id"] == memory_id
