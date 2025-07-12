"""
Ensures DB columns obey NOT NULL + dimension constraints after inserts.
Works with SQLite or SQLCipher.
"""
import pytest

import numpy as np
from memory_system.config.settings import UnifiedSettings
from memory_system.core.enhanced_store import EnhancedMemoryStore

DIM = UnifiedSettings.for_testing().model.vector_dim


@pytest.mark.asyncio
async def test_db_invariants(tmp_path):
    cfg = UnifiedSettings.for_testing()
    cfg.storage.database_url = f"sqlite:///{tmp_path/'inv.db'}"
    store = EnhancedMemoryStore(cfg)

    await store.add_memory(text="inv", embedding=np.random.rand(DIM).tolist())

    # direct SQL query for invariant check
    rows = await store._db.fetch_all("SELECT text, embedding FROM memories")
    text, emb = rows[0]

    # column text must never be NULL
    assert text is not None

    # embedding must always hold `DIM` floats
    assert len(emb) == DIM
