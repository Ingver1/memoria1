"""
Runs a benchmark and fails CI if performance regresses >10 %
vs. the stored baseline (pytest-benchmark handles comparison).
"""
import pytest

import numpy as np
from memory_system.config.settings import UnifiedSettings
from memory_system.core.enhanced_store import EnhancedMemoryStore

DIM = UnifiedSettings.for_testing().model.vector_dim
VECTOR = np.random.rand(DIM).astype("float32").tolist()


@pytest.fixture(scope="session")
async def bench_store():
    s = EnhancedMemoryStore(UnifiedSettings.for_testing())
    for _ in range(5000):
        await s.add_memory(text="bench", embedding=np.random.rand(DIM).tolist())
    yield s
    await s.close()


@pytest.mark.asyncio
def test_semantic_search_speed(benchmark, bench_store):
    async def run():
        await bench_store.semantic_search(vector=VECTOR, k=5)

    benchmark(bench_store.loop.run_until_complete, run)
