"""Memory-related API routes."""

from __future__ import annotations

# ─────────────────────────────── stdlib ────────────────────────────────
import logging
from datetime import UTC, datetime
from typing import Annotated, cast

# ────────────────────────────── third-party ─────────────────────────────
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from memory_system.api.dependencies import get_pii_filter

# ──────────────────────────────── local ────────────────────────────────
from memory_system.api.schemas import (
    MemoryCreate,
    MemoryQuery,
    MemoryRead,
    MemorySearchResult,
)
from memory_system.config.settings import UnifiedSettings
from memory_system.core.embedding import EnhancedEmbeddingService
from memory_system.core.store import EnhancedMemoryStore
from memory_system.memory_helpers import MemoryStoreProtocol, list_best
from memory_system.utils.exceptions import (
    EmbeddingError,
    MemorySystemError,
    StorageError,
    ValidationError,
)
from memory_system.utils.security import EnhancedPIIFilter

log = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["Memory Management"])

Settings = UnifiedSettings  # alias for brevity

# ────────────────────────────────────────────────────────────────────────


@router.post("/", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
async def create_memory(
    payload: MemoryCreate,
    store: EnhancedMemoryStore,
    embedding_service: EnhancedEmbeddingService,
    pii_filter: Annotated[EnhancedPIIFilter, Depends(get_pii_filter)],
) -> MemoryRead:
    """Persist a single memory row and return the stored record."""
    try:
        clean_text, _found, _types = pii_filter.redact(payload.text)
        embedding = await embedding_service.encode([clean_text])
        now = datetime.now(UTC).timestamp()
        mem = await store.add_memory(
            text=clean_text,
            role=payload.role,
            tags=payload.tags,
            importance=0.0,
            valence=payload.valence,
            emotional_intensity=payload.emotional_intensity,
            embedding=embedding[0].tolist(),
            created_at=now,
            updated_at=now,
        )
        log.info("Created memory %s", mem.id)
        return MemoryRead.model_validate(mem)
    except (MemorySystemError, StorageError) as e:  # pragma: no cover
        log.error("Failed to create memory: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ────────────────────────────────────────────────────────────────────────


@router.post("/search", response_model=list[MemorySearchResult])
async def search_memories(
    query: MemoryQuery,
    store: EnhancedMemoryStore,
    embedding_service: EnhancedEmbeddingService,
) -> list[MemorySearchResult]:
    """Search memories using semantic similarity."""
    try:
        query_embedding = await embedding_service.encode([query.query])
        results = await store.semantic_search(
            vector=query_embedding[0].tolist(),
            k=query.top_k,
            include_embeddings=query.include_embeddings,
        )
        log.info("Search query '%s' returned %s results", query.query, len(results))
        return [MemorySearchResult.model_validate(r) for r in results]
    except EmbeddingError as e:  # pragma: no cover
        log.error("Failed to generate query embedding: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to generate query embedding",
        ) from e
    except Exception as e:  # pragma: no cover
        log.error("Failed to search memories: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search memories",
        ) from e


# ────────────────────────────────────────────────────────────────────────


@router.get("/", response_model=list[MemoryRead])
async def list_memories(
    request: Request,
    user_id: str | None = Query(None, description="User ID filter"),
) -> list[MemoryRead]:
    """Return all memories, optionally filtered by user_id."""
    store = cast(EnhancedMemoryStore, request.app.state.store)
    try:
        raw_rows = await store.list_memories(user_id=user_id)
        result = [MemoryRead.model_validate(r) for r in raw_rows]
        log.info("Listed %s memories for user_id=%s", len(result), user_id)
        return result
    except Exception as e:  # pragma: no cover
        log.error("Failed to list memories: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list memories",
        ) from e


# ────────────────────────────────────────────────────────────────────────


@router.post("/batch", response_model=list[MemoryRead])
async def create_memories_batch(
    memories: list[MemoryCreate],
    store: EnhancedMemoryStore,
    embedding_service: EnhancedEmbeddingService,
    pii_filter: Annotated[EnhancedPIIFilter, Depends(get_pii_filter)],
) -> list[MemoryRead]:
    """Create multiple memories in one call (limit 100)."""
    if len(memories) > 100:
        raise ValidationError("Batch size cannot exceed 100 memories")
    try:
        clean_texts = [pii_filter.redact(m.text)[0] for m in memories]
        vectors = await embedding_service.encode(clean_texts)
        now = datetime.now(UTC).timestamp()
        inserted: list[MemoryRead] = []
        for src, vec, clean in zip(memories, vectors, clean_texts, strict=False):
            row = await store.add_memory(
                text=clean,
                role=src.role,
                tags=src.tags,
                importance=0.0,
                valence=src.valence,
                emotional_intensity=src.emotional_intensity,
                embedding=vec.tolist(),
                created_at=now,
                updated_at=now,
            )
            inserted.append(MemoryRead.model_validate(row))
        log.info("Batch-imported %s memories", len(inserted))
        return inserted
    except Exception as e:  # pragma: no cover
        log.error("Failed batch import: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to import batch memories",
        ) from e


# ────────────────────────────────────────────────────────────────────────


@router.get("/best", response_model=list[MemoryRead])
async def get_best_memories(
    store: EnhancedMemoryStore,
    limit: int = Query(5, ge=1, le=50),
) -> list[MemoryRead]:
    """Return top memories ranked by importance and emotion."""
    records = await list_best(limit, store=cast(MemoryStoreProtocol, store))
    return [MemoryRead.model_validate(r) for r in records]


# ────────────────────────────────────────────────────────────────────────


@router.get("/stats", response_model=dict[str, int])
async def get_memory_stats(
    store: EnhancedMemoryStore,
    user_id: str | None = None,
) -> dict[str, int]:
    """Return basic usage statistics."""
    log.info("Stats requested for user_id=%s", user_id)
    stats = await store.get_stats()
    if user_id:
        stats["filter_user"] = user_id
    return stats
