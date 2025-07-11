"""Simplified memory management routes used in tests."""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, List

from fastapi import APIRouter, HTTPException, Query, Request, status
from memory_system.api.schemas import MemoryCreate, MemoryQuery, MemoryRead, MemorySearchResult
from memory_system.core.store import Memory, SQLiteMemoryStore, get_memory_store
from memory_system.utils.security import EnhancedPIIFilter

log = logging.getLogger(__name__)
router = APIRouter(tags=["Memory Management"])


async def _store(request: Request) -> SQLiteMemoryStore:
    return get_memory_store(request)


@router.post("/", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
async def create_memory(
    payload: MemoryCreate,
    request: Request,
) -> MemoryRead:
    store = await _store(request)
    pii_filter = EnhancedPIIFilter()
    clean_text, _found, _types = pii_filter.redact(payload.text)
    mem = Memory(
        id=str(uuid.uuid4()),
        text=clean_text,
    metadata={"tags": payload.tags, "role": payload.role, "user_id": payload.user_id},
    )
    await store.add(mem)
    log.info("Created memory %s", mem.id)
    return MemoryRead.model_validate(mem)


@router.get("/", response_model=list[MemoryRead])
async def list_memories(
    request: Request,
    user_id: str | None = Query(None),
) -> List[MemoryRead]:
    store = await _store(request)
    records = await store.search(metadata_filters={"user_id": user_id} if user_id else None)
    return [MemoryRead.model_validate(r) for r in records]


@router.post("/search", response_model=list[MemorySearchResult])
async def search_memories(
    query: MemoryQuery,
    request: Request,
) -> List[MemorySearchResult]:
    if not query.query:
        raise HTTPException(status_code=422, detail="Query must not be empty")
    store = await _store(request)
    results = await store.search(text_query=query.query, limit=query.top_k)
    return [MemorySearchResult.model_validate(r) for r in results]


@router.get("/best", response_model=list[MemoryRead])
async def best_memories(request: Request, limit: int = Query(5, ge=1, le=50)) -> List[MemoryRead]:
    store = await _store(request)
    records = await store.list_recent(n=limit)
    return [MemoryRead.model_validate(r) for r in records]
