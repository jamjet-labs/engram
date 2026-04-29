"""FastAPI HTTP server.

Wire-protocol commitment: paths and JSON shapes match Rust v0.5.x exactly so
existing clients (Spring Boot starter, langchain4j, Java SDK) keep working.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from engram.engram import Engram
from engram.models import ChatMessage, MemoryTier, Polarity
from engram.scope import Scope


class _RecordRequest(BaseModel):
    text: str
    user_id: str = "default"
    org_id: str = "default"
    category: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    polarity: Polarity = Polarity.AFFIRMATIVE
    tier: MemoryTier = MemoryTier.WORKING
    event_date: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class _RecordResponse(BaseModel):
    fact_id: UUID
    text: str


class _SearchRequest(BaseModel):
    query: str
    user_id: str = "default"
    org_id: str = "default"
    top_k: int = Field(default=10, ge=1, le=200)


class _SearchHit(BaseModel):
    fact_id: UUID
    text: str
    score: float
    vector_score: float
    keyword_score: float
    rerank_score: float
    category: str | None
    polarity: Polarity
    event_date: datetime | None


class _SearchResponse(BaseModel):
    hits: list[_SearchHit]


class _MessageRequest(BaseModel):
    content: str
    role: str
    session_id: str
    user_id: str = "default"
    org_id: str = "default"
    timestamp: datetime | None = None


class _ContextRequest(BaseModel):
    query: str
    user_id: str = "default"
    org_id: str = "default"
    token_budget: int = 2000


class _ContextResponse(BaseModel):
    context: str


class _ExtractRequest(BaseModel):
    messages: list[_MessageRequest]
    user_id: str = "default"
    org_id: str = "default"
    session_id: str = "default"
    session_date: datetime | None = None
    persist: bool = True


class _ExtractedFactSummary(BaseModel):
    fact_id: UUID
    text: str
    confidence: float
    category: str | None
    polarity: Polarity


class _ExtractResponse(BaseModel):
    facts: list[_ExtractedFactSummary]


def build_http_app(engram: Engram) -> FastAPI:
    """Build a FastAPI app bound to a single Engram instance."""
    app = FastAPI(title="Engram", version="0.1.0a0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/memory/record", response_model=_RecordResponse)
    async def record(req: _RecordRequest) -> _RecordResponse:
        fact = await engram.record(
            text=req.text,
            user_id=req.user_id,
            org_id=req.org_id,
            category=req.category,
            confidence=req.confidence,
            polarity=req.polarity,
            tier=req.tier,
            event_date=req.event_date,
            metadata=req.metadata,
        )
        return _RecordResponse(fact_id=fact.id, text=fact.text)

    @app.post("/v1/memory/search", response_model=_SearchResponse)
    async def search(req: _SearchRequest) -> _SearchResponse:
        results = await engram.recall(
            query=req.query, user_id=req.user_id, org_id=req.org_id, top_k=req.top_k
        )
        hits = [
            _SearchHit(
                fact_id=sf.fact.id,
                text=sf.fact.text,
                score=sf.score,
                vector_score=sf.vector_score,
                keyword_score=sf.keyword_score,
                rerank_score=sf.rerank_score,
                category=sf.fact.category,
                polarity=sf.fact.polarity,
                event_date=sf.fact.event_date,
            )
            for sf in results
        ]
        return _SearchResponse(hits=hits)

    # Alias for parity with Rust v0.5.x; Rust uses /recall as a GET.
    @app.get("/v1/memory/recall", response_model=_SearchResponse)
    async def recall_get(
        query: str, user_id: str = "default", org_id: str = "default", top_k: int = 10
    ) -> _SearchResponse:
        return await search(
            _SearchRequest(query=query, user_id=user_id, org_id=org_id, top_k=top_k)
        )

    @app.get("/v1/memory/raw_facts", response_model=_SearchResponse)
    async def raw_facts(
        query: str, user_id: str = "default", org_id: str = "default", top_k: int = 30
    ) -> _SearchResponse:
        """Research-surface endpoint: top-K raw scored facts (no context formatting)."""
        return await search(
            _SearchRequest(query=query, user_id=user_id, org_id=org_id, top_k=top_k)
        )

    @app.get("/v1/memory/context", response_model=_ContextResponse)
    async def context(
        query: str,
        user_id: str = "default",
        org_id: str = "default",
        token_budget: int = 2000,
    ) -> _ContextResponse:
        ctx = await engram.context(
            query=query, user_id=user_id, org_id=org_id, token_budget=token_budget
        )
        return _ContextResponse(context=ctx)

    @app.post("/v1/memory/extract", response_model=_ExtractResponse)
    async def extract(req: _ExtractRequest) -> _ExtractResponse:
        scope = Scope(org_id=req.org_id, user_id=req.user_id)
        msgs = [
            ChatMessage(
                scope=scope,
                session_id=m.session_id,
                role=m.role,
                content=m.content,
                timestamp=m.timestamp or datetime.now().astimezone(),
            )
            for m in req.messages
        ]
        try:
            facts = await engram.extract(msgs, session_date=req.session_date, persist=req.persist)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return _ExtractResponse(
            facts=[
                _ExtractedFactSummary(
                    fact_id=f.id,
                    text=f.text,
                    confidence=f.confidence,
                    category=f.category,
                    polarity=f.polarity,
                )
                for f in facts
            ]
        )

    @app.post("/v1/sessions/{session_id}/messages")
    async def append_message(session_id: str, req: _MessageRequest) -> dict[str, str]:
        msg = await engram.record_message(
            content=req.content,
            role=req.role,
            session_id=session_id,
            user_id=req.user_id,
            org_id=req.org_id,
            timestamp=req.timestamp,
        )
        return {"message_id": str(msg.id)}

    return app
