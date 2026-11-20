"""FastAPI HTTP server.

Native routes live here. The Rust v0.5.x routes that the Java SDK and Spring
Boot starter call are in legacy_v1.py, and scripts/java-contract.sh runs the
published jar against them. `GET /v1/memory/recall` is shared: it answers in
both shapes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from starlette.routing import Route

from engram.engram import Engram
from engram.errors import NotFoundError, ScopeErasedError
from engram.governance import Provenance
from engram.models import ChatMessage, MemoryTier, Polarity
from engram.retrieve.base import ScoredFact
from engram.scope import Scope
from engram.server.auth import (
    HEALTHZ_PATH,
    LOOPBACK_HOSTNAMES,
    BearerAuthMiddleware,
    HostGuardMiddleware,
    OriginGuardMiddleware,
    check_token,
    normalize_host,
    normalize_origin,
)
from engram.server.legacy_v1 import LEGACY_HEALTH_PATH, fact_json, register_legacy_routes
from engram.server.transport import build_streamable_http_mount


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
    # Optional, additive in 0.3: who wrote it, and what it was concluded from.
    actor: str | None = None
    derived_from: list[UUID] | None = None


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


def _hit(sf: ScoredFact) -> _SearchHit:
    return _SearchHit(
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


class _ProvenanceHit(_SearchHit):
    provenance: Provenance


class _ProvenanceSearchResponse(BaseModel):
    hits: list[_ProvenanceHit]


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


# Health checks are open: no token, Origin or Host check. They reveal nothing.
_OPEN_PATHS = (HEALTHZ_PATH, LEGACY_HEALTH_PATH)


def build_http_app(
    engram: Engram,
    *,
    mcp_server: Any | None = None,
    auth_token: str | None = None,
    allow_unauthenticated: bool = False,
    allowed_origins: Iterable[str] = (),
    allowed_hosts: Iterable[str] | None = None,
) -> FastAPI:
    """Build a FastAPI app bound to a single Engram instance.

    Args:
        engram: The Engram instance to serve.
        mcp_server: Optional MCP `Server` (e.g. from `build_mcp_server`). If
            provided, a Streamable HTTP MCP endpoint is mounted at `/mcp`.
        auth_token: Bearer token required on every route, `/v1/*` and `/mcp`
            alike. Only the health checks, `/healthz` and Rust's `/health`,
            are open. One token is one trust domain: any holder can read and
            write every `(org_id, user_id)` scope, because the scope comes from
            the request. Run one server per tenant to separate them.
        allow_unauthenticated: Required to build an app without
            `auth_token`. Such an app serves anyone who can reach it. Its Host,
            Origin and Fetch Metadata checks stop a web page from using a
            browser to reach it (DNS rebinding, cross-site requests); they are
            not access control, since any other client can send
            `Host: localhost`. Only bind it to loopback or put real auth in
            front of it.
        allowed_origins: Browser origins (`scheme://host[:port]`) allowed to
            call the app. A request carrying any other `Origin` header gets 403;
            requests with no `Origin` (non-browser clients) are not affected.
            Empty by default, so every browser origin is refused. This adds no
            CORS headers.
        allowed_hosts: Hostnames, besides 127.0.0.1, localhost and ::1, that
            requests may name in `Host` (any port). Others get 421. Checked
            when there is no `auth_token`, or when this is given.

    Raises:
        ValueError: neither `auth_token` nor `allow_unauthenticated=True` was
            given, `auth_token` is not a usable token, or an allowed origin or
            host is malformed.
    """
    # A truthy string such as "false" from a config file must not open the app.
    if not isinstance(allow_unauthenticated, bool):
        raise TypeError("allow_unauthenticated must be True or False")
    if auth_token is not None:
        try:
            check_token(auth_token)
        except ValueError as e:
            raise ValueError(f"auth_token: {e}") from e
    elif allow_unauthenticated is not True:
        raise ValueError(
            "build_http_app needs auth_token=... or allow_unauthenticated=True. Without a "
            "token anyone who can reach the server can read and write all memory; the Host "
            "and Origin checks only stop browsers being used against it, they are not "
            "access control."
        )
    # A str is iterable, so "a.example" would become the hosts "a", ".", ...
    for name, value in (("allowed_origins", allowed_origins), ("allowed_hosts", allowed_hosts)):
        if isinstance(value, str | bytes):
            raise TypeError(f"{name} must be a list of strings, not a single string")
    # Validate now: Starlette builds middleware lazily, on the first request.
    origins = tuple(normalize_origin(o) for o in allowed_origins)
    hosts = None if allowed_hosts is None else {normalize_host(h) for h in allowed_hosts}

    if mcp_server is not None:
        manager, handler = build_streamable_http_mount(mcp_server)

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            async with manager.run():
                yield

        app = FastAPI(title="Engram", version="0.3.0", lifespan=lifespan)

        # Route (not Mount) so that POST /mcp hits the handler without a
        # 307 redirect. Starlette's Mount matches /mcp/ and below, but
        # redirects bare /mcp; Route matches the exact path.
        # The handler is wrapped in a class so Starlette treats it as an ASGI
        # app (methods=None = all HTTP methods) rather than a GET-only function.
        # Auth, Origin and Host checks are app-wide middleware, added below.
        class _MCPHandler:
            async def __call__(self, scope_: Any, receive_: Any, send_: Any) -> None:
                await handler(scope_, receive_, send_)

        app.router.routes.append(Route("/mcp", endpoint=_MCPHandler()))
    else:
        app = FastAPI(title="Engram", version="0.3.0")

    # Middleware added last runs first: Host, Origin, auth, then routing.
    if auth_token is not None:
        app.add_middleware(
            BearerAuthMiddleware, expected_token=auth_token, exempt_paths=_OPEN_PATHS
        )
    app.add_middleware(
        OriginGuardMiddleware,
        allowed_origins=origins,
        exempt_paths=_OPEN_PATHS,
        refuse_cross_site=auth_token is None,
    )
    # Without a token, nothing else stops a DNS-rebound page from reading over
    # same-origin GETs, so the Host check is on by default in that mode.
    if auth_token is None or hosts is not None:
        app.add_middleware(
            HostGuardMiddleware,
            allowed_hosts=LOOPBACK_HOSTNAMES | (hosts or set()),
            exempt_paths=_OPEN_PATHS,
        )

    # First, so a native route added later cannot shadow a path the Java SDK calls.
    register_legacy_routes(app, engram)

    @app.api_route(HEALTHZ_PATH, methods=["GET", "HEAD"], response_model=None)
    async def healthz(request: Request) -> Response | dict[str, str]:
        if request.method == "HEAD":
            return Response(status_code=200)
        return {"status": "ok"}

    @app.post("/v1/memory/record", response_model=_RecordResponse)
    async def record(req: _RecordRequest) -> _RecordResponse:
        try:
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
                actor=req.actor,
                derived_from=req.derived_from,
            )
        except NotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ScopeErasedError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return _RecordResponse(fact_id=fact.id, text=fact.text)

    @app.post("/v1/memory/search", response_model=_SearchResponse)
    async def search(req: _SearchRequest) -> _SearchResponse:
        results = await engram.recall(
            query=req.query, user_id=req.user_id, org_id=req.org_id, top_k=req.top_k
        )
        return _SearchResponse(hits=[_hit(sf) for sf in results])

    @app.post("/v1/memory/recall_with_provenance", response_model=_ProvenanceSearchResponse)
    async def recall_with_provenance(req: _SearchRequest) -> _ProvenanceSearchResponse:
        """Like /search, plus where each hit came from and when its status was checked.

        A separate route so the /search and /recall shapes stay identical to
        Rust v0.5.x for existing clients.
        """
        results = await engram.recall(
            query=req.query, user_id=req.user_id, org_id=req.org_id, top_k=req.top_k
        )
        return _ProvenanceSearchResponse(
            hits=[
                _ProvenanceHit(**_hit(sf).model_dump(), provenance=sf.provenance)
                for sf in results
                if sf.provenance is not None
            ]
        )

    # Rust v0.5.x clients send q/max_results and read results/total; native
    # clients send query/top_k and read hits. One route serves both.
    @app.get("/v1/memory/recall")
    async def recall_get(
        query: str | None = None,
        q: str | None = None,
        user_id: str = "default",
        org_id: str = "default",
        top_k: Annotated[int | None, Query(ge=1, le=200)] = None,
        max_results: Annotated[int | None, Query(ge=1, le=200)] = None,
    ) -> dict[str, Any]:
        text = query if query is not None else q
        if text is None:
            raise HTTPException(status_code=422, detail="query (or q) is required")
        found = await engram.recall(
            query=text, user_id=user_id, org_id=org_id, top_k=top_k or max_results or 10
        )
        results = [fact_json(sf) for sf in found]
        return {
            "hits": [_hit(sf).model_dump(mode="json") for sf in found],
            "results": results,
            "total": len(results),
        }

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
