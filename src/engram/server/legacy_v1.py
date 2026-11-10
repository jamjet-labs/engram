"""The v1 REST routes the Java SDK and Spring Boot starter call.

Paths, methods, status codes and JSON keys copy the Rust v0.5.x server
(jamjet/runtime/engram-server/src/rest.rs), so `dev.jamjet:jamjet-sdk` works
against this server unchanged. scripts/java-contract.sh runs the published jar
against it. The native routes in http.py are separate and unchanged, except
`GET /v1/memory/recall`, which http.py serves in both shapes.

Every difference from Rust, all deliberate:

Scope
- `org_id` defaults to "default", as in Rust. A missing `user_id` means the
  Python user "default"; Rust used an org-wide scope there, which Python does
  not have.
- The message routes are scoped to `(org_id, user_id)`. Rust filtered on
  `org_id` alone.

Facts
- `POST /v1/memory` without `session_id` extracts under a fresh session,
  `v1add<hex>`, per call. Rust used no session; "default" would collide with
  Spring AI's default conversation id, and rule (c) below would then let
  forgetting such a fact withhold that conversation. `timestamp` becomes the
  session date when it parses as an ISO 8601 time with a UTC offset (RFC 3339
  times do; Python also takes a few looser forms) and is ignored otherwise, as
  in Rust. With no LLM configured, add is 400.
- `max_results` and `top_k` must be 1 to 200 (Rust took any usize).
- `GET /v1/memory/search` is the hybrid recall the native
  `POST /v1/memory/search` runs, not a keyword-only search, and its hits carry
  `confidence` too.
- `tier` values are Python's (working, episodic, semantic, archival).
- `POST /v1/memory/context` accepts `format` and ignores it: Python renders
  one format. `token_count` is the ~4 characters per token estimate the
  budget uses. There is no `tier_breakdown`.
- `DELETE /v1/memory/facts/{id}` is 404 for an unknown id (Rust: 200).
- `GET /v1/memory/stats` has no `total_entities` or `total_relationships`.
- `POST /v1/memory/consolidate` is 501: Python has no consolidation.
- `DELETE /v1/memory/users/{id}` also erases the user's chat turns, events,
  entities and relationships, leaving content-free erasure records. It is 409
  or 503, never success, when the erase is refused or interrupted.

Messages
- `role` must be user, assistant, system or tool (Rust took any string).
- `metadata` is accepted and not stored, and comes back as `{}`. The
  revocation matcher reads turn content only, so stored metadata could carry
  revoked text past it.
- `seq` is a turn's 0-based position among the conversation's visible turns,
  oldest first. Rust stored the index within each save call, which restarts
  at 0; the two agree for a conversation saved in one call.
- `GET /v1/memory/messages` lists conversations by first turn (Rust: by id).
- Rule (c) of the `live_messages` view: forgetting a fact extracted from a
  conversation withholds every turn of it, turns saved later included, until
  the fact is reinstated. Extraction sets no `source_message_id`, so the view
  cannot tell which turn carried the fact, and fails closed. Rust had no
  revocation.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, Path, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from engram.engram import Engram, context_lines
from engram.errors import EmbeddingError, ExtractionError, StoreError
from engram.models import ChatMessage
from engram.retrieve.base import ScoredFact
from engram.scope import Scope

logger = logging.getLogger(__name__)

LEGACY_HEALTH_PATH = "/health"
# One alphanumeric token each, so neither can quote a user's fact text (the
# erasure rule refuses audit fields that do).
_ACTOR = "restv1"
_FORGET_REASON = "forgotten through the v1 REST API"
_DEFAULT_BUDGET = 2000

# Scope ids: Scope allows 1-128 characters; an empty one means the default.
_QueryId = Annotated[str | None, Query(max_length=128)]
_BodyId = Annotated[str | None, Field(default=None, max_length=128)]
_Role = Literal["user", "assistant", "system", "tool"]


class _Msg(BaseModel):
    role: _Role
    content: str


class _MsgWithMetadata(_Msg):
    metadata: dict[str, Any] | None = None


class _AddRequest(BaseModel):
    messages: list[_Msg]
    user_id: _BodyId
    org_id: _BodyId
    session_id: str | None = Field(default=None, max_length=256)
    timestamp: str | None = None


class _ContextRequest(BaseModel):
    query: str
    user_id: _BodyId
    org_id: _BodyId
    token_budget: int | None = Field(default=None, ge=0)
    format: str | None = None


class _ForgetRequest(BaseModel):
    reason: str | None = None


class _SaveMessagesRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=256)
    messages: list[_MsgWithMetadata]
    user_id: _BodyId
    org_id: _BodyId


def _scope(user_id: str | None, org_id: str | None) -> Scope:
    return Scope(org_id=org_id or "default", user_id=user_id or "default")


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


def fact_json(sf: ScoredFact) -> dict[str, Any]:
    """One fact in the Rust v0.5.x recall/search shape."""
    f = sf.fact
    return {
        "fact_id": str(f.id),
        "text": f.text,
        "tier": f.tier.value,
        "category": f.category,
        "confidence": f.confidence,
    }


def _session_date(value: str | None) -> datetime | None:
    """Rust's `AddRequest.timestamp`: an ISO 8601 time with an offset, else ignored."""
    if value is None:
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    return None if ts.tzinfo is None else ts.astimezone(UTC)


def _extraction_failed(e: Exception) -> JSONResponse:
    # The cause can quote a provider's reply (a masked API key, say), so it
    # goes to the log and the client gets the error class.
    logger.warning("fact extraction failed", exc_info=True)
    return _error(500, f"fact extraction failed ({type(e).__name__}); the server log has the cause")


def _rfc3339(ts: datetime) -> str:
    """UTC, as Rust's `to_rfc3339()` wrote it. A naive time is taken as UTC."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).isoformat()


def register_legacy_routes(app: FastAPI, engram: Engram) -> None:
    """Add the Rust v0.5.x routes to ``app``. Call before the native routes."""

    @app.api_route(LEGACY_HEALTH_PATH, methods=["GET", "HEAD"], response_model=None)
    async def health(request: Request) -> Response | dict[str, str]:
        """Rust's health check. Open like /healthz: it reveals nothing."""
        if request.method == "HEAD":
            return Response(status_code=200)
        return {"status": "ok", "service": "engram"}

    @app.post("/v1/memory", status_code=201, response_model=None)
    async def add(req: _AddRequest) -> dict[str, Any] | JSONResponse:
        """Extract facts from messages. Like Rust, the messages are not stored.

        Without `session_id` the facts get a session of their own, so no
        stored conversation shares it (see rule (c) in the module docstring).
        """
        if not req.messages:
            return _error(400, "messages must not be empty")
        scope = _scope(req.user_id, req.org_id)
        session_id = req.session_id or f"v1add{uuid4().hex}"
        now = datetime.now().astimezone()
        chat = [
            ChatMessage(
                scope=scope,
                session_id=session_id,
                role=m.role,
                content=m.content,
                timestamp=now,
            )
            for m in req.messages
        ]
        try:
            facts = await engram.extract(chat, session_date=_session_date(req.timestamp))
        except RuntimeError as e:  # no LLM configured
            return _error(400, str(e))
        except (ExtractionError, EmbeddingError) as e:
            return _extraction_failed(e)
        ids = [str(f.id) for f in facts]
        return {"success": True, "fact_count": len(ids), "fact_ids": ids}

    @app.get("/v1/memory/search")
    async def search(
        q: str,
        user_id: _QueryId = None,
        org_id: _QueryId = None,
        top_k: Annotated[int, Query(ge=1, le=200)] = 10,
    ) -> dict[str, Any]:
        s = _scope(user_id, org_id)
        hits = await engram.recall(query=q, user_id=s.user_id, org_id=s.org_id, top_k=top_k)
        results = [fact_json(sf) for sf in hits]
        return {"results": results, "total": len(results)}

    @app.post("/v1/memory/context")
    async def context(req: _ContextRequest) -> dict[str, Any]:
        """The native context text, with Rust's counts. One retrieval feeds both."""
        s = _scope(req.user_id, req.org_id)
        budget = _DEFAULT_BUDGET if req.token_budget is None else req.token_budget
        entries = await engram.context_entries(req.query, user_id=s.user_id, org_id=s.org_id)
        lines = context_lines(entries, budget)
        text = "\n".join(lines)
        return {
            "text": text,
            "token_count": (len(text) + 3) // 4,
            "facts_included": len(lines),
            "facts_omitted": len(entries) - len(lines),
        }

    @app.delete("/v1/memory/facts/{fact_id}", response_model=None)
    async def forget(
        fact_id: str, req: _ForgetRequest | None = None
    ) -> dict[str, Any] | JSONResponse:
        """Revoke a fact named by id alone. Soft and reversible, like Rust's invalidate.

        There is no scope check: `Engram.forget` finds the fact's owner in
        any scope. That is safe only because one server token is one trust
        domain, so the token already grants every scope. Do not reuse this
        handler behind per-tenant auth.
        """
        try:
            parsed = UUID(fact_id)
        except ValueError as e:
            return _error(400, f"invalid fact_id: {e}")
        reason = ((req.reason if req else None) or "").strip() or _FORGET_REASON
        if not await engram.forget(parsed, reason=reason, actor=_ACTOR):
            return _error(404, f"fact {parsed} not found")
        return {"success": True, "deleted_fact_id": fact_id}

    @app.get("/v1/memory/stats")
    async def stats() -> dict[str, int]:
        """Fact counts across every scope, as Rust reported them (one token, one domain)."""
        total, valid = await engram.fact_counts()
        return {"total_facts": total, "valid_facts": valid, "invalidated_facts": total - valid}

    @app.post("/v1/memory/consolidate", response_model=None)
    async def consolidate() -> JSONResponse:
        return _error(501, "consolidation is not implemented in the Python Engram server")

    @app.delete("/v1/memory/users/{user_id}", response_model=None)
    async def delete_user(
        user_id: Annotated[str, Path(min_length=1, max_length=128)],
    ) -> dict[str, Any] | JSONResponse:
        """Hard-erase everything held for (default org, user_id), as Rust did.

        Never reports success unless `erase_user` finished. 409 when the audit
        fields quote a fact: from the check before anything is erased, or, if
        a quoted fact was written meanwhile, part-way. 503 when a write landed
        mid-way. Calling again is safe in every case.
        """
        try:
            n = await engram.erase_user(
                user_id=user_id,
                org_id="default",
                reason=f"deleteuser{uuid4().hex}",
                actor=_ACTOR,
            )
        except ValueError as e:
            return _error(
                409,
                f"delete user refused: {e}. Part of the user's data may already be erased. "
                "Calling again is safe.",
            )
        except StoreError:
            logger.warning("delete user: data changed mid-erase", exc_info=True)
            return _error(
                503,
                "the user's data changed while it was being erased, so the erase did not "
                "finish; some of it may already be gone. Retrying is safe and finishes it.",
            )
        return {"success": True, "deleted_facts": n}

    @app.post("/v1/memory/messages", status_code=201, response_model=None)
    async def save_messages(req: _SaveMessagesRequest) -> dict[str, Any] | JSONResponse:
        """Store each turn, then extract facts from the stored turns.

        As in Rust, the turns stay saved when extraction is off or fails, and
        `fact_ids` is null then. An error here would make clients retry and
        store the turns twice.

        Turns are stamped one microsecond apart from one clock reading, so
        they list in the order sent even when the clock is coarse.

        The facts carry `conversation_id` as their session and no
        `source_message_id`, so the view cannot tell which turn a fact came
        from. Forgetting any of them withholds every turn of the conversation,
        turns saved later included, until the fact is reinstated (rule (c) of
        the `live_messages` view). That fails closed.
        """
        if not req.messages:
            return _error(400, "messages must not be empty")
        s = _scope(req.user_id, req.org_id)
        base = datetime.now().astimezone()
        stored = [
            await engram.record_message(
                content=m.content,
                role=m.role,
                session_id=req.conversation_id,
                user_id=s.user_id,
                org_id=s.org_id,
                timestamp=base + timedelta(microseconds=i),
            )
            for i, m in enumerate(req.messages)
        ]
        if any(m.metadata for m in req.messages):
            logger.debug("save messages: message metadata is not stored, dropped it")
        fact_ids: list[str] | None
        try:
            # From before the first turn was stored: an erase that removes the
            # turns before extraction starts must also refuse what it extracts.
            extracted = await engram.extract(stored, not_erased_since=base)
            fact_ids = [str(f.id) for f in extracted]
        except RuntimeError as e:  # no LLM configured: expected, not a fault
            logger.debug("save messages: turns kept, no extraction: %s", e)
            fact_ids = None
        except Exception:
            logger.warning("save messages: extraction failed, turns kept", exc_info=True)
            fact_ids = None
        return {
            "success": True,
            "message_ids": [str(m.id) for m in stored],
            "fact_ids": fact_ids,
        }

    @app.get("/v1/memory/messages/{conversation_id}")
    async def get_messages(
        conversation_id: str,
        last_n: Annotated[int | None, Query(ge=1)] = None,
        user_id: _QueryId = None,
        org_id: _QueryId = None,
    ) -> dict[str, Any]:
        s = _scope(user_id, org_id)
        # Read every visible turn once so `seq` is the position in the whole
        # conversation, then keep the newest `last_n`.
        turns = list(
            enumerate(
                await engram.list_messages(conversation_id, user_id=s.user_id, org_id=s.org_id)
            )
        )
        if last_n is not None:
            turns = turns[-last_n:]
        out = [
            {
                "id": str(m.id),
                "conversation_id": m.session_id,
                "role": m.role,
                "content": m.content,
                "seq": seq,
                "created_at": _rfc3339(m.timestamp),
                "metadata": {},
            }
            for seq, m in turns
        ]
        return {"messages": out, "total": len(out)}

    @app.get("/v1/memory/messages")
    async def list_conversations(
        user_id: _QueryId = None, org_id: _QueryId = None
    ) -> dict[str, Any]:
        s = _scope(user_id, org_id)
        ids = await engram.list_conversations(user_id=s.user_id, org_id=s.org_id)
        return {"conversation_ids": ids, "total": len(ids)}

    @app.delete("/v1/memory/messages/{conversation_id}")
    async def delete_messages(
        conversation_id: str, user_id: _QueryId = None, org_id: _QueryId = None
    ) -> dict[str, Any]:
        """Hard-delete one conversation's turns. Facts extracted from them stay, as in Rust."""
        s = _scope(user_id, org_id)
        n = await engram.delete_conversation(conversation_id, user_id=s.user_id, org_id=s.org_id)
        return {"success": True, "deleted_count": n}
