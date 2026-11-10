"""The v1 REST routes the Java SDK and Spring Boot starter call.

Paths, methods, status codes and JSON keys copy the Rust v0.5.x server
(jamjet/runtime/engram-server/src/rest.rs), so `dev.jamjet:jamjet-sdk` works
against this server unchanged. scripts/java-contract.sh runs the published jar
against it. The native routes in http.py are separate and unchanged, except
`GET /v1/memory/recall`, which http.py serves in both shapes.

Every difference from Rust, all deliberate:

Errors
- A request that fails validation (a missing field, a wrong type, a value
  out of range) is 422 with FastAPI's `{"detail": ...}`, not Rust's 400
  `{"error": ...}`. Errors the handlers raise themselves keep `{"error"}`.

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
- `DELETE /v1/memory/facts/{id}` is 404 for an unknown id (Rust: 200), and
  422 for a `reason` over 1000 characters.
- `GET /v1/memory/stats` has no `total_entities` or `total_relationships`.
- `POST /v1/memory/consolidate` is 501: Python has no consolidation.
- `DELETE /v1/memory/users/{id}` also erases the user's chat turns, events,
  entities and relationships, leaving content-free erasure records. It never
  reports success unless the erase finished: 409 if it is refused, 503 if a
  fact was written while it ran (calling again finishes it), 500 for any
  other storage error.

Path ids (compatibility with the published Java client)
- The client puts the user id in the delete-user path without encoding it,
  and the conversation id in the message paths with `URLEncoder`, which is
  form encoding (space as `+`, `/` as `%2F`, `+` as `%2B`). Both ids may
  contain `/`.
- The message routes read `+` in the conversation id as a space: the exact
  inverse of `URLEncoder`, decoded from the raw path. Other clients send a
  literal `+` as `%2B` and a space as `%20` or `+`.
- `DELETE /v1/memory/users/{id}` with any query string is 400 and erases
  nothing: the client would send `bob?x=1` as the user `bob` and a query.
  A `#` cannot be caught: the client drops it and the rest of the id before
  sending, so `bob#1` arrives as `bob`. The same goes for an id that ends in
  `?`. The server decodes percent escapes in the path and the client sends
  `%` unencoded, so `a%2Fb` erases the user `a/b`. Do not use `?`, `#` or `%`
  in user ids with that client.

Messages
- `role` must be user, assistant, system or tool (Rust took any string).
- `metadata` is accepted and not stored, and comes back as `{}`. The
  revocation matcher reads turn content only, so stored metadata could carry
  revoked text past it.
- `seq` is a turn's 0-based position among the conversation's visible turns,
  oldest first. Rust stored the index within each save call, which restarts
  at 0; the two agree for a conversation saved in one call.
- `GET /v1/memory/messages` lists conversations by first turn (Rust: by id).
- `GET /v1/memory/messages/{id}` for a conversation with no visible turns,
  unknown ids included, is 200 with an empty list.
- `DELETE /v1/memory/messages/{id}` with an empty `user_id` or `org_id` is
  400, not the "default" scope.
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
from urllib.parse import unquote, unquote_plus
from uuid import UUID, uuid4

from fastapi import FastAPI, Path, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from engram.engram import Engram, context_lines
from engram.errors import (
    EmbeddingError,
    ExtractionError,
    ScopeErasedError,
    ScopeResidueRefusedError,
    StoreError,
)
from engram.models import ChatMessage
from engram.retrieve.base import ScoredFact
from engram.scope import Scope

logger = logging.getLogger(__name__)

LEGACY_HEALTH_PATH = "/health"
# Forget only revokes, and no check refuses a tombstone's actor or reason for
# matching fact text, so these can be fixed. Delete user cannot use a fixed
# string: see _erase_audit_fields.
_FORGET_ACTOR = "restv1"
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
    # Every later erase in the scope re-reads tombstone reasons, so keep them short.
    reason: str | None = Field(default=None, max_length=1000)


class _SaveMessagesRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=256)
    messages: list[_MsgWithMetadata]
    user_id: _BodyId
    org_id: _BodyId


def _scope(user_id: str | None, org_id: str | None) -> Scope:
    return Scope(org_id=org_id or "default", user_id=user_id or "default")


def _erase_audit_fields() -> tuple[str, str]:
    """A fresh (reason, actor) for one erase_user call, one random token each.

    erase_user refuses a reason or actor that quotes any fact text in the
    scope, and any token holder can record a fact. A fixed string could be
    planted as a fact to block the erase for good, so each call makes new ones.
    """
    return f"deleteuser{uuid4().hex}", f"restv1x{uuid4().hex}"


def _form_decoded_tail(request: Request, decoded: str) -> str:
    """The route's last path parameter, decoded as `URLDecoder` would.

    The ASGI server has already percent-decoded ``decoded`` and kept ``+``,
    so a ``+`` there may have been sent as ``+`` (a space, from
    `URLEncoder`) or as ``%2B`` (a plus). The raw path tells them apart: find
    the raw tail that percent-decodes to ``decoded`` and form-decode it.
    Without a raw path, every ``+`` is read as a space.
    """
    raw = request.scope.get("raw_path")
    if isinstance(raw, bytes):
        path = raw.decode("latin-1").split("?", 1)[0]
        cut = len(path)
        # Each step left adds at least one character to the decoded tail, so
        # at most one tail can match.
        while (cut := path.rfind("/", 0, cut)) >= 0:
            tail = path[cut + 1 :]
            if unquote(tail) == decoded:
                return unquote_plus(tail)
    return decoded.replace("+", " ")


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
        if not await engram.forget(parsed, reason=reason, actor=_FORGET_ACTOR):
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

    @app.delete("/v1/memory/users/{user_id:path}", response_model=None)
    async def delete_user(
        request: Request,
        user_id: Annotated[str, Path(max_length=128)],
    ) -> dict[str, Any] | JSONResponse:
        """Hard-erase everything held for (default org, user_id), as Rust did.

        The id is the rest of the path, `/` included, as the Java client sends
        it (unencoded). Any query string is refused with 400 before anything
        is erased: from that client it means the id held a `?`.

        Never reports success unless `erase_user` finished. 409 if it refuses
        the audit fields as quoting a fact; they are fresh random tokens on
        every call, so a stored fact cannot be made to match them. 503 when a
        fact was written mid-way; calling again is safe and finishes. 500 for
        any other storage error, which a retry may not fix.
        """
        if request.scope.get("query_string"):
            return _error(
                400,
                "delete user takes no query string; a '?' in the user id would cut it "
                "short. Nothing was erased.",
            )
        if not user_id:
            return _error(400, "user_id must not be empty. Nothing was erased.")
        reason, actor = _erase_audit_fields()
        try:
            n = await engram.erase_user(
                user_id=user_id, org_id="default", reason=reason, actor=actor
            )
        except ValueError as e:
            return _error(
                409,
                f"delete user refused: {e}. Part of the user's data may already be erased. "
                "Calling again is safe.",
            )
        except ScopeResidueRefusedError:
            logger.warning("delete user: a fact was written mid-erase", exc_info=True)
            return _error(
                503,
                "the user's data changed while it was being erased, so the erase did not "
                "finish; some of it may already be gone. Retrying is safe and finishes it.",
            )
        except StoreError:
            logger.error("delete user: storage error", exc_info=True)
            return _error(
                500,
                "delete user failed with a storage error and did not finish; some of the "
                "user's data may already be gone. The server log has the cause.",
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
        stored: list[ChatMessage] = []
        try:
            for i, m in enumerate(req.messages):
                stored.append(
                    await engram.record_message(
                        content=m.content,
                        role=m.role,
                        session_id=req.conversation_id,
                        user_id=s.user_id,
                        org_id=s.org_id,
                        timestamp=base + timedelta(microseconds=i),
                        not_erased_since=base,
                    )
                )
        except ScopeErasedError:
            # The erase removed the turns stored before it; store no more.
            logger.info("save messages: the user was deleted while the save ran; stopped")
            return _error(
                409,
                "the user was deleted while this save ran, so the save stopped. Nothing "
                "from it is kept.",
            )
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

    @app.get("/v1/memory/messages/{conversation_id:path}", response_model=None)
    async def get_messages(
        request: Request,
        conversation_id: str,
        last_n: Annotated[int | None, Query(ge=1)] = None,
        user_id: _QueryId = None,
        org_id: _QueryId = None,
    ) -> dict[str, Any] | JSONResponse:
        conversation_id = _form_decoded_tail(request, conversation_id)
        if not conversation_id:
            return _error(400, "conversation_id must not be empty")
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

    @app.delete("/v1/memory/messages/{conversation_id:path}", response_model=None)
    async def delete_messages(
        request: Request,
        conversation_id: str,
        user_id: _QueryId = None,
        org_id: _QueryId = None,
    ) -> dict[str, Any] | JSONResponse:
        """Hard-delete one conversation's turns. Facts extracted from them stay, as in Rust.

        An empty `user_id` or `org_id` is refused rather than read as "default",
        so a client bug cannot delete the default user's conversation.
        """
        conversation_id = _form_decoded_tail(request, conversation_id)
        if not conversation_id:
            return _error(400, "conversation_id must not be empty")
        if user_id == "" or org_id == "":
            return _error(400, "user_id and org_id must not be empty when given")
        s = _scope(user_id, org_id)
        n = await engram.delete_conversation(conversation_id, user_id=s.user_id, org_id=s.org_id)
        return {"success": True, "deleted_count": n}
