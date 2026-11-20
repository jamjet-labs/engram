"""Property-based tests for the stores.

We don't fuzz LLM-generated text, but we do verify that arbitrary fact texts
round-trip through the store cleanly, including unicode and FTS5 edge cases.

Each example opens its own fresh target on the backend under test and drops it
afterwards. The `backend` fixture holds no state, so one value serves every
example.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from engram.models import Fact
from engram.scope import Scope
from engram.store import EngramStore, open_store
from tests.conformance.backends import Backend

# The stores refuse NUL by contract, so the strategy leaves it out.
fact_text = st.text(
    min_size=1,
    max_size=512,
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
)

scope_token = st.text(
    min_size=1, max_size=64, alphabet=st.characters(whitelist_categories=("L", "N"))
)

_STATELESS_BACKEND = [HealthCheck.function_scoped_fixture]


def _now() -> datetime:
    return datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)


@asynccontextmanager
async def _fresh_store(backend: Backend) -> AsyncIterator[EngramStore]:
    with tempfile.TemporaryDirectory() as tmp:
        target = await backend.new_target(Path(tmp))
        store = await open_store(target)
        try:
            yield store
        finally:
            await store.close()
            await backend.drop(target)


@settings(max_examples=50, deadline=None, suppress_health_check=_STATELESS_BACKEND)
@given(text=fact_text)
async def test_fact_roundtrip(backend: Backend, text: str) -> None:
    async with _fresh_store(backend) as s:
        scope = Scope(org_id="acme", user_id="alice")
        f = Fact(text=text, scope=scope, valid_from=_now())
        await s.upsert_fact(f)
        got = await s.get_fact(f.id, scope)
        assert got is not None
        assert got.text == text


@settings(max_examples=20, deadline=None, suppress_health_check=_STATELESS_BACKEND)
@given(org=scope_token, user=scope_token, text=fact_text)
async def test_scope_isolation(backend: Backend, org: str, user: str, text: str) -> None:
    async with _fresh_store(backend) as s:
        scope_a = Scope(org_id=org, user_id=user)
        scope_b = Scope(org_id=org, user_id=user + "_other")
        f = Fact(text=text, scope=scope_a, valid_from=_now())
        await s.upsert_fact(f)
        assert await s.get_fact(f.id, scope_a) is not None
        assert await s.get_fact(f.id, scope_b) is None
