"""Property-based tests for SqliteStore.

We don't fuzz LLM-generated text, but we do verify that arbitrary fact texts
round-trip through the store cleanly, including unicode and FTS5 edge cases.
"""

from __future__ import annotations

from datetime import UTC, datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from engram.models import Fact
from engram.scope import Scope
from engram.store.sqlite import SqliteStore


def _now() -> datetime:
    return datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)


fact_text = st.text(
    min_size=1,
    max_size=512,
    alphabet=st.characters(blacklist_categories=("Cs",)),
)

scope_token = st.text(
    min_size=1, max_size=64, alphabet=st.characters(whitelist_categories=("L", "N"))
)


@settings(max_examples=50, deadline=None)
@given(text=fact_text)
async def test_fact_roundtrip(text: str) -> None:
    s = await SqliteStore.open(":memory:")
    try:
        scope = Scope(org_id="acme", user_id="alice")
        f = Fact(text=text, scope=scope, valid_from=_now())
        await s.upsert_fact(f)
        got = await s.get_fact(f.id, scope)
        assert got is not None
        assert got.text == text
    finally:
        await s.close()


@settings(max_examples=20, deadline=None)
@given(org=scope_token, user=scope_token, text=fact_text)
async def test_scope_isolation(org: str, user: str, text: str) -> None:
    s = await SqliteStore.open(":memory:")
    try:
        scope_a = Scope(org_id=org, user_id=user)
        scope_b = Scope(org_id=org, user_id=user + "_other")
        f = Fact(text=text, scope=scope_a, valid_from=_now())
        await s.upsert_fact(f)
        assert await s.get_fact(f.id, scope_a) is not None
        assert await s.get_fact(f.id, scope_b) is None
    finally:
        await s.close()
