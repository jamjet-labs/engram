from engram.scope import Scope


def test_scope_default_is_anonymous() -> None:
    s = Scope()
    assert s.org_id == "default"
    assert s.user_id == "default"


def test_scope_explicit_construction() -> None:
    s = Scope(org_id="acme", user_id="alice")
    assert s.org_id == "acme"
    assert s.user_id == "alice"


def test_scope_is_hashable() -> None:
    s1 = Scope(org_id="acme", user_id="alice")
    s2 = Scope(org_id="acme", user_id="alice")
    assert hash(s1) == hash(s2)
    assert s1 == s2


def test_scope_distinct_when_user_differs() -> None:
    s1 = Scope(org_id="acme", user_id="alice")
    s2 = Scope(org_id="acme", user_id="bob")
    assert s1 != s2


def test_scope_serialization_roundtrip() -> None:
    s = Scope(org_id="acme", user_id="alice")
    data = s.model_dump()
    s2 = Scope.model_validate(data)
    assert s == s2
