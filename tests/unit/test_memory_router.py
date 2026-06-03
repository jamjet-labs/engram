from __future__ import annotations

from engram.memory_router import RuleBasedMemoryRouter
from engram.models import LifecycleState, MemorySystem, PromotionState


def test_router_maps_preference_tooling() -> None:
    route = RuleBasedMemoryRouter().route("For this repo, always use pnpm instead of npm.")
    assert route.memory_system == MemorySystem.PREFERENCE
    assert route.memory_subtype == "tool_preference"
    assert route.lifecycle_state == LifecycleState.DURABLE
    assert route.promotion_state == PromotionState.CANDIDATE


def test_router_maps_procedural_workflow() -> None:
    route = RuleBasedMemoryRouter().route(
        "Release workflow: run tests, update changelog, bump version."
    )
    assert route.memory_system == MemorySystem.PROCEDURAL
    assert route.memory_subtype == "release_process"


def test_router_maps_prospective_trigger() -> None:
    route = RuleBasedMemoryRouter().route("After tests pass, update the docs.")
    assert route.memory_system == MemorySystem.PROSPECTIVE
    assert route.memory_subtype == "trigger"
    assert route.lifecycle_state == LifecycleState.TEMPORARY


def test_router_explicit_metadata_takes_precedence() -> None:
    route = RuleBasedMemoryRouter().route(
        "anything",
        metadata={"memory_system": "semantic", "memory_subtype": "config", "tags": ["api"]},
    )
    assert route.memory_system == MemorySystem.SEMANTIC
    assert route.memory_subtype == "config"
    assert route.tags == ["api"]
