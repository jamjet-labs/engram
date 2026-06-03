from __future__ import annotations

from engram.models import MemorySystem
from engram.query_intent import RuleBasedQueryIntentClassifier


def test_query_intent_detects_tool_preference() -> None:
    intent = RuleBasedQueryIntentClassifier().classify(
        "In this repo, what package manager should I use?"
    )
    assert intent.memory_systems == (MemorySystem.PREFERENCE,)
    assert intent.memory_subtypes == ("tool_preference",)


def test_query_intent_detects_release_process() -> None:
    intent = RuleBasedQueryIntentClassifier().classify("What is the release workflow?")
    assert intent.memory_systems == (MemorySystem.PROCEDURAL,)
    assert intent.memory_subtypes == ("release_process",)


def test_query_intent_detects_prospective_trigger() -> None:
    intent = RuleBasedQueryIntentClassifier().classify("What should I follow up on next?")
    assert intent.memory_systems == (MemorySystem.PROSPECTIVE,)
    assert intent.memory_subtypes == ("trigger",)


def test_query_intent_is_conservative_for_plain_keyword() -> None:
    intent = RuleBasedQueryIntentClassifier().classify("espresso")
    assert intent.memory_systems is None
    assert intent.memory_subtypes is None
    assert intent.tags is None
