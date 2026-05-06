"""Tests for is_preference_question — the permissive preference gate."""

from __future__ import annotations

import pytest

from engram.classify.base import QuestionType
from engram.read.preference_gate import is_preference_question


@pytest.mark.parametrize(
    "query",
    [
        "Can you recommend a show or movie for me to watch tonight?",
        "Any tips on what to look for in a new guitar?",
        "I was thinking of trying a new coffee creamer recipe. Any recommendations?",
        "Can you recommend some recent publications I might find interesting?",
        "What should I cook for dinner?",
    ],
)
def test_implicit_preference_questions_detected(query: str):
    qt = QuestionType.SINGLE_SESSION_USER  # what classifier returns for these
    assert is_preference_question(query, qt) is True


def test_explicit_preference_via_classifier():
    qt = QuestionType.SINGLE_SESSION_PREFERENCE
    assert is_preference_question("What's my favorite color?", qt) is True


def test_multi_session_takes_priority():
    qt = QuestionType.MULTI_SESSION
    assert (
        is_preference_question("Across all my conversations, any tips you remember giving me?", qt)
        is False
    )


def test_temporal_takes_priority():
    qt = QuestionType.TEMPORAL_REASONING
    assert is_preference_question("What did you recommend before March 1?", qt) is False


def test_no_preference_signal_returns_false():
    qt = QuestionType.SINGLE_SESSION_USER
    assert is_preference_question("How old am I?", qt) is False
    assert is_preference_question("Where do I live?", qt) is False


def test_handles_none_qt():
    assert is_preference_question("any tips on guitars?", None) is True
    assert is_preference_question("how old am I?", None) is False


def test_negative_guards():
    """Documented false-positive boundaries."""
    qt = QuestionType.SINGLE_SESSION_USER
    # Bare temporal queries do not trigger
    assert is_preference_question("Is it going to rain tomorrow?", qt) is False
    assert is_preference_question("What's the weather tonight?", qt) is False
    # Temporal classifier label blocks even when marker words appear
    qt_temp = QuestionType.TEMPORAL_REASONING
    assert is_preference_question("How many tips did I get last month?", qt_temp) is False
    # Multi-session classifier label blocks marker words
    qt_multi = QuestionType.MULTI_SESSION
    assert (
        is_preference_question("How many recipes did I try across all sessions?", qt_multi) is False
    )
