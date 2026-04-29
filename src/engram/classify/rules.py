"""Rule-based question classifier — no LLM, fast, offline-safe.

Accuracy ceiling is moderate (~70-80% on hand-labeled samples). Use the
`LLMClassifier` for higher accuracy at the cost of an extra LLM call.
"""

from __future__ import annotations

import re

from engram.classify.base import QuestionType
from engram.retrieve.temporal import detect_temporal_intent

_PREFERENCE_RE = re.compile(
    r"\b(prefer|favorite|like|dislike|love|hate|enjoy|favor|preference)\b",
    re.IGNORECASE,
)
_KNOWLEDGE_UPDATE_RE = re.compile(
    r"\b(now|currently|latest|update|change|since|new|recent|after)\b",
    re.IGNORECASE,
)
_ASSISTANT_RE = re.compile(
    r"\b(you (recommend|said|told|suggested|advised|explained|mentioned)|"
    r"what did you say|previous (advice|conversation|answer)|"
    r"the recipe (you )?gave|the answer (you )?provided)\b",
    re.IGNORECASE,
)
_MULTI_SESSION_RE = re.compile(
    r"\b(across|combined|sum of|total of|"
    r"all (the )?(times|conversations|sessions)|"
    r"how many (times|sessions|conversations)|"
    r"both (visits|sessions|conversations|trips))\b",
    re.IGNORECASE,
)


class RuleBasedClassifier:
    """Heuristic classifier. Order of precedence (most specific first):

    1. Multi-session if 'across', 'both', 'combined', 'sum/total of', 'sessions'
       (these signals beat temporal even when 'how many' is present)
    2. Knowledge-update if 'now', 'currently', 'latest', 'change', 'update'
       (a "what is the latest X" question is asking for current state, not time math)
    3. Temporal-reasoning if `detect_temporal_intent` matches
    4. Single-session-assistant if 'you recommended/said'
    5. Single-session-preference if 'prefer/favorite/like'
    6. Default: single-session-user
    """

    async def classify(self, query: str) -> QuestionType:
        if _MULTI_SESSION_RE.search(query):
            return QuestionType.MULTI_SESSION
        if _KNOWLEDGE_UPDATE_RE.search(query):
            return QuestionType.KNOWLEDGE_UPDATE
        if detect_temporal_intent(query) is not None:
            return QuestionType.TEMPORAL_REASONING
        if _ASSISTANT_RE.search(query):
            return QuestionType.SINGLE_SESSION_ASSISTANT
        if _PREFERENCE_RE.search(query):
            return QuestionType.SINGLE_SESSION_PREFERENCE
        return QuestionType.SINGLE_SESSION_USER
