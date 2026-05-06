"""Preference gate — permissive predicate for activating the synthesis-mode
read path on LongMemEval single-session-preference questions.

Used by the smoke runner to route classifier-mis-tagged conversational
recommendation questions ("any tips?", "recommend a movie") onto a
recommendation-friendly read path, while leaving non-preference questions
on the existing fact-recall pipeline.

History: attempt-1 (`feat/preference-uplift`, tagged
`preference-uplift-attempt-1`) shipped this predicate as part of a larger
expander+tool+preamble experiment that didn't move the metric. attempt-2
keeps just this predicate and pairs it with synthesis-mode reading.
"""

from __future__ import annotations

import re

from engram.classify.base import QuestionType

# Conversational advice / recommendation markers — questions of this shape
# almost always assume the user has stored preferences in memory and the
# assistant should retrieve them. The existing classifier's preference regex
# (prefer/favorite/like/...) misses all of these.
_IMPLICIT_PREF_RE = re.compile(
    r"\b("
    r"recommend(ation|ations)?|"
    r"tips?|advice|advise|"
    r"suggest(ion|ions)?|"
    r"recipe|"
    r"what should i|"
    r"any (suggestions|ideas|thoughts|recommendations|tips)"
    r")\b",
    re.IGNORECASE,
)


def is_preference_question(query: str, qt: QuestionType | None) -> bool:
    """Permissive predicate for activating the preference synthesis path.

    Returns True when:
    - the existing classifier returned ``SINGLE_SESSION_PREFERENCE``, OR
    - the query contains implicit-preference markers
      (recommend / tips / advice / etc.) AND the classifier did NOT return
      a stronger signal (multi-session or temporal-reasoning).

    Multi-session and temporal-reasoning override because the existing
    pipeline handles those better — promoting them into the synthesis path
    would lose category-specific retrieval logic.
    """
    if qt == QuestionType.SINGLE_SESSION_PREFERENCE:
        return True
    if qt in (QuestionType.MULTI_SESSION, QuestionType.TEMPORAL_REASONING):
        return False
    return bool(_IMPLICIT_PREF_RE.search(query))
