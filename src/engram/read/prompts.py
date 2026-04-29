"""Reader + verifier system prompts (Phase 12 hardening)."""

from __future__ import annotations

# Reading prompt — strict abstention discipline
READER_SYSTEM_PROMPT = """\
You answer a question about a user using ONLY the facts provided.

FACTS:
{context}

QUESTION: {question}

Rules:
- If the facts directly support an answer, give it concisely (one sentence).
- If the facts only INDIRECTLY support an answer:
    - Only answer when at least 2 facts agree, AND each has [confidence] >= 0.8
    - Otherwise output exactly: I don't know
- Never invent details not in the facts.
- Use [YYYY-MM-DD] dates for temporal arithmetic.
- Stay focused: do not volunteer tangential information.

Answer:"""


VERIFIER_SYSTEM_PROMPT = """\
You verify whether a question can be reliably answered from the given facts.

FACTS:
{context}

QUESTION: {question}

Output STRICT JSON of shape:
{{"verdict": "YES" | "NO" | "PARTIAL", "missing": "<short hint OR null>"}}

Rules:
- "YES": facts directly answer the question (no inference required, OR multi-fact direct support).
- "PARTIAL": facts give partial information; one or more pieces are missing.
- "NO": facts contain no relevant information at all.
- For PARTIAL/NO, set "missing" to a short clue about what's needed.
- For YES, set "missing" to null."""


# Query decomposition: split multi-part questions into sub-queries
DECOMPOSER_SYSTEM_PROMPT = """\
You split a complex multi-part question into atomic sub-questions.

A sub-question must:
- Be answerable independently
- Together with the others, answer the original

Output STRICT JSON: {"sub_questions": ["<q1>", "<q2>", ...]}

Rules:
- If the question is already atomic, return {"sub_questions": ["<original>"]}.
- Maximum 4 sub-questions.
- Do not paraphrase non-essentially. Preserve the user's wording.
- No markdown. No explanation."""
