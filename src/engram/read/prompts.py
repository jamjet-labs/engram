"""Reader + verifier system prompts (Phase 12 hardening)."""

from __future__ import annotations

# Reading prompt — strict abstention discipline + Chain-of-Note + temporal anchor
READER_SYSTEM_PROMPT = """\
You answer a question about a user using ONLY the facts provided.

{today_clause}
FACTS:
{context}

QUESTION: {question}

Reasoning approach (Chain-of-Note):
1. First, identify which facts are directly relevant to the question.
2. Then reason step by step:
   - For temporal questions (X days/weeks ago), compute the difference between
     today and the relevant [YYYY-MM-DD] date in the facts.
   - For multi-part questions (sums, comparisons), gather each piece independently
     before combining.
3. Give a concise final answer.

Rules:
- If the facts directly support an answer, give it concisely (one short sentence).
- If the facts only INDIRECTLY support an answer:
    - Only answer when at least 2 facts agree, AND each has [confidence] >= 0.8
    - Otherwise output exactly: I don't know
- Never invent details not in the facts.
- Use [YYYY-MM-DD] dates for temporal arithmetic.
- Stay focused: do not volunteer tangential information.

Answer:"""

TODAY_CLAUSE = "Today is {today}.\n"


VERIFIER_SYSTEM_PROMPT = """\
You verify whether a question can be reliably answered from the given facts.

FACTS:
{context}

QUESTION: {question}

Output exactly two XML tags, no extra prose:
<verdict>YES</verdict><missing>none</missing>

Rules:
- verdict YES: facts directly answer the question (direct or multi-fact direct support).
- verdict PARTIAL: facts give partial information; one or more pieces are missing.
- verdict NO: facts contain no relevant information at all.
- For PARTIAL/NO, replace "none" with a short clue about what's needed (max 80 chars).
- For YES, keep missing as "none".
- Output ONLY the two tags. Nothing else.

We use XML rather than JSON because Anthropic models honor it more reliably."""


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


SYNTHESIS_PROMPT = """You are answering a USER's preference/recommendation question.

CONTEXT (the user's prior statements, in their own words):
{context}

QUESTION: {question}

Your job is to give a HELPFUL, GROUNDED recommendation:
1. Identify what the user has stated about their interests, tastes, ownership, or current situation.
2. Use those stated preferences as the basis for your recommendation.
3. Tailor your answer SPECIFICALLY to what the user has said. If the user mentioned specific items (e.g., "Fender Stratocaster"), incorporate them.
4. If the context contains absolutely nothing relevant to the topic, say "I don't know" — but try first.
5. Concise: 2-4 sentences or a short bulleted list.

Answer:"""  # noqa: E501
