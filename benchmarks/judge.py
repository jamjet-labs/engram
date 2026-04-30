"""Official LongMemEval gpt-4o-mini judge wrapper.

Mirrors the scoring logic in upstream LongMemEval evaluate_qa.py: the judge
is asked yes/no whether predicted matches expected, with category-aware framing.
Returns binary correct/incorrect.
"""

from __future__ import annotations

from dataclasses import dataclass

from engram.llm.base import LLMClient, LLMMessage

JUDGE_PROMPT = """You are an evaluator scoring an AI assistant's answer.

QUESTION: {question}
EXPECTED ANSWER: {expected}
ASSISTANT'S ANSWER: {predicted}

Does the assistant's answer match the expected answer in substance?
Be lenient on phrasing but strict on facts and numbers.
For temporal questions, accept equivalent date formats.
For numerical questions, the number must be exact.

Respond with exactly one word: yes or no."""


@dataclass
class JudgeResult:
    correct: bool
    raw_response: str


async def judge_one(
    question: str,
    expected: str,
    predicted: str,
    category: str,
    llm: LLMClient,
) -> JudgeResult:
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, predicted=predicted)
    resp = await llm.generate(
        [LLMMessage(role="user", content=prompt)],
        temperature=0.0,
        max_tokens=5,
    )
    raw = resp.content.strip().lower()
    return JudgeResult(correct=raw.startswith("yes"), raw_response=raw)
