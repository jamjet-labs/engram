"""Extraction prompts.

Ported from Engram Rust v0.6 + LongMemEval research harness with light cleanup.
The system prompt instructs strict JSON output. We never trust the model's
response — `pipeline.py` validates against the Pydantic schema and discards
malformed entries.
"""

from __future__ import annotations

EXTRACTION_SYSTEM_PROMPT = """\
You extract durable memory facts from a conversation between a user and an assistant.

Extract ALL discrete facts from BOTH user and assistant turns:
  - User personal details, preferences, plans, history, possessions, relationships
  - Assistant recommendations, explanations, computed answers, factual claims
  - Stated event dates and locations
  - Emotional state and stated intentions
  - Negations ("user does NOT like X")

Rules:
  - One fact per JSON object (atomic, self-contained, makes sense out of context)
  - Resolve pronouns where possible ("she" -> "Alice", "my brother" -> "the user's brother")
  - For temporal expressions, populate event_date with an absolute ISO date if it can
    be resolved from the session date; otherwise leave it null
  - Set polarity to "negative" for explicit negations, "hypothetical" for conditionals
  - Set confidence in [0,1] reflecting how sure the model is the fact is correct
  - Pick a category from this set when applicable (else null):
      "user_preference", "user_personal", "user_plan", "user_history",
      "assistant_recommendation", "assistant_explanation", "assistant_factual",
      "knowledge_update", "relationship", "location", "temporal_event"

Output STRICTLY valid JSON of shape:
{
  "facts": [
    {
      "text": "<atomic fact, declarative sentence>",
      "category": "<from list above OR null>",
      "polarity": "affirmative" | "negative" | "hypothetical",
      "confidence": <float 0..1>,
      "entities": [<entity names mentioned>],
      "event_date": "<ISO 8601 datetime OR null>"
    },
    ...
  ]
}

If there are no extractable facts, return {"facts": []}.
Do not include explanations, markdown, or surrounding prose.
"""


def build_extraction_user_prompt(turns: list[dict[str, str]], session_date: str | None) -> str:
    """Format the conversation turns + session date for the extraction model."""
    header = []
    if session_date:
        header.append(
            f"Session date (treat as 'today' for relative time resolution): {session_date}"
        )
    header.append("Conversation:")
    body = "\n".join(f"[{t['role']}] {t['content']}" for t in turns)
    return "\n\n".join([*header, body, "Extract all facts now."])
