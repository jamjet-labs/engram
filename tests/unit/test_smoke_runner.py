import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchmarks.smoke_runner import stratified_sample
from engram.errors import ExtractionError
from engram.llm.base import LLMResponse

def test_stratified_sample_covers_all_categories(tmp_path):
    fake_oracle = [
        {"question_id": f"q{i}", "question_type": t,
         "haystack_sessions": [[{"role": "user", "content": "x"}]],
         "haystack_session_ids": ["s1"], "haystack_dates": ["2024/01/01 (Mon) 10:00"],
         "question": "Q?", "question_date": "2024/06/01 (Sat) 10:00", "answer": "x"}
        for i, t in enumerate(["temporal-reasoning"] * 50 + ["multi-session"] * 50 +
                              ["single-session-user"] * 50 + ["single-session-assistant"] * 50 +
                              ["knowledge-update"] * 50 + ["single-session-preference"] * 50)
    ]
    p = tmp_path / "oracle.json"
    p.write_text(json.dumps(fake_oracle))
    sample = stratified_sample(str(p), n=100)
    assert len(sample) == 100
    types = [q["question_type"] for q in sample]
    for cat in {"temporal-reasoning", "multi-session", "single-session-user",
                "single-session-assistant", "knowledge-update", "single-session-preference"}:
        assert types.count(cat) >= 10, f"{cat}: {types.count(cat)}"


from benchmarks.smoke_runner import write_trace_line, TraceRecord

def test_trace_record_round_trip(tmp_path):
    p = tmp_path / "trace.jsonl"
    rec: TraceRecord = {
        "qid": "abc", "category": "temporal-reasoning",
        "decomposer": {"fired": False, "subqueries": ["Q?"]},
        "recall": [{"sq": "Q?", "n_candidates": 50, "top_k": 5, "fact_ids": []}],
        "reader": {"model": "gpt-4o-mini", "tool_calls": [], "answer": "A", "tokens": {"in": 10, "out": 5}},
        "verifier": {"verdict": "YES", "missing": None},
        "escalation": [],
        "react": None,
        "final": {"answer": "A", "abstained": False, "judge_verdict": None, "cost_usd": 0.001},
    }
    write_trace_line(p, rec)
    line = p.read_text().strip()
    parsed = json.loads(line)
    assert parsed["qid"] == "abc"
    assert parsed["final"]["cost_usd"] == 0.001


# ── Task 2: SYNTHESIS_PROMPT + _synthesis_read ──────────────────────────────


@pytest.mark.asyncio
async def test_synthesis_read_formats_prompt_and_returns_answer():
    """_synthesis_read calls tier.reader.generate with SYNTHESIS_PROMPT formatted
    around context+question and returns the trimmed reader output."""
    from benchmarks.smoke_runner import SYNTHESIS_PROMPT, _synthesis_read  # noqa: F401

    captured = {"system": None, "user": None}
    fake_reader = AsyncMock()

    async def fake_gen(messages, **_kw):
        for m in messages:
            if m.role == "system":
                captured["system"] = m.content
            elif m.role == "user":
                captured["user"] = m.content
        return LLMResponse(content="  black coffee, please.  ", input_tokens=10, output_tokens=10)

    fake_reader.generate = fake_gen
    tier = type("Tier", (), {"reader": fake_reader})()

    out = await _synthesis_read(tier, "the user prefers dark roast", "what coffee should I order?")

    assert out == "black coffee, please."  # trimmed
    assert captured["system"] is not None
    assert "the user prefers dark roast" in captured["system"]
    assert "what coffee should I order?" in captured["system"]
    assert "preference/recommendation" in captured["system"].lower()


@pytest.mark.asyncio
async def test_synthesis_read_returns_idk_on_extraction_error():
    """If the LLM call raises ExtractionError, return a clean abstention string."""
    from benchmarks.smoke_runner import _synthesis_read

    fake_reader = AsyncMock()
    fake_reader.generate = AsyncMock(side_effect=ExtractionError("network"))
    tier = type("Tier", (), {"reader": fake_reader})()

    out = await _synthesis_read(tier, "ctx", "q")
    assert out == "I don't know"


# ── Task 3: _ingest_chunks role_filter ──────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_chunks_role_filter_user_only_skips_assistant_turns():
    """When role_filter=('user',), only user turns are recorded."""
    from benchmarks.smoke_runner import _ingest_chunks

    fake_memory = AsyncMock()
    fake_memory.record = AsyncMock()
    fake_q = {
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2026/05/06 (Wed)"],
        "haystack_sessions": [[
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]],
    }

    n = await _ingest_chunks(fake_memory, fake_q, "alice", role_filter=("user",))

    # Only 2 user turns ingested, not all 4
    assert n == 2
    assert fake_memory.record.await_count == 2
    # Both calls must be for user turns
    texts = [call.kwargs["text"] for call in fake_memory.record.await_args_list]
    assert "u1" in texts
    assert "u2" in texts
    assert "a1" not in texts
    assert "a2" not in texts


@pytest.mark.asyncio
async def test_ingest_chunks_no_role_filter_keeps_existing_behavior():
    """role_filter=None (default) ingests every turn — unchanged from before."""
    from benchmarks.smoke_runner import _ingest_chunks

    fake_memory = AsyncMock()
    fake_memory.record = AsyncMock()
    fake_q = {
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2026/05/06 (Wed)"],
        "haystack_sessions": [[
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]],
    }

    n = await _ingest_chunks(fake_memory, fake_q, "alice")  # no role_filter

    assert n == 2  # both user and assistant ingested
    assert fake_memory.record.await_count == 2
