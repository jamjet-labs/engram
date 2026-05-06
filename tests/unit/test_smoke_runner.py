import json

from benchmarks.smoke_runner import (
    TraceRecord,
    stratified_sample,
    write_trace_line,
)


def test_stratified_sample_covers_all_categories(tmp_path):
    fake_oracle = [
        {
            "question_id": f"q{i}",
            "question_type": t,
            "haystack_sessions": [[{"role": "user", "content": "x"}]],
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2024/01/01 (Mon) 10:00"],
            "question": "Q?",
            "question_date": "2024/06/01 (Sat) 10:00",
            "answer": "x",
        }
        for i, t in enumerate(
            ["temporal-reasoning"] * 50
            + ["multi-session"] * 50
            + ["single-session-user"] * 50
            + ["single-session-assistant"] * 50
            + ["knowledge-update"] * 50
            + ["single-session-preference"] * 50
        )
    ]
    p = tmp_path / "oracle.json"
    p.write_text(json.dumps(fake_oracle))
    sample = stratified_sample(str(p), n=100)
    assert len(sample) == 100
    types = [q["question_type"] for q in sample]
    for cat in {
        "temporal-reasoning",
        "multi-session",
        "single-session-user",
        "single-session-assistant",
        "knowledge-update",
        "single-session-preference",
    }:
        assert types.count(cat) >= 10, f"{cat}: {types.count(cat)}"


def test_trace_record_round_trip(tmp_path):
    p = tmp_path / "trace.jsonl"
    rec: TraceRecord = {
        "qid": "abc",
        "category": "temporal-reasoning",
        "decomposer": {"fired": False, "subqueries": ["Q?"]},
        "recall": [{"sq": "Q?", "n_candidates": 50, "top_k": 5, "fact_ids": []}],
        "reader": {
            "model": "gpt-4o-mini",
            "tool_calls": [],
            "answer": "A",
            "tokens": {"in": 10, "out": 5},
        },
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
