from datetime import UTC, datetime
from uuid import uuid4

from engram.solve.temporal import SolverResult, TemporalQuery


def test_temporal_query_minimal():
    q = TemporalQuery(op="count")
    assert q.op == "count"
    assert q.subject is None


def test_temporal_query_with_window():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 12, 31, tzinfo=UTC)
    q = TemporalQuery(op="count", verb="run", window=(start, end))
    assert q.window == (start, end)


def test_solver_result_round_trip():
    r = SolverResult(answer=4, confidence=0.95, evidence_event_ids=[uuid4()])
    assert r.answer == 4
    assert r.confidence == 0.95
    assert len(r.evidence_event_ids) == 1


def test_solver_result_string_answer():
    r = SolverResult(answer="2024-05-01", confidence=0.8)
    assert r.answer == "2024-05-01"
