import pytest

from training.cross_encoder.eval import dcg_at_k, ndcg_at_k


def test_ndcg_perfect_ranking():
    relevances = [3, 2, 1, 0, 0]
    assert abs(ndcg_at_k(relevances, k=5) - 1.0) < 1e-6


def test_ndcg_worst_ranking():
    relevances = [0, 0, 1, 2, 3]
    score = ndcg_at_k(relevances, k=5)
    assert 0.0 < score < 1.0


def test_ndcg_zero_relevances():
    assert ndcg_at_k([0, 0, 0], k=3) == 0.0


def test_dcg_truncates_at_k():
    """Raw DCG truncates at position k (numerator only — nDCG normalises against
    the ideal which sees the full list)."""
    a = dcg_at_k([3, 2, 1, 0, 0, 99], k=3)
    b = dcg_at_k([3, 2, 1], k=3)
    assert abs(a - b) < 1e-9


def test_dcg_first_position_full_weight():
    """dcg position 0 has weight 1/log2(2) = 1.0."""
    assert dcg_at_k([5], k=1) == 5.0


@pytest.mark.slow
def test_evaluate_model_smoke(tmp_path):
    """Smoke test: tiny eval set, verify the helper runs end-to-end without error."""
    pytest.importorskip("sentence_transformers")
    from training.cross_encoder.eval import evaluate_model

    p = tmp_path / "eval.jsonl"
    p.write_text(
        '{"question": "What did I eat?", "passage": "I ate pizza", "label": 1}\n'
        '{"question": "What did I eat?", "passage": "It rained", "label": 0}\n'
        '{"question": "Where do I live?", "passage": "I live in Berlin", "label": 1}\n'
        '{"question": "Where do I live?", "passage": "Goats are nice", "label": 0}\n'
    )
    score = evaluate_model(model_path=None, eval_jsonl=p, k=10)
    assert 0.0 <= score <= 1.0
