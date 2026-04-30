from uuid import uuid4

from engram.read.decomposer import should_decompose
from engram.retrieve.rrf import reciprocal_rank_fusion


def test_short_atomic_skips():
    assert should_decompose("What is my favourite color?") is False
    assert should_decompose("How old am I?") is False


def test_compound_with_and_decomposes():
    assert (
        should_decompose("Where do I live and what is my brother's name today?") is True
    )


def test_two_question_marks_decomposes():
    assert should_decompose("What's my job? And where do I work?") is True


def test_long_question_with_or_decomposes():
    assert (
        should_decompose("Did I prefer espresso or filter coffee in the morning yesterday?")
        is True
    )


def test_long_question_no_conjunction_skips():
    # 13 words, no conjunction
    assert (
        should_decompose("What was the name of the restaurant we visited last week here?")
        is False
    )


def test_short_with_conjunction_skips():
    # Has 'and' but only 6 words → still atomic
    assert should_decompose("Tea and biscuits today?") is False


# RRF helper tests
def test_rrf_fuses_overlapping_lists():
    a, b, c = uuid4(), uuid4(), uuid4()
    fused = reciprocal_rank_fusion([[a, b, c], [b, a, c]], k=60)
    # a and b each appear at high ranks; c always ranks lowest
    assert fused.index(c) > fused.index(a)
    assert fused.index(c) > fused.index(b)


def test_rrf_with_one_list_preserves_order():
    a, b, c = uuid4(), uuid4(), uuid4()
    fused = reciprocal_rank_fusion([[a, b, c]], k=60)
    assert fused == [a, b, c]


def test_rrf_with_empty_input_returns_empty():
    assert reciprocal_rank_fusion([], k=60) == []


def test_rrf_handles_partial_overlap():
    a, b, c, d = uuid4(), uuid4(), uuid4(), uuid4()
    # a in both lists at top; d only in second; c only in first
    fused = reciprocal_rank_fusion([[a, b, c], [a, d]], k=60)
    assert fused[0] == a  # appears highest in both
