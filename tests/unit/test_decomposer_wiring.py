from engram.read.decomposer import should_decompose


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
