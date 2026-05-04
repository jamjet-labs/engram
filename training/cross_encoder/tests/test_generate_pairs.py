import json

from training.cross_encoder.generate_pairs import (
    build_synthetic_pair_seed,
    split_train_eval,
    write_seed_file,
)


def _haystack_one_question():
    return [
        {
            "question_id": "q1",
            "question": "How many marathons?",
            "answer": "4",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_dates": [
                "2024/05/01 (Mon) 10:00",
                "2024/06/01 (Sat) 11:00",
            ],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I ran 4 marathons today"},
                    {"role": "assistant", "content": "great"},
                ],
                [
                    {"role": "user", "content": "Talked about pasta"},
                    {"role": "assistant", "content": "yum"},
                ],
            ],
        }
    ]


def test_build_synthetic_pair_seed_yields_positive_and_hard_negative():
    seeds = build_synthetic_pair_seed(_haystack_one_question(), n_per_question=2)
    assert len(seeds) >= 2
    labels = sorted(s["label"] for s in seeds)
    assert 1 in labels and 0 in labels


def test_pair_seed_skips_question_with_no_matching_positive():
    haystack = _haystack_one_question()
    haystack[0]["answer"] = "this string appears nowhere"
    seeds = build_synthetic_pair_seed(haystack, n_per_question=2)
    assert seeds == []


def test_pair_seed_deterministic_for_same_seed():
    h = _haystack_one_question()
    a = build_synthetic_pair_seed(h, n_per_question=2, seed=42)
    b = build_synthetic_pair_seed(h, n_per_question=2, seed=42)
    assert a == b


def test_write_seed_file_round_trip(tmp_path):
    seeds = [{"question": "Q?", "passage": "P", "label": 1}]
    out = tmp_path / "seed.jsonl"
    write_seed_file(seeds, out)
    lines = out.read_text().strip().splitlines()
    assert json.loads(lines[0])["question"] == "Q?"


def test_split_train_eval_keeps_questions_disjoint():
    seeds = [
        {"question": f"q{i}", "passage": "p", "label": 1} for i in range(10)
    ] + [
        {"question": f"q{i}", "passage": "p", "label": 0} for i in range(10)
    ]
    train, eval_ = split_train_eval(seeds, eval_frac=0.3, seed=0)
    train_qs = {s["question"] for s in train}
    eval_qs = {s["question"] for s in eval_}
    # Every question is in exactly one split
    assert train_qs.isdisjoint(eval_qs)
    assert len(train_qs) + len(eval_qs) == 10
