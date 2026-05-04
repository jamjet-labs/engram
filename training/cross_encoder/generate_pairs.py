"""Generate synthetic (question, passage, label) pairs for cross-encoder fine-tune.

Walks LongMemEval haystacks. For each question, the gold-answer-bearing turn
is a positive (label=1); sibling turns from the same haystack are hard negatives
(label=0). Hard negatives matter more than random negatives because they share
domain vocabulary.

Output format: JSONL, one record per line:
    {"question": "...", "passage": "...", "label": 0 or 1}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def build_synthetic_pair_seed(
    haystack: list[dict[str, Any]],
    n_per_question: int = 2,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """For each question, emit up to ``n_per_question`` positive + matching
    hard-negative pairs from its haystack sessions.

    A turn is positive when the (lowercased) gold answer string appears in it;
    other turns from the same haystack become hard negatives.
    """
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for q in haystack:
        question = q["question"]
        answer = str(q["answer"]).lower().strip()
        if not answer:
            continue
        positives: list[str] = []
        negatives: list[str] = []
        for session in q.get("haystack_sessions", []):
            for turn in session:
                content = turn["content"]
                if answer in content.lower():
                    positives.append(content)
                else:
                    negatives.append(content)
        if not positives:
            continue
        rng.shuffle(positives)
        rng.shuffle(negatives)
        for i in range(min(n_per_question, len(positives))):
            out.append({"question": question, "passage": positives[i], "label": 1})
            if i < len(negatives):
                out.append({"question": question, "passage": negatives[i], "label": 0})
    return out


def write_seed_file(seeds: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in seeds:
            f.write(json.dumps(s) + "\n")


def split_train_eval(
    seeds: list[dict[str, Any]],
    eval_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group-aware split — all pairs for one question stay in the same shard.

    Stops a question's positive ending up in train and its negative in eval
    (which would let the model memorise the question rather than generalise).
    """
    rng = random.Random(seed)
    by_q: dict[str, list[dict[str, Any]]] = {}
    for s in seeds:
        by_q.setdefault(s["question"], []).append(s)
    questions = list(by_q.keys())
    rng.shuffle(questions)
    split = int(len(questions) * (1 - eval_frac))
    train_qs = set(questions[:split])
    train: list[dict[str, Any]] = []
    eval_: list[dict[str, Any]] = []
    for s in seeds:
        if s["question"] in train_qs:
            train.append(s)
        else:
            eval_.append(s)
    return train, eval_


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", type=Path, required=True, help="LongMemEval oracle JSON")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-per-question", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="0 = use all questions")
    args = p.parse_args()

    data = json.loads(args.oracle.read_text())
    if args.limit > 0:
        data = data[: args.limit]
    seeds = build_synthetic_pair_seed(data, n_per_question=args.n_per_question)
    train, eval_ = split_train_eval(seeds)
    write_seed_file(train, args.out_dir / "ce_train.jsonl")
    write_seed_file(eval_, args.out_dir / "ce_eval.jsonl")
    print(f"train: {len(train)} pairs ({len({s['question'] for s in train})} questions)")
    print(f"eval:  {len(eval_)} pairs ({len({s['question'] for s in eval_})} questions)")


if __name__ == "__main__":
    main()
