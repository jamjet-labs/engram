"""Evaluate a fine-tuned cross-encoder vs the base on held-out pairs.

Reports nDCG@k over per-question groups. Run::

    uv run python -m training.cross_encoder.eval --eval benchmarks/cache/ce_eval.jsonl
    uv run python -m training.cross_encoder.eval --eval benchmarks/cache/ce_eval.jsonl \
        --model training/cross_encoder/checkpoints/v1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances[:k]))


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """Standard nDCG@k. Returns 0.0 if there is no relevant item in the list."""
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(relevances, k) / idcg


def evaluate_model(model_path: str | None, eval_jsonl: Path, k: int = 10) -> float:
    """Mean nDCG@k over per-question groups in ``eval_jsonl``.

    Each line is ``{"question", "passage", "label"}``. Pairs are grouped by
    question; the cross-encoder scores each (question, passage), the labels
    are reordered by score descending, and nDCG@k is computed per question.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_path or "cross-encoder/ms-marco-MiniLM-L-6-v2")

    by_q: dict[str, list[tuple[str, int]]] = {}
    for line in eval_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_q.setdefault(r["question"], []).append((r["passage"], int(r["label"])))

    if not by_q:
        return 0.0

    ndcgs: list[float] = []
    for q, pairs in by_q.items():
        passages = [p for p, _ in pairs]
        labels = [lbl for _, lbl in pairs]
        scores = model.predict([(q, p) for p in passages])
        order = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
        ranked_labels = [labels[i] for i in order]
        ndcgs.append(ndcg_at_k(ranked_labels, k))

    return sum(ndcgs) / len(ndcgs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=None,
        help="Path to fine-tuned model dir; default = base MS MARCO MiniLM",
    )
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("-k", type=int, default=10)
    args = p.parse_args()
    score = evaluate_model(args.model, args.eval, args.k)
    print(f"nDCG@{args.k}: {score:.4f}")


if __name__ == "__main__":
    main()
