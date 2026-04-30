"""Fine-tune a cross-encoder on synthetic LongMemEval pairs.

Run::

    uv pip install '.[training]'
    uv run python -m training.cross_encoder.train \
        --seed benchmarks/cache/ce_train.jsonl \
        --out training/cross_encoder/checkpoints/v1 \
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def train(
    seed_path: Path,
    out_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    show_progress: bool = False,
) -> None:
    """Fine-tune ``base_model`` on (question, passage, label) pairs in ``seed_path``.

    Lazy-imports torch + sentence-transformers so importing this module doesn't
    pay the heavy dep cost on machines that only need the eval helper.
    """
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    examples: list[InputExample] = []
    for line in seed_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        examples.append(
            InputExample(texts=[rec["question"], rec["passage"]], label=float(rec["label"]))
        )
    if not examples:
        raise ValueError(f"no training examples in {seed_path}")

    model = CrossEncoder(base_model, num_labels=1)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=loader,
        epochs=epochs,
        warmup_steps=max(1, len(loader) // 10),
        output_path=str(out_dir),
        show_progress_bar=show_progress,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--base-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HF model id for the cross-encoder base",
    )
    p.add_argument("--progress", action="store_true")
    args = p.parse_args()
    train(
        seed_path=args.seed,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_model=args.base_model,
        show_progress=args.progress,
    )


if __name__ == "__main__":
    main()
