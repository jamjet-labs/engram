import pytest


@pytest.mark.slow
def test_train_smoke_runs_without_error(tmp_path):
    """Smoke test: train for 1 epoch on 4 pairs, verify it produces a checkpoint dir.

    Marked @slow because it imports torch + sentence-transformers and downloads
    the base model on first run. Skipped automatically when the deps aren't installed.
    """
    pytest.importorskip("sentence_transformers")
    from training.cross_encoder.train import train

    seed_path = tmp_path / "seed.jsonl"
    seed_path.write_text(
        '{"question": "What did I eat?", "passage": "I ate pizza", "label": 1}\n'
        '{"question": "What did I eat?", "passage": "I went hiking", "label": 0}\n'
        '{"question": "Where do I live?", "passage": "I live in Berlin", "label": 1}\n'
        '{"question": "Where do I live?", "passage": "It rained today", "label": 0}\n'
    )
    out = tmp_path / "ckpt"
    train(seed_path=seed_path, out_dir=out, epochs=1, batch_size=2)
    assert out.exists()


def test_train_raises_on_empty_seed(tmp_path):
    """Empty seed → ValueError, not silent zero-epoch run."""
    pytest.importorskip("sentence_transformers")
    from training.cross_encoder.train import train

    seed_path = tmp_path / "empty.jsonl"
    seed_path.write_text("")
    with pytest.raises(ValueError, match="no training examples"):
        train(seed_path=seed_path, out_dir=tmp_path / "ckpt", epochs=1)
