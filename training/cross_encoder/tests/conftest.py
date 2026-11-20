"""Fixtures for the cross-encoder training tests."""

from __future__ import annotations

import pytest

# The default of train(base_model=...) and evaluate_model(model_path=None).
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.fixture
def offline_base_model(monkeypatch: pytest.MonkeyPatch) -> str:
    """Load the base model from the local Hugging Face cache, with the hub offline.

    Skips when sentence-transformers or the cached model is missing. Tests
    never download it: the network guard would refuse.
    """
    pytest.importorskip("sentence_transformers")
    import huggingface_hub.constants
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(BASE_MODEL, "config.json"), str):
        pytest.skip(f"{BASE_MODEL} is not in the local Hugging Face cache")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    return BASE_MODEL
