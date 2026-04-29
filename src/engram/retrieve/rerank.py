"""Cross-encoder reranking via sentence-transformers.

Lazy-imports `sentence_transformers` so the dep is optional.
Install with: `pip install jamjet-engram[rerank]`.

Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22 MB).
"""

from __future__ import annotations

import asyncio
from typing import Any

from engram.errors import StoreError
from engram.retrieve.base import ScoredFact


class CrossEncoderReranker:
    """Sentence-transformers cross-encoder reranker.

    The model is loaded once at construction and reused. `predict` is sync
    (CPU/GPU bound), so we run it via `asyncio.to_thread` to keep the event
    loop responsive.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise StoreError(
                "sentence-transformers not installed; pip install 'jamjet-engram[rerank]'"
            ) from e
        self._model: Any = CrossEncoder(model_name)
        self._batch_size = batch_size

    async def rerank(
        self, query: str, scored: list[ScoredFact], top_k: int = 10
    ) -> list[ScoredFact]:
        if not scored:
            return scored
        pairs = [(query, sf.fact.text) for sf in scored]
        scores: list[float] = await asyncio.to_thread(self._predict_sync, pairs, self._batch_size)
        for sf, sc in zip(scored, scores, strict=True):
            sf.rerank_score = float(sc)
            sf.score = float(sc)
        scored.sort(key=lambda s: s.rerank_score, reverse=True)
        return scored[:top_k]

    def _predict_sync(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        result = self._model.predict(pairs, batch_size=batch_size)
        # CrossEncoder.predict returns a numpy array
        return [float(x) for x in result]
