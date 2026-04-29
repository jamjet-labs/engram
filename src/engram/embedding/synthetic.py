"""Deterministic synthetic embeddings for tests + offline development.

Produces stable, normalized vectors derived from text bytes. Quality is just
good enough for retrieval correctness tests (substrings hash to nearby vectors)
without needing a real model.
"""

from __future__ import annotations

import hashlib

import numpy as np


class SyntheticEmbedding:
    """Hash-based pseudo-embeddings. Deterministic; fast; no network."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        # Mix text bytes into the vector at hash-determined positions
        v = np.zeros(self._dim, dtype=np.float32)
        # Token-level mixing for substring locality
        for tok in text.lower().split() or [""]:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            seed = int.from_bytes(h, "big")
            rng = np.random.default_rng(seed)
            v += rng.standard_normal(self._dim, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            v[0] = 1.0
            norm = 1.0
        result: list[float] = (v / norm).tolist()
        return result
