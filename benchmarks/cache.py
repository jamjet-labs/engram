"""Per-component caches: ingestion snapshots + reader responses.

Per-component (not end-to-end) so swapping one component invalidates only
steps downstream of it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def dataset_hash(oracle_path: str) -> str:
    h = hashlib.sha256()
    with open(oracle_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def snapshot_path(dataset_hash: str, extract_mode: bool, cache_dir: Path) -> Path:
    mode = "extract" if extract_mode else "chunks"
    return cache_dir / f"sqlite_dump_{dataset_hash}_{mode}.db"


class IngestionCache:
    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def has(self, dataset_hash: str, extract_mode: bool) -> bool:
        return snapshot_path(dataset_hash, extract_mode, self._dir).exists()

    def save(self, src_db_path: Path, dataset_hash: str, extract_mode: bool) -> None:
        shutil.copy2(src_db_path, snapshot_path(dataset_hash, extract_mode, self._dir))

    def restore(self, out_path: Path, dataset_hash: str, extract_mode: bool) -> None:
        shutil.copy2(snapshot_path(dataset_hash, extract_mode, self._dir), out_path)


class ReaderCache:
    """Content-addressed reader response cache. Hits = $0."""

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def key(
        self,
        *,
        model: str,
        system: str,
        context: str,
        question: str,
        tools_signature: str,
        n_sample_index: int,
    ) -> str:
        h = hashlib.sha256()
        h.update(
            f"{model}|{system}|{context}|{question}|{tools_signature}|{n_sample_index}".encode()
        )
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / f"reader_{key[:16]}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text())  # type: ignore[no-any-return]

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._path(key).write_text(json.dumps(value))
