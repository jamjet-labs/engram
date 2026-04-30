"""Per-component caches: ingestion snapshots + reader responses.

Per-component (not end-to-end) so swapping one component invalidates only
steps downstream of it.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


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
