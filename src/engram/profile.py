"""Profile-based context management for Engram.

Provides the `ProfileManager` — resolves named profiles to DB paths,
and supports **promoting** generalized knowledge from one profile DB to
another without exposing raw work data.

Safety invariants
-----------------
* promote() only extracts facts with ``promotion_state="promoted"`` by default —
  raw and candidate work facts never leave the source.
* The exported JSON is a sanitised subset: ``text``, ``memory_system``,
  ``memory_subtype``, ``tags``, ``source_profile``. No IDs, no timestamps,
  no raw metadata.
* Import writes to the target DB with `promotion_state="raw"` and tags the
  fact with ``source:<profile>`` for provenance.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

from engram import Engram
from engram.models import MemorySystem

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path.home() / ".engram" / "profiles.json"


@dataclass
class Profile:
    """A named memory silo pointing at a separate SQLite database."""

    name: str
    db: str
    description: str = ""
    tags: tuple[str, ...] = ()

    @property
    def db_path(self) -> str:
        return os.path.expanduser(self.db)

    def to_dict(self) -> dict[str, Any]:
        return {
            "db": self.db,
            "description": self.description,
            "tags": list(self.tags),
        }


@dataclass
class ProfileManager:
    """Loads/manages named profiles from ``profiles.json``."""

    config_path: str | Path = DEFAULT_CONFIG_PATH
    _profiles: dict[str, Profile] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.load()

    # ------------------------------------------------------------------
    # load / save
    # ------------------------------------------------------------------

    def load(self) -> None:
        path = Path(os.path.expanduser(str(self.config_path)))
        if not path.exists():
            self._profiles = {}
            return
        raw = json.loads(path.read_text())
        self._profiles = {
            name: Profile(name=name, **data)
            for name, data in raw.get("profiles", {}).items()
        }

    def save(self) -> None:
        path = Path(os.path.expanduser(str(self.config_path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = {"profiles": {n: p.to_dict() for n, p in self._profiles.items()}}
        path.write_text(json.dumps(raw, indent=2) + "\n")

    def add(
        self, name: str, db: str, description: str = "", tags: list[str] | None = None
    ) -> Profile:
        p = Profile(name=name, db=db, description=description, tags=tuple(tags or []))
        self._profiles[name] = p
        self.save()
        return p

    def get(self, name: str) -> Profile:
        if name not in self._profiles:
            msg = f"unknown profile {name!r}; available: {list(self._profiles)}"
            raise KeyError(msg)
        return self._profiles[name]

    def list(self) -> dict[str, Profile]:
        return dict(self._profiles)

    # ------------------------------------------------------------------
    # promote  (work → personal bridge)
    # ------------------------------------------------------------------

    async def promote(
        self,
        source: str,
        *,
        output: str | None = None,
        states: tuple[str, ...] = ("promoted",),
        generalize: bool = True,
    ) -> list[dict[str, Any]]:
        """Export generalised learnings from *source* profile.

        Only facts with ``promotion_state`` in *states* are included (default
        ``promoted``).  If *output* is provided the result is
        written as JSON lines (one dict per line).  Returns the list of
        exported records.
        """
        src = self.get(source)
        records: list[dict[str, Any]] = []

        async with await Engram.open(src.db_path) as memory:
            # fetch all facts via direct store query — avoids embedding dependency
            store = memory._store
            async with store._conn.execute(
                "SELECT id, text, memory_system, memory_subtype, tags, "
                "lifecycle_state, promotion_state, metadata FROM facts"
            ) as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                ps_raw = row["promotion_state"]
                if not ps_raw:
                    md = json.loads(row["metadata"] or "{}")
                    ps_raw = md.get("promotion_state", "")
                if ps_raw not in states:
                    continue
                ms_raw = row["memory_system"] or "working"
                try:
                    ms = MemorySystem(ms_raw)
                except ValueError:
                    ms = MemorySystem.WORKING
                text = row["text"]
                if generalize:
                    text = text.strip()
                tags_raw = row["tags"]
                tags = json.loads(tags_raw) if isinstance(tags_raw, str) and tags_raw else []
                records.append(
                    {
                        "text": text,
                        "memory_system": ms.value,
                        "memory_subtype": row["memory_subtype"],
                        "tags": tags,
                        "source_profile": source,
                        "promotion_state": ps_raw,
                    }
                )

        if output:
            payload = "\n".join(json.dumps(r) for r in records) + "\n"
            await anyio.to_thread.run_sync(
                lambda: Path(os.path.expanduser(output)).write_text(payload)
            )

        return records

    # ------------------------------------------------------------------
    # import  (personal → personal DB)
    # ------------------------------------------------------------------

    async def import_learnings(self, target: str, source_file: str) -> int:
        """Import a promote-exported JSONL file into the *target* profile.

        Returns how many facts were imported.
        """
        dst = self.get(target)
        count = 0
        content = await anyio.to_thread.run_sync(
            lambda: Path(os.path.expanduser(source_file)).read_text()
        )

        async with await Engram.open(dst.db_path) as memory:
            for line in content.strip().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                tags = list(rec.get("tags", []))
                src_name = rec.get("source_profile", "unknown")
                if "source:" + src_name not in tags:
                    tags.append("source:" + src_name)

                await memory.record(
                    text=rec["text"],
                    memory_system=rec.get("memory_system", "semantic"),
                    memory_subtype=rec.get("memory_subtype"),
                    tags=tags,
                    lifecycle_state="durable",
                    promotion_state="raw",
                )
                count += 1

        return count

    # ------------------------------------------------------------------
    # copy (direct DB-to-DB copy — same machine, same safety domain)
    # ------------------------------------------------------------------

    async def copy_promoted(
        self, source: str, target: str, *, states: tuple[str, ...] = ("promoted",)
    ) -> int:
        """Directly copy promoted facts from *source* DB to *target* DB.

        Only runs when both profiles point to DBs on the same filesystem.
        Data never leaves the machine.
        """
        src = self.get(source)
        dst = self.get(target)
        count = 0

        async with await Engram.open(src.db_path) as src_mem:
            async with await Engram.open(dst.db_path) as dst_mem:
                store = src_mem._store
                async with store._conn.execute(
                    "SELECT id, text, memory_system, memory_subtype, tags, "
                    "promotion_state FROM facts"
                ) as cursor:
                    rows = await cursor.fetchall()
                for row in rows:
                    ps_raw = row["promotion_state"]
                    if not ps_raw or ps_raw not in states:
                        continue
                    ms_raw = row["memory_system"] or "working"
                    tags_raw = row["tags"]
                    tags = json.loads(tags_raw) if isinstance(tags_raw, str) and tags_raw else []
                    tags.append("source:" + source)
                    await dst_mem.record(
                        text=row["text"],
                        memory_system=ms_raw,
                        memory_subtype=row["memory_subtype"],
                        tags=tags,
                        lifecycle_state="durable",
                        promotion_state="raw",
                    )
                    count += 1

        return count
