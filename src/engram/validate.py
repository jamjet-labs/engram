"""Text, JSON and limit rules shared by the facade and every store.

The facade calls these before any embedder or LLM call. Each store calls them
again in every write, so a caller that skips the facade meets the same rules.
Every check raises ``ValueError``. A message names the field and the rule,
never the value.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

MAX_FACT_TEXT_BYTES = 65_536
"""UTF-8 bytes. One limit on every backend (Postgres refuses a tsvector over 1 MiB)."""

MAX_MESSAGE_BYTES = 1_048_576
MAX_METADATA_BYTES = 65_536
"""Size of ``json.dumps(metadata, allow_nan=False)``."""

MAX_TOP_K = 1_000
"""Most results one recall returns."""

MAX_STORE_LIMIT = 10_000
"""Most rows a store call is asked for. A recall of MAX_TOP_K asks the store
for up to ten times that many candidates."""


def check_text(name: str, value: str, *, max_bytes: int, allow_blank: bool = False) -> str:
    """Return ``value`` if it is storable text, else raise ``ValueError``.

    Refused: a value that is not a ``str``, a NUL character, a lone surrogate,
    more than ``max_bytes`` UTF-8 bytes, and (unless ``allow_blank``) text that
    is empty or only whitespace.
    """
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain a NUL character")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        raise ValueError(f"{name} must not contain a lone surrogate") from None
    if size > max_bytes:
        raise ValueError(f"{name} is {size} bytes; the limit is {max_bytes}")
    if not allow_blank and not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value


def check_json(name: str, value: Mapping[str, Any], *, max_bytes: int = MAX_METADATA_BYTES) -> None:
    """Raise ``ValueError`` unless ``value`` stores as the same JSON everywhere.

    Refused: NaN and Infinity, a NUL character or lone surrogate in any key or
    string value, a value JSON cannot hold, and more than ``max_bytes`` of
    ``json.dumps`` output.
    """
    try:
        _check_strings(name, value)
        dumped = json.dumps(value, allow_nan=False)
    except RecursionError:
        raise ValueError(f"{name} is nested too deeply") from None
    except (TypeError, ValueError) as e:
        if isinstance(e, ValueError) and str(e).startswith(name):
            raise
        raise ValueError(f"{name} must be plain JSON without NaN or Infinity") from None
    size = len(dumped.encode("utf-8"))
    if size > max_bytes:
        raise ValueError(f"{name} is {size} bytes as JSON; the limit is {max_bytes}")


def _check_strings(name: str, node: Any) -> None:
    if isinstance(node, str):
        check_text(name, node, max_bytes=MAX_METADATA_BYTES, allow_blank=True)
    elif isinstance(node, Mapping):
        for key, item in node.items():
            _check_strings(name, key)
            _check_strings(name, item)
    elif isinstance(node, list | tuple):
        for item in node:
            _check_strings(name, item)


def check_limit(name: str, value: int, *, maximum: int) -> int:
    """Return ``value`` if it is an int from 0 to ``maximum``, else raise ``ValueError``.

    A bool is refused. Zero is allowed: the caller returns an empty result.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    if value > maximum:
        raise ValueError(f"{name} is {value}; the limit is {maximum}")
    return value
