"""EmbeddingProvider protocol, embedder identity, and the helpers that call an embedder."""

from __future__ import annotations

import math
import warnings
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from engram.errors import EmbeddingResponseError


@dataclass(frozen=True, slots=True)
class EmbedderIdentity:
    """Which embedder made a vector: provider, model and vector size.

    Vectors from two identities do not compare, so the key is what a store
    records and checks.
    """

    provider: str  # "openai", "ollama", "synthetic", or "<module>.<Class>" for others
    model: str
    dim: int

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}/{self.dim}"


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Convert text to dense vector embeddings.

    Implementations live in `engram.embedding.{ollama,openai,synthetic,...}`.

    Three more members are optional:

    - ``identity`` (an `EmbedderIdentity`): names the embedder's vectors.
      Without one, `embedder_identity` makes one from the class and warns.
    - ``async embed_query(text) -> list[float]``: for search queries.
    - ``async embed_documents(texts) -> list[list[float]]``: for memory
      being stored.

    Without the last two, ``embed`` does both. Engram calls an embedder
    through `embed_query` and `embed_documents` in this module, which use
    them when present and check the answer.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimensionality. All vectors returned by `embed` are of this length."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input, each of length `self.dim`."""
        ...


def embedder_identity(e: EmbeddingProvider) -> EmbedderIdentity:
    """The embedder's ``identity``, or one made from its class, with a warning.

    The stand-in is ``("<module>.<Class>", "", e.dim)``: it cannot tell two
    models of one class apart. An ``identity`` that is not an
    `EmbedderIdentity` is a TypeError; one whose size is not ``e.dim`` is a
    ValueError.
    """
    identity: object = getattr(e, "identity", None)
    if identity is None:
        cls = type(e)
        made = EmbedderIdentity(f"{cls.__module__}.{cls.__qualname__}", "", e.dim)
        warnings.warn(
            f"{made.provider} has no identity, so Engram names its vectors {made.key!r}. "
            "Give it an identity (an EmbedderIdentity of provider, model and dim) "
            "so a change of model is caught.",
            UserWarning,
            stacklevel=2,
        )
        return made
    if not isinstance(identity, EmbedderIdentity):
        raise TypeError(f"{type(e).__qualname__}.identity must be an EmbedderIdentity")
    if identity.dim != e.dim:
        raise ValueError(
            f"{type(e).__qualname__}.identity says {identity.dim} dimensions, "
            f"but its dim is {e.dim}"
        )
    return identity


def check_vector(values: Sequence[float], dim: int) -> tuple[float, ...]:
    """``values`` as floats, if they are ``dim`` finite numbers, not all zero.

    Else EmbeddingResponseError. The message holds no value and no text.
    """
    try:
        out = tuple(float(x) for x in values)
    except (TypeError, ValueError):
        raise EmbeddingResponseError("the vector is not a list of numbers") from None
    if len(out) != dim:
        raise EmbeddingResponseError(f"the vector has {len(out)} values, not {dim}")
    if not all(math.isfinite(x) for x in out):
        raise EmbeddingResponseError("the vector holds a NaN or infinite value")
    if math.hypot(*out) == 0.0:
        raise EmbeddingResponseError("the vector is all zeros")
    return out


def _checked(vectors: Sequence[Sequence[float]], count: int, dim: int) -> list[list[float]]:
    # Each vector checked, or none returned. Index, never text, in the message.
    if len(vectors) != count:
        raise EmbeddingResponseError(
            f"the embedder returned {len(vectors)} vectors, expected {count}"
        )
    out: list[list[float]] = []
    for i, values in enumerate(vectors):
        try:
            out.append(list(check_vector(values, dim)))
        except EmbeddingResponseError as e:
            raise EmbeddingResponseError(f"vector {i} of {count}: {e}") from None
    return out


async def embed_query(e: EmbeddingProvider, text: str) -> list[float]:
    """One search query's vector: ``e.embed_query`` if present, else ``e.embed([text])[0]``.

    The vector is checked (`check_vector`); a bad one, or a count other than
    one, raises EmbeddingResponseError.
    """
    method = getattr(e, "embed_query", None)
    if callable(method):
        vectors: list[list[float]] = [await method(text)]
    else:
        vectors = await e.embed([text])
    [vector] = _checked(vectors, 1, e.dim)
    return vector


async def embed_documents(e: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
    """Vectors for memory being stored: ``e.embed_documents`` if present, else ``e.embed``.

    One vector per text, in order, each checked (`check_vector`). A bad
    vector or a wrong count raises EmbeddingResponseError and returns none of
    them. No texts, no call.
    """
    if not texts:
        return []
    method = getattr(e, "embed_documents", None)
    if callable(method):
        vectors: list[list[float]] = await method(list(texts))
    else:
        vectors = await e.embed(list(texts))
    return _checked(vectors, len(texts), e.dim)
