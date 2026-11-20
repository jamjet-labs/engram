"""Embedding providers — turn text into vectors."""

from engram.embedding.base import (
    EmbedderIdentity,
    EmbeddingProvider,
    check_vector,
    embed_documents,
    embed_query,
    embedder_identity,
)
from engram.embedding.ollama import OllamaEmbedding
from engram.embedding.synthetic import SyntheticEmbedding

__all__ = [
    "EmbedderIdentity",
    "EmbeddingProvider",
    "OllamaEmbedding",
    "SyntheticEmbedding",
    "check_vector",
    "embed_documents",
    "embed_query",
    "embedder_identity",
]
