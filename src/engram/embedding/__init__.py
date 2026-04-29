"""Embedding providers — turn text into vectors."""

from engram.embedding.base import EmbeddingProvider
from engram.embedding.ollama import OllamaEmbedding
from engram.embedding.synthetic import SyntheticEmbedding

__all__ = ["EmbeddingProvider", "OllamaEmbedding", "SyntheticEmbedding"]
