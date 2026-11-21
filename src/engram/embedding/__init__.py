"""Embedding providers: turn text into vectors.

`OllamaEmbedding` and `SyntheticEmbedding` are here. Providers with their own
dependencies are imported from their modules, so importing this package does
not load them:

- ``engram.embedding.openai.OpenAIEmbedding``
- ``engram.embedding.vertex.VertexEmbedding`` (``pip install 'jamjet-engram[vertex]'``)
"""

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
