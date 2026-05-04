"""Offline training pipeline for a domain-specialised cross-encoder reranker.

Lives outside ``src/engram/`` because it pulls heavy training-only deps
(sentence-transformers train mode, torch). Install via the ``training`` extra:

    uv pip install '.[training]'
"""
