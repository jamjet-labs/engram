"""Check Vertex AI's embedding answer against the test fixtures, with one real call.

Not run by the tests or CI. It sends one short text to Vertex AI with your
application default credentials, so it needs a project with the Vertex AI API
on, and it costs a fraction of a cent.

    uv sync --extra vertex
    gcloud auth application-default login
    uv run python scripts/vertex_smoke.py --project <project> [--location us-central1]
        [--model gemini-embedding-001] [--dim 768]

It prints the shape of the raw answer (keys and value types, not the vector),
whether that shape matches tests/fixtures/vertex/predict_response.json, the
raw vector's length before VertexEmbedding scales it, and the token count.
It exits 1 if the shape differs or the call fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

import httpx

from engram.embedding.vertex import VertexEmbedding
from engram.errors import EmbeddingError

FIXTURE = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "vertex" / "predict_response.json"
)
TEXT = "alice prefers espresso"


class KeepAnswer(httpx.AsyncBaseTransport):
    """Sends through a real transport and keeps the last answer's JSON."""

    def __init__(self, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._inner = inner or httpx.AsyncHTTPTransport()
        self.answer: Any = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        content = await response.aread()  # decoded: Vertex sends gzip
        try:
            self.answer = json.loads(content)
        except ValueError:
            self.answer = None
        # Hand on the decoded body without the headers that describe the
        # encoded one, or the client decodes it a second time and fails.
        headers = [
            (name, value)
            for name, value in response.headers.multi_items()
            if name.lower() not in ("content-encoding", "content-length", "transfer-encoding")
        ]
        return httpx.Response(
            response.status_code, headers=headers, content=content, request=request
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


def shape(value: Any) -> Any:
    """Keys and types, with each list shown by its first item."""
    if isinstance(value, dict):
        return {k: shape(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return (
        "number"
        if isinstance(value, int | float) and not isinstance(value, bool)
        else type(value).__name__
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model", default="gemini-embedding-001")
    parser.add_argument("--dim", type=int, default=768)
    args = parser.parse_args()

    transport = KeepAnswer()
    embedder = VertexEmbedding(
        project=args.project,
        location=args.location,
        model=args.model,
        dim=args.dim,
        transport=transport,
        max_attempts=1,
    )
    try:
        [vector] = await embedder.embed_documents([TEXT])
    except EmbeddingError as e:
        print(f"the call failed: {type(e).__name__}: {e}")
        return 1
    finally:
        await embedder.aclose()

    answer = transport.answer
    expected = shape(json.loads(FIXTURE.read_text(encoding="utf-8")))
    got = shape(answer)
    print("answer shape:")
    print(json.dumps(got, indent=2))
    print("matches the fixture:", got == expected)
    embeddings = answer["predictions"][0]["embeddings"]
    raw = embeddings["values"]
    print(f"raw vector: {len(raw)} values, length {math.hypot(*raw):.6f} before scaling")
    print(f"scaled vector: {len(vector)} values, length {math.hypot(*vector):.6f}")
    print("statistics:", embeddings.get("statistics"))
    return 0 if got == expected else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
