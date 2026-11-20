"""VertexEmbedding against a fake transport. No test here reaches Google.

The JSON files in tests/fixtures/vertex follow the documented shape of
Vertex AI's predict call for text embeddings and of Google's error body.
They are written from the documentation, not recorded, and the answer's
shape has been checked against a real one. scripts/vertex_smoke.py checks it
with one real call.

Every embedder here gets fake credentials, an httpx.MockTransport, a sleep
that records its delays and moves a fake clock, and a fixed random source,
so retries and deadlines run without waiting.
"""

from __future__ import annotations

import asyncio
import copy
import gzip
import hashlib
import inspect
import json
import math
import subprocess
import sys
import threading
import traceback
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import google.auth
import google.auth.exceptions
import google.auth.transport
import httpcore
import httpx
import pytest
import pytest_asyncio

from engram.embedding import EmbedderIdentity, EmbeddingProvider, embed_query
from engram.embedding.vertex import VertexEmbedding, _auth_request
from engram.errors import (
    EmbeddingAuthError,
    EmbeddingError,
    EmbeddingQuotaError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
)
from scripts.vertex_smoke import KeepAnswer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "vertex"
URL = (
    "https://us-central1-aiplatform.googleapis.com/v1/projects/acme-test/locations/"
    "us-central1/publishers/google/models/gemini-embedding-001:predict"
)
CLOUD_PLATFORM = "https://www.googleapis.com/auth/cloud-platform"
DIM = 8


# ── fakes ───────────────────────────────────────────────────────────


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def utcnow() -> datetime:
    # google-auth keeps expiry as a naive UTC datetime.
    return datetime.now(UTC).replace(tzinfo=None)


class FakeCredentials:
    """Stands in for google-auth credentials: a token, an expiry and refresh()."""

    def __init__(
        self,
        token: str | None = "token-1",
        expires_in: timedelta | None = timedelta(hours=1),
        quota_project_id: str | None = None,
    ) -> None:
        self.token = token
        self.expiry = None if expires_in is None else utcnow() + expires_in
        self.quota_project_id = quota_project_id
        self.refresh_threads: list[int] = []
        self.refresh_requests: list[object] = []
        self.fail_with: Exception | None = None
        self._issued = 1 if token else 0

    def refresh(self, request: object) -> None:
        self.refresh_threads.append(threading.get_ident())
        self.refresh_requests.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        self._issued += 1
        self.token = f"token-{self._issued}"
        self.expiry = utcnow() + timedelta(hours=1)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class RecordedSleep:
    """Records each delay and moves the fake clock by it, without waiting."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        self.clock.now += seconds


def body_of(request: httpx.Request) -> Any:
    return json.loads(request.content)


def texts_of(request: httpx.Request) -> list[str]:
    return [instance["content"] for instance in body_of(request)["instances"]]


def raw_vector(text: str, dim: int) -> list[float]:
    """A model-like vector for `text`: never zero, never unit length."""
    return [(b - 127.5) / 400.0 for b in hashlib.shake_256(text.encode()).digest(dim)]


def unit(values: list[float]) -> list[float]:
    norm = math.hypot(*values)
    return [x / norm for x in values]


def answer(*vectors: list[float], truncated: bool = False) -> Any:
    """The fixture answer, with one prediction per vector."""
    body = load("predict_response.json")
    template = body["predictions"][0]
    body["predictions"] = []
    for values in vectors:
        prediction = copy.deepcopy(template)
        prediction["embeddings"]["values"] = list(values)
        prediction["embeddings"]["statistics"]["truncated"] = truncated
        body["predictions"].append(prediction)
    return body


def ok(request: httpx.Request) -> httpx.Response:
    """A good answer: one vector per text, sized as the request asked."""
    dim = body_of(request)["parameters"]["outputDimensionality"]
    return httpx.Response(200, json=answer(*(raw_vector(t, dim) for t in texts_of(request))))


def error(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    path = FIXTURES / f"error_{status}.json"
    if path.exists():
        body = load(path.name)
    else:  # 500, 502 and 504 share the 503 shape
        body = load("error_503.json")
        body["error"]["code"] = status
    return httpx.Response(status, json=body, headers=headers)


class Script:
    """Answers request n with step n; the last step repeats.

    A step is "ok", a status code, (status code, headers), or an httpx
    exception class to raise.
    """

    def __init__(self, *steps: Any) -> None:
        self.steps = steps
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        if step == "ok":
            return ok(request)
        if isinstance(step, int):
            return error(step)
        if isinstance(step, tuple):
            return error(step[0], step[1])
        raise step("fake transport failure", request=request)


def assert_close(actual: list[list[float]], expected: list[list[float]]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got == pytest.approx(want, abs=1e-12)


def assert_names_no_text(err: BaseException, *texts: str) -> None:
    """No message a traceback of `err` would print holds any of `texts`."""
    messages: list[str] = []
    e: BaseException | None = err
    while e is not None and len(messages) < 20:
        messages.extend(traceback.format_exception_only(e))
        e = e.__cause__ or (None if e.__suppress_context__ else e.__context__)
    shown = "".join(messages)
    for text in texts:
        assert text not in shown


class Harness:
    """Builds VertexEmbeddings on fakes and remembers every request sent."""

    def __init__(self) -> None:
        self.clock = FakeClock()
        self.sleeps = RecordedSleep(self.clock)
        self.credentials = FakeCredentials()
        self.requests: list[httpx.Request] = []
        self.built: list[VertexEmbedding] = []

    def embedder(self, handler: Callable[[httpx.Request], Any], **kw: Any) -> VertexEmbedding:
        async def record(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            result = handler(request)
            if inspect.isawaitable(result):
                result = await result
            assert isinstance(result, httpx.Response)
            return result

        kw.setdefault("project", "acme-test")
        kw.setdefault("dim", DIM)
        kw.setdefault("credentials", self.credentials)
        kw.setdefault("sleep", self.sleeps)
        kw.setdefault("rand", lambda: 1.0)
        kw.setdefault("clock", self.clock)
        e = VertexEmbedding(transport=httpx.MockTransport(record), **kw)
        self.built.append(e)
        return e


@pytest_asyncio.fixture
async def vx() -> AsyncIterator[Harness]:
    h = Harness()
    try:
        yield h
    finally:
        for e in h.built:
            await e.aclose()


@pytest.fixture
def no_project_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)


# ── the request ─────────────────────────────────────────────────────


async def test_request_matches_the_documented_predict_call(vx: Harness) -> None:
    e = vx.embedder(ok, dim=768)
    await e.embed_documents(["alice prefers espresso"])

    [request] = vx.requests
    assert request.method == "POST"
    assert str(request.url) == URL
    assert request.headers["authorization"] == "Bearer token-1"
    assert request.headers["content-type"] == "application/json"
    assert body_of(request) == load("predict_request.json")
    assert "x-goog-user-project" not in request.headers
    # No API key, anywhere.
    assert not request.url.query
    assert not [name for name in request.headers if "key" in name.lower()]
    assert not [p for p in inspect.signature(VertexEmbedding).parameters if "key" in p.lower()]

    assert e.dim == 768
    assert e.identity == EmbedderIdentity("vertex", "gemini-embedding-001", 768)
    assert isinstance(e, EmbeddingProvider)


async def test_the_credentials_quota_project_is_sent(vx: Harness) -> None:
    vx.credentials.quota_project_id = "acme-billing"
    e = vx.embedder(ok)
    await e.embed_documents(["alice prefers espresso"])
    assert vx.requests[0].headers["x-goog-user-project"] == "acme-billing"


@pytest.mark.parametrize(
    "bad",
    [
        {"location": "us-central1.evil.example"},
        {"location": "evil.example/x"},
        {"location": "US-CENTRAL1"},
        {"location": ""},
        {"project": "acme/locations/x"},
        {"project": "../acme"},
        {"project": "acme test"},
        {"project": ""},
        {"model": "gemini-embedding-001/../other"},
        {"model": "gemini-embedding-001?alt=sse"},
        {"model": ""},
    ],
)
def test_location_project_and_model_cannot_change_the_url(bad: dict[str, str]) -> None:
    kw: dict[str, Any] = {"project": "acme-test", "credentials": FakeCredentials(), **bad}
    with pytest.raises(ValueError, match=next(iter(bad))):
        VertexEmbedding(**kw)


@pytest.mark.parametrize(
    "bad",
    [
        {"dim": 0},
        {"dim": -8},
        {"dim": True},
        {"dim": 8.0},
        {"max_batch": 0},
        {"concurrency": 0},
        {"max_attempts": 0},
        {"timeout_s": 0},
        {"deadline_s": -1.0},
        {"timeout_s": float("nan")},
        {"deadline_s": float("inf")},
        {"auto_truncate": "no"},
    ],
)
def test_bad_options_are_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=next(iter(bad))):
        VertexEmbedding(project="acme-test", credentials=FakeCredentials(), **bad)


# ── the answer ──────────────────────────────────────────────────────


async def test_answer_is_normalised_and_in_order(vx: Harness) -> None:
    served = load("predict_response.json")
    fixture_values = served["predictions"][0]["embeddings"]["values"]
    assert math.hypot(*fixture_values) != pytest.approx(
        1.0
    )  # the model's vector is not unit length

    one = vx.embedder(lambda request: httpx.Response(200, json=served))
    [vector] = await one.embed_documents(["alice prefers espresso"])
    assert vector == pytest.approx(unit(fixture_values), abs=1e-12)
    assert math.hypot(*vector) == pytest.approx(1.0, abs=1e-12)

    first, second = raw_vector("first", DIM), [2.5 * x for x in raw_vector("second", DIM)]
    two = vx.embedder(lambda request: httpx.Response(200, json=answer(first, second)), max_batch=2)
    vectors = await two.embed_documents(["note one", "note two"])
    assert_close(vectors, [unit(first), unit(second)])


async def test_gemini_embedding_001_sends_one_text_per_request(vx: Harness) -> None:
    e = vx.embedder(ok)
    await e.embed_documents(["note 0", "note 1", "note 2"])
    assert sorted(texts_of(r)[0] for r in vx.requests) == ["note 0", "note 1", "note 2"]
    assert [len(texts_of(r)) for r in vx.requests] == [1, 1, 1]


async def test_batches_split_by_max_batch_and_reassemble_in_input_order_when_answers_arrive_out_of_order(  # noqa: E501
    vx: Harness,
) -> None:
    texts = [f"note {i}" for i in range(5)]
    arrived: list[int] = []
    finished: list[int] = []
    everyone_in = asyncio.Event()
    turn = {4: asyncio.Event(), 2: asyncio.Event(), 0: asyncio.Event()}
    after = {4: 2, 2: 0}

    async def last_batch_first(request: httpx.Request) -> httpx.Response:
        first = texts.index(texts_of(request)[0])
        arrived.append(first)
        if len(arrived) == 3:
            everyone_in.set()
            turn[4].set()
        await everyone_in.wait()
        await turn[first].wait()
        finished.append(first)
        if first in after:
            turn[after[first]].set()
        return ok(request)

    e = vx.embedder(last_batch_first, max_batch=2)
    vectors = await e.embed_documents(texts)

    assert sorted(tuple(texts_of(r)) for r in vx.requests) == [
        ("note 0", "note 1"),
        ("note 2", "note 3"),
        ("note 4",),
    ]
    assert finished == [4, 2, 0]
    assert_close(vectors, [unit(raw_vector(t, DIM)) for t in texts])


async def test_no_more_than_concurrency_requests_at_once(vx: Harness) -> None:
    in_flight = 0
    peak = 0

    async def slow(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        for _ in range(5):
            await asyncio.sleep(0)
        in_flight -= 1
        return ok(request)

    e = vx.embedder(slow, concurrency=2)
    await asyncio.gather(
        e.embed_documents([f"note {i}" for i in range(4)]),
        e.embed_documents([f"other {i}" for i in range(4)]),
    )
    assert len(vx.requests) == 8
    assert peak == 2


async def test_partial_batch_raises_and_returns_nothing(vx: Harness) -> None:
    def one_short(request: httpx.Request) -> httpx.Response:
        texts = texts_of(request)
        if texts[0] == "note 2":
            texts = texts[:-1]
        return httpx.Response(200, json=answer(*(raw_vector(t, DIM) for t in texts)))

    e = vx.embedder(one_short, max_batch=2)
    with pytest.raises(EmbeddingResponseError) as info:
        await e.embed_documents([f"note {i}" for i in range(6)])
    assert "1 vectors for texts 2 to 3, expected 2" in str(info.value)
    assert_names_no_text(info.value, "note 2", "note 3")

    def one_extra(request: httpx.Request) -> httpx.Response:
        texts = texts_of(request)
        return httpx.Response(200, json=answer(*(raw_vector(t, DIM) for t in [*texts, "x"])))

    extra = vx.embedder(one_extra)
    with pytest.raises(EmbeddingResponseError, match="2 vectors for text 0, expected 1"):
        await extra.embed_documents(["note 0"])


async def test_a_failed_batch_stops_the_others(vx: Harness) -> None:
    started = asyncio.Event()
    stopped: list[str] = []

    async def slow_or_refused(request: httpx.Request) -> httpx.Response:
        if texts_of(request) == ["note 0"]:
            await started.wait()
            return error(400)
        started.set()
        try:
            await asyncio.Event().wait()  # never answers
        except asyncio.CancelledError:
            stopped.append(texts_of(request)[0])
            raise
        raise AssertionError("unreachable")

    e = vx.embedder(slow_or_refused)
    with pytest.raises(EmbeddingError, match="text 0: HTTP 400"):
        await e.embed_documents(["note 0", "note 1"])
    assert stopped == ["note 1"]


@pytest.mark.parametrize("length", [DIM - 1, DIM + 1, 0])
async def test_different_dimension_than_asked_raises(vx: Harness, length: int) -> None:
    e = vx.embedder(lambda request: httpx.Response(200, json=answer([0.25] * length)))
    with pytest.raises(EmbeddingResponseError) as info:
        await e.embed_documents(["alice prefers espresso"])
    assert f"text 0: the vector has {length} values, not {DIM}" in str(info.value)
    assert_names_no_text(info.value, "espresso")


def one_bad(token: str) -> list[str]:
    return ["0.25"] * 3 + [token] + ["0.25"] * (DIM - 4)


# JSON value lists, as Vertex might send them.
BAD_VALUES = {
    "NaN": one_bad("NaN"),
    "infinite": one_bad("Infinity"),
    "all zeros": ["0.0"] * DIM,
    "a string": one_bad('"0.25"'),
    "true": one_bad("true"),
    "null": one_bad("null"),
    "too long to scale": ["1e308"] * DIM,
}


@pytest.mark.parametrize("bad", list(BAD_VALUES), ids=list(BAD_VALUES))
async def test_nan_or_zero_vector_raises(vx: Harness, bad: str) -> None:
    values = BAD_VALUES[bad]
    content = '{"predictions": [{"embeddings": {"values": [' + ", ".join(values) + "]}}]}"
    e = vx.embedder(lambda request: httpx.Response(200, content=content.encode()))
    with pytest.raises(EmbeddingResponseError, match="text 0") as info:
        await e.embed_documents(["alice prefers espresso"])
    assert_names_no_text(info.value, "espresso")


async def test_truncated_flag_raises_when_auto_truncate_is_off(vx: Harness) -> None:
    def cut(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=answer(raw_vector("x", DIM), truncated=True))

    off = vx.embedder(cut)
    with pytest.raises(EmbeddingResponseError, match="cut text 0") as info:
        await off.embed_documents(["a very long note"])
    assert_names_no_text(info.value, "very long")
    assert body_of(vx.requests[-1])["parameters"]["autoTruncate"] is False

    on = vx.embedder(cut, auto_truncate=True)
    [vector] = await on.embed_documents(["a very long note"])
    assert math.hypot(*vector) == pytest.approx(1.0)
    assert body_of(vx.requests[-1])["parameters"]["autoTruncate"] is True


# ── retries ─────────────────────────────────────────────────────────


async def test_429_retried_with_growing_jittered_delays_then_succeeds(vx: Harness) -> None:
    jitter = iter([0.5, 0.25, 0.75])
    e = vx.embedder(Script(429, 429, 429, "ok"), rand=lambda: next(jitter))
    [vector] = await e.embed_documents(["alice prefers espresso"])
    assert len(vx.requests) == 4
    # 0.5, 1 and 2 seconds, each times the random draw
    assert vx.sleeps.delays == [0.25, 0.25, 1.5]
    assert vector == pytest.approx(unit(raw_vector("alice prefers espresso", DIM)))


async def test_jittered_delays_stop_growing_at_the_cap(vx: Harness) -> None:
    e = vx.embedder(Script(*[429] * 7, "ok"), max_attempts=8, deadline_s=1000.0)
    await e.embed_documents(["alice prefers espresso"])
    assert vx.sleeps.delays == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


async def test_retry_after_is_honoured_up_to_the_cap(vx: Harness) -> None:
    later = format_datetime(datetime.now(UTC) + timedelta(hours=1), usegmt=True)
    earlier = format_datetime(datetime.now(UTC) - timedelta(hours=1), usegmt=True)
    e = vx.embedder(
        Script(
            (429, {"Retry-After": "3"}),
            (503, {"Retry-After": "30"}),
            (429, {"Retry-After": later}),
            (429, {"Retry-After": earlier}),
            (429, {"Retry-After": "soon"}),
            "ok",
        ),
        max_attempts=6,
        rand=lambda: 0.1,
    )
    await e.embed_documents(["alice prefers espresso"])
    # Seconds and dates are honoured up to 8 s; one that cannot be read falls back to jitter.
    assert vx.sleeps.delays == [3.0, 8.0, 8.0, 0.0, pytest.approx(8.0 * 0.1)]


async def test_429_until_the_deadline_raises_quota_error(vx: Harness) -> None:
    e = vx.embedder(Script(429), max_attempts=100, deadline_s=10.0)
    with pytest.raises(EmbeddingQuotaError) as info:
        await e.embed_documents(["alice prefers espresso"])
    assert len(vx.requests) == 5
    # The next delay, 8 s, would end past the 10 s deadline, so it is not taken.
    assert vx.sleeps.delays == [0.5, 1.0, 2.0, 4.0]
    assert "text 0" in str(info.value)
    assert_names_no_text(info.value, "espresso")


async def test_429_after_max_attempts_raises_quota_error(vx: Harness) -> None:
    e = vx.embedder(Script(429), max_attempts=3)
    with pytest.raises(EmbeddingQuotaError, match="text 0"):
        await e.embed_documents(["alice prefers espresso"])
    assert len(vx.requests) == 3
    assert vx.sleeps.delays == [0.5, 1.0]


async def test_503_and_timeout_are_retried(vx: Harness) -> None:
    steps = [
        503,
        httpx.ReadTimeout,
        500,
        502,
        504,
        httpx.ConnectTimeout,
        httpx.RemoteProtocolError,
        httpx.ReadError,
        "ok",
    ]
    e = vx.embedder(Script(*steps), max_attempts=len(steps), deadline_s=1000.0)
    [vector] = await e.embed_documents(["alice prefers espresso"])
    assert len(vx.requests) == len(steps)
    assert len(vx.sleeps.delays) == len(steps) - 1
    assert vector == pytest.approx(unit(raw_vector("alice prefers espresso", DIM)))


async def test_timeouts_until_the_deadline_raise_timeout_error(vx: Harness) -> None:
    e = vx.embedder(Script(httpx.ReadTimeout), max_attempts=100, timeout_s=20.0, deadline_s=10.0)
    with pytest.raises(EmbeddingTimeoutError) as info:
        await e.embed_documents(["alice prefers espresso"])
    assert len(vx.requests) == 5
    assert vx.sleeps.delays == [0.5, 1.0, 2.0, 4.0]
    # No try waits longer than the time left before the deadline.
    assert [r.extensions["timeout"]["read"] for r in vx.requests] == [10.0, 9.5, 8.5, 6.5, 2.5]
    assert "text 0" in str(info.value)
    assert_names_no_text(info.value, "espresso")


async def test_5xx_until_max_attempts_raises_timeout_error(vx: Harness) -> None:
    e = vx.embedder(Script(503), max_attempts=3)
    with pytest.raises(EmbeddingTimeoutError, match="HTTP 503"):
        await e.embed_documents(["alice prefers espresso"])
    assert len(vx.requests) == 3


@pytest.mark.parametrize("status", [400, 404])
async def test_400_is_not_retried_and_the_error_names_the_index_not_the_text(
    vx: Harness, status: int
) -> None:
    secret = "alice's door code is 4417"

    def refuse_the_secret(request: httpx.Request) -> httpx.Response:
        if texts_of(request) == [secret]:
            body = load(f"error_{status}.json")
            body["error"]["message"] += f" Input: {secret}"  # the worst case: the text echoed
            return httpx.Response(status, json=body)
        return ok(request)

    e = vx.embedder(refuse_the_secret)
    with pytest.raises(EmbeddingError) as info:
        await e.embed_documents(["alice prefers espresso", secret])
    err = info.value
    assert type(err) is EmbeddingError
    assert f"text 1: HTTP {status}" in str(err)
    assert_names_no_text(err, secret, "4417")
    assert [texts_of(r) for r in vx.requests].count([secret]) == 1
    assert vx.sleeps.delays == []


# ── credentials ─────────────────────────────────────────────────────


async def test_401_refreshes_once_then_raises_auth_error(vx: Harness) -> None:
    e = vx.embedder(Script(401))
    with pytest.raises(EmbeddingAuthError, match="HTTP 401"):
        await e.embed_documents(["alice prefers espresso"])
    assert [r.headers["authorization"] for r in vx.requests] == [
        "Bearer token-1",
        "Bearer token-2",
    ]
    assert len(vx.credentials.refresh_threads) == 1
    assert vx.sleeps.delays == []


async def test_401_then_a_good_answer_after_one_refresh(vx: Harness) -> None:
    e = vx.embedder(Script(401, "ok"))
    await e.embed_documents(["alice prefers espresso"])
    assert len(vx.requests) == 2
    assert len(vx.credentials.refresh_threads) == 1


async def test_requests_that_get_401_together_refresh_once(vx: Harness) -> None:
    both_sent = asyncio.Event()
    sent: list[str] = []

    async def old_token_refused(request: httpx.Request) -> httpx.Response:
        if request.headers["authorization"] == "Bearer token-1":
            sent.append(texts_of(request)[0])
            if len(sent) == 2:
                both_sent.set()
            await both_sent.wait()
            return error(401)
        return ok(request)

    e = vx.embedder(old_token_refused)
    await e.embed_documents(["note 0", "note 1"])
    assert sorted(sent) == ["note 0", "note 1"]
    assert len(vx.credentials.refresh_threads) == 1
    assert len(vx.requests) == 4


async def test_403_is_an_auth_error_at_once(vx: Harness) -> None:
    e = vx.embedder(Script(403))
    with pytest.raises(EmbeddingAuthError, match="HTTP 403 PERMISSION_DENIED"):
        await e.embed_documents(["alice prefers espresso"])
    assert len(vx.requests) == 1
    assert vx.credentials.refresh_threads == []
    assert vx.sleeps.delays == []


@pytest.mark.parametrize("aware", [False, True], ids=["naive expiry", "aware expiry"])
async def test_token_near_expiry_is_refreshed_off_the_event_loop(vx: Harness, aware: bool) -> None:
    expiry = utcnow() + timedelta(minutes=4)
    vx.credentials.expiry = expiry.replace(tzinfo=UTC) if aware else expiry
    e = vx.embedder(ok)
    await e.embed_documents(["note 0", "note 1", "note 2"])  # three requests at once

    assert len(vx.credentials.refresh_threads) == 1
    assert vx.credentials.refresh_threads[0] != threading.get_ident()
    assert isinstance(vx.credentials.refresh_requests[0], google.auth.transport.Request)
    assert {r.headers["authorization"] for r in vx.requests} == {"Bearer token-2"}


async def test_token_with_more_than_five_minutes_left_is_not_refreshed(vx: Harness) -> None:
    vx.credentials.expiry = utcnow() + timedelta(minutes=6)
    e = vx.embedder(ok)
    await e.embed_documents(["alice prefers espresso"])
    assert vx.credentials.refresh_threads == []
    assert vx.requests[0].headers["authorization"] == "Bearer token-1"


async def test_missing_token_is_fetched_before_the_first_request(vx: Harness) -> None:
    vx.credentials = FakeCredentials(token=None, expires_in=None)  # as google-auth hands them out
    e = vx.embedder(ok)
    await e.embed_documents(["alice prefers espresso"])
    assert len(vx.credentials.refresh_threads) == 1
    assert vx.requests[0].headers["authorization"] == "Bearer token-1"


async def test_a_failed_refresh_is_an_auth_error(vx: Harness) -> None:
    vx.credentials.token = None
    vx.credentials.fail_with = google.auth.exceptions.RefreshError("invalid_grant")
    e = vx.embedder(ok)
    with pytest.raises(EmbeddingAuthError, match="refresh"):
        await e.embed_documents(["alice prefers espresso"])
    assert vx.requests == []


async def test_credentials_are_fetched_on_first_use_off_the_event_loop(
    vx: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, tuple[str, ...], object]] = []

    def fake_default(scopes: Any = None, request: Any = None, **kw: Any) -> Any:
        calls.append((threading.get_ident(), tuple(scopes), request))
        return FakeCredentials(token="adc-token"), "adc-project"

    monkeypatch.setattr(google.auth, "default", fake_default)
    e = vx.embedder(ok, credentials=None)
    assert calls == []  # building the embedder makes no call

    await e.embed_documents(["note 0"])
    await e.embed_documents(["note 1"])
    [(thread, scopes, request)] = calls
    assert thread != threading.get_ident()
    assert scopes == (CLOUD_PLATFORM,)
    assert isinstance(request, google.auth.transport.Request)
    assert [r.headers["authorization"] for r in vx.requests] == ["Bearer adc-token"] * 2
    assert "/projects/acme-test/" in str(vx.requests[0].url)  # the project given, not ADC's


async def test_no_application_default_credentials_is_an_auth_error(
    vx: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_adc(*args: Any, **kw: Any) -> Any:
        raise google.auth.exceptions.DefaultCredentialsError("not found")

    monkeypatch.setattr(google.auth, "default", no_adc)
    e = vx.embedder(ok, credentials=None)
    with pytest.raises(EmbeddingAuthError, match="application-default login"):
        await e.embed_documents(["alice prefers espresso"])
    assert vx.requests == []


@pytest.mark.usefixtures("no_project_env")
async def test_project_comes_from_the_argument_then_application_default_credentials(
    vx: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    adc_calls: list[object] = []

    def fake_default(scopes: Any = None, request: Any = None, **kw: Any) -> Any:
        adc_calls.append(request)
        return FakeCredentials(token="adc-token"), "adc-project"

    monkeypatch.setattr(google.auth, "default", fake_default)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")

    given = vx.embedder(ok, project="arg-project", credentials=None)
    await given.embed_documents(["note"])
    assert "/projects/arg-project/" in str(vx.requests[-1].url)

    # google-auth's own project prefers GOOGLE_CLOUD_PROJECT, so the variable
    # is read without asking ADC at construction.
    adc_calls.clear()
    from_env = vx.embedder(ok, project=None, credentials=None)
    assert adc_calls == []
    await from_env.embed_documents(["note"])
    assert "/projects/env-project/" in str(vx.requests[-1].url)

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT")
    adc_calls.clear()
    from_adc = vx.embedder(ok, project=None, credentials=None)
    assert len(adc_calls) == 1  # asked once, at construction, for the project
    await from_adc.embed_documents(["note"])
    assert len(adc_calls) == 1  # and its credentials are kept
    assert "/projects/adc-project/" in str(vx.requests[-1].url)
    assert vx.requests[-1].headers["authorization"] == "Bearer adc-token"


@pytest.mark.usefixtures("no_project_env")
@pytest.mark.parametrize("case", ["credentials given", "no ADC", "ADC without a project"])
def test_no_project_is_refused_at_construction(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    def fake_default(*args: Any, **kw: Any) -> Any:
        if case == "no ADC":
            raise google.auth.exceptions.DefaultCredentialsError("not found")
        return FakeCredentials(), None

    monkeypatch.setattr(google.auth, "default", fake_default)
    credentials = FakeCredentials() if case == "credentials given" else None
    with pytest.raises(EmbeddingError, match="GOOGLE_CLOUD_PROJECT"):
        VertexEmbedding(credentials=credentials)


# ── inputs ──────────────────────────────────────────────────────────


async def test_empty_list_makes_no_request(vx: Harness) -> None:
    vx.credentials.token = None  # a token fetch would show as a refresh
    e = vx.embedder(ok)
    assert await e.embed([]) == []
    assert await e.embed_documents([]) == []
    assert vx.requests == []
    assert vx.credentials.refresh_threads == []


@pytest.mark.parametrize("blank", ["", " ", "\n\t "], ids=["empty", "space", "whitespace"])
async def test_empty_or_blank_string_is_refused_before_any_request(vx: Harness, blank: str) -> None:
    e = vx.embedder(ok)
    with pytest.raises(ValueError, match="text 1 is empty or blank"):
        await e.embed_documents(["alice prefers espresso", blank])
    with pytest.raises(ValueError, match="text 0 is empty or blank"):
        await e.embed_query(blank)
    assert vx.requests == []


async def test_a_text_that_is_not_a_string_is_refused(vx: Harness) -> None:
    e = vx.embedder(ok)
    with pytest.raises(TypeError, match="text 1 is not a string"):
        await e.embed_documents(["alice prefers espresso", None])  # type: ignore[list-item]
    assert vx.requests == []


async def test_embed_query_uses_retrieval_query(vx: Harness) -> None:
    e = vx.embedder(ok)
    vector = await e.embed_query("where does alice work")
    await e.embed(["alice works at acme"])
    await e.embed_documents(["alice prefers espresso"])
    await embed_query(e, "what does alice drink")  # Engram's helper uses it too

    assert vector == pytest.approx(unit(raw_vector("where does alice work", DIM)))
    assert [body_of(r)["instances"] for r in vx.requests] == [
        [{"content": "where does alice work", "task_type": "RETRIEVAL_QUERY"}],
        [{"content": "alice works at acme", "task_type": "RETRIEVAL_DOCUMENT"}],
        [{"content": "alice prefers espresso", "task_type": "RETRIEVAL_DOCUMENT"}],
        [{"content": "what does alice drink", "task_type": "RETRIEVAL_QUERY"}],
    ]


async def test_aclose_closes_the_client(vx: Harness) -> None:
    e = vx.embedder(ok)
    await e.aclose()
    await e.aclose()
    with pytest.raises(EmbeddingError, match="closed"):
        await e.embed_documents(["alice prefers espresso"])
    assert vx.requests == []


# ── packaging ───────────────────────────────────────────────────────


def test_missing_extra_gives_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("google.auth", "google.auth.exceptions", "google.auth.transport"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(EmbeddingError, match=r"pip install 'jamjet-engram\[vertex\]'"):
        VertexEmbedding(project="acme-test", credentials=FakeCredentials())


def test_engram_embedding_imports_neither_vertex_nor_google_auth() -> None:
    code = (
        "import sys, engram, engram.embedding\n"
        "assert 'engram.embedding.vertex' not in sys.modules, 'engram imported vertex'\n"
        "import engram.embedding.vertex\n"
        "assert 'google.auth' not in sys.modules, 'importing vertex imported google.auth'\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_google_auth_requests_go_through_httpx() -> None:
    seen: list[httpx.Request] = []

    def server(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"Metadata-Flavor": "Google", "Content-Type": "application/json"},
            content=b'{"access_token": "t"}',
        )

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        request = _auth_request(client)
        assert isinstance(request, google.auth.transport.Request)
        got = request("http://metadata.google.internal/x", headers={"A": "b"}, timeout=3)
        assert isinstance(got, google.auth.transport.Response)
        assert got.status == 200
        assert got.headers.get("metadata-flavor") == "Google"
        assert got.headers["content-type"] == "application/json"
        assert got.data == b'{"access_token": "t"}'
        request("https://oauth2.googleapis.com/token", method="POST", body=b"grant_type=x")

    assert [r.method for r in seen] == ["GET", "POST"]
    assert seen[0].headers["a"] == "b"
    assert seen[0].extensions["timeout"]["read"] == 3
    assert seen[1].content == b"grant_type=x"

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with httpx.Client(transport=httpx.MockTransport(unreachable)) as client:
        with pytest.raises(google.auth.exceptions.TransportError):
            _auth_request(client)("https://oauth2.googleapis.com/token", method="POST")


# ── the smoke script (scripts/vertex_smoke.py) ──────────────────────


def gzipped(body: Any) -> httpx.Response:
    """A 200 as Vertex really sends it: the JSON gzipped, and Content-Encoding says so."""
    return httpx.Response(
        200,
        headers={"content-type": "application/json; charset=UTF-8", "content-encoding": "gzip"},
        content=gzip.compress(json.dumps(body).encode()),
    )


async def test_the_smoke_script_keeps_a_gzipped_answer_and_hands_it_on(vx: Harness) -> None:
    served = load("predict_response.json")
    keep = KeepAnswer(httpx.MockTransport(lambda request: gzipped(served)))
    e = VertexEmbedding(
        project="acme-test", dim=DIM, credentials=vx.credentials, transport=keep, max_attempts=1
    )
    try:
        [vector] = await e.embed_documents(["alice prefers espresso"])
    finally:
        await e.aclose()
    assert keep.answer == served
    assert vector == pytest.approx(unit(served["predictions"][0]["embeddings"]["values"]))


# ── the network guard (tests/network_guard.py) ──────────────────────


class ReachedTheConnectionPool(Exception):
    pass


async def test_network_guard_blocks_a_real_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The autouse guard stops real HTTP and Google credential lookups.

    Backstops keep this test safe without the guard: the connection pools
    raise before a socket opens, and ADC points at a file that does not
    exist. With the guard gone the test fails on those, and nothing leaves
    the machine.
    """

    def sync_pool(*args: Any, **kw: Any) -> Any:
        raise ReachedTheConnectionPool

    async def async_pool(*args: Any, **kw: Any) -> Any:
        raise ReachedTheConnectionPool

    monkeypatch.setattr(httpcore.ConnectionPool, "handle_request", sync_pool)
    monkeypatch.setattr(httpcore.AsyncConnectionPool, "handle_async_request", async_pool)
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "absent.json"))
    guard = "network guard"

    def blocked_without_a_loop() -> None:
        with pytest.raises(RuntimeError, match=guard):
            httpx.HTTPTransport().handle_request(httpx.Request("POST", URL))
        with httpx.Client() as client, pytest.raises(RuntimeError, match=guard):
            client.post(URL, json={})
        with pytest.raises(RuntimeError, match=guard):
            google.auth.default()

    with pytest.raises(RuntimeError, match=guard):
        await httpx.AsyncHTTPTransport().handle_async_request(httpx.Request("POST", URL))
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError, match=guard):
            await client.post(URL, json={})
    blocked_without_a_loop()

    # An embedder left to find its own credentials and transport stops there too.
    e = VertexEmbedding(project="acme-test")
    try:
        with pytest.raises(RuntimeError, match=guard):
            await e.embed_documents(["alice prefers espresso"])
    finally:
        await e.aclose()
    e = VertexEmbedding(project="acme-test", credentials=FakeCredentials())
    try:
        with pytest.raises(RuntimeError, match=guard):
            await e.embed_documents(["alice prefers espresso"])
    finally:
        await e.aclose()

    # Loopback passes the guard (a local Ollama), and stops at the backstop here.
    with pytest.raises(ReachedTheConnectionPool):
        await httpx.AsyncHTTPTransport().handle_async_request(
            httpx.Request("GET", "http://127.0.0.1:1/")
        )
    with pytest.raises(ReachedTheConnectionPool):
        httpx.HTTPTransport().handle_request(httpx.Request("GET", "http://localhost:1/"))

    # In-process transports are untouched.
    mock = httpx.MockTransport(lambda request: httpx.Response(204))
    async with httpx.AsyncClient(transport=mock) as client:
        assert (await client.get(URL)).status_code == 204
