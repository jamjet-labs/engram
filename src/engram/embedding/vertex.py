"""Vertex AI text embeddings, with application default credentials.

Install with ``pip install 'jamjet-engram[vertex]'`` and import from this
module (``engram.embedding`` does not import it)::

    from engram.embedding.vertex import VertexEmbedding

    embedder = VertexEmbedding(project="acme-prod", location="us-central1")

Credentials come from ``google.auth.default()`` the first time the embedder
sends a request, unless you pass ``credentials=``. There is no API key
option. On Cloud Run the service account is used; on a laptop run
``gcloud auth application-default login`` first.

The request and answer follow the documented shape of Vertex's ``predict``
call. The tests run against that shape, and ``scripts/vertex_smoke.py``
checks it with one real call.
"""

from __future__ import annotations

import asyncio
import email.utils
import importlib
import math
import os
import random
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from engram.embedding.base import EmbedderIdentity, check_vector
from engram.errors import (
    EmbeddingAuthError,
    EmbeddingError,
    EmbeddingQuotaError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
)

if TYPE_CHECKING:
    from engram.embedding.base import EmbeddingProvider

INSTALL_HINT = "VertexEmbedding needs google-auth: pip install 'jamjet-engram[vertex]'"
SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)

# Texts per predict request. gemini-embedding-001 takes one. A model not
# listed here gets one too, unless the caller passes max_batch.
MAX_BATCH_BY_MODEL: dict[str, int] = {"gemini-embedding-001": 1}

_DOCUMENT = "RETRIEVAL_DOCUMENT"
_QUERY = "RETRIEVAL_QUERY"
_REFRESH_BEFORE_EXPIRY = timedelta(minutes=5)
_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 8.0
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
# A timeout, or a connection that could not be made or was lost.
_RETRY_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
# The location is part of the host name; the project and model are path segments.
_LOCATION = re.compile(r"[a-z][a-z0-9-]{0,62}")
_PROJECT = re.compile(r"[a-z0-9][a-z0-9.:-]{0,127}")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]{0,127}")
_STATUS_NAME = re.compile(r"[A-Z][A-Z_]{0,63}")


class VertexEmbedding:
    """Embed text with a Vertex AI embedding model (default ``gemini-embedding-001``).

    Memory is embedded with task type ``RETRIEVAL_DOCUMENT`` and search
    queries with ``RETRIEVAL_QUERY``. ``embed`` is ``embed_documents``. Each
    vector is checked and scaled to unit length, which leaves cosine scores
    as they are.

    Texts go out in batches of ``max_batch`` (one per request for
    gemini-embedding-001), with at most ``concurrency`` requests in flight
    for this embedder, and come back in input order.

    A request answered 429, 500, 502, 503 or 504, or one that times out or
    loses its connection, is tried again after a jittered delay
    (``Retry-After`` is honoured up to 8 s). It stops after ``max_attempts``
    tries, or when the next delay would end ``deadline_s`` seconds or more
    after its first try, and raises EmbeddingQuotaError if the last answer
    was 429, else EmbeddingTimeoutError. Each try waits at most
    ``timeout_s``. A 401 refreshes the token and tries once more; a second
    401, or a 403, is EmbeddingAuthError. Any other answer, such as 400 or
    404, is EmbeddingError at once.

    If one request fails, or one answer has the wrong count, the wrong size,
    a value that is not a finite number, a zero vector, or is cut to fit the
    model (``truncated``) while ``auto_truncate`` is off, the whole call
    raises and returns nothing. Messages name the text's index, never the
    text.

    ``transport``, ``sleep``, ``rand`` and ``clock`` are for tests. Call
    ``aclose()`` when done.
    """

    def __init__(
        self,
        *,
        project: str | None = None,
        location: str = "us-central1",
        model: str = "gemini-embedding-001",
        dim: int = 768,
        max_batch: int | None = None,
        concurrency: int = 4,
        timeout_s: float = 20.0,
        deadline_s: float = 60.0,
        max_attempts: int = 5,
        auto_truncate: bool = False,
        credentials: Any | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rand: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _require_google_auth()
        _check_name("location", location, _LOCATION)
        _check_name("model", model, _MODEL)
        if project is not None:
            _check_name("project", project, _PROJECT)
        self._dim = _whole_number("dim", dim)
        self._max_batch = _whole_number(
            "max_batch", MAX_BATCH_BY_MODEL.get(model, 1) if max_batch is None else max_batch
        )
        self._concurrency = _whole_number("concurrency", concurrency)
        self._max_attempts = _whole_number("max_attempts", max_attempts)
        self._timeout_s = _seconds("timeout_s", timeout_s)
        self._deadline_s = _seconds("deadline_s", deadline_s)
        if not isinstance(auto_truncate, bool):
            raise ValueError("auto_truncate must be a bool")
        self._auto_truncate = auto_truncate
        self._model = model
        self._credentials = credentials

        # google-auth's own project prefers GOOGLE_CLOUD_PROJECT, so reading
        # the variable first gives the same answer without asking ADC.
        if project is None:
            project = os.environ.get("GOOGLE_CLOUD_PROJECT") or None
            if project is None and credentials is None:
                try:
                    self._credentials, project = _application_default(self._timeout_s)
                except EmbeddingAuthError as e:
                    raise EmbeddingError(
                        _NO_PROJECT + " Application default credentials were not found either."
                    ) from e
            if project is None:
                raise EmbeddingError(_NO_PROJECT)
            _check_name("project", project, _PROJECT)

        self._url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{location}/publishers/google/models/{model}:predict"
        )
        self._sleep = sleep
        self._rand = rand
        self._clock = clock
        self._client = httpx.AsyncClient(transport=transport, timeout=self._timeout_s)
        self._auth_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(self._concurrency)
        self._closed = False

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def identity(self) -> EmbedderIdentity:
        return EmbedderIdentity("vertex", self._model, self._dim)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Same as `embed_documents`."""
        return await self.embed_documents(texts)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """One unit vector per text, in order, for memory being stored."""
        return await self._embed(texts, _DOCUMENT)

    async def embed_query(self, text: str) -> list[float]:
        """One unit vector for a search query."""
        [vector] = await self._embed([text], _QUERY)
        return vector

    async def aclose(self) -> None:
        """Close the HTTP client. Later calls raise EmbeddingError."""
        self._closed = True
        await self._client.aclose()

    async def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        texts = list(texts)
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"text {i} is not a string")
            if not text.strip():
                raise ValueError(f"text {i} is empty or blank; Vertex cannot embed it")
        if not texts:
            return []
        if self._closed:
            raise EmbeddingError("this VertexEmbedding is closed")
        size = self._max_batch
        starts = range(0, len(texts), size)
        if len(starts) == 1:
            return await self._batch(0, texts, task_type)
        tasks = [
            asyncio.create_task(self._batch(start, texts[start : start + size], task_type))
            for start in starts
        ]
        try:
            parts = await asyncio.gather(*tasks)
        except BaseException:
            # One batch failed, or the call was cancelled: stop the rest, return nothing.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return [vector for part in parts for vector in part]

    async def _batch(self, start: int, texts: list[str], task_type: str) -> list[list[float]]:
        async with self._slots:
            return await self._predict(start, texts, task_type)

    async def _predict(self, start: int, texts: list[str], task_type: str) -> list[list[float]]:
        where = _where(start, len(texts))
        body = {
            "instances": [{"content": text, "task_type": task_type} for text in texts],
            "parameters": {"outputDimensionality": self._dim, "autoTruncate": self._auto_truncate},
        }
        deadline = self._clock() + self._deadline_s
        failures = 0
        refreshed = False
        while True:
            token, headers = await self._headers()
            left = deadline - self._clock()
            if left <= 0:
                raise EmbeddingTimeoutError(
                    f"Vertex gave no answer for {where} within {self._deadline_s:g} s"
                )
            last_error: Exception | None = None
            try:
                response = await self._client.post(
                    self._url, json=body, headers=headers, timeout=min(self._timeout_s, left)
                )
            except _RETRY_ERRORS as e:
                last_error, failure, retry_after, quota = e, type(e).__name__, None, False
            except httpx.HTTPError as e:
                raise EmbeddingError(
                    f"Vertex request for {where} failed: {type(e).__name__}"
                ) from e
            else:
                status = response.status_code
                if status == 200:
                    return self._vectors(response, start, len(texts))
                failure = f"HTTP {status}{_status_name(response)}"
                if status == 401 and not refreshed:
                    refreshed = True
                    await self._refresh_after_401(token)
                    continue
                if status in (401, 403):
                    raise EmbeddingAuthError(
                        f"Vertex refused the credentials for {where}: {failure}. Check the "
                        "project, that the Vertex AI API is on, and that the account may "
                        "call it (roles/aiplatform.user)."
                    )
                if status not in _RETRY_STATUSES:
                    raise EmbeddingError(f"Vertex refused {where}: {failure}")
                retry_after = response.headers.get("retry-after")
                quota = status == 429
            failures += 1
            delay = self._delay(failures, retry_after)
            if failures >= self._max_attempts or self._clock() + delay >= deadline:
                if quota:
                    raise EmbeddingQuotaError(
                        f"Vertex answered 429 (over quota) {failures} times for {where}; "
                        "try again later or ask for more quota"
                    ) from last_error
                raise EmbeddingTimeoutError(
                    f"Vertex gave no answer for {where} in {failures} tries (last: {failure})"
                ) from last_error
            await self._sleep(delay)

    def _delay(self, failures: int, retry_after: str | None) -> float:
        if retry_after is not None:
            seconds = _retry_after_s(retry_after)
            if seconds is not None:
                return min(_BACKOFF_CAP_S, seconds)
        # Full jitter: min(cap, base * 2**n) * rand(), n = 0 for the first retry.
        return min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * 2.0 ** min(failures - 1, 16)) * self._rand()

    def _vectors(self, response: httpx.Response, start: int, count: int) -> list[list[float]]:
        where = _where(start, count)
        try:
            payload = response.json()
        except ValueError:
            raise EmbeddingResponseError(f"Vertex's answer for {where} is not JSON") from None
        predictions = payload.get("predictions") if isinstance(payload, dict) else None
        if not isinstance(predictions, list):
            raise EmbeddingResponseError(f"Vertex's answer for {where} has no predictions")
        if len(predictions) != count:
            raise EmbeddingResponseError(
                f"Vertex returned {len(predictions)} vectors for {where}, expected {count}"
            )
        return [self._vector(p, start + j) for j, p in enumerate(predictions)]

    def _vector(self, prediction: object, index: int) -> list[float]:
        embeddings = prediction.get("embeddings") if isinstance(prediction, dict) else None
        values = embeddings.get("values") if isinstance(embeddings, dict) else None
        if not isinstance(embeddings, dict) or not isinstance(values, list):
            raise EmbeddingResponseError(f"Vertex's answer for text {index} has no vector")
        statistics = embeddings.get("statistics")
        if (
            not self._auto_truncate
            and isinstance(statistics, dict)
            and statistics.get("truncated") is True
        ):
            raise EmbeddingResponseError(
                f"Vertex cut text {index} to fit the model's input limit; shorten it, "
                "or pass auto_truncate=True"
            )
        if not all(isinstance(x, int | float) and not isinstance(x, bool) for x in values):
            raise EmbeddingResponseError(f"text {index}: the vector is not a list of numbers")
        try:
            checked = check_vector(values, self._dim)
        except EmbeddingResponseError as e:
            raise EmbeddingResponseError(f"text {index}: {e}") from None
        norm = math.hypot(*checked)
        if not math.isfinite(norm):
            raise EmbeddingResponseError(f"text {index}: the vector is too long to scale")
        return [x / norm for x in checked]

    async def _headers(self) -> tuple[str, dict[str, str]]:
        """The token to send, fetched or refreshed off the event loop when due."""
        async with self._auth_lock:
            if self._credentials is None:
                self._credentials, _ = await asyncio.to_thread(
                    _application_default, self._timeout_s
                )
            credentials = self._credentials
            if _needs_refresh(credentials):
                await asyncio.to_thread(_refresh, credentials, self._timeout_s)
            token = getattr(credentials, "token", None)
            quota_project = getattr(credentials, "quota_project_id", None)
        if not isinstance(token, str) or not token:
            raise EmbeddingAuthError("the Google credentials gave no access token")
        headers = {"authorization": f"Bearer {token}"}
        if isinstance(quota_project, str) and quota_project:
            headers["x-goog-user-project"] = quota_project
        return token, headers

    async def _refresh_after_401(self, sent: str) -> None:
        async with self._auth_lock:
            credentials = self._credentials
            # Another request may have refreshed it while this one waited.
            if getattr(credentials, "token", None) == sent:
                await asyncio.to_thread(_refresh, credentials, self._timeout_s)


_NO_PROJECT = (
    "VertexEmbedding needs a Google Cloud project: pass project=, or set GOOGLE_CLOUD_PROJECT."
)


def _require_google_auth() -> None:
    try:
        for name in ("google.auth", "google.auth.exceptions", "google.auth.transport"):
            importlib.import_module(name)
    except ImportError as e:
        raise EmbeddingError(INSTALL_HINT) from e


def _check_name(what: str, value: object, pattern: re.Pattern[str]) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{what} {value!r:.80} is not a valid Vertex {what} name")


def _whole_number(what: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{what} must be a whole number of at least 1, not {value!r:.40}")
    return value


def _seconds(what: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{what} must be a finite number of seconds above 0, not {value!r:.40}")
    return float(value)


def _where(start: int, count: int) -> str:
    return f"text {start}" if count == 1 else f"texts {start} to {start + count - 1}"


def _status_name(response: httpx.Response) -> str:
    """The status name from Google's error body, as ``" INVALID_ARGUMENT"``, else ``""``.

    Only the name: the error's message may quote the input.
    """
    try:
        payload = response.json()
    except ValueError:
        return ""
    error = payload.get("error") if isinstance(payload, dict) else None
    status = error.get("status") if isinstance(error, dict) else None
    return f" {status}" if isinstance(status, str) and _STATUS_NAME.fullmatch(status) else ""


def _retry_after_s(value: str) -> float | None:
    """Seconds to wait from a Retry-After header (seconds or an HTTP date), else None."""
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - datetime.now(UTC)).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _needs_refresh(credentials: Any) -> bool:
    if not getattr(credentials, "token", None):
        return True
    expiry = getattr(credentials, "expiry", None)
    if not isinstance(expiry, datetime):
        return False
    now = datetime.now(UTC)
    if expiry.tzinfo is None:  # google-auth keeps expiry as naive UTC
        now = now.replace(tzinfo=None)
    return expiry - now <= _REFRESH_BEFORE_EXPIRY


def _application_default(timeout_s: float) -> tuple[Any, str | None]:
    """``google.auth.default()`` for the cloud-platform scope. Blocking: run it off the loop."""
    import google.auth
    import google.auth.exceptions

    try:
        with httpx.Client(timeout=timeout_s) as client:
            credentials, project = google.auth.default(
                scopes=list(SCOPES), request=_auth_request(client)
            )
    except google.auth.exceptions.DefaultCredentialsError as e:
        raise EmbeddingAuthError(
            f"no Google application default credentials ({type(e).__name__}); run "
            "'gcloud auth application-default login', or pass credentials="
        ) from e
    return credentials, project


def _refresh(credentials: Any, timeout_s: float) -> None:
    """Refresh the access token. Blocking: run it off the loop."""
    import google.auth.exceptions

    try:
        with httpx.Client(timeout=timeout_s) as client:
            credentials.refresh(_auth_request(client))
    except google.auth.exceptions.GoogleAuthError as e:
        raise EmbeddingAuthError(
            f"could not refresh the Google access token ({type(e).__name__})"
        ) from e


def _auth_request(client: httpx.Client) -> Any:
    """A google-auth transport Request that sends through ``client``.

    google-auth's own transports need the ``requests`` package; this keeps
    the extra to google-auth alone and every request on httpx.
    """
    from google.auth import exceptions, transport

    class _Response(transport.Response):
        def __init__(self, response: httpx.Response) -> None:
            self._response = response

        @property
        def status(self) -> int:
            return self._response.status_code

        @property
        def headers(self) -> Mapping[str, str]:
            return self._response.headers

        @property
        def data(self) -> bytes:
            return self._response.content

    class _Request(transport.Request):
        def __call__(
            self,
            url: str,
            method: str = "GET",
            body: bytes | None = None,
            headers: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: Any,
        ) -> _Response:
            try:
                response = client.request(
                    method,
                    url,
                    content=body,
                    headers=headers,
                    timeout=httpx.USE_CLIENT_DEFAULT if timeout is None else timeout,
                )
            except httpx.HTTPError as e:
                # google-auth's exception classes take untyped *args.
                raise exceptions.TransportError(  # type: ignore[no-untyped-call]
                    f"{type(e).__name__} from {httpx.URL(url).host}"
                ) from e
            return _Response(response)

    return _Request()


if TYPE_CHECKING:

    def _is_an_embedding_provider(e: VertexEmbedding) -> EmbeddingProvider:
        return e
