"""engram-server --embedder and --llm: flags, env, failures and what reaches the server."""

from __future__ import annotations

import logging
import sys
from typing import Any

import pytest
from typer.testing import CliRunner

import engram.cli as cli
from engram.cli import app
from engram.embedding.base import EmbedderIdentity
from engram.llm.base import LLMMessage, LLMResponse

runner = CliRunner()

_ENV = (
    "ENGRAM_AUTH_TOKEN",
    "ENGRAM_ALLOW_UNAUTHENTICATED",
    "ENGRAM_EMBEDDER",
    "ENGRAM_LLM",
    "ENGRAM_VERTEX_LOCATION",
    "GOOGLE_CLOUD_PROJECT",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def _serve_capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def fake_serve(**kwargs: Any) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(cli, "serve_engram", fake_serve)
    return seen


def _run(*args: str) -> Any:
    return runner.invoke(app, ["--no-auth", "--db-path", "x.db", *args])


class _FakeLLM:
    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="[]")


def test_embedder_flag_parses_each_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _serve_capture(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")

    def identity(*args: str) -> EmbedderIdentity:
        result = _run(*args)
        assert result.exit_code == 0, result.output
        ident: EmbedderIdentity = seen["embedder"].identity
        return ident

    assert identity() == EmbedderIdentity("synthetic", "blake2b-v1", 384)
    assert identity("--embedder", "synthetic") == EmbedderIdentity("synthetic", "blake2b-v1", 384)
    assert identity("--embedder", "synthetic:64") == EmbedderIdentity("synthetic", "blake2b-v1", 64)
    assert identity("--embedder", "synthetic", "--embedder-dim", "32").dim == 32
    assert identity("--embedder", "ollama:nomic-embed-text") == EmbedderIdentity(
        "ollama", "nomic-embed-text", 768
    )
    assert identity("--embedder", "ollama:mxbai-embed-large", "--embedder-dim", "1024") == (
        EmbedderIdentity("ollama", "mxbai-embed-large", 1024)
    )
    assert identity("--embedder", "openai:text-embedding-3-small") == EmbedderIdentity(
        "openai", "text-embedding-3-small", 1536
    )
    assert identity("--embedder", "openai:text-embedding-3-small", "--embedder-dim", "256") == (
        EmbedderIdentity("openai", "text-embedding-3-small", 256)
    )
    vertex = identity("--embedder", "vertex:gemini-embedding-001", "--vertex-project", "acme-prod")
    assert vertex == EmbedderIdentity("vertex", "gemini-embedding-001", 768)
    assert "us-central1-aiplatform" in seen["embedder"]._url
    assert "/projects/acme-prod/" in seen["embedder"]._url
    assert seen["on_embedder_change"] == "refuse"
    assert seen["llm"] is None


def test_embedder_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _serve_capture(monkeypatch)
    monkeypatch.setenv("ENGRAM_EMBEDDER", "vertex:gemini-embedding-001")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "acme-env")
    monkeypatch.setenv("ENGRAM_VERTEX_LOCATION", "europe-west4")
    result = _run("--embedder-dim", "512", "--on-embedder-change", "switch")
    assert result.exit_code == 0, result.output
    embedder = seen["embedder"]
    assert embedder.identity == EmbedderIdentity("vertex", "gemini-embedding-001", 512)
    assert "europe-west4-aiplatform" in embedder._url
    assert "/projects/acme-env/" in embedder._url
    assert seen["on_embedder_change"] == "switch"


def test_dry_run_builds_both_objects_without_a_call(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[str] = []
    real_embedder, real_llm = cli.build_embedder, cli.build_llm

    def spy_embedder(*a: Any, **k: Any) -> Any:
        built.append("embedder")
        return real_embedder(*a, **k)

    def spy_llm(*a: Any, **k: Any) -> Any:
        built.append("llm")
        return real_llm(*a, **k)

    monkeypatch.setattr(cli, "build_embedder", spy_embedder)
    monkeypatch.setattr(cli, "build_llm", spy_llm)
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    result = _run(
        "--dry-run",
        "--embedder",
        "vertex:gemini-embedding-001",
        "--vertex-project",
        "acme-prod",
        "--llm",
        "openai:gpt-4o-mini",
    )
    assert result.exit_code == 0, result.output
    assert built == ["embedder", "llm"]


@pytest.mark.parametrize(
    "args",
    [
        ["--embedder", "cohere:embed-v3"],
        ["--embedder", "ollama"],
        ["--embedder", "ollama:"],
        ["--embedder", "synthetic:abc"],
        ["--embedder", "synthetic:0"],
        ["--embedder", "synthetic:64", "--embedder-dim", "32"],
        ["--embedder", "ollama:m", "--embedder-dim", "0"],
        ["--embedder", "openai:text-embedding-ada-002", "--embedder-dim", "256"],
        ["--embedder", "vertex:gemini-embedding-001"],
        ["--on-embedder-change", "merge"],
        ["--llm", "gemini:flash"],
        ["--llm", "openai"],
        ["--llm", "ollama:"],
    ],
)
def test_unknown_scheme_exits_2(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    seen = _serve_capture(monkeypatch)
    result = _run(*args)
    assert result.exit_code == 2, result.output
    assert seen == {}


def test_unknown_scheme_names_the_valid_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run("--embedder", "cohere:embed-v3", "--dry-run")
    assert result.exit_code == 2
    for scheme in ("synthetic", "ollama", "openai", "vertex"):
        assert scheme in result.output
    result = _run("--llm", "gemini:flash", "--dry-run")
    assert result.exit_code == 2
    for scheme in ("none", "openai", "anthropic", "ollama"):
        assert scheme in result.output


def test_vertex_without_the_extra_exits_2_with_the_install_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "google.auth", None)
    seen = _serve_capture(monkeypatch)
    result = _run("--embedder", "vertex:gemini-embedding-001", "--vertex-project", "acme-prod")
    assert result.exit_code == 2, result.output
    assert "pip install 'jamjet-engram[vertex]'" in result.output
    assert seen == {}


def test_llm_openai_without_key_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _serve_capture(monkeypatch)
    result = _run("--llm", "openai:gpt-4o-mini")
    assert result.exit_code == 2, result.output
    assert "OPENAI_API_KEY" in result.output
    assert seen == {}
    result = _run("--llm", "anthropic:claude-haiku-4-5")
    assert result.exit_code == 2, result.output
    assert "ANTHROPIC_API_KEY" in result.output


def test_a_key_is_never_taken_from_a_flag_and_never_printed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run("--llm", "openai:gpt-4o-mini", "--api-key", "sk-secret")
    assert result.exit_code == 2
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    seen = _serve_capture(monkeypatch)
    result = _run("--llm", "openai:gpt-4o-mini")
    assert result.exit_code == 0, result.output
    assert "sk-secret-value" not in result.output
    assert seen["llm"] is not None


def test_llm_ollama_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _serve_capture(monkeypatch)
    monkeypatch.setenv("ENGRAM_LLM", "ollama:llama3.2:3b")
    result = _run()
    assert result.exit_code == 0, result.output
    assert seen["llm"]._model == "llama3.2:3b"


def _patch_server(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import uvicorn

    import engram.server.http as http

    seen: dict[str, Any] = {}
    real_build = http.build_http_app

    def build(engram: Any, **kwargs: Any) -> Any:
        seen["engram"] = engram
        return real_build(engram, **kwargs)

    async def fake_serve(self: Any, sockets: Any = None) -> None:
        return None

    monkeypatch.setattr(http, "build_http_app", build)
    monkeypatch.setattr(uvicorn.Server, "serve", fake_serve)
    return seen


async def _serve(**kwargs: Any) -> None:
    await cli.serve_engram(
        host="127.0.0.1",
        port=9999,
        db_path=":memory:",
        auth_token=None,
        allow_unauthenticated=True,
        log_level="INFO",
        **kwargs,
    )


async def test_llm_flag_turns_extraction_on(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_server(monkeypatch)
    llm = _FakeLLM()
    await _serve(llm=llm)
    assert seen["engram"]._extract is not None
    assert seen["engram"]._events is not None
    seen.clear()
    await _serve()
    assert seen["engram"]._extract is None


async def test_the_embedder_reaches_the_engram_and_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engram.embedding.synthetic import SyntheticEmbedding

    class Closing(SyntheticEmbedding):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    seen = _patch_server(monkeypatch)
    embedder = Closing(dim=48)
    await _serve(embedder=embedder)
    assert seen["engram"]._embed is embedder
    assert embedder.closed


async def test_no_llm_warning_only_without_llm(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_server(monkeypatch)
    with caplog.at_level(logging.INFO, logger="engram.cli"):
        await _serve(llm=_FakeLLM())
    assert "no LLM configured" not in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="engram.cli"):
        await _serve()
    assert "no LLM configured" in caplog.text
    assert "--llm" in caplog.text


async def test_synthetic_embedder_is_logged_as_not_semantic(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from engram.embedding.synthetic import SyntheticEmbedding

    class Other(SyntheticEmbedding):
        @property
        def identity(self) -> EmbedderIdentity:
            return EmbedderIdentity("acme", "stub", self.dim)

    _patch_server(monkeypatch)
    with caplog.at_level(logging.INFO, logger="engram.cli"):
        await _serve()
    assert "not semantic" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="engram.cli"):
        await _serve(embedder=Other(dim=16))
    assert "not semantic" not in caplog.text


def test_bare_command_still_starts_as_the_docker_cmd_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENGRAM_ALLOW_UNAUTHENTICATED", "1")
    result = runner.invoke(
        app, ["--transport", "http", "--host", "0.0.0.0", "--port", "8765", "--dry-run"]
    )
    assert result.exit_code == 0, result.output


async def test_health_and_openapi_report_the_package_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata

    from fastapi.testclient import TestClient

    from engram import Engram
    from engram.server.http import build_http_app
    from engram.server.mcp import build_mcp_server

    real = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "9.9.9" if name == "jamjet-engram" else real(name),
    )
    memory = await Engram.open(":memory:", allow_default_scope=True)
    try:
        for mcp in (None, build_mcp_server(memory, name="engram")):
            app_ = build_http_app(memory, mcp_server=mcp, allow_unauthenticated=True)
            client = TestClient(app_, base_url="http://127.0.0.1")
            assert client.get("/healthz").json() == {"status": "ok"}
            assert client.get("/openapi.json").json()["info"]["version"] == "9.9.9"
    finally:
        await memory.close()
