"""CLI flag-validation tests. Does not exercise uvicorn/server lifecycle."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from engram.cli import app

runner = CliRunner()
GOOD_TOKEN = "Zt8qL2vNw5RkX0pHc7YmB3sJd9FgA1eU"


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.3.0" in result.stdout


def test_transport_must_be_http():
    result = runner.invoke(app, ["--transport", "stdio"])
    assert result.exit_code == 2
    assert "reserved for a future release" in result.stderr or "reserved" in result.output


def test_no_auth_requires_loopback_127(monkeypatch):
    monkeypatch.delenv("ENGRAM_AUTH_TOKEN", raising=False)
    result = runner.invoke(app, ["--no-auth", "--host", "0.0.0.0", "--dry-run"])
    assert result.exit_code == 2
    output = result.stderr + result.output
    assert "loopback" in output.lower() or "127.0.0.1" in output


def test_no_auth_rejects_localhost_string(monkeypatch):
    monkeypatch.delenv("ENGRAM_AUTH_TOKEN", raising=False)
    result = runner.invoke(app, ["--no-auth", "--host", "localhost", "--dry-run"])
    assert result.exit_code == 2


def test_no_auth_accepts_ipv4_loopback(monkeypatch):
    monkeypatch.delenv("ENGRAM_AUTH_TOKEN", raising=False)
    result = runner.invoke(app, ["--no-auth", "--host", "127.0.0.1", "--dry-run"])
    assert result.exit_code == 0


def test_no_auth_accepts_ipv6_loopback(monkeypatch):
    monkeypatch.delenv("ENGRAM_AUTH_TOKEN", raising=False)
    result = runner.invoke(app, ["--no-auth", "--host", "::1", "--dry-run"])
    assert result.exit_code == 0


def test_missing_token_rejects(monkeypatch):
    monkeypatch.delenv("ENGRAM_AUTH_TOKEN", raising=False)
    result = runner.invoke(app, ["--dry-run"])
    assert result.exit_code == 2
    assert "ENGRAM_AUTH_TOKEN" in (result.stderr + result.output)


def test_token_present_starts(monkeypatch):
    monkeypatch.setenv("ENGRAM_AUTH_TOKEN", "x" * 40)
    result = runner.invoke(app, ["--dry-run"])
    assert result.exit_code == 0


def test_short_token_warns_but_starts(monkeypatch):
    monkeypatch.setenv("ENGRAM_AUTH_TOKEN", "short")
    result = runner.invoke(app, ["--dry-run"])
    assert result.exit_code == 0
    output = result.stderr + result.output
    assert "32" in output or "weak" in output.lower()


def _clear_env(monkeypatch):
    for name in (
        "ENGRAM_AUTH_TOKEN",
        "ENGRAM_ALLOW_UNAUTHENTICATED",
        "ENGRAM_ALLOWED_ORIGINS",
        "ENGRAM_ALLOWED_HOSTS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_non_loopback_without_token_refuses_and_says_why(monkeypatch):
    _clear_env(monkeypatch)
    result = runner.invoke(app, ["--host", "0.0.0.0", "--dry-run"])
    assert result.exit_code != 0
    output = result.stderr + result.output
    assert "ENGRAM_AUTH_TOKEN" in output
    assert "--allow-unauthenticated" in output


def test_no_auth_on_non_loopback_names_the_override(monkeypatch):
    _clear_env(monkeypatch)
    result = runner.invoke(app, ["--no-auth", "--host", "0.0.0.0", "--dry-run"])
    assert result.exit_code != 0
    assert "--allow-unauthenticated" in (result.stderr + result.output)


def test_allow_unauthenticated_flag_permits_non_loopback(monkeypatch):
    _clear_env(monkeypatch)
    result = runner.invoke(app, ["--allow-unauthenticated", "--host", "0.0.0.0", "--dry-run"])
    assert result.exit_code == 0
    assert "without auth" in (result.stderr + result.output).lower()


def test_allow_unauthenticated_env_permits_non_loopback(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ENGRAM_ALLOW_UNAUTHENTICATED", "1")
    result = runner.invoke(app, ["--host", "0.0.0.0", "--dry-run"])
    assert result.exit_code == 0


def test_allow_unauthenticated_env_needs_an_explicit_yes(monkeypatch):
    _clear_env(monkeypatch)
    for value in ("", "0", "false", "no", "off", "maybe"):
        monkeypatch.setenv("ENGRAM_ALLOW_UNAUTHENTICATED", value)
        result = runner.invoke(app, ["--host", "0.0.0.0", "--dry-run"])
        assert result.exit_code != 0, value


def test_docker_cmd_without_token_fails_fast(monkeypatch):
    # The image's default command must not come up open on 0.0.0.0.
    import json
    import re
    from pathlib import Path

    dockerfile = Path(__file__).resolve().parents[2] / "Dockerfile"
    cmd_line = re.search(r"^CMD (\[.*\])$", dockerfile.read_text(), re.MULTILINE)
    assert cmd_line is not None
    argv = json.loads(cmd_line.group(1))
    assert argv[0] == "engram-server"
    _clear_env(monkeypatch)
    result = runner.invoke(app, [*argv[1:], "--dry-run"])
    assert result.exit_code != 0
    assert "ENGRAM_AUTH_TOKEN" in (result.stderr + result.output)


def test_bad_allowed_origin_refuses(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ENGRAM_AUTH_TOKEN", "x" * 40)
    result = runner.invoke(app, ["--allowed-origin", "*", "--dry-run"])
    assert result.exit_code != 0
    monkeypatch.setenv("ENGRAM_ALLOWED_ORIGINS", "https://ok.example, not-an-origin")
    result = runner.invoke(app, ["--dry-run"])
    assert result.exit_code != 0


def test_settings_reach_the_server(monkeypatch):
    # Validation is worthless if the values never reach build_http_app.
    import engram.cli as cli

    _clear_env(monkeypatch)
    monkeypatch.setenv("ENGRAM_AUTH_TOKEN", GOOD_TOKEN)
    monkeypatch.setenv("ENGRAM_ALLOWED_ORIGINS", "https://b.example")
    seen = {}

    async def fake_serve(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(cli, "serve_engram", fake_serve)
    result = runner.invoke(
        app, ["--host", "0.0.0.0", "--allowed-origin", "https://a.example", "--db-path", "x.db"]
    )
    assert result.exit_code == 0, result.output
    assert seen["auth_token"] == GOOD_TOKEN
    # Bound to 0.0.0.0, so the loopback origins are not added.
    assert seen["allowed_origins"] == ["https://a.example", "https://b.example"]
    assert seen["host"] == "0.0.0.0"


def test_no_auth_ignores_the_token_and_passes_none(monkeypatch):
    import engram.cli as cli

    _clear_env(monkeypatch)
    monkeypatch.setenv("ENGRAM_AUTH_TOKEN", "t" * 40)
    seen = {}

    async def fake_serve(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(cli, "serve_engram", fake_serve)
    result = runner.invoke(app, ["--no-auth", "--host", "127.0.0.1", "--db-path", "x.db"])
    assert result.exit_code == 0, result.output
    assert seen["auth_token"] is None


def test_token_with_outer_whitespace_refuses(monkeypatch):
    # A trailing newline from a secrets file would otherwise start a server
    # that rejects every request.
    _clear_env(monkeypatch)
    monkeypatch.setenv("ENGRAM_AUTH_TOKEN", "t" * 40 + "\n")
    result = runner.invoke(app, ["--dry-run"])
    assert result.exit_code != 0
    assert "printable ASCII" in (result.stderr + result.output)


def test_allowed_hosts_reach_the_server_and_are_validated(monkeypatch):
    import engram.cli as cli

    _clear_env(monkeypatch)
    seen = {}

    async def fake_serve(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(cli, "serve_engram", fake_serve)
    monkeypatch.setenv("ENGRAM_ALLOWED_HOSTS", "b.internal")
    result = runner.invoke(app, ["--no-auth", "--allowed-host", "A.internal", "--db-path", "x.db"])
    assert result.exit_code == 0, result.output
    assert seen["allowed_hosts"] == ["a.internal", "b.internal"]

    result = runner.invoke(app, ["--no-auth", "--allowed-host", "http://x", "--dry-run"])
    assert result.exit_code != 0


def test_unauthenticated_warning_names_the_host_check(monkeypatch):
    _clear_env(monkeypatch)
    result = runner.invoke(app, ["--allow-unauthenticated", "--host", "0.0.0.0", "--dry-run"])
    assert result.exit_code == 0
    assert "--allowed-host" in (result.stderr + result.output)


def test_short_token_refused_on_non_loopback_host(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ENGRAM_AUTH_TOKEN", "x" * 15)
    result = runner.invoke(app, ["--host", "0.0.0.0", "--dry-run"])
    assert result.exit_code != 0
    assert "16" in (result.stderr + result.output)
    monkeypatch.setenv("ENGRAM_AUTH_TOKEN", "k7Qp2xVb9LmZ4tRw")
    result = runner.invoke(app, ["--host", "0.0.0.0", "--dry-run"])
    assert result.exit_code == 0


def test_short_token_only_warns_on_loopback(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ENGRAM_AUTH_TOKEN", "x" * 8)
    result = runner.invoke(app, ["--host", "127.0.0.1", "--dry-run"])
    assert result.exit_code == 0
    assert "weak" in (result.stderr + result.output).lower()


def test_own_origin_is_allowed_so_docs_work(monkeypatch):
    # /docs "Try it out" posts with the page's own origin.
    import engram.cli as cli

    _clear_env(monkeypatch)
    seen = {}

    async def fake_serve(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(cli, "serve_engram", fake_serve)
    result = runner.invoke(app, ["--no-auth", "--port", "9999", "--db-path", "x.db"])
    assert result.exit_code == 0, result.output
    for origin in ("http://127.0.0.1:9999", "http://localhost:9999", "http://[::1]:9999"):
        assert origin in seen["allowed_origins"]


async def test_serve_engram_builds_an_unauthenticated_app_on_purpose(monkeypatch):
    # The CLI has already decided no token is allowed; the library default
    # (refuse) must not stop it, and the app it builds must work.
    import uvicorn
    from httpx import ASGITransport, AsyncClient

    import engram.cli as cli

    monkeypatch.setenv("ENGRAM_LLM_PROVIDER", "mock")
    statuses = []

    async def fake_serve(self, sockets=None):
        transport = ASGITransport(app=self.config.app)
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1:9999") as c:
            statuses.append((await c.get("/v1/memory/recall", params={"query": "q"})).status_code)
            statuses.append(
                (
                    await c.post(
                        "/v1/memory/record",
                        headers={"Origin": "http://127.0.0.1:9999"},
                        json={"text": "from the docs page"},
                    )
                ).status_code
            )

    monkeypatch.setattr(uvicorn.Server, "serve", fake_serve)
    await cli.serve_engram(
        host="127.0.0.1",
        port=9999,
        db_path=":memory:",
        auth_token=None,
        allow_unauthenticated=True,
        log_level="INFO",
        allowed_origins=cli._own_origins(9999),
    )
    assert statuses == [200, 200]


@pytest.mark.parametrize(
    "token", ["a" * 16, "ab" * 20, "changemechangeme", "CHANGEME" * 4, "your-token-here-please"]
)
def test_low_entropy_token_refused_on_non_loopback_host(monkeypatch, token):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ENGRAM_AUTH_TOKEN", token)
    result = runner.invoke(app, ["--host", "0.0.0.0", "--dry-run"])
    assert result.exit_code != 0, token
    result = runner.invoke(app, ["--host", "127.0.0.1", "--dry-run"])
    assert result.exit_code == 0, token


def test_own_origins_only_when_bound_to_loopback(monkeypatch):
    import engram.cli as cli

    _clear_env(monkeypatch)
    seen = {}

    async def fake_serve(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(cli, "serve_engram", fake_serve)
    result = runner.invoke(app, ["--allow-unauthenticated", "--host", "0.0.0.0", "--db-path", "x"])
    assert result.exit_code == 0, result.output
    assert seen["allowed_origins"] == []


async def test_serve_engram_applies_the_cli_rules(monkeypatch):
    # Called directly, it must not be a way around the CLI's checks.
    import uvicorn

    import engram.cli as cli

    async def no_serve(self, sockets=None):
        raise AssertionError("server started")

    async def no_open(*args, **kwargs):
        raise AssertionError("opened the database before checking")

    monkeypatch.setattr(uvicorn.Server, "serve", no_serve)
    monkeypatch.setattr("engram.engram.Engram.open", no_open)
    base = {"port": 9999, "db_path": ":memory:", "log_level": "INFO"}
    with pytest.raises(ValueError, match="allow_unauthenticated"):
        await cli.serve_engram(host="127.0.0.1", auth_token=None, **base)
    with pytest.raises(ValueError, match="16"):
        await cli.serve_engram(host="0.0.0.0", auth_token="k7Qp2xVb9", **base)
    with pytest.raises(ValueError):
        await cli.serve_engram(host="0.0.0.0", auth_token="a" * 40, **base)
    with pytest.raises(ValueError, match="printable ASCII"):
        await cli.serve_engram(host="127.0.0.1", auth_token="bad token", **base)
