"""CLI flag-validation tests. Does not exercise uvicorn/server lifecycle."""

from __future__ import annotations

from typer.testing import CliRunner

from engram.cli import app

runner = CliRunner()


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
    for name in ("ENGRAM_AUTH_TOKEN", "ENGRAM_ALLOW_UNAUTHENTICATED", "ENGRAM_ALLOWED_ORIGINS"):
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
    monkeypatch.setenv("ENGRAM_AUTH_TOKEN", "t" * 40)
    monkeypatch.setenv("ENGRAM_ALLOWED_ORIGINS", "https://b.example")
    seen = {}

    async def fake_serve(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(cli, "serve_engram", fake_serve)
    result = runner.invoke(
        app, ["--host", "0.0.0.0", "--allowed-origin", "https://a.example", "--db-path", "x.db"]
    )
    assert result.exit_code == 0, result.output
    assert seen["auth_token"] == "t" * 40
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
