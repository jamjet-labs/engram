"""CLI flag-validation tests. Does not exercise uvicorn/server lifecycle."""

from __future__ import annotations

from typer.testing import CliRunner

from engram.cli import app

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.stdout


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
