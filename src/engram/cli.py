"""`engram-server` CLI — typer-based.

Resolves flags + env, validates startup conditions (fail-closed on missing
auth token, loopback-only for --no-auth), then hands off to uvicorn.

Use `--dry-run` in tests to validate flags without actually binding a port.
"""

from __future__ import annotations

import asyncio
import os

import typer

LOOPBACK_HOSTS = {"127.0.0.1", "::1"}
MIN_TOKEN_LEN = 32

app = typer.Typer(add_completion=False, help="Engram MCP HTTP server")


def _eprint(msg: str) -> None:
    typer.echo(msg, err=True)


def _print_version() -> None:
    from importlib.metadata import version as _pkg_version

    typer.echo(_pkg_version("jamjet-engram"))
    raise typer.Exit(0)


@app.command()
def main(
    transport: str = typer.Option("http", "--transport", help="Only `http` valid in v0.2."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8765, "--port", help="Bind port."),
    auth_token_env: str = typer.Option(
        "ENGRAM_AUTH_TOKEN", "--auth-token-env", help="Env var name for bearer token."
    ),
    no_auth: bool = typer.Option(False, "--no-auth", help="Disable auth (loopback only)."),
    db_path: str | None = typer.Option(None, "--db-path", help="SQLite path."),
    log_level: str = typer.Option("INFO", "--log-level"),
    version: bool = typer.Option(False, "--version", help="Print version and exit."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate flags, do not bind."),
) -> None:
    """Start the Engram MCP HTTP server."""
    if version:
        _print_version()

    # Rule 1: transport must be http
    if transport != "http":
        _eprint(f"transport={transport} is reserved for a future release")
        raise typer.Exit(2)

    # Rule 2: --no-auth requires loopback (exact IPs only; "localhost" rejected)
    if no_auth and host not in LOOPBACK_HOSTS:
        _eprint(
            "--no-auth is only allowed with --host 127.0.0.1 or ::1 "
            "(localhost is rejected because /etc/hosts can redirect it)"
        )
        raise typer.Exit(2)

    # Rule 3: token required unless --no-auth (fail-closed)
    auth_token: str | None = None
    if not no_auth:
        auth_token = os.environ.get(auth_token_env)
        if not auth_token:
            _eprint(
                f"${auth_token_env} is unset; set it or pass --no-auth (loopback only).\n"
                "Generate a token with: "
                "python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
            raise typer.Exit(2)

        # Rule 4: short-token warning (not blocking)
        if len(auth_token) < MIN_TOKEN_LEN:
            _eprint(
                f"warning: ${auth_token_env} is shorter than {MIN_TOKEN_LEN} chars; "
                "this is weak. Suggested: python -c 'import secrets; "
                "print(secrets.token_urlsafe(32))'"
            )

    if dry_run:
        return

    resolved_db = db_path or os.environ.get("ENGRAM_DB_PATH") or "engram.db"

    asyncio.run(
        serve_engram(
            host=host,
            port=port,
            db_path=resolved_db,
            auth_token=auth_token,
            log_level=log_level,
        )
    )


async def serve_engram(
    *,
    host: str,
    port: int,
    db_path: str,
    auth_token: str | None,
    log_level: str,
) -> None:
    """Open Engram and serve the FastAPI app via uvicorn — all on one event loop.

    Engram.open() creates loop-bound resources (an aiosqlite Connection lives on
    the loop that opened it). If we used asyncio.run(open_engram) followed by
    uvicorn.run(...), the second call would create a different event loop and
    every MCP tool call would explode with "Event loop is closed". Keeping the
    full lifecycle inside one asyncio.run(serve_engram(...)) avoids that.
    """
    # Late imports — keep --version, --help, and dry-run fast.
    import uvicorn

    from engram.engram import Engram
    from engram.server.http import build_http_app
    from engram.server.mcp import build_mcp_server

    engram = await Engram.open(path=db_path)
    try:
        mcp_server = build_mcp_server(engram, name="engram")
        fast_app = build_http_app(engram, mcp_server=mcp_server, auth_token=auth_token)
        config = uvicorn.Config(fast_app, host=host, port=port, log_level=log_level.lower())
        server = uvicorn.Server(config)
        await server.serve()
    finally:
        await engram.close()


if __name__ == "__main__":
    app()
