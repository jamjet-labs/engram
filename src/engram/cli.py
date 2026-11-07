"""`engram-server` CLI — typer-based.

Resolves flags + env, validates startup conditions, then hands off to uvicorn.
Fail-closed: no token means no start, unless the operator asks for an
unauthenticated server explicitly (`--no-auth` on loopback, or
`--allow-unauthenticated` anywhere).

Use `--dry-run` in tests to validate flags without actually binding a port.
"""

from __future__ import annotations

import asyncio
import os

import typer

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
MIN_TOKEN_LEN = 32
ALLOW_UNAUTHENTICATED_ENV = "ENGRAM_ALLOW_UNAUTHENTICATED"
ALLOWED_ORIGINS_ENV = "ENGRAM_ALLOWED_ORIGINS"
ALLOWED_HOSTS_ENV = "ENGRAM_ALLOWED_HOSTS"
_YES = frozenset({"1", "true", "yes", "on"})
_TOKEN_HINT = "python -c 'import secrets; print(secrets.token_urlsafe(32))'"

app = typer.Typer(add_completion=False, help="Engram MCP HTTP server")


def _eprint(msg: str) -> None:
    typer.echo(msg, err=True)


def _print_version() -> None:
    from importlib.metadata import version as _pkg_version

    typer.echo(_pkg_version("jamjet-engram"))
    raise typer.Exit(0)


def _flag_and_env(flag_values: list[str] | None, env_name: str) -> list[str]:
    raw = list(flag_values or [])
    return raw + [v for v in os.environ.get(env_name, "").split(",") if v.strip()]


@app.command()
def main(
    transport: str = typer.Option("http", "--transport", help="Only `http` valid in v0.2."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8765, "--port", help="Bind port."),
    auth_token_env: str = typer.Option(
        "ENGRAM_AUTH_TOKEN", "--auth-token-env", help="Env var name for bearer token."
    ),
    no_auth: bool = typer.Option(False, "--no-auth", help="Disable auth (loopback only)."),
    allow_unauthenticated: bool = typer.Option(
        False,
        "--allow-unauthenticated",
        help=(
            "Start without a token even on a non-loopback host. Anyone who can reach "
            f"the port can read and write all memory. Env: {ALLOW_UNAUTHENTICATED_ENV}=1."
        ),
    ),
    allowed_origin: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--allowed-origin",
        help=(
            "Browser origin (scheme://host[:port]) allowed to call the server. "
            "Repeatable. Other Origin headers get 403; requests without one are "
            f"unaffected. Env: {ALLOWED_ORIGINS_ENV} (comma-separated)."
        ),
    ),
    allowed_host: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--allowed-host",
        help=(
            "Hostname besides 127.0.0.1, localhost and ::1 that requests may use in "
            "the Host header when running without a token. Repeatable. Env: "
            f"{ALLOWED_HOSTS_ENV} (comma-separated)."
        ),
    ),
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

    allow_unauth = (
        allow_unauthenticated
        or os.environ.get(ALLOW_UNAUTHENTICATED_ENV, "").strip().lower() in _YES
    )

    from engram.server.auth import check_token, normalize_host, normalize_origin

    try:
        origins = [normalize_origin(o) for o in _flag_and_env(allowed_origin, ALLOWED_ORIGINS_ENV)]
        hosts = [normalize_host(h) for h in _flag_and_env(allowed_host, ALLOWED_HOSTS_ENV)]
    except ValueError as e:
        _eprint(str(e))
        raise typer.Exit(2) from e

    # Rule 2: a token is required unless the operator opts out explicitly.
    auth_token: str | None = None if no_auth else (os.environ.get(auth_token_env) or None)
    if auth_token is None:
        if not (no_auth or allow_unauth):
            _eprint(
                f"refusing to start: ${auth_token_env} is unset, so the server would "
                "accept requests from anyone who can reach it.\n"
                f"Set ${auth_token_env} to a random token (generate one with: {_TOKEN_HINT}).\n"
                f"With Docker: docker run -e {auth_token_env}=<token> ...\n"
                "To run without a token on this machine only: "
                "--no-auth --host 127.0.0.1.\n"
                "To run without a token on another interface (for example behind a proxy "
                f"that does auth): --allow-unauthenticated or {ALLOW_UNAUTHENTICATED_ENV}=1."
            )
            raise typer.Exit(2)

        # Rule 3: unauthenticated + non-loopback needs the explicit override.
        # Exact IPs only; "localhost" is rejected because /etc/hosts can redirect it.
        if host not in LOOPBACK_HOSTS and not allow_unauth:
            _eprint(
                f"refusing to start: --no-auth with --host {host} would let anyone who can "
                "reach this port read and write every user's memory.\n"
                "Use --host 127.0.0.1 or ::1 (localhost is rejected because /etc/hosts can "
                f"redirect it), set ${auth_token_env}, or pass --allow-unauthenticated "
                f"({ALLOW_UNAUTHENTICATED_ENV}=1) if something in front of the server "
                "does the auth."
            )
            raise typer.Exit(2)

        _eprint(
            f"warning: running without auth on {host}:{port}. Anyone who can reach it can "
            "read and write all memory. Requests must name 127.0.0.1, localhost or ::1 "
            "in the Host header; add other names with --allowed-host."
        )
    else:
        try:
            check_token(auth_token)
        except ValueError as e:
            _eprint(f"refusing to start: ${auth_token_env}: {e}")
            raise typer.Exit(2) from e

    if auth_token is not None and len(auth_token) < MIN_TOKEN_LEN:
        # Rule 4: short-token warning (not blocking)
        _eprint(
            f"warning: ${auth_token_env} is shorter than {MIN_TOKEN_LEN} chars; "
            f"this is weak. Suggested: {_TOKEN_HINT}"
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
            allowed_origins=origins,
            allowed_hosts=hosts,
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
    allowed_origins: list[str] | None = None,
    allowed_hosts: list[str] | None = None,
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
        fast_app = build_http_app(
            engram,
            mcp_server=mcp_server,
            auth_token=auth_token,
            allowed_origins=allowed_origins or (),
            allowed_hosts=allowed_hosts or None,
        )
        config = uvicorn.Config(fast_app, host=host, port=port, log_level=log_level.lower())
        server = uvicorn.Server(config)
        await server.serve()
    finally:
        await engram.close()


if __name__ == "__main__":
    app()
