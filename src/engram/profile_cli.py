"""CLI for profile-based Engram context separation."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import typer

from engram.profile import DEFAULT_CONFIG_PATH, ProfileManager

app = typer.Typer(add_completion=False, help="Engram profile/context launcher")


def _profile_manager() -> ProfileManager:
    return ProfileManager(os.environ.get("ENGRAM_PROFILES", DEFAULT_CONFIG_PATH))


@app.command("list")
def list_profiles() -> None:
    """List configured profiles."""
    profiles = _profile_manager().list()
    if not profiles:
        typer.echo("No profiles configured")
        raise typer.Exit(1)
    for name, p in profiles.items():
        typer.echo(f"{name:<12} {p.db:<30} {p.description}")


@app.command()
def init() -> None:
    """Create a default work/personal profile config if missing."""
    path = Path(os.path.expanduser(str(DEFAULT_CONFIG_PATH)))
    if path.exists():
        typer.echo(f"Profiles already exist at {path}")
        return
    pm = ProfileManager(path)
    pm.add(
        "default",
        "~/.engram/engram.db",
        "Default/shared memory (migration from single-DB setup)",
        ["default"],
    )
    pm.add("work", "~/.engram/work.db", "Work projects and context", ["work"])
    pm.add("personal", "~/.engram/personal.db", "Personal learning and projects", ["personal"])
    typer.echo(f"Created profiles at {path}")


@app.command()
def start(profile: str = typer.Argument("default")) -> None:
    """Start the stdio MCP server for a profile."""
    p = _profile_manager().get(profile)
    env = dict(os.environ)
    env["ENGRAM_PROFILE"] = profile
    cmd = [sys.executable, "-m", "engram.server.stdio_mcp", p.db_path]
    raise typer.Exit(subprocess.call(cmd, env=env))


@app.command()
def promote(
    profile: str,
    output: str | None = typer.Option(None, "--output", "-o", help="Write JSONL export."),
) -> None:
    """Export promoted learnings from a profile as sanitized JSONL."""
    records = asyncio.run(_profile_manager().promote(profile, output=output))
    typer.echo(f"Exported {len(records)} promoted learnings from {profile!r}")
    if output:
        typer.echo(f"Written to {output}")
    else:
        for r in records:
            typer.echo(json.dumps(r))


@app.command("import")
def import_learnings(profile: str, source_file: str) -> None:
    """Import promoted learnings into a profile DB."""
    count = asyncio.run(_profile_manager().import_learnings(profile, source_file))
    typer.echo(f"Imported {count} learnings into {profile!r}")


@app.command("copy-promoted")
def copy_promoted(source: str, target: str) -> None:
    """Copy promoted facts between local profile DBs."""
    count = asyncio.run(_profile_manager().copy_promoted(source, target))
    typer.echo(f"Copied {count} promoted facts from {source!r} to {target!r}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
