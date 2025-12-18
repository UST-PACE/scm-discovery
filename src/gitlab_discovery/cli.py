from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

from .gitlab_client import GitLabAPIError, GitLabClient
from .reporting import audit_gitlab

console = Console()
app = typer.Typer(add_completion=False, help="Audit GitLab groups, users, and memberships.")


@app.command("features")
def list_features() -> None:
    """List supported features and parameters."""
    console.print("[bold]Features[/]")
    console.print("[cyan]*[/] Recurses a root group to include all subgroups")
    console.print("[cyan]*[/] Captures group memberships (including inherited access)")
    console.print("[cyan]*[/] Detects bots where available and can enrich with user emails")
    console.print("[cyan]*[/] Writes CSV reports for groups, users, memberships plus a JSON summary")
    console.print("[cyan]*[/] Retries on rate limits and common API errors")
    console.print()
    console.print("[bold]Parameters[/]")
    console.print("[green]>[/] --token: GitLab PAT with read_api (required)")
    console.print("[green]>[/] --root-group: Group ID or full path to start discovery (required)")
    console.print("[green]>[/] --base-url: GitLab instance URL if not using gitlab.com")
    console.print("[green]>[/] --output: Directory to write reports")
    console.print("[green]>[/] --include-emails: Attempt to fetch user emails and bot flags (admin token needed)")


@app.command()
def audit(
    root_group: str | None = typer.Option(
        None,
        "--root-group",
        "-g",
        envvar="GITLAB_ROOT_GROUP",
        help="Root group id or full path to start discovery. Defaults to your top-level group if omitted.",
    ),
    token: str = typer.Option(
        None,
        "--token",
        envvar="GITLAB_TOKEN",
        help="GitLab personal access token with read_api scope.",
    ),
    base_url: str = typer.Option(
        "https://gitlab.com",
        "--base-url",
        envvar="GITLAB_BASE_URL",
        help="GitLab base URL (defaults to gitlab.com). Can be set via GITLAB_BASE_URL.",
    ),
    output: Path = typer.Option(
        Path("reports"),
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
        envvar="OUTPUT_DIR",
        help="Directory to write CSV/JSON reports. Can be set via OUTPUT_DIR.",
    ),
    include_emails: bool = typer.Option(
        False,
        "--include-emails",
        envvar="INCLUDE_EMAILS",
        help="Attempt to fetch user emails (admin token required; best effort). Can be set via INCLUDE_EMAILS.",
    ),
) -> None:
    """
    Fetch all groups/subgroups from ROOT_GROUP and emit membership/user reports.
    """
    if not token:
        raise typer.BadParameter("GitLab token is required via --token or env GITLAB_TOKEN")

    try:
        with GitLabClient(base_url=base_url, token=token) as client:
            resolved_root = root_group
            if resolved_root is None:
                top_levels = list(client.iter_top_level_groups(membership_only=True))
                if not top_levels:
                    console.print("[red]No top-level groups found for this token. Provide --root-group explicitly.[/]")
                    raise typer.Exit(code=1)
                if len(top_levels) > 1:
                    console.print("[yellow]Multiple top-level groups found. Please choose one with --root-group:[/]")
                    for g in top_levels:
                        console.print(f"- {g.get('full_path') or g.get('path')} (id={g.get('id')})")
                    raise typer.Exit(code=1)
                resolved_root = top_levels[0].get("full_path") or str(top_levels[0].get("id"))

            result = audit_gitlab(
                client=client,
                root_group_ref=resolved_root,
                include_emails=include_emails,
                output_dir=output,
                console=console,
            )
    except GitLabAPIError as exc:
        console.print(f"[red]GitLab API error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - safety net
        console.print(f"[red]Unexpected error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("\n[bold green]Audit complete.[/]")
    console.print(f"Groups: {result.summary['groups']}, Users: {result.summary['users']}, Bots: {result.summary['bots']}, Memberships: {result.summary['memberships']}")
    console.print(f"Reports written to {result.output_dir}")
    sys.exit(0)


if __name__ == "__main__":
    app()
