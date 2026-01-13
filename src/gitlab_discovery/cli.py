from __future__ import annotations

import sys
import logging
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv
import json
from dataclasses import asdict
from urllib.parse import urlparse
from collections import defaultdict
from base64 import b64decode

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

from .gitlab_client import GitLabAPIError, GitLabClient
from .reporting import AuditResult, audit_gitlab, write_combined_membership_report
from .repo_reporting import LargeFileReport, audit_repositories

LOG_LEVEL = os.getenv("GITLAB_DISCOVERY_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s",
)
log = logging.getLogger(__name__)

DOTENV_PATH = Path(os.getenv("GITLAB_DISCOVERY_DOTENV", ".env"))
if DOTENV_PATH.exists():
    load_dotenv(dotenv_path=DOTENV_PATH, override=False)
    log.info("Loaded environment variables from %s", DOTENV_PATH)

LFS_THRESHOLD_MB = int(os.getenv("GITLAB_DISCOVERY_LFS_THRESHOLD_MB", "100"))
LFS_THRESHOLD_BYTES = LFS_THRESHOLD_MB * 1024 * 1024

console = Console()
app = typer.Typer(add_completion=False, help="Audit GitLab groups, users, and memberships.")


def _safe_group_dir_name(ref: str) -> str:
    clean = ref.strip().strip("/")
    clean = clean.replace("/", "__")
    return clean or "group"


def _normalize_ref(ref: str | None) -> str | None:
    if not ref:
        return None
    if ref.startswith("http"):
        parsed = urlparse(ref)
        path = parsed.path.lstrip("/")
        if path.endswith(".git"):
            path = path[:-4]
        return path
    return ref.strip()


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
    console.print("[green]>[/] --root-group: Group ID or full path to start discovery (optional; omit to audit every accessible top-level group)")
    console.print("[green]>[/] --base-url: GitLab instance URL if not using gitlab.com")
    console.print("[green]>[/] --output: Directory to write reports")
    console.print("[green]>[/] --include-emails: Attempt to fetch user emails and bot flags (admin token needed)")


@app.command("list-repos")
def list_repositories(
    root_group: str | None = typer.Option(
        None,
        "--root-group",
        "-g",
        envvar="GITLAB_ROOT_GROUP",
        help="Group id or full path to list repositories under. Omit to list every accessible repository.",
    ),
    include_subgroups: bool = typer.Option(
        True,
        "--include-subgroups/--no-include-subgroups",
        help="When using --root-group, include repositories from all nested subgroups.",
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
        Path("output"),
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
        envvar="OUTPUT_DIR",
        help="Directory to write repository listing CSV/JSON. Can be set via OUTPUT_DIR.",
    ),
    parallel_pages: int = typer.Option(
        1,
        "--parallel-pages",
        envvar="GITLAB_DISCOVERY_PARALLEL_PAGES",
        help="Number of API pages to fetch concurrently (>=1). Use 1 to disable.",
    ),
) -> None:
    """Count and list repositories accessible to the token (optionally scoped to a group)."""
    if not token:
        raise typer.BadParameter("GitLab token is required via --token or env GITLAB_TOKEN")
    if parallel_pages < 1:
        raise typer.BadParameter("--parallel-pages must be >= 1")

    parallel = parallel_pages if parallel_pages > 1 else None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    scope_dir = _safe_group_dir_name(root_group) if root_group else "all-projects"
    run_root = output / timestamp / f"repo-list-{scope_dir}"
    run_root.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "name",
        "path_with_namespace",
        "visibility",
        "web_url",
        "archived",
        "empty_repo",
        "default_branch",
        "last_activity_at",
        "created_at",
        "namespace_full_path",
    ]

    try:
        with GitLabClient(base_url=base_url, token=token) as client:
            if root_group:
                group = client.get_group(root_group)
                root_label = group.get("full_path") or group.get("path") or str(group.get("id"))
                console.print(f"[cyan]Listing repositories under group[/] [bold]{root_label}[/] (include_subgroups={include_subgroups})")
                projects = list(
                    client.iter_group_projects(
                        int(group["id"]),
                        include_subgroups=include_subgroups,
                        parallel_pages=parallel,
                    )
                )
            else:
                root_label = "all-accessible"
                console.print("[cyan]Listing all repositories accessible to this token.[/]")
                projects = list(
                    client.iter_projects(
                        membership_only=True,
                        include_statistics=False,
                        simple=True,
                        parallel_pages=parallel,
                    )
                )
    except GitLabAPIError as exc:
        console.print(f"[red]GitLab API error while listing repositories:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - safety net
        console.print(f"[red]Unexpected error while listing repositories:[/] {exc}")
        raise typer.Exit(code=1) from exc

    csv_path = run_root / "repositories.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for repo in projects:
            namespace = repo.get("namespace") or {}
            writer.writerow(
                {
                    "id": repo.get("id"),
                    "name": repo.get("name"),
                    "path_with_namespace": repo.get("path_with_namespace") or repo.get("path"),
                    "visibility": repo.get("visibility"),
                    "web_url": repo.get("web_url"),
                    "archived": repo.get("archived"),
                    "empty_repo": repo.get("empty_repo"),
                    "default_branch": repo.get("default_branch"),
                    "last_activity_at": repo.get("last_activity_at"),
                    "created_at": repo.get("created_at"),
                    "namespace_full_path": namespace.get("full_path"),
                }
            )

    summary_path = run_root / "repositories.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "root": root_label,
                "include_subgroups": include_subgroups,
                "count": len(projects),
                "repositories": [
                    {
                        "id": repo.get("id"),
                        "name": repo.get("name"),
                        "path_with_namespace": repo.get("path_with_namespace") or repo.get("path"),
                        "visibility": repo.get("visibility"),
                        "web_url": repo.get("web_url"),
                        "archived": repo.get("archived"),
                        "empty_repo": repo.get("empty_repo"),
                        "default_branch": repo.get("default_branch"),
                        "last_activity_at": repo.get("last_activity_at"),
                        "created_at": repo.get("created_at"),
                        "namespace_full_path": (repo.get("namespace") or {}).get("full_path"),
                    }
                    for repo in projects
                ],
            },
            fh,
            indent=2,
            sort_keys=True,
            default=str,
        )

    console.print("\n[bold green]Repository listing complete.[/]")
    console.print(f"Repositories found: {len(projects)}")
    for repo in projects:
        console.print(f"- {repo.get('path_with_namespace') or repo.get('path')} (id={repo.get('id')}, visibility={repo.get('visibility')})")
    console.print(f"[blue]Reports written to {run_root}[/]")


@app.command("list-groups")
def list_groups(
    root_group: str | None = typer.Option(
        None,
        "--root-group",
        "-g",
        envvar="GITLAB_ROOT_GROUP",
        help="Group id or full path to list groups under. Omit to list top-level groups.",
    ),
    include_subgroups: bool = typer.Option(
        True,
        "--include-subgroups/--no-include-subgroups",
        help="Include subgroups recursively when listing groups.",
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
        Path("output"),
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
        envvar="OUTPUT_DIR",
        help="Directory to write group listing CSV/JSON. Can be set via OUTPUT_DIR.",
    ),
    parallel_pages: int = typer.Option(
        1,
        "--parallel-pages",
        envvar="GITLAB_DISCOVERY_PARALLEL_PAGES",
        help="Number of API pages to fetch concurrently (>=1). Use 1 to disable.",
    ),
) -> None:
    """List accessible groups (optionally scoped to a root group)."""
    if not token:
        raise typer.BadParameter("GitLab token is required via --token or env GITLAB_TOKEN")
    if parallel_pages < 1:
        raise typer.BadParameter("--parallel-pages must be >= 1")

    parallel = parallel_pages if parallel_pages > 1 else None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    scope_dir = _safe_group_dir_name(root_group) if root_group else "top-level"
    run_root = output / timestamp / f"groups-{scope_dir}"
    run_root.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "name",
        "path",
        "full_path",
        "visibility",
        "web_url",
        "parent_id",
    ]

    def _flatten_groups(
        client: GitLabClient,
        roots: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        collected: list[dict[str, object]] = []
        queue = list(roots)
        while queue:
            group = queue.pop(0)
            collected.append(group)
            if not include_subgroups:
                continue
            for subgroup in client.iter_subgroups(int(group["id"]), parallel_pages=parallel):
                queue.append(subgroup)
        return collected

    try:
        with GitLabClient(base_url=base_url, token=token) as client:
            if root_group:
                root = client.get_group(root_group)
                root_label = root.get("full_path") or root.get("path") or str(root.get("id"))
                console.print(
                    f"[cyan]Listing groups under[/] [bold]{root_label}[/] (include_subgroups={include_subgroups})"
                )
                groups = _flatten_groups(client, [root])
            else:
                console.print("[cyan]Listing top-level groups accessible to this token.[/]")
                top_levels = list(client.iter_top_level_groups(membership_only=True, parallel_pages=parallel))
                groups = _flatten_groups(client, top_levels)
    except GitLabAPIError as exc:
        console.print(f"[red]GitLab API error while listing groups:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - safety net
        console.print(f"[red]Unexpected error while listing groups:[/] {exc}")
        raise typer.Exit(code=1) from exc

    csv_path = run_root / "groups.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            writer.writerow(
                {
                    "id": group.get("id"),
                    "name": group.get("name"),
                    "path": group.get("path"),
                    "full_path": group.get("full_path"),
                    "visibility": group.get("visibility"),
                    "web_url": group.get("web_url"),
                    "parent_id": group.get("parent_id"),
                }
            )

    summary_path = run_root / "groups.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "root": root_group or "top-level",
                "include_subgroups": include_subgroups,
                "count": len(groups),
                "groups": [
                    {
                        "id": group.get("id"),
                        "name": group.get("name"),
                        "path": group.get("path"),
                        "full_path": group.get("full_path"),
                        "visibility": group.get("visibility"),
                        "web_url": group.get("web_url"),
                        "parent_id": group.get("parent_id"),
                    }
                    for group in groups
                ],
            },
            fh,
            indent=2,
            sort_keys=True,
            default=str,
        )

    console.print("\n[bold green]Group listing complete.[/]")
    console.print(f"Groups found: {len(groups)}")
    for group in groups:
        console.print(f"- {group.get('full_path') or group.get('path')} (id={group.get('id')})")
    console.print(f"[blue]Reports written to {run_root}[/]")


@app.command("list-repo-urls")
def list_repo_urls(
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
        Path("output"),
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
        envvar="OUTPUT_DIR",
        help="Directory to write repository URL listing. Can be set via OUTPUT_DIR.",
    ),
    membership_only: bool = typer.Option(
        True,
        "--membership-only/--all-accessible",
        help="Limit to projects you are a member of (default). Disable to list all accessible projects.",
    ),
    parallel_pages: int = typer.Option(
        1,
        "--parallel-pages",
        envvar="GITLAB_DISCOVERY_PARALLEL_PAGES",
        help="Number of API pages to fetch concurrently (>=1). Use 1 to disable.",
    ),
) -> None:
    """List all accessible repositories (name + URLs only)."""
    if not token:
        raise typer.BadParameter("GitLab token is required via --token or env GITLAB_TOKEN")
    if parallel_pages < 1:
        raise typer.BadParameter("--parallel-pages must be >= 1")

    parallel = parallel_pages if parallel_pages > 1 else None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_root = output / timestamp / "repo-urls"
    run_root.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "path_with_namespace",
        "web_url",
        "ssh_url_to_repo",
        "http_url_to_repo",
    ]

    try:
        with GitLabClient(base_url=base_url, token=token) as client:
            console.print(
                f"[cyan]Listing repository names and URLs (membership_only={membership_only}).[/]"
            )
            projects = list(
                client.iter_projects(
                    membership_only=membership_only,
                    include_statistics=False,
                    simple=True,
                    parallel_pages=parallel,
                )
            )
    except GitLabAPIError as exc:
        console.print(f"[red]GitLab API error while listing repository URLs:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - safety net
        console.print(f"[red]Unexpected error while listing repository URLs:[/] {exc}")
        raise typer.Exit(code=1) from exc

    csv_path = run_root / "repository_urls.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for repo in projects:
            writer.writerow(
                {
                    "name": repo.get("name"),
                    "path_with_namespace": repo.get("path_with_namespace") or repo.get("path"),
                    "web_url": repo.get("web_url"),
                    "ssh_url_to_repo": repo.get("ssh_url_to_repo"),
                    "http_url_to_repo": repo.get("http_url_to_repo"),
                }
            )

    summary_path = run_root / "repository_urls.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "count": len(projects),
                "membership_only": membership_only,
                "repositories": [
                    {
                        "name": repo.get("name"),
                        "path_with_namespace": repo.get("path_with_namespace") or repo.get("path"),
                        "web_url": repo.get("web_url"),
                        "ssh_url_to_repo": repo.get("ssh_url_to_repo"),
                        "http_url_to_repo": repo.get("http_url_to_repo"),
                    }
                    for repo in projects
                ],
            },
            fh,
            indent=2,
            sort_keys=True,
            default=str,
        )

    console.print("\n[bold green]Repository URL listing complete.[/]")
    console.print(f"Repositories found: {len(projects)}")
    for repo in projects:
        console.print(f"- {repo.get('path_with_namespace') or repo.get('path')}: {repo.get('web_url')}")
    console.print(f"[blue]Reports written to {run_root}[/]")


@app.command("find-duplicate-repos")
def find_duplicate_repos(
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
        Path("output"),
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
        envvar="OUTPUT_DIR",
        help="Directory to write duplicate repository report. Can be set via OUTPUT_DIR.",
    ),
    membership_only: bool = typer.Option(
        True,
        "--membership-only/--all-accessible",
        help="Limit to repositories you are a member of (default). Disable to list all accessible repositories.",
    ),
) -> None:
    """Find duplicate repository names across accessible projects."""
    if not token:
        raise typer.BadParameter("GitLab token is required via --token or env GITLAB_TOKEN")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_root = output / timestamp / "duplicate-repos"
    run_root.mkdir(parents=True, exist_ok=True)

    try:
        with GitLabClient(base_url=base_url, token=token) as client:
            console.print(
                f"[cyan]Scanning repositories for duplicate names (membership_only={membership_only}).[/]"
            )
            projects = list(
                client.iter_projects(
                    membership_only=membership_only,
                    include_statistics=False,
                    simple=True,
                )
            )
    except GitLabAPIError as exc:
        console.print(f"[red]GitLab API error while listing repositories:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - safety net
        console.print(f"[red]Unexpected error while listing repositories:[/] {exc}")
        raise typer.Exit(code=1) from exc

    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for repo in projects:
        name = repo.get("name") or repo.get("path") or ""
        groups[name].append(repo)

    duplicates = {name: repos for name, repos in groups.items() if len(repos) > 1}

    csv_path = run_root / "duplicate_repositories.csv"
    fieldnames = ["name", "path_with_namespace", "id", "visibility", "web_url"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for name, repos in sorted(duplicates.items()):
            for repo in repos:
                writer.writerow(
                    {
                        "name": name,
                        "path_with_namespace": repo.get("path_with_namespace") or repo.get("path"),
                        "id": repo.get("id"),
                        "visibility": repo.get("visibility"),
                        "web_url": repo.get("web_url"),
                    }
                )

    summary_path = run_root / "duplicate_repositories.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "membership_only": membership_only,
                "total_repositories": len(projects),
                "duplicate_names": len(duplicates),
                "duplicates": [
                    {
                        "name": name,
                        "count": len(repos),
                        "repositories": [
                            {
                                "path_with_namespace": repo.get("path_with_namespace") or repo.get("path"),
                                "id": repo.get("id"),
                                "visibility": repo.get("visibility"),
                                "web_url": repo.get("web_url"),
                            }
                            for repo in repos
                        ],
                    }
                    for name, repos in sorted(duplicates.items())
                ],
            },
            fh,
            indent=2,
            sort_keys=True,
            default=str,
        )

    console.print("\n[bold green]Duplicate repository scan complete.[/]")
    if duplicates:
        console.print(f"Duplicate names found: {len(duplicates)} (see {csv_path})")
    else:
        console.print("No duplicate repository names found.")
    console.print(f"[blue]Reports written to {run_root}[/]")


@app.command("check-lfs")
def check_lfs_usage(
    repo: str = typer.Argument(
        ...,
        metavar="<repo-url|path|id>",
        help="Repository URL, path_with_namespace, short path, or id to check for Git LFS usage.",
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
        Path("output"),
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
        envvar="OUTPUT_DIR",
        help="Directory to write the LFS check report. Can be set via OUTPUT_DIR.",
    ),
) -> None:
    """
    Check a repository for Git LFS usage:
    - Reports project.lfs_enabled flag
    - Looks for 'filter=lfs' rules in .gitattributes
    - Detects LFS pointer files on the default branch
    """
    if not token:
        raise typer.BadParameter("GitLab token is required via --token or env GITLAB_TOKEN")

    repo_ref = _normalize_ref(repo)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    scope_dir = _safe_group_dir_name(repo_ref or "unknown")
    run_root = output / timestamp / "lfs-check" / scope_dir
    run_root.mkdir(parents=True, exist_ok=True)

    try:
        with GitLabClient(base_url=base_url, token=token) as client:
            project = client.get_project(repo_ref, include_statistics=True)
            project_id = int(project["id"])
            project_path = project.get("path_with_namespace") or project.get("path") or str(project_id)
            default_branch = project.get("default_branch")
            if not default_branch:
                console.print(f"[red]Project {project_path} has no default branch; cannot inspect for LFS.[/]")
                raise typer.Exit(code=1)

            console.print(
                f"[cyan]Checking Git LFS usage in[/] [bold]{project_path}[/] (branch {default_branch})."
            )
            lfs_enabled = bool(project.get("lfs_enabled"))

            has_lfs_gitattributes = False
            gitattributes_rules: list[dict[str, object]] = []
            lfs_pointer_files: list[dict[str, object]] = []
            pointer_limit = 20

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=False,
                console=console,
            ) as progress:
                task = progress.add_task("Scanning repository tree...", start=True)
                for entry in client.iter_project_tree(
                    project_id=project_id,
                    ref=default_branch,
                    recursive=True,
                ):
                    if entry.get("type") != "blob":
                        continue
                    path = entry.get("path", "")
                    size = int(entry.get("size") or 0)
                    blob_sha = entry.get("id")
                    if not blob_sha:
                        continue

                    if path.endswith(".gitattributes") and size <= 128 * 1024:
                        blob = client.get_blob(project_id=project_id, blob_sha=blob_sha)
                        content = blob.get("content")
                        if content and blob.get("encoding") == "base64":
                            decoded = b64decode(content).decode("utf-8", errors="replace")
                            gitattributes_rules.append({"path": path, "content": decoded})
                            if "filter=lfs" in decoded:
                                has_lfs_gitattributes = True

                    if len(lfs_pointer_files) >= pointer_limit:
                        continue
                    if 80 <= size <= 2048:
                        blob = client.get_blob(project_id=project_id, blob_sha=blob_sha)
                        content = blob.get("content")
                        if content and blob.get("encoding") == "base64":
                            decoded = b64decode(content).decode("utf-8", errors="replace")
                            if (
                                "version https://git-lfs.github.com/spec/v1" in decoded
                                and "oid sha256:" in decoded
                            ):
                                lfs_pointer_files.append(
                                    {
                                        "path": path,
                                        "size_bytes": size,
                                        "blob_sha": blob_sha,
                                    }
                                )
                progress.update(task, advance=100)
    except GitLabAPIError as exc:
        console.print(f"[red]GitLab API error while checking LFS:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - safety net
        console.print(f"[red]Unexpected error while checking LFS:[/] {exc}")
        raise typer.Exit(code=1) from exc

    lfs_detected = lfs_enabled or has_lfs_gitattributes or bool(lfs_pointer_files)

    summary = {
        "project_id": project_id,
        "project_path": project_path,
        "default_branch": default_branch,
        "lfs_enabled": lfs_enabled,
        "gitattributes_with_lfs": has_lfs_gitattributes,
        "lfs_pointer_files_found": bool(lfs_pointer_files),
        "lfs_pointer_samples": lfs_pointer_files,
        "gitattributes_rules": gitattributes_rules,
        "lfs_detected": lfs_detected,
    }

    summary_path = run_root / "lfs_check.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)

    console.print("\n[bold green]LFS check complete.[/]")
    console.print(
        f"LFS enabled flag: {lfs_enabled} | .gitattributes contains filter=lfs: {has_lfs_gitattributes} | LFS pointers: {len(lfs_pointer_files)}"
    )
    if lfs_pointer_files:
        for lf in lfs_pointer_files:
            console.print(f"- {lf['path']} ({lf['size_bytes']} bytes)")
    console.print(f"[blue]Report written to {summary_path}[/]")


@app.command("check-rate-limit")
def check_rate_limit(
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
        Path("output"),
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
        envvar="OUTPUT_DIR",
        help="Directory to write the rate limit report. Can be set via OUTPUT_DIR.",
    ),
) -> None:
    """Check current PAT user and rate-limit headers returned by the GitLab API."""
    if not token:
        raise typer.BadParameter("GitLab token is required via --token or env GITLAB_TOKEN")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_root = output / timestamp / "rate-limit"
    run_root.mkdir(parents=True, exist_ok=True)

    def _parse_reset_headers(headers: dict[str, str]) -> tuple[str | None, str | None, float | None]:
        reset_at_utc = headers.get("RateLimit-ResetTime") or None
        reset_epoch = headers.get("RateLimit-Reset")
        seconds_until_reset: float | None = None
        reset_at_ist: str | None = None
        if reset_epoch and str(reset_epoch).isdigit():
            try:
                reset_ts = int(reset_epoch)
                now_ts = int(datetime.now(timezone.utc).timestamp())
                seconds_until_reset = max(reset_ts - now_ts, 0)
                reset_dt_utc = datetime.fromtimestamp(reset_ts, tz=timezone.utc)
                if not reset_at_utc:
                    reset_at_utc = reset_dt_utc.isoformat()
                reset_dt_ist = reset_dt_utc.astimezone(timezone(timedelta(hours=5, minutes=30)))
                reset_at_ist = reset_dt_ist.isoformat()
            except Exception:
                seconds_until_reset = None
        return reset_at_utc, reset_at_ist, seconds_until_reset

    try:
        with GitLabClient(base_url=base_url, token=token) as client:
            user_payload, headers = client.get_current_user()
    except GitLabAPIError as exc:
        console.print(f"[red]GitLab API error while checking rate limit:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - safety net
        console.print(f"[red]Unexpected error while checking rate limit:[/] {exc}")
        raise typer.Exit(code=1) from exc

    reset_time_utc, reset_time_ist, seconds_until_reset = _parse_reset_headers(headers)

    limit_info = {
        "limit": headers.get("RateLimit-Limit"),
        "observed": headers.get("RateLimit-Observed"),
        "remaining": headers.get("RateLimit-Remaining"),
        "reset": headers.get("RateLimit-Reset"),
        "reset_time": reset_time_utc,
        "reset_time_ist": reset_time_ist,
        "seconds_until_reset": seconds_until_reset,
        "window": headers.get("RateLimit-Window"),
    }

    summary = {
        "user": {
            "id": user_payload.get("id"),
            "username": user_payload.get("username"),
            "name": user_payload.get("name"),
            "state": user_payload.get("state"),
        },
        "rate_limit_headers": limit_info,
    }

    summary_path = run_root / "rate_limit.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)

    console.print("\n[bold green]Rate limit check complete.[/]")
    console.print(
        f"User: {user_payload.get('username')} (id={user_payload.get('id')}) | Remaining: {limit_info.get('remaining')} / {limit_info.get('limit')}"
    )
    if reset_time_utc or reset_time_ist or seconds_until_reset is not None:
        console.print(
            f"Resets at UTC: {reset_time_utc or 'unknown'}; IST: {reset_time_ist or 'unknown'}"
            + (f" (in ~{int(seconds_until_reset)}s)" if seconds_until_reset is not None else "")
        )
    console.print(f"[blue]Report written to {summary_path}[/]")


@app.command("list-users")
def list_users(
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
        Path("output"),
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
        envvar="OUTPUT_DIR",
        help="Directory to write user listing. Can be set via OUTPUT_DIR.",
    ),
    include_inactive: bool = typer.Option(
        False,
        "--include-inactive",
        help="Include inactive users (default lists active only).",
    ),
    include_email: bool = typer.Option(
        True,
        "--include-email/--no-include-email",
        help="Best-effort to fetch email for each user (requires admin on GitLab SaaS).",
    ),
) -> None:
    """
    List users (active by default) including bots, with username, email (best-effort), and profile URL.
    Email visibility on GitLab SaaS generally requires an admin token.
    """
    if not token:
        raise typer.BadParameter("GitLab token is required via --token or env GITLAB_TOKEN")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_root = output / timestamp / "users"
    run_root.mkdir(parents=True, exist_ok=True)

    fieldnames = ["id", "username", "name", "state", "is_bot", "web_url", "email"]

    users: list[dict[str, object]] = []
    email_errors: list[dict[str, str]] = []
    active_param: bool | None = None if include_inactive else True
    warned_email_permission = False

    try:
        with GitLabClient(base_url=base_url, token=token) as client:
            console.print(
                f"[cyan]Listing users (active_only={active_param is True}, include_email={include_email}).[/]"
            )
            raw_users = list(client.iter_users(active=active_param))
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=False,
                console=console,
            ) as progress:
                task = progress.add_task("Fetching user details...", total=len(raw_users))
                for user in raw_users:
                    detail = user
                    if include_email:
                        try:
                            detail = client.get_user(int(user["id"]))
                        except GitLabAPIError as exc:
                            email_errors.append({"user": user.get("username", ""), "error": str(exc)})
                            if not warned_email_permission and exc.status_code in (401, 403):
                                console.print(
                                    "[yellow]Warning:[/] Email visibility is restricted on GitLab SaaS; admin token required."
                                )
                                warned_email_permission = True
                    is_bot = detail.get("is_bot")
                    if is_bot is None and "bot" in detail:
                        is_bot = detail.get("bot")
                    users.append(
                        {
                            "id": detail.get("id"),
                            "username": detail.get("username"),
                            "name": detail.get("name"),
                            "state": detail.get("state"),
                            "is_bot": is_bot,
                            "web_url": detail.get("web_url"),
                            "email": detail.get("email"),
                        }
                    )
                    progress.advance(task)
    except GitLabAPIError as exc:
        console.print(f"[red]GitLab API error while listing users:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - safety net
        console.print(f"[red]Unexpected error while listing users:[/] {exc}")
        raise typer.Exit(code=1) from exc

    csv_path = run_root / "users.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for user in users:
            writer.writerow(
                {
                    **user,
                    "is_bot": "" if user.get("is_bot") is None else str(user.get("is_bot")).lower(),
                }
            )

    summary_path = run_root / "users.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "count": len(users),
                "active_only": active_param is True,
                "include_email": include_email,
                "email_errors": email_errors,
                "users": users,
            },
            fh,
            indent=2,
            sort_keys=True,
            default=str,
        )

    console.print("\n[bold green]User listing complete.[/]")
    console.print(f"Users found: {len(users)} (active_only={active_param is True})")
    if email_errors:
        console.print(f"[yellow]Email fetch warnings for {len(email_errors)} user(s); see users.json for details.[/]")
    console.print(f"[blue]Reports written to {run_root}[/]")


@app.command("find-large-files")
def find_large_files(
    target: str | None = typer.Argument(
        None,
        metavar="[<group/project|id>]",
        help="Repository path/ID to scan, or a group path/ID when using --group. Optional when using --repo-name.",
    ),
    repo_name: str | None = typer.Option(
        None,
        "--repo-name",
        help="Repository path/ID to scan (alternative to positional argument).",
    ),
    threshold_mb: int = typer.Option(
        50,
        "--threshold-mb",
        "-t",
        help="Size threshold in megabytes to flag files (default: 50 MB).",
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
        Path("output"),
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
        envvar="OUTPUT_DIR",
        help="Directory to write large file scan results. Can be set via OUTPUT_DIR.",
    ),
    group: bool = typer.Option(
        False,
        "--group",
        "-g",
        help="Treat the target as a group and scan all projects under it.",
    ),
    include_subgroups: bool = typer.Option(
        True,
        "--include-subgroups/--no-include-subgroups",
        help="When using --group, include repositories from nested subgroups.",
    ),
) -> None:
    """Scan a repository (or every repo in a group) for files larger than the given threshold (Git LFS candidates)."""
    if not token:
        raise typer.BadParameter("GitLab token is required via --token or env GITLAB_TOKEN")
    if threshold_mb <= 0:
        raise typer.BadParameter("--threshold-mb must be positive")
    if not target and not repo_name:
        raise typer.BadParameter("Provide a repository/group via positional argument or --repo-name")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    scope_dir = _safe_group_dir_name(target or repo_name or "unknown")
    run_root = output / timestamp / "large-files" / scope_dir
    run_root.mkdir(parents=True, exist_ok=True)

    threshold_bytes = threshold_mb * 1024 * 1024
    large_files: list[LargeFileReport] = []
    scanned_projects: list[str] = []
    skipped_projects: list[str] = []
    skipped_projects_errors: list[dict[str, str]] = []
    repo_filter = repo_name

    target = _normalize_ref(target)
    repo_name = _normalize_ref(repo_name)

    def _resolve_project_by_name(client: GitLabClient, name: str) -> dict[str, object]:
        candidates = [
            p
            for p in client.iter_projects(
                membership_only=True,
                include_statistics=False,
                simple=True,
                search=name,
            )
            if p.get("path") == name or p.get("name") == name
        ]
        if not candidates:
            raise GitLabAPIError(f"Repository named '{name}' not found or not accessible")
        if len(candidates) > 1:
            raise GitLabAPIError(
                f"Multiple repositories named '{name}' found; please pass the full path_with_namespace"
            )
        return candidates[0]

    try:
        with GitLabClient(base_url=base_url, token=token) as client:
            if group:
                grp = client.get_group(target or repo_name)
                group_label = grp.get("full_path") or grp.get("path") or str(grp.get("id"))
                console.print(
                    f"[cyan]Scanning group[/] [bold]{group_label}[/] "
                    f"for files >= {threshold_mb} MB (include_subgroups={include_subgroups})."
                )
                projects = list(client.iter_group_projects(int(grp["id"]), include_subgroups=include_subgroups))
                if repo_name:
                    repo_filter = repo_name
                    projects = [
                        p
                        for p in projects
                        if (p.get("path_with_namespace") or p.get("path")) == repo_filter
                        or ("/" not in repo_filter and (p.get("path") == repo_filter or p.get("name") == repo_filter))
                        or str(p.get("id")) == repo_filter
                    ]
                    if not projects:
                        raise GitLabAPIError(
                            f"Repository '{repo_filter}' not found in group {group_label} (or not accessible)"
                        )
            else:
                project_ref = repo_name or target
                if project_ref and "/" not in project_ref and not str(project_ref).isdigit():
                    project = _resolve_project_by_name(client, project_ref)
                else:
                    project = client.get_project(project_ref, include_statistics=False)
                projects = [project]

            if not projects:
                console.print("[yellow]No projects found to scan.[/]")
                raise typer.Exit(code=1)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=False,
                console=console,
            ) as progress:
                task = progress.add_task(f"Scanning {len(projects)} project(s)...", total=len(projects))
                for project in projects:
                    project_id = int(project["id"])
                    project_path = project.get("path_with_namespace") or project.get("path") or str(project_id)
                    default_branch = project.get("default_branch")
                    if not default_branch:
                        skipped_projects.append(project_path)
                        progress.update(task, advance=1)
                        continue
                    scanned_projects.append(project_path)
                    try:
                        for entry in client.iter_project_tree(project_id=project_id, ref=default_branch, recursive=True):
                            if entry.get("type") != "blob":
                                continue
                            blob_sha = entry.get("id")
                            if not blob_sha:
                                continue
                            blob = client.get_blob(project_id=project_id, blob_sha=blob_sha)
                            size = int(blob.get("size") or 0)
                            if size >= threshold_bytes:
                                large_files.append(
                                    LargeFileReport(
                                        project_id=project_id,
                                        project_path=project_path,
                                        file_path=entry.get("path", ""),
                                        size_bytes=size,
                                        size_mb=round(size / (1024 * 1024), 2),
                                        blob_sha=blob_sha,
                                    )
                                )
                    except GitLabAPIError as exc:
                        skipped_projects_errors.append({"project": project_path, "error": str(exc)})
                        console.print(
                            f"[yellow]Warning:[/] skipping project {project_path} due to API error while walking tree: {exc}"
                        )
                    progress.advance(task, 1)
    except GitLabAPIError as exc:
        console.print(f"[red]GitLab API error while scanning repository:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - safety net
        console.print(f"[red]Unexpected error while scanning repository:[/] {exc}")
        raise typer.Exit(code=1) from exc

    csv_path = run_root / "large_files.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["project_id", "project_path", "file_path", "size_bytes", "size_mb", "blob_sha"],
        )
        writer.writeheader()
        for lf in large_files:
            writer.writerow(asdict(lf))

    summary_path = run_root / "large_files.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "target": target or repo_name,
                "group_mode": group,
                "threshold_mb": threshold_mb,
                "count": len(large_files),
                "scanned_projects": scanned_projects,
                "repo_filter": repo_filter,
                "skipped_projects_no_default_branch": skipped_projects,
                "skipped_projects_errors": skipped_projects_errors,
                "large_files": [asdict(lf) for lf in large_files],
            },
            fh,
            indent=2,
            sort_keys=True,
        )

    console.print("\n[bold green]Large file scan complete.[/]")
    console.print(
        f"Projects scanned: {len(scanned_projects)} "
        f"(skipped {len(skipped_projects)} without default branch, {len(skipped_projects_errors)} with errors)"
    )
    console.print(f"Files >= {threshold_mb} MB: {len(large_files)}")
    if large_files:
        for lf in large_files:
            console.print(f"- {lf.file_path} ({lf.size_mb} MB)")
    console.print(f"[blue]Reports written to {run_root}[/]")


@app.command()
def audit(
    target: str = typer.Argument(
        "groups",
        metavar="[groups|repo|<group/repo>]",
        help="Audit scope: default 'groups' for membership. Use 'repo' or provide a specific repo path to audit repositories.",
    ),
    subject: str | None = typer.Argument(
        None,
        metavar="[all|<group/repo>]",
        help="When auditing repositories, provide 'all' or a single <group/project>.",
    ),
    root_group: str | None = typer.Option(
        None,
        "--root-group",
        "-g",
        envvar="GITLAB_ROOT_GROUP",
        help="Root group id or full path to start discovery. Omit to audit every accessible top-level group.",
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
        Path("output"),
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
    all_top_groups: bool = typer.Option(
        False,
        "--all-top-groups",
        hidden=True,
        help="Deprecated; audits all accessible top-level groups by default when --root-group is not provided.",
    ),
) -> None:
    """
    Fetch all groups/subgroups from ROOT_GROUP and emit membership/user reports.
    """
    if not token:
        raise typer.BadParameter("GitLab token is required via --token or env GITLAB_TOKEN")
    if root_group and all_top_groups:
        raise typer.BadParameter("--all-top-groups cannot be combined with --root-group")

    repo_mode = False
    repo_scope: str | None = None
    if target == "repo":
        repo_mode = True
        repo_scope = subject
    elif "/" in target and target not in {"groups"}:
        repo_mode = True
        repo_scope = target
    elif subject and "/" in subject and target == "groups":
        repo_mode = True
        repo_scope = subject

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_root = output / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Run output directory:[/] {run_root}")
    log.info("Run output directory initialized at %s", run_root)
    log.debug("Audit target=%s subject=%s repo_mode=%s scope=%s", target, subject, repo_mode, repo_scope)

    if repo_mode:
        _run_repo_audit(
            token=token,
            base_url=base_url,
            repo_scope=repo_scope,
            output_root=run_root,
            console=console,
        )
        sys.exit(0)

    try:
        with GitLabClient(base_url=base_url, token=token) as client:
            target_roots: list[tuple[str, Path]] = []
            if all_top_groups:
                log.warning("--all-top-groups flag is deprecated; the CLI now audits all accessible top-level groups by default.")
            if root_group:
                log.info("Using provided root group %s", root_group)
                group_dir = run_root / _safe_group_dir_name(str(root_group))
                target_roots.append((root_group, group_dir))
            else:
                top_levels = list(client.iter_top_level_groups(membership_only=True))
                log.info("Discovered %s top-level groups for token", len(top_levels))
                if not top_levels:
                    console.print("[red]No top-level groups found for this token. Provide --root-group explicitly.[/]")
                    raise typer.Exit(code=1)
                console.print(
                    f"[cyan]No --root-group provided; auditing all {len(top_levels)} accessible top-level groups.[/]"
                )
                for group in top_levels:
                    ref = group.get("full_path") or group.get("path") or str(group.get("id"))
                    group_dir = run_root / _safe_group_dir_name(str(ref))
                    log.info("Queueing top-level group %s -> %s", ref, group_dir)
                    target_roots.append((str(ref), group_dir))

            results: list[AuditResult] = []
            for root_ref, output_dir in target_roots:
                log.info("Starting audit for root %s (output=%s)", root_ref, output_dir)
                results.append(
                    audit_gitlab(
                        client=client,
                        root_group_ref=root_ref,
                        include_emails=include_emails,
                        output_dir=output_dir,
                        console=console,
                    )
                )
                log.info("Finished audit for root %s (output=%s)", root_ref, output_dir)
    except typer.Exit:
        raise
    except GitLabAPIError as exc:
        console.print(f"[red]GitLab API error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - safety net
        console.print(f"[red]Unexpected error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if not results:
        console.print("[red]No audits were executed.[/]")
        raise typer.Exit(code=1)

    combined_report = write_combined_membership_report(results, run_root / "client_report.csv")
    console.print(f"[cyan]Client-friendly CSV generated at {combined_report}[/]")

    if len(results) == 1:
        result = results[0]
        console.print("\n[bold green]Audit complete.[/]")
        console.print(f"Groups: {result.summary['groups']}, Users: {result.summary['users']}, Bots: {result.summary['bots']}, Memberships: {result.summary['memberships']}")
        console.print(f"Reports written to {result.output_dir}")
    else:
        console.print(f"\n[bold green]Audit complete for {len(results)} top-level groups.[/]")
        total_groups = sum(int(r.summary.get("groups", 0)) for r in results)
        total_users = sum(int(r.summary.get("users", 0)) for r in results)
        total_bots = sum(int(r.summary.get("bots", 0)) for r in results)
        total_memberships = sum(int(r.summary.get("memberships", 0)) for r in results)
        for result in results:
            root_name = result.groups[0].full_path if result.groups else "unknown"
            console.print(
                f"- {root_name}: Groups={result.summary['groups']}, Users={result.summary['users']}, Bots={result.summary['bots']}, Memberships={result.summary['memberships']} (reports: {result.output_dir})"
            )
        console.print(
            f"Combined totals -> Groups: {total_groups}, Users: {total_users}, Bots: {total_bots}, Memberships: {total_memberships}"
        )
    console.print(f"[blue]Run artifacts saved under {run_root}[/]")
    sys.exit(0)


def _run_repo_audit(
    token: str,
    base_url: str,
    repo_scope: str | None,
    output_root: Path,
    console: Console,
) -> None:
    scope = _normalize_ref(repo_scope) or "all"
    tree_parallel_raw = os.getenv("GITLAB_DISCOVERY_TREE_PARALLEL_PAGES", "4").strip()
    tree_parallel_pages: int | None = None
    try:
        value = int(tree_parallel_raw)
        if value < 1:
            raise ValueError("GITLAB_DISCOVERY_TREE_PARALLEL_PAGES must be >= 1")
        tree_parallel_pages = value if value > 1 else None
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    skip_large_files_raw = os.getenv("GITLAB_DISCOVERY_SKIP_LARGE_FILES", "true").strip().lower()
    if skip_large_files_raw not in {"true", "false", "1", "0", "yes", "no"}:
        raise typer.BadParameter("GITLAB_DISCOVERY_SKIP_LARGE_FILES must be true/false")
    skip_large_file_scan = skip_large_files_raw in {"true", "1", "yes"}
    lfs_config_scan_raw = os.getenv("GITLAB_DISCOVERY_LFS_CONFIG_SCAN", "true").strip().lower()
    if lfs_config_scan_raw not in {"true", "false", "1", "0", "yes", "no"}:
        raise typer.BadParameter("GITLAB_DISCOVERY_LFS_CONFIG_SCAN must be true/false")
    lfs_config_scan = lfs_config_scan_raw in {"true", "1", "yes"}
    include_csv_raw = os.getenv("GITLAB_DISCOVERY_INCLUDE_CSV", "false").strip().lower()
    if include_csv_raw not in {"true", "false", "1", "0", "yes", "no"}:
        raise typer.BadParameter("GITLAB_DISCOVERY_INCLUDE_CSV must be true/false")
    include_csv = include_csv_raw in {"true", "1", "yes"}
    repo_output = output_root / "repos"
    try:
        with GitLabClient(base_url=base_url, token=token) as client:
            if scope.lower() == "all":
                projects = list(client.iter_projects(membership_only=True, include_statistics=True))
                if not projects:
                    console.print("[red]No repositories found for this token.[/]")
                    raise typer.Exit(code=1)
                log.info("Auditing %s repositories (all accessible).", len(projects))
            else:
                log.info("Auditing repository %s", scope)
                try:
                    project = client.get_project(scope, include_statistics=True)
                except GitLabAPIError as exc:
                    console.print(f"[red]Could not load repository {scope}:[/] {exc}")
                    raise typer.Exit(code=1) from exc
                projects = [project]

            console.print(
                f"[cyan]Auditing {len(projects)} repository{'ies' if len(projects) != 1 else ''} "
                f"(LFS threshold {LFS_THRESHOLD_MB} MB).[/]"
            )
            repo_result = audit_repositories(
                client=client,
                projects=projects,
                output_dir=repo_output,
                console=console,
                lfs_threshold_bytes=LFS_THRESHOLD_BYTES,
                tree_parallel_pages=tree_parallel_pages,
                skip_large_file_scan=skip_large_file_scan,
                lfs_config_scan=lfs_config_scan,
                include_csv=include_csv,
            )
    except typer.Exit:
        raise
    except GitLabAPIError as exc:
        console.print(f"[red]GitLab API error during repository audit:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[red]Unexpected error during repository audit:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("\n[bold green]Repository audit complete.[/]")
    console.print(
        f"Repositories: {repo_result.summary['projects']}, Large files >= {LFS_THRESHOLD_MB} MB: {repo_result.summary['large_files']}"
    )
    if repo_result.summary.get("csv_enabled"):
        console.print(f"Combined CSV for client: {repo_result.combined_csv}")
    else:
        console.print("CSV outputs disabled (set GITLAB_DISCOVERY_INCLUDE_CSV=true to enable).")
    console.print(f"[blue]Run artifacts saved under {output_root}[/]")


if __name__ == "__main__":
    app()
