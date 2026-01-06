from __future__ import annotations

import sys
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
from dataclasses import asdict

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
) -> None:
    """Count and list repositories accessible to the token (optionally scoped to a group)."""
    if not token:
        raise typer.BadParameter("GitLab token is required via --token or env GITLAB_TOKEN")

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
                projects = list(client.iter_group_projects(int(group["id"]), include_subgroups=include_subgroups))
            else:
                root_label = "all-accessible"
                console.print("[cyan]Listing all repositories accessible to this token.[/]")
                projects = list(client.iter_projects(membership_only=True, include_statistics=False, simple=True))
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
                    or str(p.get("id")) == repo_filter
                ]
            else:
                project_ref = repo_name or target
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
    scope = repo_scope or "all"
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
    console.print(f"Combined CSV for client: {repo_result.combined_csv}")
    console.print(f"[blue]Run artifacts saved under {output_root}[/]")


if __name__ == "__main__":
    app()
