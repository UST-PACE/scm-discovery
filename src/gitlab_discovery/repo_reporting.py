from __future__ import annotations

import csv
import json
import logging
from base64 import b64decode
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Iterable, Sequence

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .gitlab_client import GitLabAPIError, GitLabClient

log = logging.getLogger(__name__)

ACCESS_LEVELS = {
    0: "no_access",
    5: "minimal",
    10: "guest",
    20: "reporter",
    30: "developer",
    40: "maintainer",
    50: "owner",
}


@dataclass
class ProjectReport:
    id: int
    name: str
    path_with_namespace: str
    visibility: str
    default_branch: str | None
    web_url: str | None
    ssh_url_to_repo: str | None
    http_url_to_repo: str | None
    archived: bool
    empty_repo: bool
    forked_from_project: str | None
    last_activity_at: str | None
    created_at: str | None
    repository_size_mb: float | None
    storage_size_mb: float | None
    lfs_objects_size_mb: float | None
    packages_size_mb: float | None
    wiki_size_mb: float | None
    job_artifacts_size_mb: float | None
    namespace_full_path: str | None
    lfs_enabled: bool
    lfs_gitattributes: bool
    lfs_pointer_files_found: bool
    lfs_pointer_files_count: int
    lfs_detected: bool
    lfs_config_present: bool
    lfs_config_url: str | None
    lfs_config_scanned: bool


@dataclass
class LargeFileReport:
    project_id: int
    project_path: str
    file_path: str
    size_bytes: int
    size_mb: float
    blob_sha: str


@dataclass
class ProjectMemberReport:
    project_id: int
    project_path: str
    user_id: int
    username: str
    name: str
    state: str
    access_level: int
    access_label: str
    expires_at: str | None
    is_bot: bool | None


@dataclass
class ProjectHookReport:
    project_id: int
    project_path: str
    hook_id: int
    url: str
    push_events: bool
    tag_push_events: bool
    merge_requests_events: bool
    issues_events: bool
    note_events: bool
    pipeline_events: bool
    wiki_page_events: bool
    job_events: bool
    releases_events: bool
    enable_ssl_verification: bool
    created_at: str | None


@dataclass
class ProjectIntegrationReport:
    project_id: int
    project_path: str
    name: str
    slug: str | None
    category: str | None
    active: bool
    created_at: str | None
    updated_at: str | None
    url: str | None
    properties: dict[str, object]


@dataclass
class PipelineReport:
    project_id: int
    project_path: str
    pipeline_id: int
    status: str
    ref: str | None
    sha: str | None
    source: str | None
    created_at: str | None
    updated_at: str | None
    duration: float | None
    queued_duration: float | None
    web_url: str | None
    user_id: int | None
    username: str | None


@dataclass
class PipelineJobReport:
    project_id: int
    project_path: str
    pipeline_id: int
    job_id: int
    name: str
    stage: str
    status: str
    duration: float | None
    started_at: str | None
    finished_at: str | None
    artifacts_file: str | None
    runner_description: str | None


@dataclass
class PipelineScheduleReport:
    project_id: int
    project_path: str
    schedule_id: int
    description: str
    ref: str | None
    cron: str | None
    cron_timezone: str | None
    active: bool
    next_run_at: str | None
    created_at: str | None


@dataclass
class ProtectedBranchReport:
    project_id: int
    project_path: str
    name: str
    push_access_levels: list[str]
    merge_access_levels: list[str]


@dataclass
class ProtectedTagReport:
    project_id: int
    project_path: str
    name: str
    create_access_levels: list[str]


@dataclass
class EnvironmentReport:
    project_id: int
    project_path: str
    environment_id: int
    name: str
    state: str | None
    external_url: str | None
    tier: str | None
    last_deployment_id: int | None
    created_at: str | None
    updated_at: str | None


@dataclass
class DeploymentReport:
    project_id: int
    project_path: str
    deployment_id: int
    iid: int | None
    status: str | None
    ref: str | None
    sha: str | None
    environment: str | None
    created_at: str | None
    updated_at: str | None
    user_id: int | None
    username: str | None


@dataclass
class PackageReport:
    project_id: int
    project_path: str
    package_id: int
    name: str
    version: str | None
    package_type: str
    status: str | None
    created_at: str | None
    pipeline_id: int | None


@dataclass
class RegistryRepositoryReport:
    project_id: int
    project_path: str
    registry_id: int
    path: str
    location: str
    tags_count: int


@dataclass
class RegistryTagReport:
    project_id: int
    project_path: str
    registry_id: int
    repository_path: str
    name: str
    digest: str | None
    media_type: str | None
    size_bytes: int | None
    created_at: str | None


@dataclass
class ReleaseReport:
    project_id: int
    project_path: str
    name: str
    tag_name: str
    description: str | None
    released_at: str | None
    created_at: str | None
    author_id: int | None
    author_username: str | None
    web_url: str | None


@dataclass
class TagReport:
    project_id: int
    project_path: str
    name: str
    message: str | None
    target: str | None
    release_description: str | None
    protected: bool


@dataclass
class BranchReport:
    project_id: int
    project_path: str
    name: str
    default: bool
    merged: bool
    protected: bool
    developers_can_push: bool
    developers_can_merge: bool


@dataclass
class VariableReport:
    project_id: int
    project_path: str
    key: str
    environment_scope: str
    protected: bool
    masked: bool
    variable_type: str


@dataclass
class RepoAuditResult:
    projects: list[ProjectReport]
    large_files: list[LargeFileReport]
    members: list["ProjectMemberReport"]
    hooks: list["ProjectHookReport"]
    integrations: list["ProjectIntegrationReport"]
    pipelines: list["PipelineReport"]
    pipeline_jobs: list["PipelineJobReport"]
    pipeline_schedules: list["PipelineScheduleReport"]
    protected_branches: list["ProtectedBranchReport"]
    protected_tags: list["ProtectedTagReport"]
    environments: list["EnvironmentReport"]
    deployments: list["DeploymentReport"]
    packages: list["PackageReport"]
    registry_repositories: list["RegistryRepositoryReport"]
    registry_tags: list["RegistryTagReport"]
    releases: list["ReleaseReport"]
    tags: list["TagReport"]
    branches: list["BranchReport"]
    variables: list["VariableReport"]
    summary: dict[str, object]
    output_dir: Path
    combined_csv: Path


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_csv_if_rows(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
    has_rows: bool,
) -> None:
    if not has_rows:
        return
    _write_csv(path, fieldnames, rows)


def _mb(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024 * 1024), 2)


def audit_repositories(
    client: GitLabClient,
    projects: Sequence[dict],
    output_dir: Path,
    console: Console | None = None,
    lfs_threshold_bytes: int = 100 * 1024 * 1024,
    tree_parallel_pages: int | None = None,
    skip_large_file_scan: bool = True,
    lfs_config_scan: bool = True,
    include_csv: bool = False,
) -> RepoAuditResult:
    console = console or Console(stderr=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_start = datetime.now(timezone.utc)
    projects_report: list[ProjectReport] = []
    large_files: list[LargeFileReport] = []
    members: list[ProjectMemberReport] = []
    hooks: list[ProjectHookReport] = []
    integrations: list[ProjectIntegrationReport] = []
    pipelines: list[PipelineReport] = []
    pipeline_jobs: list[PipelineJobReport] = []
    pipeline_schedules: list[PipelineScheduleReport] = []
    protected_branches: list[ProtectedBranchReport] = []
    protected_tags: list[ProtectedTagReport] = []
    environments: list[EnvironmentReport] = []
    deployments: list[DeploymentReport] = []
    packages: list[PackageReport] = []
    registry_repositories: list[RegistryRepositoryReport] = []
    registry_tags: list[RegistryTagReport] = []
    releases: list[ReleaseReport] = []
    tags: list[TagReport] = []
    branches: list[BranchReport] = []
    variables: list[VariableReport] = []
    run_id = output_dir.parent.name
    feature_flags_by_project: dict[int, dict[str, bool]] = {}
    project_settings_by_project: dict[int, dict[str, object]] = {}
    work_counts_by_project: dict[int, dict[str, int]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
        console=console,
    ) as progress:
        task = progress.add_task("Gathering repositories...", total=len(projects))
        for project in projects:
            project_id = int(project["id"])
            lfs_enabled = bool(project.get("lfs_enabled"))
            lfs_gitattributes = False
            lfs_pointer_files_count = 0
            lfs_config_present = False
            lfs_config_url: str | None = None
            default_branch = project.get("default_branch")
            if not skip_large_file_scan:
                if not default_branch:
                    log.warning("Project %s has no default branch; skipping large file scanning.", project.get("path_with_namespace"))
                else:
                    for entry in client.iter_project_tree(
                        project_id=project_id,
                        ref=default_branch,
                        recursive=True,
                        parallel_pages=tree_parallel_pages,
                    ):
                        if entry.get("type") != "blob":
                            continue
                        blob_sha = entry.get("id")
                        if not blob_sha:
                            continue
                        blob = client.get_blob(project_id=project_id, blob_sha=blob_sha)
                        size = int(blob.get("size") or 0)
                        path = entry.get("path", "")
                        content = blob.get("content")
                        encoding = blob.get("encoding")
                        if content and encoding == "base64":
                            if path == ".lfsconfig" and size <= 128 * 1024:
                                decoded = b64decode(content).decode("utf-8", errors="replace")
                                lfs_config_present = True
                                for line in decoded.splitlines():
                                    if "=" not in line:
                                        continue
                                    key, value = line.split("=", 1)
                                    if key.strip().lower() == "lfs.url":
                                        lfs_config_url = value.strip()
                            if path.endswith(".gitattributes") and size <= 128 * 1024:
                                decoded = b64decode(content).decode("utf-8", errors="replace")
                                if "filter=lfs" in decoded:
                                    lfs_gitattributes = True
                            if 80 <= size <= 2048:
                                decoded = b64decode(content).decode("utf-8", errors="replace")
                                if (
                                    "version https://git-lfs.github.com/spec/v1" in decoded
                                    and "oid sha256:" in decoded
                                ):
                                    lfs_pointer_files_count += 1
                        if size >= lfs_threshold_bytes:
                            large_files.append(
                                LargeFileReport(
                                    project_id=project_id,
                                    project_path=project.get("path_with_namespace") or project.get("path") or "",
                                    file_path=path,
                                    size_bytes=size,
                                    size_mb=round(size / (1024 * 1024), 2),
                                    blob_sha=blob_sha,
                                )
                            )
            elif lfs_config_scan and default_branch:
                for entry in client.iter_project_tree(
                    project_id=project_id,
                    ref=default_branch,
                    recursive=False,
                ):
                    if entry.get("type") != "blob":
                        continue
                    if entry.get("path") != ".lfsconfig":
                        continue
                    blob_sha = entry.get("id")
                    if not blob_sha:
                        continue
                    blob = client.get_blob(project_id=project_id, blob_sha=blob_sha)
                    content = blob.get("content")
                    encoding = blob.get("encoding")
                    if content and encoding == "base64":
                        decoded = b64decode(content).decode("utf-8", errors="replace")
                        lfs_config_present = True
                        for line in decoded.splitlines():
                            if "=" not in line:
                                continue
                            key, value = line.split("=", 1)
                            if key.strip().lower() == "lfs.url":
                                lfs_config_url = value.strip()
                    break
            lfs_pointer_files_found = lfs_pointer_files_count > 0
            lfs_detected = lfs_enabled or lfs_gitattributes or lfs_pointer_files_found
            if skip_large_file_scan:
                lfs_detected = lfs_detected or bool(project.get("statistics", {}).get("lfs_objects_size"))
            feature_flags_by_project[project_id] = {
                key: bool(value)
                for key, value in project.items()
                if isinstance(value, bool)
            }
            project_settings_by_project[project_id] = {
                "default_branch": project.get("default_branch") or "",
                "merge_method": project.get("merge_method") or "",
                "squash_option": project.get("squash_option") or "",
                "approvals_before_merge": project.get("approvals_before_merge"),
                "visibility": project.get("visibility"),
                "archived": bool(project.get("archived")),
                "request_access_enabled": bool(project.get("request_access_enabled")),
                "resolve_outdated_diff_discussions": bool(project.get("resolve_outdated_diff_discussions")),
                "lfs_enabled": bool(project.get("lfs_enabled")),
                "shared_runners_enabled": bool(project.get("shared_runners_enabled")),
                "public_jobs": bool(project.get("public_jobs")),
                "printing_merge_request_link_enabled": bool(project.get("printing_merge_request_link_enabled")),
                "analytics_access_level": project.get("analytics_access_level"),
                "ci_forward_deployment_enabled": bool(project.get("ci_forward_deployment_enabled")),
                "only_allow_merge_if_pipeline_succeeds": bool(project.get("only_allow_merge_if_pipeline_succeeds")),
                "only_allow_merge_if_all_discussions_are_resolved": bool(
                    project.get("only_allow_merge_if_all_discussions_are_resolved")
                ),
                "remove_source_branch_after_merge": bool(project.get("remove_source_branch_after_merge")),
                "auto_devops_enabled": bool(project.get("auto_devops_enabled")),
                "auto_devops_deploy_strategy": project.get("auto_devops_deploy_strategy") or "",
            }
            try:
                work_counts_by_project[project_id] = {
                    "issues": client.get_list_total(f"/projects/{project_id}/issues"),
                    "merge_requests": client.get_list_total(f"/projects/{project_id}/merge_requests"),
                    "milestones": client.get_list_total(f"/projects/{project_id}/milestones"),
                }
            except GitLabAPIError as exc:
                log.warning("Could not fetch work item counts for %s: %s", project.get("path_with_namespace"), exc)
                work_counts_by_project[project_id] = {"issues": 0, "merge_requests": 0, "milestones": 0}
            projects_report.append(
                _project_payload(
                    project,
                    {
                        "lfs_enabled": lfs_enabled,
                        "lfs_gitattributes": lfs_gitattributes,
                        "lfs_pointer_files_found": lfs_pointer_files_found,
                        "lfs_pointer_files_count": lfs_pointer_files_count,
                        "lfs_detected": lfs_detected,
                        "lfs_config_present": lfs_config_present,
                        "lfs_config_url": lfs_config_url,
                        "lfs_config_scanned": (not skip_large_file_scan) or lfs_config_scan,
                    },
                )
            )
            _collect_project_members(client, project, members)
            _collect_project_hooks(client, project, hooks, console=console)
            _collect_project_integrations(client, project, integrations, console=console)
            _collect_project_pipelines(
                client,
                project,
                pipelines=pipelines,
                pipeline_jobs=pipeline_jobs,
                pipeline_schedules=pipeline_schedules,
                console=console,
            )
            _collect_project_protections(
                client,
                project,
                protected_branches=protected_branches,
                protected_tags=protected_tags,
                console=console,
            )
            _collect_project_envs_and_deployments(
                client,
                project,
                environments=environments,
                deployments=deployments,
                console=console,
            )
            _collect_project_packages_and_registry(
                client,
                project,
                packages=packages,
                registry_repos=registry_repositories,
                registry_tags=registry_tags,
                console=console,
            )
            _collect_project_releases_and_tags(
                client,
                project,
                releases=releases,
                tags=tags,
                console=console,
            )
            _collect_project_branches(client, project, branches, console=console)
            _collect_project_variables(client, project, variables, console=console)
            progress.advance(task)

    if include_csv:
        _write_csv_if_rows(
            output_dir / "repositories.csv",
            [
                "id",
                "name",
                "path_with_namespace",
                "visibility",
                "default_branch",
                "web_url",
                "ssh_url_to_repo",
                "http_url_to_repo",
                "archived",
                "empty_repo",
                "forked_from_project",
                "last_activity_at",
                "created_at",
                "repository_size_mb",
                "storage_size_mb",
                "lfs_objects_size_mb",
                "packages_size_mb",
                "wiki_size_mb",
                "job_artifacts_size_mb",
                "namespace_full_path",
                "lfs_enabled",
                "lfs_gitattributes",
                "lfs_pointer_files_found",
                "lfs_pointer_files_count",
                "lfs_detected",
                "lfs_config_present",
                "lfs_config_url",
                "lfs_config_scanned",
            ],
            (asdict(r) for r in projects_report),
            bool(projects_report),
        )
        _write_csv_if_rows(
            output_dir / "large_files.csv",
            ["project_id", "project_path", "file_path", "size_bytes", "size_mb", "blob_sha"],
            (asdict(l) for l in large_files),
            bool(large_files),
        )
        _write_csv_if_rows(
            output_dir / "project_members.csv",
            [
                "project_id",
                "project_path",
                "user_id",
                "username",
                "name",
                "state",
                "access_level",
                "access_label",
                "expires_at",
                "is_bot",
            ],
            (
                {
                    **asdict(member),
                    "is_bot": "" if member.is_bot is None else str(member.is_bot).lower(),
                }
                for member in members
            ),
            bool(members),
        )
        _write_csv_if_rows(
            output_dir / "project_hooks.csv",
            [
                "project_id",
                "project_path",
                "hook_id",
                "url",
                "push_events",
                "tag_push_events",
                "merge_requests_events",
                "issues_events",
                "note_events",
                "pipeline_events",
                "wiki_page_events",
                "job_events",
                "releases_events",
                "enable_ssl_verification",
                "created_at",
            ],
            (asdict(hook) for hook in hooks),
            bool(hooks),
        )
        _write_csv_if_rows(
            output_dir / "project_integrations.csv",
            [
                "project_id",
                "project_path",
                "name",
                "slug",
                "category",
                "active",
                "created_at",
                "updated_at",
                "url",
                "properties",
            ],
            (
                {
                    **asdict(integration),
                    "properties": json.dumps(integration.properties, sort_keys=True),
                }
                for integration in integrations
            ),
            bool(integrations),
        )
        _write_csv_if_rows(
            output_dir / "pipelines.csv",
            [
                "project_id",
                "project_path",
                "pipeline_id",
                "status",
                "ref",
                "sha",
                "source",
                "created_at",
                "updated_at",
                "duration",
                "queued_duration",
                "web_url",
                "user_id",
                "username",
            ],
            (asdict(pipeline) for pipeline in pipelines),
            bool(pipelines),
        )
        _write_csv_if_rows(
            output_dir / "pipeline_jobs.csv",
            [
                "project_id",
                "project_path",
                "pipeline_id",
                "job_id",
                "name",
                "stage",
                "status",
                "duration",
                "started_at",
                "finished_at",
                "artifacts_file",
                "runner_description",
            ],
            (asdict(job) for job in pipeline_jobs),
            bool(pipeline_jobs),
        )
        _write_csv_if_rows(
            output_dir / "pipeline_schedules.csv",
            [
                "project_id",
                "project_path",
                "schedule_id",
                "description",
                "ref",
                "cron",
                "cron_timezone",
                "active",
                "next_run_at",
                "created_at",
            ],
            (asdict(schedule) for schedule in pipeline_schedules),
            bool(pipeline_schedules),
        )
        _write_csv_if_rows(
            output_dir / "protected_branches.csv",
            ["project_id", "project_path", "name", "push_access_levels", "merge_access_levels"],
            (
                {
                    "project_id": pb.project_id,
                    "project_path": pb.project_path,
                    "name": pb.name,
                    "push_access_levels": json.dumps(pb.push_access_levels),
                    "merge_access_levels": json.dumps(pb.merge_access_levels),
                }
                for pb in protected_branches
            ),
            bool(protected_branches),
        )
        _write_csv_if_rows(
            output_dir / "protected_tags.csv",
            ["project_id", "project_path", "name", "create_access_levels"],
            (
                {
                    "project_id": pt.project_id,
                    "project_path": pt.project_path,
                    "name": pt.name,
                    "create_access_levels": json.dumps(pt.create_access_levels),
                }
                for pt in protected_tags
            ),
            bool(protected_tags),
        )
        _write_csv_if_rows(
            output_dir / "environments.csv",
            [
                "project_id",
                "project_path",
                "environment_id",
                "name",
                "state",
                "external_url",
                "tier",
                "last_deployment_id",
                "created_at",
                "updated_at",
            ],
            (asdict(env) for env in environments),
            bool(environments),
        )
        _write_csv_if_rows(
            output_dir / "deployments.csv",
            [
                "project_id",
                "project_path",
                "deployment_id",
                "iid",
                "status",
                "ref",
                "sha",
                "environment",
                "created_at",
                "updated_at",
                "user_id",
                "username",
            ],
            (asdict(dep) for dep in deployments),
            bool(deployments),
        )
        _write_csv_if_rows(
            output_dir / "packages.csv",
            [
                "project_id",
                "project_path",
                "package_id",
                "name",
                "version",
                "package_type",
                "status",
                "created_at",
                "pipeline_id",
            ],
            (asdict(pkg) for pkg in packages),
            bool(packages),
        )
        _write_csv_if_rows(
            output_dir / "registry_repositories.csv",
            [
                "project_id",
                "project_path",
                "registry_id",
                "path",
                "location",
                "tags_count",
            ],
            (asdict(repo) for repo in registry_repositories),
            bool(registry_repositories),
        )
        _write_csv_if_rows(
            output_dir / "registry_tags.csv",
            [
                "project_id",
                "project_path",
                "registry_id",
                "repository_path",
                "name",
                "digest",
                "media_type",
                "size_bytes",
                "created_at",
            ],
            (asdict(tag) for tag in registry_tags),
            bool(registry_tags),
        )
        _write_csv_if_rows(
            output_dir / "releases.csv",
            [
                "project_id",
                "project_path",
                "name",
                "tag_name",
                "description",
                "released_at",
                "created_at",
                "author_id",
                "author_username",
                "web_url",
            ],
            (asdict(rel) for rel in releases),
            bool(releases),
        )
        _write_csv_if_rows(
            output_dir / "tags.csv",
            [
                "project_id",
                "project_path",
                "name",
                "message",
                "target",
                "release_description",
                "protected",
            ],
            (asdict(tag) for tag in tags),
            bool(tags),
        )
        _write_csv_if_rows(
            output_dir / "branches.csv",
            [
                "project_id",
                "project_path",
                "name",
                "default",
                "merged",
                "protected",
                "developers_can_push",
                "developers_can_merge",
            ],
            (asdict(branch) for branch in branches),
            bool(branches),
        )
        _write_csv_if_rows(
            output_dir / "ci_variables.csv",
            [
                "project_id",
                "project_path",
                "key",
                "environment_scope",
                "protected",
                "masked",
                "variable_type",
            ],
            (asdict(var) for var in variables),
            bool(variables),
        )

    combined_csv = output_dir / "repo_client_report.csv"
    if include_csv:
        _write_csv_if_rows(
            combined_csv,
            ["project_path", "file_path", "size_mb", "size_bytes", "blob_sha"],
            (
                {
                    "project_path": lf.project_path,
                    "file_path": lf.file_path,
                    "size_mb": lf.size_mb,
                    "size_bytes": lf.size_bytes,
                    "blob_sha": lf.blob_sha,
                }
                for lf in large_files
            ),
            bool(large_files),
        )

    summary = {
        "projects": len(projects_report),
        "large_files": len(large_files),
        "members": len(members),
        "hooks": len(hooks),
        "integrations": len(integrations),
        "pipelines": len(pipelines),
        "pipeline_jobs": len(pipeline_jobs),
        "pipeline_schedules": len(pipeline_schedules),
        "protected_branches": len(protected_branches),
        "protected_tags": len(protected_tags),
        "environments": len(environments),
        "deployments": len(deployments),
        "packages": len(packages),
        "registry_repositories": len(registry_repositories),
        "registry_tags": len(registry_tags),
        "releases": len(releases),
        "tags": len(tags),
        "branches": len(branches),
        "variables": len(variables),
        "lfs_threshold_bytes": lfs_threshold_bytes,
        "large_file_scan_skipped": skip_large_file_scan,
        "csv_enabled": include_csv,
    }
    with (output_dir / "repo_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                **summary,
                "repositories": [asdict(r) for r in projects_report],
                "large_files": [asdict(l) for l in large_files],
                "project_members": [asdict(m) for m in members],
                "project_hooks": [asdict(h) for h in hooks],
                "project_integrations": [
                    {
                        **asdict(integration),
                        "properties": integration.properties,
                    }
                    for integration in integrations
                ],
                "pipelines": [asdict(p) for p in pipelines],
                "pipeline_jobs": [asdict(j) for j in pipeline_jobs],
                "pipeline_schedules": [asdict(s) for s in pipeline_schedules],
                "protected_branches": [
                    {
                        "project_id": pb.project_id,
                        "project_path": pb.project_path,
                        "name": pb.name,
                        "push_access_levels": pb.push_access_levels,
                        "merge_access_levels": pb.merge_access_levels,
                    }
                    for pb in protected_branches
                ],
                "protected_tags": [
                    {
                        "project_id": pt.project_id,
                        "project_path": pt.project_path,
                        "name": pt.name,
                        "create_access_levels": pt.create_access_levels,
                    }
                    for pt in protected_tags
                ],
                "environments": [asdict(env) for env in environments],
                "deployments": [asdict(dep) for dep in deployments],
                "packages": [asdict(pkg) for pkg in packages],
                "registry_repositories": [asdict(repo) for repo in registry_repositories],
                "registry_tags": [asdict(tag) for tag in registry_tags],
                "releases": [asdict(rel) for rel in releases],
                "tags_detail": [asdict(tag) for tag in tags],
                "branches": [asdict(branch) for branch in branches],
                "ci_variables": [asdict(var) for var in variables],
            },
            fh,
            indent=2,
            sort_keys=True,
        )

    _write_repo_html_reports(
        output_dir=output_dir,
        run_id=run_id,
        projects=projects_report,
        large_files=large_files,
        members=members,
        hooks=hooks,
        integrations=integrations,
        pipelines=pipelines,
        pipeline_jobs=pipeline_jobs,
        pipeline_schedules=pipeline_schedules,
        protected_branches=protected_branches,
        protected_tags=protected_tags,
        environments=environments,
        deployments=deployments,
        packages=packages,
        registry_repositories=registry_repositories,
        registry_tags=registry_tags,
        releases=releases,
        tags=tags,
        branches=branches,
        variables=variables,
        feature_flags_by_project=feature_flags_by_project,
        skip_large_file_scan=skip_large_file_scan,
        project_settings_by_project=project_settings_by_project,
        work_counts_by_project=work_counts_by_project,
        duration_seconds=(datetime.now(timezone.utc) - report_start).total_seconds(),
    )

    console.print(f"Repository reports written to {output_dir}")
    return RepoAuditResult(
        projects=projects_report,
        large_files=large_files,
        members=members,
        hooks=hooks,
        integrations=integrations,
        pipelines=pipelines,
        pipeline_jobs=pipeline_jobs,
        pipeline_schedules=pipeline_schedules,
        protected_branches=protected_branches,
        protected_tags=protected_tags,
        environments=environments,
        deployments=deployments,
        packages=packages,
        registry_repositories=registry_repositories,
        registry_tags=registry_tags,
        releases=releases,
        tags=tags,
        branches=branches,
        variables=variables,
        summary=summary,
        output_dir=output_dir,
        combined_csv=combined_csv,
    )


def _project_payload(project: dict, lfs_info: dict[str, object] | None = None) -> ProjectReport:
    stats = project.get("statistics") or {}
    namespace = project.get("namespace") or {}
    forked_from = project.get("forked_from_project")
    forked_path = None
    if isinstance(forked_from, dict):
        forked_path = forked_from.get("path_with_namespace")
    lfs_info = lfs_info or {}

    return ProjectReport(
        id=int(project["id"]),
        name=project.get("name", ""),
        path_with_namespace=project.get("path_with_namespace") or project.get("path") or "",
        visibility=project.get("visibility", "private"),
        default_branch=project.get("default_branch"),
        web_url=project.get("web_url"),
        ssh_url_to_repo=project.get("ssh_url_to_repo"),
        http_url_to_repo=project.get("http_url_to_repo"),
        archived=bool(project.get("archived")),
        empty_repo=bool(project.get("empty_repo")),
        forked_from_project=forked_path,
        last_activity_at=project.get("last_activity_at"),
        created_at=project.get("created_at"),
        repository_size_mb=_mb(stats.get("repository_size")),
        storage_size_mb=_mb(stats.get("storage_size")),
        lfs_objects_size_mb=_mb(stats.get("lfs_objects_size")),
        packages_size_mb=_mb(stats.get("packages_size")),
        wiki_size_mb=_mb(stats.get("wiki_size")),
        job_artifacts_size_mb=_mb(stats.get("job_artifacts_size")),
        namespace_full_path=namespace.get("full_path"),
        lfs_enabled=bool(lfs_info.get("lfs_enabled")),
        lfs_gitattributes=bool(lfs_info.get("lfs_gitattributes")),
        lfs_pointer_files_found=bool(lfs_info.get("lfs_pointer_files_found")),
        lfs_pointer_files_count=int(lfs_info.get("lfs_pointer_files_count") or 0),
        lfs_detected=bool(lfs_info.get("lfs_detected")),
        lfs_config_present=bool(lfs_info.get("lfs_config_present")),
        lfs_config_url=(lfs_info.get("lfs_config_url") or None),
        lfs_config_scanned=bool(lfs_info.get("lfs_config_scanned")),
    )


def _safe_report_filename(project_path: str) -> str:
    safe = project_path.strip().replace("/", "__").replace("\\", "__")
    return safe or "repository"


def _collect_counts(items: Iterable[object], key: str = "project_id") -> dict[int, int]:
    counts: dict[int, int] = {}
    for item in items:
        value = getattr(item, key, None)
        if value is None:
            continue
        counts[int(value)] = counts.get(int(value), 0) + 1
    return counts


def _format_bool(value: bool) -> str:
    return "Yes" if value else "No"


def _fmt_mb(value: float | None) -> str:
    return "N/A" if value is None else f"{value}"


def _format_run_id_ist(run_id: str) -> str:
    try:
        run_dt = datetime.strptime(run_id, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        run_dt = datetime.now(timezone.utc)
    ist = timezone(timedelta(hours=5, minutes=30))
    return run_dt.astimezone(ist).strftime("%Y-%m-%d %H:%M:%S IST")


def _write_repo_html_reports(
    output_dir: Path,
    run_id: str,
    projects: list[ProjectReport],
    large_files: list[LargeFileReport],
    members: list[ProjectMemberReport],
    hooks: list[ProjectHookReport],
    integrations: list[ProjectIntegrationReport],
    pipelines: list[PipelineReport],
    pipeline_jobs: list[PipelineJobReport],
    pipeline_schedules: list[PipelineScheduleReport],
    protected_branches: list[ProtectedBranchReport],
    protected_tags: list[ProtectedTagReport],
    environments: list[EnvironmentReport],
    deployments: list[DeploymentReport],
    packages: list[PackageReport],
    registry_repositories: list[RegistryRepositoryReport],
    registry_tags: list[RegistryTagReport],
    releases: list[ReleaseReport],
    tags: list[TagReport],
    branches: list[BranchReport],
    variables: list[VariableReport],
    feature_flags_by_project: dict[int, dict[str, bool]],
    skip_large_file_scan: bool,
    project_settings_by_project: dict[int, dict[str, object]],
    work_counts_by_project: dict[int, dict[str, int]],
    duration_seconds: float,
) -> None:
    counts = {
        "large_files": _collect_counts(large_files),
        "members": _collect_counts(members),
        "hooks": _collect_counts(hooks),
        "integrations": _collect_counts(integrations),
        "pipelines": _collect_counts(pipelines),
        "pipeline_jobs": _collect_counts(pipeline_jobs),
        "pipeline_schedules": _collect_counts(pipeline_schedules),
        "protected_branches": _collect_counts(protected_branches),
        "protected_tags": _collect_counts(protected_tags),
        "environments": _collect_counts(environments),
        "deployments": _collect_counts(deployments),
        "packages": _collect_counts(packages),
        "registry_repositories": _collect_counts(registry_repositories),
        "registry_tags": _collect_counts(registry_tags),
        "releases": _collect_counts(releases),
        "tags": _collect_counts(tags),
        "branches": _collect_counts(branches),
        "variables": _collect_counts(variables),
    }

    for project in projects:
        project_id = int(project.id)
        project_path = project.path_with_namespace
        safe_name = _safe_report_filename(project_path)
        report_path = output_dir / f"{safe_name}_{run_id}.html"

        icon_repo = (
            "<span class='icon'><svg viewBox='0 0 24 24' aria-hidden='true'>"
            "<path d='M4 4h12a4 4 0 0 1 4 4v10a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4zm2 4h12'/>"
            "</svg></span>"
        )
        icon_wiki = (
            "<span class='icon'><svg viewBox='0 0 24 24' aria-hidden='true'>"
            "<path d='M4 5h12a2 2 0 0 1 2 2v12H6a2 2 0 0 1-2-2V5zm4 3h8M8 11h8M8 15h6'/>"
            "</svg></span>"
        )
        icon_lfs = (
            "<span class='icon'><svg viewBox='0 0 24 24' aria-hidden='true'>"
            "<path d='M4 16l8-10 8 10H4zm8-7v6'/>"
            "</svg></span>"
        )
        icon_pkg = (
            "<span class='icon'><svg viewBox='0 0 24 24' aria-hidden='true'>"
            "<path d='M12 3l8 4-8 4-8-4 8-4zm-8 8l8 4 8-4v8l-8 4-8-4v-8z'/>"
            "</svg></span>"
        )
        icon_features = (
            "<span class='icon'><svg viewBox='0 0 24 24' aria-hidden='true'>"
            "<path d='M12 2l2.5 5 5.5.8-4 3.9.9 5.6-4.9-2.6-4.9 2.6.9-5.6-4-3.9 5.5-.8L12 2z'/>"
            "</svg></span>"
        )
        icon_table = (
            "<span class='icon'><svg viewBox='0 0 24 24' aria-hidden='true'>"
            "<path d='M4 5h16v14H4V5zm0 4h16M9 5v14M15 5v14'/>"
            "</svg></span>"
        )

        def count_for(key: str) -> int:
            return counts[key].get(project_id, 0)

        feature_flags = feature_flags_by_project.get(project_id, {})
        project_settings = project_settings_by_project.get(project_id, {})
        work_counts = work_counts_by_project.get(project_id, {"issues": 0, "merge_requests": 0, "milestones": 0})
        feature_rows = "".join(
            f"<tr><td>{escape(name)}</td>"
            f"<td><span class='status {('yes' if enabled else 'no')}'>{_format_bool(enabled)}</span></td></tr>"
            for name, enabled in sorted(feature_flags.items())
        ) or "<tr><td>No feature flags returned</td><td><span class='status no'>No</span></td></tr>"

        settings_rows = "".join(
            f"<tr><td>{escape(str(key).replace('_', ' ').title())}</td><td>{escape(str(value))}</td></tr>"
            for key, value in sorted(project_settings.items())
        ) or "<tr><td>No settings returned</td><td>N/A</td></tr>"

        if skip_large_file_scan:
            large_files_rows = "<tr><td colspan='3'>Large file scan skipped.</td></tr>"
        else:
            large_files_rows = "".join(
                f"<tr><td>{escape(lf.file_path)}</td><td>{lf.size_mb}</td><td>{escape(lf.blob_sha)}</td></tr>"
                for lf in large_files
                if lf.project_id == project_id
            )
            if not large_files_rows:
                large_files_rows = "<tr><td colspan='3'>No large files detected.</td></tr>"

        ist_time = _format_run_id_ist(run_id)
        duration_label = f"{duration_seconds:.1f} seconds"
        if duration_seconds >= 60:
            duration_label = f"{duration_seconds / 60:.1f} minutes"
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(project.name)} - Repository Discovery</title>
  <style>
    :root {{
      --bg: #0f1216;
      --panel: #161b22;
      --panel-2: #1d2430;
      --text: #e6edf3;
      --muted: #9aa7b5;
      --accent: #2ec4b6;
      --accent-2: #f6ae2d;
      --danger: #ff6b6b;
      --ok: #2ec4b6;
      --border: #273040;
      --shadow: 0 14px 30px rgba(0,0,0,0.35);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 20% 20%, #1b2430 0%, #0f1216 55%, #0b0d11 100%);
      color: var(--text);
      font-family: "Georgia", "Times New Roman", serif;
      line-height: 1.5;
    }}
    header {{
      padding: 32px 24px 12px;
    }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 0 24px 48px;
    }}
    .headline {{
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding: 24px;
      border-radius: 18px;
      background: linear-gradient(135deg, rgba(46,196,182,0.2), rgba(246,174,45,0.12));
      border: 1px solid rgba(46,196,182,0.3);
      box-shadow: var(--shadow);
    }}
    .notice {{
      font-size: 12px;
      letter-spacing: 0.3px;
      padding: 8px 12px;
      border-radius: 999px;
      background: linear-gradient(135deg, rgba(246,174,45,0.3), rgba(255,107,107,0.25));
      color: #ffd166;
      width: fit-content;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      border: 1px solid rgba(246,174,45,0.5);
      box-shadow: 0 10px 18px rgba(246,174,45,0.2);
    }}
    .headline h1 {{
      margin: 0;
      font-size: 30px;
      letter-spacing: 0.3px;
    }}
    .headline p {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-top: 22px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }}
    .card h3 {{
      margin: 0 0 6px;
      font-size: 16px;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      letter-spacing: 0.4px;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .icon {{
      display: inline-flex;
      width: 18px;
      height: 18px;
      margin-right: 8px;
      color: var(--accent);
    }}
    .icon svg {{
      width: 18px;
      height: 18px;
      fill: currentColor;
    }}
    .card .value {{
      font-size: 22px;
      font-weight: 700;
    }}
    .card .chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(46,196,182,0.18);
      color: var(--text);
      font-size: 13px;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }}
    .chip.warning {{
      background: rgba(246,174,45,0.2);
      color: var(--accent-2);
    }}
    .chip.danger {{
      background: rgba(255,107,107,0.18);
      color: var(--danger);
    }}
    .section {{
      margin-top: 28px;
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 20px;
    }}
    .section h2 {{
      margin: 0 0 14px;
      font-size: 18px;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      color: var(--accent);
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }}
    .table th,
    .table td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid rgba(39,48,64,0.7);
    }}
    .table th {{
      color: var(--muted);
      font-weight: 600;
    }}
    .status {{
      padding: 2px 10px;
      border-radius: 999px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }}
    .status.yes {{
      background: rgba(46,196,182,0.2);
      color: var(--ok);
    }}
    .status.no {{
      background: rgba(255,107,107,0.18);
      color: var(--danger);
    }}
    .tag {{
      display: inline-block;
      margin: 4px 6px 0 0;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(46,196,182,0.12);
      color: var(--text);
      font-size: 12px;
    }}
    .footer {{
      margin-top: 22px;
      color: var(--muted);
      font-size: 12px;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }}
    .footer-highlight {{
      margin-top: 14px;
      padding: 12px 16px;
      border-radius: 14px;
      background: rgba(46,196,182,0.12);
      border: 1px solid rgba(46,196,182,0.35);
      color: #d9f7f4;
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0.3px;
    }}
    @media (max-width: 700px) {{
      .headline h1 {{
        font-size: 24px;
      }}
      .card .value {{
        font-size: 20px;
      }}
    }}
  </style>
</head>
<body>
  <header class="wrap">
    <div class="headline">
      <h1>{escape(project.name)}</h1>
      <table class="table">
        <tr><th>Repository</th><td>{escape(project_path)}</td></tr>
        <tr><th>Repository URL</th><td><a href="{escape(project.web_url or '')}">{escape(project.web_url or 'N/A')}</a></td></tr>
        <tr><th>Run ID</th><td>{escape(run_id)}</td></tr>
        <tr><th>Generated (IST)</th><td>{escape(ist_time)}</td></tr>
      </table>
      <div class="chip">{escape(project.visibility)} visibility</div>
      <div class="notice">CONFIDENTIAL – This report contains proprietary information intended solely for internal use within client network and authorized personnel.</div>
    </div>
  </header>
  <main class="wrap">
    <section class="grid">
      <div class="card">
        <h3>{icon_repo}Repository Size</h3>
        <div class="value">{escape(_fmt_mb(project.repository_size_mb))} MB</div>
        <div class="chip">Storage {escape(_fmt_mb(project.storage_size_mb))} MB</div>
      </div>
      <div class="card">
        <h3>{icon_wiki}Wiki Size</h3>
        <div class="value">{escape(_fmt_mb(project.wiki_size_mb))} MB</div>
        <div class="chip">Job artifacts {escape(_fmt_mb(project.job_artifacts_size_mb))} MB</div>
      </div>
      <div class="card">
        <h3>{icon_lfs}LFS Signals</h3>
        <div class="value">{_format_bool(project.lfs_detected)}</div>
        <div class="chip">Pointers {project.lfs_pointer_files_count}</div>
      </div>
      <div class="card">
        <h3>{icon_pkg}Registry & Packages</h3>
        <div class="value">{count_for("packages")}</div>
        <div class="chip">Registries {count_for("registry_repositories")}</div>
      </div>
      <div class="card">
        <h3>{icon_repo}Branches & Tags</h3>
        <div class="value">{count_for("branches")}</div>
        <div class="chip">Tags {count_for("tags")}</div>
      </div>
    </section>

    <section class="section">
      <h2>{icon_features}Features Snapshot</h2>
      <table class="table">
        <tr><th>Feature</th><th>Count</th></tr>
        <tr><td>Hooks</td><td>{count_for("hooks")}</td></tr>
        <tr><td>Integrations</td><td>{count_for("integrations")}</td></tr>
        <tr><td>Pipelines</td><td>{count_for("pipelines")}</td></tr>
        <tr><td>Pipeline Jobs</td><td>{count_for("pipeline_jobs")}</td></tr>
        <tr><td>Pipeline Schedules</td><td>{count_for("pipeline_schedules")}</td></tr>
        <tr><td>Protected Branches</td><td>{count_for("protected_branches")}</td></tr>
        <tr><td>Protected Tags</td><td>{count_for("protected_tags")}</td></tr>
        <tr><td>Environments</td><td>{count_for("environments")}</td></tr>
        <tr><td>Deployments</td><td>{count_for("deployments")}</td></tr>
        <tr><td>Releases</td><td>{count_for("releases")}</td></tr>
        <tr><td>Tags</td><td>{count_for("tags")}</td></tr>
        <tr><td>Branches</td><td>{count_for("branches")}</td></tr>
        <tr><td>CI Variables</td><td>{count_for("variables")}</td></tr>
        <tr><td>Members</td><td>{count_for("members")}</td></tr>
      </table>
    </section>

    <section class="section">
      <h2>{icon_features}Project Feature Flags</h2>
      <table class="table">
        <tr><th>Feature</th><th>Enabled</th></tr>
        {feature_rows}
      </table>
    </section>

    <section class="section">
      <h2>{icon_table}Project Settings</h2>
      <table class="table">
        <tr><th>Setting</th><th>Value</th></tr>
        {settings_rows}
      </table>
    </section>

    <section class="section">
      <h2>{icon_table}Work Item Counts</h2>
      <table class="table">
        <tr><th>Type</th><th>Count</th></tr>
        <tr><td>Issues</td><td>{work_counts.get("issues", 0)}</td></tr>
        <tr><td>Merge Requests</td><td>{work_counts.get("merge_requests", 0)}</td></tr>
        <tr><td>Milestones</td><td>{work_counts.get("milestones", 0)}</td></tr>
      </table>
    </section>

    <section class="section">
      <h2>{icon_table}LFS Details</h2>
      <table class="table">
        <tr><th>Flag</th><th>Value</th></tr>
        <tr><td>Project LFS Enabled</td><td>{_format_bool(project.lfs_enabled)}</td></tr>
        <tr><td>.gitattributes has LFS</td><td>{_format_bool(project.lfs_gitattributes)}</td></tr>
        <tr><td>LFS Pointer Files Found</td><td>{_format_bool(project.lfs_pointer_files_found)}</td></tr>
        <tr><td>Large File Scan</td><td>{_format_bool(not skip_large_file_scan)}</td></tr>
        <tr><td>.lfsconfig Present</td><td>{_format_bool(project.lfs_config_present)}</td></tr>
        <tr><td>.lfsconfig URL</td><td>{escape(project.lfs_config_url or "N/A")}</td></tr>
        <tr><td>.lfsconfig Scanned</td><td>{_format_bool(project.lfs_config_scanned)}</td></tr>
        <tr><td>LFS Signal Note</td><td>Project-level LFS is enabled, but a full scan is needed to confirm actual LFS usage or a custom LFS server.</td></tr>
      </table>
    </section>

    <section class="section">
      <h2>{icon_table}Large Files (LFS Candidates)</h2>
      <table class="table">
        <tr><th>File</th><th>Size (MB)</th><th>SHA</th></tr>
        {large_files_rows}
      </table>
    </section>

    <div class="footer">
      <div class="footer-highlight">
        Generated by UST Git-Lift Discovery CLI Tool.<br>
        Report generation time: {escape(duration_label)}
      </div>
    </div>
  </main>
</body>
</html>
"""
        report_path.write_text(html, encoding="utf-8")


def _collect_project_members(
    client: GitLabClient,
    project: dict,
    members: list[ProjectMemberReport],
) -> None:
    project_id = int(project["id"])
    path = project.get("path_with_namespace") or project.get("path") or ""
    try:
        for member in client.iter_project_members(project_id, include_inherited=True):
            access_level = int(member.get("access_level", 0))
            members.append(
                ProjectMemberReport(
                    project_id=project_id,
                    project_path=path,
                    user_id=int(member.get("id")),
                    username=member.get("username", ""),
                    name=member.get("name", ""),
                    state=member.get("state", "unknown"),
                    access_level=access_level,
                    access_label=ACCESS_LEVELS.get(access_level, f"unknown_{access_level}"),
                    expires_at=member.get("expires_at"),
                    is_bot=_resolve_bot_flag(member),
                )
            )
    except GitLabAPIError as exc:
        log.warning("Could not fetch members for %s: %s", path, exc)


def _collect_project_hooks(
    client: GitLabClient,
    project: dict,
    hooks: list[ProjectHookReport],
    console: Console | None = None,
) -> None:
    project_id = int(project["id"])
    path = project.get("path_with_namespace") or project.get("path") or ""
    try:
        for hook in client.list_project_hooks(project_id):
            hooks.append(
                ProjectHookReport(
                    project_id=project_id,
                    project_path=path,
                    hook_id=int(hook.get("id")),
                    url=hook.get("url", ""),
                    push_events=bool(hook.get("push_events")),
                    tag_push_events=bool(hook.get("tag_push_events")),
                    merge_requests_events=bool(hook.get("merge_requests_events")),
                    issues_events=bool(hook.get("issues_events")),
                    note_events=bool(hook.get("note_events")),
                    pipeline_events=bool(hook.get("pipeline_events")),
                    wiki_page_events=bool(hook.get("wiki_page_events")),
                    job_events=bool(hook.get("job_events")),
                    releases_events=bool(hook.get("releases_events")),
                    enable_ssl_verification=bool(hook.get("enable_ssl_verification", True)),
                    created_at=hook.get("created_at"),
                )
            )
    except GitLabAPIError as exc:
        if console:
            console.print(f"[yellow]Warning:[/] could not fetch hooks for {path}: {exc}")
        log.warning("Could not fetch hooks for %s: %s", path, exc)


def _collect_project_integrations(
    client: GitLabClient,
    project: dict,
    integrations: list[ProjectIntegrationReport],
    console: Console | None = None,
) -> None:
    project_id = int(project["id"])
    path = project.get("path_with_namespace") or project.get("path") or ""
    try:
        for integration in client.list_project_integrations(project_id):
            props = integration.get("properties") or {}
            integrations.append(
                ProjectIntegrationReport(
                    project_id=project_id,
                    project_path=path,
                    name=integration.get("name") or integration.get("title") or "",
                    slug=integration.get("slug"),
                    category=integration.get("category"),
                    active=bool(integration.get("active")),
                    created_at=integration.get("created_at"),
                    updated_at=integration.get("updated_at"),
                    url=_integration_url(integration),
                    properties=_scrub_properties(props),
                )
            )
    except GitLabAPIError as exc:
        if console:
            console.print(f"[yellow]Warning:[/] could not fetch integrations for {path}: {exc}")
        log.warning("Could not fetch integrations for %s: %s", path, exc)


def _resolve_bot_flag(member: dict[str, object]) -> bool | None:
    if "is_bot" in member and member["is_bot"] is not None:
        return bool(member["is_bot"])
    if "bot" in member and member["bot"] is not None:
        return bool(member["bot"])
    return None


def _scrub_properties(props: dict[str, object]) -> dict[str, object]:
    sensitive = ("token", "secret", "password", "key")
    cleaned: dict[str, object] = {}
    for key, value in props.items():
        if any(part in key.lower() for part in sensitive):
            continue
        cleaned[key] = value
    return cleaned


def _integration_url(integration: dict[str, object]) -> str | None:
    for key in ("url", "webhook", "api_url", "hook"):
        value = integration.get(key)
        if value:
            return str(value)
    return None


def _collect_project_pipelines(
    client: GitLabClient,
    project: dict,
    pipelines: list[PipelineReport],
    pipeline_jobs: list[PipelineJobReport],
    pipeline_schedules: list[PipelineScheduleReport],
    console: Console | None = None,
) -> None:
    project_id = int(project["id"])
    path = project.get("path_with_namespace") or project.get("path") or ""
    try:
        for pipeline in client.iter_project_pipelines(project_id):
            pipelines.append(
                PipelineReport(
                    project_id=project_id,
                    project_path=path,
                    pipeline_id=int(pipeline.get("id")),
                    status=pipeline.get("status", ""),
                    ref=pipeline.get("ref"),
                    sha=pipeline.get("sha"),
                    source=pipeline.get("source"),
                    created_at=pipeline.get("created_at"),
                    updated_at=pipeline.get("updated_at"),
                    duration=pipeline.get("duration"),
                    queued_duration=pipeline.get("queued_duration"),
                    web_url=pipeline.get("web_url"),
                    user_id=(pipeline.get("user") or {}).get("id"),
                    username=(pipeline.get("user") or {}).get("username"),
                )
            )
            try:
                for job in client.iter_pipeline_jobs(project_id, int(pipeline.get("id"))):
                    pipeline_jobs.append(
                        PipelineJobReport(
                            project_id=project_id,
                            project_path=path,
                            pipeline_id=int(pipeline.get("id")),
                            job_id=int(job.get("id")),
                            name=job.get("name", ""),
                            stage=job.get("stage", ""),
                            status=job.get("status", ""),
                            duration=job.get("duration"),
                            started_at=job.get("started_at"),
                            finished_at=job.get("finished_at"),
                            artifacts_file=(job.get("artifacts_file") or {}).get("filename"),
                            runner_description=(job.get("runner") or {}).get("description"),
                        )
                    )
            except GitLabAPIError as exc:
                if console:
                    console.print(f"[yellow]Warning:[/] could not fetch jobs for pipeline {pipeline.get('id')} in {path}: {exc}")
                log.warning("Could not fetch jobs for pipeline %s/%s: %s", path, pipeline.get("id"), exc)
    except GitLabAPIError as exc:
        if console:
            console.print(f"[yellow]Warning:[/] could not fetch pipelines for {path}: {exc}")
        log.warning("Could not fetch pipelines for %s: %s", path, exc)

    try:
        for schedule in client.iter_project_pipeline_schedules(project_id):
            pipeline_schedules.append(
                PipelineScheduleReport(
                    project_id=project_id,
                    project_path=path,
                    schedule_id=int(schedule.get("id")),
                    description=schedule.get("description", ""),
                    ref=schedule.get("ref"),
                    cron=schedule.get("cron"),
                    cron_timezone=schedule.get("cron_timezone"),
                    active=bool(schedule.get("active")),
                    next_run_at=schedule.get("next_run_at"),
                    created_at=schedule.get("created_at"),
                )
            )
    except GitLabAPIError as exc:
        if console:
            console.print(f"[yellow]Warning:[/] could not fetch pipeline schedules for {path}: {exc}")
        log.warning("Could not fetch pipeline schedules for %s: %s", path, exc)


def _collect_project_protections(
    client: GitLabClient,
    project: dict,
    protected_branches: list[ProtectedBranchReport],
    protected_tags: list[ProtectedTagReport],
    console: Console | None = None,
) -> None:
    project_id = int(project["id"])
    path = project.get("path_with_namespace") or project.get("path") or ""
    try:
        for branch in client.iter_project_protected_branches(project_id):
            protected_branches.append(
                ProtectedBranchReport(
                    project_id=project_id,
                    project_path=path,
                    name=branch.get("name", ""),
                    push_access_levels=[_access_level_entry(entry) for entry in branch.get("push_access_levels", [])],
                    merge_access_levels=[_access_level_entry(entry) for entry in branch.get("merge_access_levels", [])],
                )
            )
    except GitLabAPIError as exc:
        if console:
            console.print(f"[yellow]Warning:[/] could not fetch protected branches for {path}: {exc}")
        log.warning("Could not fetch protected branches for %s: %s", path, exc)

    try:
        for tag in client.iter_project_protected_tags(project_id):
            protected_tags.append(
                ProtectedTagReport(
                    project_id=project_id,
                    project_path=path,
                    name=tag.get("name", ""),
                    create_access_levels=[_access_level_entry(entry) for entry in tag.get("create_access_levels", [])],
                )
            )
    except GitLabAPIError as exc:
        if console:
            console.print(f"[yellow]Warning:[/] could not fetch protected tags for {path}: {exc}")
        log.warning("Could not fetch protected tags for %s: %s", path, exc)


def _collect_project_envs_and_deployments(
    client: GitLabClient,
    project: dict,
    environments: list[EnvironmentReport],
    deployments: list[DeploymentReport],
    console: Console | None = None,
) -> None:
    project_id = int(project["id"])
    path = project.get("path_with_namespace") or project.get("path") or ""
    try:
        for env in client.iter_project_environments(project_id):
            environment_id = env.get("id")
            environments.append(
                EnvironmentReport(
                    project_id=project_id,
                    project_path=path,
                    environment_id=int(environment_id),
                    name=env.get("name", ""),
                    state=env.get("state"),
                    external_url=env.get("external_url"),
                    tier=env.get("tier"),
                    last_deployment_id=(env.get("last_deployment") or {}).get("id"),
                    created_at=env.get("created_at"),
                    updated_at=env.get("updated_at"),
                )
            )
    except GitLabAPIError as exc:
        if console:
            console.print(f"[yellow]Warning:[/] could not fetch environments for {path}: {exc}")
        log.warning("Could not fetch environments for %s: %s", path, exc)

    try:
        for deployment in client.iter_project_deployments(project_id):
            user = deployment.get("user") or {}
            env = deployment.get("environment") or {}
            deployments.append(
                DeploymentReport(
                    project_id=project_id,
                    project_path=path,
                    deployment_id=int(deployment.get("id")),
                    iid=deployment.get("iid"),
                    status=deployment.get("status"),
                    ref=deployment.get("ref"),
                    sha=deployment.get("sha"),
                    environment=env.get("name"),
                    created_at=deployment.get("created_at"),
                    updated_at=deployment.get("updated_at"),
                    user_id=user.get("id"),
                    username=user.get("username"),
                )
            )
    except GitLabAPIError as exc:
        if console:
            console.print(f"[yellow]Warning:[/] could not fetch deployments for {path}: {exc}")
        log.warning("Could not fetch deployments for %s: %s", path, exc)


def _collect_project_packages_and_registry(
    client: GitLabClient,
    project: dict,
    packages: list[PackageReport],
    registry_repos: list[RegistryRepositoryReport],
    registry_tags: list[RegistryTagReport],
    console: Console | None = None,
) -> None:
    project_id = int(project["id"])
    path = project.get("path_with_namespace") or project.get("path") or ""
    try:
        for package in client.iter_project_packages(project_id):
            packages.append(
                PackageReport(
                    project_id=project_id,
                    project_path=path,
                    package_id=int(package.get("id")),
                    name=package.get("name", ""),
                    version=package.get("version"),
                    package_type=package.get("package_type", ""),
                    status=package.get("status"),
                    created_at=package.get("created_at"),
                    pipeline_id=(package.get("pipeline") or {}).get("id"),
                )
            )
    except GitLabAPIError as exc:
        if console:
            console.print(f"[yellow]Warning:[/] could not fetch packages for {path}: {exc}")
        log.warning("Could not fetch packages for %s: %s", path, exc)

    try:
        for repo in client.list_registry_repositories(project_id):
            registry_repos.append(
                RegistryRepositoryReport(
                    project_id=project_id,
                    project_path=path,
                    registry_id=int(repo.get("id")),
                    path=repo.get("path", ""),
                    location=repo.get("location", ""),
                    tags_count=int(repo.get("tags_count") or 0),
                )
            )
            try:
                for tag in client.list_registry_repository_tags(project_id, int(repo.get("id"))):
                    registry_tags.append(
                        RegistryTagReport(
                            project_id=project_id,
                            project_path=path,
                            registry_id=int(repo.get("id")),
                            repository_path=repo.get("path", ""),
                            name=tag.get("name", ""),
                            digest=tag.get("digest"),
                            media_type=tag.get("media_type"),
                            size_bytes=tag.get("total_size"),
                            created_at=tag.get("created_at"),
                        )
                    )
            except GitLabAPIError as exc:
                if console:
                    console.print(f"[yellow]Warning:[/] could not fetch registry tags for {path}:{repo.get('path')} - {exc}")
                log.warning("Could not fetch registry tags for %s/%s: %s", path, repo.get("path"), exc)
    except GitLabAPIError as exc:
        if console:
            console.print(f"[yellow]Warning:[/] could not fetch container registry data for {path}: {exc}")
        log.warning("Could not fetch registry data for %s: %s", path, exc)


def _collect_project_releases_and_tags(
    client: GitLabClient,
    project: dict,
    releases: list[ReleaseReport],
    tags: list[TagReport],
    console: Console | None = None,
) -> None:
    project_id = int(project["id"])
    path = project.get("path_with_namespace") or project.get("path") or ""
    try:
        for release in client.iter_project_releases(project_id):
            author = release.get("author") or {}
            releases.append(
                ReleaseReport(
                    project_id=project_id,
                    project_path=path,
                    name=release.get("name", ""),
                    tag_name=release.get("tag_name", ""),
                    description=release.get("description"),
                    released_at=release.get("released_at"),
                    created_at=release.get("created_at"),
                    author_id=author.get("id"),
                    author_username=author.get("username"),
                    web_url=release.get("web_url"),
                )
            )
    except GitLabAPIError as exc:
        if console:
            console.print(f"[yellow]Warning:[/] could not fetch releases for {path}: {exc}")
        log.warning("Could not fetch releases for %s: %s", path, exc)

    try:
        for tag in client.iter_project_tags(project_id):
            release_info = tag.get("release") or {}
            tags.append(
                TagReport(
                    project_id=project_id,
                    project_path=path,
                    name=tag.get("name", ""),
                    message=tag.get("message"),
                    target=tag.get("target"),
                    release_description=release_info.get("description"),
                    protected=bool(tag.get("protected")),
                )
            )
    except GitLabAPIError as exc:
        if console:
            console.print(f"[yellow]Warning:[/] could not fetch tags for {path}: {exc}")
        log.warning("Could not fetch tags for %s: %s", path, exc)


def _collect_project_branches(
    client: GitLabClient,
    project: dict,
    branches: list[BranchReport],
    console: Console | None = None,
) -> None:
    project_id = int(project["id"])
    path = project.get("path_with_namespace") or project.get("path") or ""
    try:
        for branch in client.iter_project_branches(project_id):
            branches.append(
                BranchReport(
                    project_id=project_id,
                    project_path=path,
                    name=branch.get("name", ""),
                    default=bool(branch.get("default")),
                    merged=bool(branch.get("merged")),
                    protected=bool(branch.get("protected")),
                    developers_can_push=bool(branch.get("developers_can_push")),
                    developers_can_merge=bool(branch.get("developers_can_merge")),
                )
            )
    except GitLabAPIError as exc:
        if console:
            console.print(f"[yellow]Warning:[/] could not fetch branches for {path}: {exc}")
        log.warning("Could not fetch branches for %s: %s", path, exc)


def _collect_project_variables(
    client: GitLabClient,
    project: dict,
    variables: list[VariableReport],
    console: Console | None = None,
) -> None:
    project_id = int(project["id"])
    path = project.get("path_with_namespace") or project.get("path") or ""
    try:
        for variable in client.iter_project_variables(project_id):
            variables.append(
                VariableReport(
                    project_id=project_id,
                    project_path=path,
                    key=variable.get("key", ""),
                    environment_scope=variable.get("environment_scope", "*"),
                    protected=bool(variable.get("protected")),
                    masked=bool(variable.get("masked")),
                    variable_type=variable.get("variable_type", "env_var"),
                )
            )
    except GitLabAPIError as exc:
        if console:
            console.print(f"[yellow]Warning:[/] could not fetch CI variables for {path}: {exc}")
        log.warning("Could not fetch CI variables for %s: %s", path, exc)


def _access_level_entry(entry: dict[str, object]) -> str:
    user = entry.get("user_id")
    group = entry.get("group_id")
    access = entry.get("access_level")
    if user:
        return f"user:{user}:{access}"
    if group:
        return f"group:{group}:{access}"
    if access is not None:
        return str(access)
    return "unknown"
