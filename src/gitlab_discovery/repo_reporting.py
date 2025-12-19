from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
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
) -> RepoAuditResult:
    console = console or Console(stderr=True)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    variables: list[VariableReport] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
        console=console,
    ) as progress:
        task = progress.add_task("Gathering repositories...", total=len(projects))
        for project in projects:
            projects_report.append(_project_payload(project))
            default_branch = project.get("default_branch")
            if not default_branch:
                log.warning("Project %s has no default branch; skipping large file scanning.", project.get("path_with_namespace"))
            else:
                for entry in client.iter_project_tree(project_id=int(project["id"]), ref=default_branch, recursive=True):
                    if entry.get("type") != "blob":
                        continue
                    blob_sha = entry.get("id")
                    if not blob_sha:
                        continue
                    blob = client.get_blob(project_id=int(project["id"]), blob_sha=blob_sha)
                    size = int(blob.get("size") or 0)
                    if size >= lfs_threshold_bytes:
                        large_files.append(
                            LargeFileReport(
                                project_id=int(project["id"]),
                                project_path=project.get("path_with_namespace") or project.get("path") or "",
                                file_path=entry.get("path", ""),
                                size_bytes=size,
                                size_mb=round(size / (1024 * 1024), 2),
                                blob_sha=blob_sha,
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
            _collect_project_variables(client, project, variables, console=console)
            progress.advance(task)

    _write_csv(
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
        ],
        (asdict(r) for r in projects_report),
    )
    _write_csv(
        output_dir / "large_files.csv",
        ["project_id", "project_path", "file_path", "size_bytes", "size_mb", "blob_sha"],
        (asdict(l) for l in large_files),
    )
    _write_csv(
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
    )
    _write_csv(
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
    )
    _write_csv(
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
    )
    _write_csv(
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
    )
    _write_csv(
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
    )
    _write_csv(
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
    )
    _write_csv(
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
    )
    _write_csv(
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
    )
    _write_csv(
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
    )
    _write_csv(
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
    )
    _write_csv(
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
    )
    _write_csv(
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
    )
    _write_csv(
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
    )
    _write_csv(
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
    )
    _write_csv(
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
    )
    _write_csv(
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
    )

    combined_csv = output_dir / "repo_client_report.csv"
    _write_csv(
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
        "variables": len(variables),
        "lfs_threshold_bytes": lfs_threshold_bytes,
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
                "ci_variables": [asdict(var) for var in variables],
            },
            fh,
            indent=2,
            sort_keys=True,
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
        variables=variables,
        summary=summary,
        output_dir=output_dir,
        combined_csv=combined_csv,
    )


def _project_payload(project: dict) -> ProjectReport:
    stats = project.get("statistics") or {}
    namespace = project.get("namespace") or {}
    forked_from = project.get("forked_from_project")
    forked_path = None
    if isinstance(forked_from, dict):
        forked_path = forked_from.get("path_with_namespace")

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
    )


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
