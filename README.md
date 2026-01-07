# GitLab Discovery CLI

CLI to inventory GitLab SaaS groups, subgroups, and members for migration prep.

## Configure
- Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv) installed.
- Create and populate `.env`:
  - `GITLAB_TOKEN=` GitLab PAT with `read_api` (group Owner recommended).
  - `GITLAB_BASE_URL=https://gitlab.com` (change if self-managed, e.g., `https://gitlab.ustpace.com/`).
  - `GITLAB_ROOT_GROUP=` group ID or full path to start recursion (leave blank to audit every accessible top-level group).
  - `OUTPUT_DIR=output`
  - `INCLUDE_EMAILS=false` (set to true if you have admin to fetch emails).
  - `GITLAB_DISCOVERY_LOG_LEVEL=INFO` (optional; set to `DEBUG` for verbose line-by-line tracing).
  - `GITLAB_DISCOVERY_DOTENV=.env` (optional; override to point at a different env file to auto-load).
  - `GITLAB_DISCOVERY_LFS_THRESHOLD_MB=100` (optional; raise/lower the Git LFS candidate threshold for repo audits).
- Create env and install deps:
```bash
uv venv
source .venv/bin/activate
uv sync
```

## Run
- Environment variables from `.env` are loaded automatically (override path with `GITLAB_DISCOVERY_DOTENV` if needed).
- Group/member discovery (default):
```bash
uv run gitlab-discovery audit
```
  - If you omit `--root-group`, every top-level group you can access will be audited automatically. Reports are written beneath `OUTPUT_DIR/<timestamp>/<group_name>`.
  - Each run writes artifacts into `OUTPUT_DIR/<YYYYMMDD-HHMMSS>/<group_name>/` (default base is `output/`), keeping a full history of `groups.csv`, `users.csv`, `memberships.csv`, and `summary.json` (which now mirrors every CSV record plus per-user group/email insight). A merged `client_report.csv` lives at `OUTPUT_DIR/<timestamp>/` with one row per membership including group + user metadata for easy sharing.
- Repository-level audits (with Git LFS candidates):
```bash
# Scan every repository you can access
uv run gitlab-discovery audit repo all

# Focus on a single repository
uv run gitlab-discovery audit group/subgroup/project
```
  - Outputs live under `OUTPUT_DIR/<timestamp>/repos/` and include comprehensive CSVs: repositories, large_files, project_members, project_hooks, project_integrations, pipelines, pipeline_jobs, pipeline_schedules, protected_branches, protected_tags, environments, deployments, packages, registry_repositories, registry_tags, releases, tags, ci_variables, plus `repo_summary.json` (mirrors every record) and `repo_client_report.csv`. Large files ≥ `GITLAB_DISCOVERY_LFS_THRESHOLD_MB` are flagged for Git LFS migration.
- Find large files (Git LFS prep) for a single repository:
```bash
uv run gitlab-discovery find-large-files group/subgroup/project --threshold-mb 50
```
  - Walks the default branch and writes `large_files.csv` + `large_files.json` under `OUTPUT_DIR/<timestamp>/large-files/<project>/`.
- You can also pass the repository via `--repo-name` (instead of positional):
```bash
uv run gitlab-discovery find-large-files --repo-name group/subgroup/project --threshold-mb 50
```
- Find large files across every repo in a group (with subgroups):
```bash
uv run gitlab-discovery find-large-files portal-services --group --threshold-mb 50
```
  - Scans each project’s default branch under the group, writes one CSV/JSON, and lists any projects skipped for missing default branches.
- Check if a repository uses Git LFS (flag + .gitattributes + pointer detection):
```bash
uv run gitlab-discovery check-lfs https://gitlab.ustpace.com/group/subgroup/repo
```
  - Writes `lfs_check.json` under `OUTPUT_DIR/<timestamp>/lfs-check/<repo>/` with flags and sample pointer files if found.
- Quick repository count/listing (lightweight, read-only):
```bash
uv run gitlab-discovery list-repos --root-group my-group
# or list everything you can access
uv run gitlab-discovery list-repos
```
  - Writes a `repositories.csv` and `repositories.json` under `OUTPUT_DIR/<timestamp>/repo-list-*/` with id, path, visibility, and other metadata, and prints a count to the console.
- Just names and URLs for all accessible repos:
```bash
uv run gitlab-discovery list-repo-urls
```
  - Saves `repository_urls.csv` and `repository_urls.json` under `OUTPUT_DIR/<timestamp>/repo-urls/`.
- Find duplicate repository names (detect same repo name across projects):
```bash
uv run gitlab-discovery find-duplicate-repos
```
  - Writes `duplicate_repositories.csv` and `duplicate_repositories.json` under `OUTPUT_DIR/<timestamp>/duplicate-repos/`; defaults to membership-only, use `--all-accessible` to widen the scope.
- All users (active by default; best-effort email + bots included):
```bash
uv run gitlab-discovery list-users --include-email
```
  - Saves `users.csv` and `users.json` under `OUTPUT_DIR/<timestamp>/users/`. Email visibility on GitLab SaaS generally requires an admin token; otherwise emails may be blank and warnings are recorded.

## Supported parameters
- `--token`: GitLab PAT with `read_api` (required).
- `--root-group`: Group ID or full path to start discovery; if omitted, every accessible top-level group is audited.
- `--base-url`: GitLab instance URL (supports SaaS and self-managed, e.g., `https://gitlab.ustpace.com/`).
- `--output`: Directory to write reports.
- `--include-emails`: Attempt to fetch user emails and bot flag (needs admin; best effort).
- `audit repo all`: Repository mode covering every accessible project (finds >100 MB files for Git LFS).
- `audit <group/project>`: Repository mode scoped to one project path or ID.
