# GitLab Discovery CLI

CLI to inventory GitLab SaaS groups, subgroups, and members for migration prep.

## Configure
- Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv) installed.
- Create and populate `.env`:
  - `GITLAB_TOKEN=` GitLab PAT with `read_api` (group Owner recommended).
  - `GITLAB_BASE_URL=https://gitlab.com` (change if self-managed, e.g., `https://gitlab.ustpace.com/`).
  - `GITLAB_ROOT_GROUP=` group ID or full path to start recursion (leave blank to auto-use your only top-level group).
  - `OUTPUT_DIR=reports`
  - `INCLUDE_EMAILS=false` (set to true if you have admin to fetch emails).
- Create env and install deps:
```bash
uv venv
source .venv/bin/activate
uv sync
```

## Run
- Using env vars from `.env` (loads via Typer env support):
```bash
export $(cat .env | xargs)
uv run gitlab-discovery audit
```
- Reports land in `OUTPUT_DIR`: `groups.csv`, `users.csv`, `memberships.csv`, `summary.json`.

## Supported parameters
- `--token`: GitLab PAT with `read_api` (required).
- `--root-group`: Group ID or full path to start discovery; if omitted, uses your only top-level group (membership).
- `--base-url`: GitLab instance URL (supports SaaS and self-managed, e.g., `https://gitlab.ustpace.com/`).
- `--output`: Directory to write reports.
- `--include-emails`: Attempt to fetch user emails and bot flag (needs admin; best effort).
