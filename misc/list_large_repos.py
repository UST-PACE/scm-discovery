#!/usr/bin/env python3

# *********************************************************************************************
# DISCLAIMER: This script is provided "AS IS" without warranty of any kind.
# Use at your own risk. No liability for data loss, repo changes, or downtime.
#*********************************************************************************************

# Dependencies: Python 3.10+ (stdlib only; no pip installs required).


# What it python script does:
#   Lists GitLab repositories larger than a given size threshold (using statistics)
#   across top-level groups you can access, and writes results to a CSV file.


# Optional venv (Linux/macOS):
#   python3 -m venv .venv
#   source .venv/bin/activate


# Usage:
#   export GITLAB_TOKEN="YOUR_TOKEN"
#   export GITLAB_BASE_URL="https://gitlab.com"   # optional
#   python3 list_large_repos.py --min-repo-mb 100 --size-field storage_size --out large_repos.csv

import argparse
import csv
import json
import os
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

def api_get(base_url, token, path, params):
    url = f"{base_url.rstrip('/')}/api/v4{path}?{urlencode(params)}"
    req = Request(url, headers={"PRIVATE-TOKEN": token})
    with urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        next_page = resp.headers.get("X-Next-Page", "")
    return data, next_page

def paginate(base_url, token, path, params):
    page = 1
    while True:
        page_params = dict(params)
        page_params["page"] = page
        page_params["per_page"] = 100
        data, next_page = api_get(base_url, token, path, page_params)
        if isinstance(data, list):
            for item in data:
                yield item
        if not next_page:
            break
        page = int(next_page)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-repo-mb", type=float, default=100.0)
    parser.add_argument(
        "--size-field",
        default="storage_size",
        choices=[
            "repository_size",
            "storage_size",
            "lfs_objects_size",
            "packages_size",
            "wiki_size",
            "job_artifacts_size",
        ],
    )
    parser.add_argument("--out", default="large_repos.csv")
    parser.add_argument("--include-subgroups", action="store_true", default=True)
    parser.add_argument("--no-include-subgroups", dest="include_subgroups", action="store_false")
    parser.add_argument("--all-accessible-groups", action="store_true", default=False)
    args = parser.parse_args()

    token = os.getenv("GITLAB_TOKEN")
    if not token:
        print("Missing GITLAB_TOKEN env var", file=sys.stderr)
        sys.exit(1)
    base_url = os.getenv("GITLAB_BASE_URL", "https://gitlab.com")

    min_bytes = int(args.min_repo_mb * 1024 * 1024)
    rows = []
    seen = set()

    group_params = {"top_level_only": True}
    if not args.all_accessible_groups:
        group_params["membership"] = True

    for group in paginate(base_url, token, "/groups", group_params):
        group_id = int(group["id"])
        for proj in paginate(
            base_url,
            token,
            f"/groups/{group_id}/projects",
            {
                "include_subgroups": args.include_subgroups,
                "with_shared": False,
                "statistics": True,
            },
        ):
            pid = int(proj.get("id") or 0)
            if pid in seen:
                continue
            seen.add(pid)
            stats = proj.get("statistics") or {}
            size = stats.get(args.size_field)
            if size is None:
                continue
            try:
                size = int(size)
            except (TypeError, ValueError):
                continue
            if size < min_bytes:
                continue
            rows.append({
                "id": proj.get("id"),
                "name": proj.get("name"),
                "path_with_namespace": proj.get("path_with_namespace") or proj.get("path"),
                "web_url": proj.get("web_url"),
                "repository_size_mb": round((stats.get("repository_size") or 0) / (1024 * 1024), 2),
                "storage_size_mb": round((stats.get("storage_size") or 0) / (1024 * 1024), 2),
                "lfs_objects_size_mb": round((stats.get("lfs_objects_size") or 0) / (1024 * 1024), 2),
            })

    rows.sort(key=lambda r: r["storage_size_mb"], reverse=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [
            "id","name","path_with_namespace","web_url","repository_size_mb","storage_size_mb","lfs_objects_size_mb"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} repos to {args.out}")

if __name__ == "__main__":
    main()
