#!/usr/bin/env python3
# DISCLAIMER: This script is provided "AS IS" without warranty of any kind.
# Use at your own risk. Deletion is irreversible. No liability for data loss or downtime.
# Dependencies: Python 3.10+ (stdlib only).
# GitHub App auth requires: pyjwt + cryptography (pip install pyjwt cryptography).
# What it does:
#   Deletes all repositories in a given GitHub/GHE organization, after explicit confirmation.
# Usage:
#   export GITHUB_TOKEN="YOUR_TOKEN"
#   export GITHUB_API_URL="https://api.github.com"   # optional (e.g., https://tekion.ghe.com/api/v3)
#   python3 delete_org_repos.py --org my-org
#   python3 delete_org_repos.py --org my-org --dry-run
#   python3 delete_org_repos.py --org my-org

from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
import os
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse, parse_qs
from urllib.request import Request, urlopen


def load_env_file(path: str) -> None:
    if not path:
        return
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value


def normalize_api_base(base_url: str) -> str:
    if not base_url:
        return "https://api.github.com"
    parsed = urlparse(base_url)
    if not parsed.scheme:
        base_url = f"https://{base_url}"
        parsed = urlparse(base_url)
    if "api.github.com" in parsed.netloc:
        return base_url.rstrip("/")
    if "/api/" in parsed.path:
        return base_url.rstrip("/")
    return base_url.rstrip("/") + "/api/v3"


def api_request(
    base_url: str,
    token: str,
    method: str,
    path: str,
    params: dict | None = None,
    accept: str | None = None,
):
    if path.startswith("http://") or path.startswith("https://"):
        url = path
        if params:
            joiner = "&" if "?" in url else "?"
            url = f"{url}{joiner}{urlencode(params, doseq=True)}"
    else:
        url = f"{base_url.rstrip('/')}{path}"
        if params:
            url = f"{url}?{urlencode(params, doseq=True)}"
    accept_header = accept if accept is not None else (os.getenv("GITHUB_ACCEPT") or "application/vnd.github.v3+json")
    user_agent = os.getenv("GITHUB_USER_AGENT", "delete-org-repos")
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": user_agent,
    }
    if accept_header:
        headers["Accept"] = accept_header
    api_version = os.getenv("GITHUB_API_VERSION")
    if api_version:
        headers["X-GitHub-Api-Version"] = api_version
    req = Request(url, method=method, headers=headers)
    with urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if body and "json" in content_type:
            try:
                payload = json.loads(body)
            except JSONDecodeError:
                payload = body
        else:
            payload = body if body else None
        link = resp.headers.get("Link", "")
    return payload, link


def parse_next_link(link_header: str) -> str | None:
    if not link_header:
        return None
    parts = [p.strip() for p in link_header.split(",")]
    for part in parts:
        if 'rel="next"' not in part:
            continue
        url_part = part.split(";", 1)[0].strip()
        if url_part.startswith("<") and url_part.endswith(">"):
            return url_part[1:-1]
    return None


def list_org_repos(base_url: str, token: str, org: str) -> list[dict]:
    repos: list[dict] = []
    path = f"/orgs/{org}/repos"
    params = {"per_page": 100, "type": "all"}
    payload, link = api_request(base_url, token, "GET", path, params=params)
    if isinstance(payload, list):
        repos.extend(payload)
    next_url = parse_next_link(link)
    while next_url:
        payload, link = api_request(base_url, token, "GET", next_url)
        if isinstance(payload, list):
            repos.extend(payload)
        next_url = parse_next_link(link)
    return repos


def list_installation_repos(base_url: str, token: str) -> list[dict]:
    repos: list[dict] = []
    path = "/installation/repositories"
    params = {"per_page": 100}
    payload, link = api_request(base_url, token, "GET", path, params=params)
    if isinstance(payload, dict):
        repos.extend(payload.get("repositories") or [])
    next_url = parse_next_link(link)
    while next_url:
        payload, link = api_request(base_url, token, "GET", next_url)
        if isinstance(payload, dict):
            repos.extend(payload.get("repositories") or [])
        next_url = parse_next_link(link)
    return repos


def generate_app_jwt(app_id: str, private_key_pem: str) -> str:
    try:
        import jwt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyJWT is required for GitHub App auth. Install: pip install pyjwt cryptography") from exc
    now = int(time.time())
    payload = {"iat": now - 60, "exp": now + 540, "iss": app_id}
    token = jwt.encode(payload, private_key_pem, algorithm="RS256")
    return token.decode("utf-8") if isinstance(token, bytes) else token


def get_installation_token(base_url: str) -> str | None:
    app_id = os.getenv("GITHUB_APP_ID")
    installation_id = os.getenv("GITHUB_INSTALLATION_ID")
    private_key = os.getenv("GITHUB_APP_PRIVATE_KEY")
    if not (app_id and installation_id and private_key):
        return None
    private_key = private_key.replace("\\n", "\n")
    app_jwt = generate_app_jwt(app_id, private_key)
    accept_candidates = [
        os.getenv("GITHUB_ACCEPT"),
        "application/vnd.github.machine-man-preview+json",
        "application/vnd.github.v3+json",
        "application/vnd.github+json",
        "application/json",
        "*/*",
        "",
    ]
    payload = None
    last_error: HTTPError | None = None
    for accept in [a for a in accept_candidates if a]:
        try:
            payload, _ = api_request(
                base_url,
                app_jwt,
                "POST",
                f"/app/installations/{installation_id}/access_tokens",
                accept=accept,
            )
            break
        except HTTPError as exc:
            last_error = exc
            if exc.code != 406:
                raise
            continue
    if payload is None and last_error is not None:
        raise last_error
    if isinstance(payload, str) and "<html" in payload.lower():
        raise RuntimeError(
            "GitHub App authentication returned HTML. Check GITHUB_API_BASE (should include /api/v3) "
            "and confirm GitHub Apps are supported on this instance."
        )
    if not isinstance(payload, dict) or "token" not in payload:
        raise RuntimeError(f"GitHub App authentication failed; response: {payload!r}")
    return str(payload["token"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete all repositories in a GitHub/GHE organization.")
    parser.add_argument("--org", required=True, help="Organization name (owner).")
    parser.add_argument("--base-url", default=None, help="GitHub API base URL.")
    parser.add_argument("--token", default=None, help="GitHub token with delete_repo scope.")
    parser.add_argument("--dry-run", action="store_true", help="List repos but do not delete.")
    args = parser.parse_args()

    load_env_file(".env")
    if not args.base_url:
        args.base_url = os.getenv("GITHUB_API_BASE") or os.getenv("GITHUB_API_URL") or os.getenv("GITHUB_BASE_URL") or ""
    args.base_url = normalize_api_base(args.base_url)
    if not args.token:
        args.token = os.getenv("GITHUB_TOKEN") or ""

    token = args.token
    using_app_token = False
    if not token:
        try:
            token = get_installation_token(args.base_url) or ""
            using_app_token = True
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if not token:
        print("Missing GitHub credentials (GITHUB_TOKEN or GitHub App credentials).", file=sys.stderr)
        return 1

    try:
        if using_app_token:
            all_repos = list_installation_repos(args.base_url, token)
            repos = [r for r in all_repos if (r.get("owner") or {}).get("login") == args.org]
        else:
            repos = list_org_repos(args.base_url, token, args.org)
    except HTTPError as exc:
        print(f"GitHub API error: {exc.code} {exc.reason} at {exc.url}", file=sys.stderr)
        return 1
    if not repos:
        print(f"No repositories found under org '{args.org}'.")
        return 0

    print(f"Found {len(repos)} repositories in org '{args.org}'.")
    print("Sample (first 20):")
    for repo in repos[:20]:
        print(f"- {repo.get('name')} ({repo.get('html_url')})")
    if len(repos) > 20:
        print(f"...and {len(repos) - 20} more.")

    if args.dry_run:
        print("Dry run enabled. No deletions performed.")
        return 0

    print("\nWARNING: This will permanently delete ALL repositories listed above.")
    confirm_org = input(f"Type the org name '{args.org}' to confirm: ").strip()
    if confirm_org != args.org:
        print("Confirmation did not match. Aborting.")
        return 1
    confirm = input("Type DELETE to confirm irreversible deletion: ").strip()
    if confirm != "DELETE":
        print("Confirmation not received. Aborting.")
        return 1

    errors = 0
    for repo in repos:
        name = repo.get("name")
        if not name:
            continue
        path = f"/repos/{args.org}/{name}"
        try:
            api_request(args.base_url, token, "DELETE", path)
            print(f"Deleted: {args.org}/{name}")
        except Exception as exc:
            errors += 1
            print(f"Failed: {args.org}/{name} -> {exc}", file=sys.stderr)

    if errors:
        print(f"Completed with {errors} errors.")
        return 1
    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
