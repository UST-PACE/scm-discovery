from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

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
class GroupInfo:
    id: int
    name: str
    full_path: str
    parent_id: Optional[int]
    web_url: Optional[str]


@dataclass
class UserInfo:
    id: int
    username: str
    name: str
    state: str
    is_bot: Optional[bool] = None
    web_url: Optional[str] = None
    email: Optional[str] = None


@dataclass
class MembershipInfo:
    group_id: int
    group_full_path: str
    user_id: int
    username: str
    access_level: int
    access_label: str
    expires_at: Optional[str]


@dataclass
class AuditResult:
    groups: list[GroupInfo]
    users: dict[int, UserInfo]
    memberships: list[MembershipInfo]
    summary: dict[str, object]
    output_dir: Path


def _access_label(level: int) -> str:
    return ACCESS_LEVELS.get(level, f"unknown_{level}")


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def audit_gitlab(
    client: GitLabClient,
    root_group_ref: str,
    include_emails: bool = False,
    output_dir: Path | str = "output",
    console: Console | None = None,
) -> AuditResult:
    console = console or Console(stderr=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log.info("Output directory prepared at %s", output_path.resolve())

    groups: list[GroupInfo] = []
    users: dict[int, UserInfo] = {}
    memberships: list[MembershipInfo] = []
    user_groups: defaultdict[int, set[str]] = defaultdict(set)

    root_group = client.get_group(root_group_ref)
    console.print(f"Discovered root group: [bold]{root_group.get('full_path')}[/] (id={root_group.get('id')})")
    log.info(
        "Fetched root group %s (id=%s)",
        root_group.get("full_path") or root_group.get("path"),
        root_group.get("id"),
    )

    def upsert_user(member_obj: dict) -> UserInfo:
        user_id = int(member_obj["id"])
        user = users.get(user_id)
        is_bot = member_obj.get("is_bot")
        if is_bot is None and "bot" in member_obj:
            is_bot = member_obj.get("bot")
        email = member_obj.get("email")
        web_url = member_obj.get("web_url")

        if user is None:
            user = UserInfo(
                id=user_id,
                username=member_obj.get("username", ""),
                name=member_obj.get("name", ""),
                state=member_obj.get("state", "unknown"),
                is_bot=is_bot,
                web_url=web_url,
                email=email,
            )
            users[user_id] = user
        else:
            # Fill missing fields when available.
            if user.email is None and email:
                user.email = email
            if user.is_bot is None and is_bot is not None:
                user.is_bot = bool(is_bot)
            if user.web_url is None and web_url:
                user.web_url = web_url
        return user

    def walk_group(group_obj: dict, parent_id: Optional[int]) -> None:
        group_info = GroupInfo(
            id=int(group_obj["id"]),
            name=group_obj.get("name", ""),
            full_path=group_obj.get("full_path") or group_obj.get("path") or "",
            parent_id=parent_id,
            web_url=group_obj.get("web_url"),
        )
        groups.append(group_info)
        log.debug("Processing group %s (id=%s)", group_info.full_path, group_info.id)

        for member in client.iter_group_members(group_info.id, include_inherited=True):
            user = upsert_user(member)
            access_level = int(member.get("access_level", 0))
            membership = MembershipInfo(
                group_id=group_info.id,
                group_full_path=group_info.full_path,
                user_id=user.id,
                username=user.username,
                access_level=access_level,
                access_label=_access_label(access_level),
                expires_at=member.get("expires_at"),
            )
            memberships.append(membership)
            user_groups[user.id].add(group_info.full_path)

        for subgroup in client.iter_subgroups(group_info.id):
            walk_group(subgroup, parent_id=group_info.id)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Fetching group hierarchy and memberships...", start=True)
        walk_group(root_group, parent_id=root_group.get("parent_id"))
        progress.update(task, advance=100)

    if include_emails:
        console.print("Enriching users with email/bot flags (best effort)...")
        for user in users.values():
            try:
                detail = client.get_user(user.id)
            except GitLabAPIError as exc:  # skip users we cannot read
                console.print(f"[yellow]Warning:[/] could not fetch user {user.username} ({user.id}): {exc}")
                log.warning("Could not fetch user %s (%s): %s", user.username, user.id, exc)
                continue
            if detail.get("email"):
                user.email = detail["email"]
            if detail.get("bot") is not None:
                user.is_bot = bool(detail["bot"])
            elif detail.get("is_bot") is not None:
                user.is_bot = bool(detail["is_bot"])

    # Write reports
    _write_csv(
        output_path / "groups.csv",
        ["id", "name", "full_path", "parent_id", "web_url"],
        (asdict(g) for g in groups),
    )
    log.info("groups.csv written with %s rows", len(groups))
    _write_csv(
        output_path / "users.csv",
        ["id", "username", "name", "state", "is_bot", "web_url", "email"],
        (
            {
                **asdict(u),
                "is_bot": "" if u.is_bot is None else str(u.is_bot).lower(),
            }
            for u in users.values()
        ),
    )
    log.info("users.csv written with %s rows", len(users))
    _write_csv(
        output_path / "memberships.csv",
        ["group_id", "group_full_path", "user_id", "username", "access_level", "access_label", "expires_at"],
        (asdict(m) for m in memberships),
    )
    log.info("memberships.csv written with %s rows", len(memberships))

    user_group_details = [
        {
            "id": user.id,
            "username": user.username,
            "name": user.name,
            "email": user.email,
            "groups": sorted(user_groups.get(user.id, [])),
        }
        for user in users.values()
    ]

    summary = {
        "groups": len(groups),
        "users": len(users),
        "bots": sum(1 for u in users.values() if u.is_bot),
        "memberships": len(memberships),
        "states": Counter(u.state for u in users.values()),
        "user_group_details": user_group_details,
        "groups_data": [asdict(g) for g in groups],
        "users_data": [
            {
                **asdict(u),
                "is_bot": None if u.is_bot is None else bool(u.is_bot),
            }
            for u in users.values()
        ],
        "memberships_data": [asdict(m) for m in memberships],
    }
    with (output_path / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)
    log.info("summary.json written (groups=%s users=%s memberships=%s)", summary["groups"], summary["users"], summary["memberships"])

    console.print(f"Wrote reports to {output_path.resolve()}")
    return AuditResult(groups=groups, users=users, memberships=memberships, summary=summary, output_dir=output_path)


def write_combined_membership_report(results: Sequence[AuditResult], destination: Path | str) -> Path:
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "root_group",
        "group_id",
        "group_full_path",
        "group_parent_id",
        "group_web_url",
        "user_id",
        "username",
        "name",
        "email",
        "state",
        "is_bot",
        "user_web_url",
        "access_level",
        "access_label",
        "expires_at",
    ]
    with dest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            root_group = result.groups[0].full_path if result.groups else "unknown"
            group_lookup = {g.id: g for g in result.groups}
            for membership in result.memberships:
                group_info = group_lookup.get(membership.group_id)
                user = result.users.get(membership.user_id)
                writer.writerow(
                    {
                        "root_group": root_group,
                        "group_id": membership.group_id,
                        "group_full_path": membership.group_full_path,
                        "group_parent_id": group_info.parent_id if group_info else None,
                        "group_web_url": group_info.web_url if group_info else None,
                        "user_id": membership.user_id,
                        "username": membership.username,
                        "name": user.name if user else "",
                        "email": user.email if user else "",
                        "state": user.state if user else "",
                        "is_bot": "" if user is None or user.is_bot is None else str(user.is_bot).lower(),
                        "user_web_url": user.web_url if user else "",
                        "access_level": membership.access_level,
                        "access_label": membership.access_label,
                        "expires_at": membership.expires_at,
                    }
                )

    log.info("Combined client report written to %s", dest_path)
    return dest_path
