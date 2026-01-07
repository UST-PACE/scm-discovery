from __future__ import annotations

import time
import typing as t
from dataclasses import dataclass
from urllib.parse import quote

import httpx


@dataclass
class GitLabAPIError(Exception):
    message: str
    status_code: int | None = None
    body: str | None = None

    def __str__(self) -> str:  # pragma: no cover - simple wrapper
        base = self.message
        if self.status_code:
            base += f" (status {self.status_code})"
        if self.body:
            base += f": {self.body}"
        return base


def _encode_group_ref(ref: str | int) -> str:
    if isinstance(ref, int) or str(ref).isdigit():
        return str(ref)
    # Preserve slashes in group path
    return quote(str(ref), safe="/")


def _encode_project_ref(ref: str | int) -> str:
    if isinstance(ref, int) or str(ref).isdigit():
        return str(ref)
    # Encode slashes for project refs
    return quote(str(ref), safe="")


class GitLabClient:
    """
    Thin wrapper around GitLab REST API (v4) with pagination and retry handling.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        timeout: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        api_base = base_url.rstrip("/") + "/api/v4"
        self.api_base = api_base
        self._client = httpx.Client(
            headers={"PRIVATE-TOKEN": token},
            timeout=timeout,
        )
        self._max_retries = max_retries

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GitLabClient":
        return self

    def __exit__(self, *_: t.Any) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, t.Any] | None = None,
    ) -> httpx.Response:
        url = f"{self.api_base}{path}"
        attempt = 0
        backoff = 1.0
        while True:
            try:
                response = self._client.request(method, url, params=params)
            except httpx.RequestError as exc:  # network/timeout errors
                if attempt >= self._max_retries:
                    raise GitLabAPIError(f"Request to {url} failed: {exc}") from exc
                time.sleep(backoff)
                attempt += 1
                backoff = min(backoff * 2, 10.0)
                continue

            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= self._max_retries:
                    raise GitLabAPIError(
                        f"GitLab API error at {url}",
                        status_code=response.status_code,
                        body=response.text,
                    )
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else backoff
                time.sleep(delay)
                attempt += 1
                backoff = min(backoff * 2, 10.0)
                continue

            if response.status_code >= 400:
                raise GitLabAPIError(
                    f"GitLab API error at {url}",
                    status_code=response.status_code,
                    body=response.text,
                )
            return response

    def _paginate(
        self,
        path: str,
        params: dict[str, t.Any] | None = None,
    ) -> t.Iterator[dict[str, t.Any]]:
        page = 1
        params = params or {}
        while True:
            page_params = {**params, "page": page, "per_page": 100}
            response = self._request("GET", path, params=page_params)
            items = response.json()
            if not isinstance(items, list):
                raise GitLabAPIError(f"Expected list response for {path}")
            for item in items:
                yield item
            next_page = response.headers.get("X-Next-Page")
            if not next_page:
                break
            page = int(next_page)

    def get_group(self, ref: str | int) -> dict[str, t.Any]:
        ref_enc = _encode_group_ref(ref)
        response = self._request("GET", f"/groups/{ref_enc}")
        return response.json()

    def iter_subgroups(self, group_id: int) -> t.Iterator[dict[str, t.Any]]:
        return self._paginate(
            f"/groups/{group_id}/subgroups",
            params={"with_projects": False},
        )

    def iter_group_projects(
        self,
        group_id: int,
        include_subgroups: bool = False,
    ) -> t.Iterator[dict[str, t.Any]]:
        params: dict[str, t.Any] = {
            "include_subgroups": include_subgroups,
            "with_shared": False,
        }
        return self._paginate(f"/groups/{group_id}/projects", params=params)

    def iter_group_projects(
        self,
        group_id: int,
        include_subgroups: bool = False,
    ) -> t.Iterator[dict[str, t.Any]]:
        params: dict[str, t.Any] = {
            "include_subgroups": include_subgroups,
            "with_shared": False,
        }
        return self._paginate(f"/groups/{group_id}/projects", params=params)

    def iter_top_level_groups(self, membership_only: bool = True) -> t.Iterator[dict[str, t.Any]]:
        params: dict[str, t.Any] = {"top_level_only": True}
        if membership_only:
            params["membership"] = True
        return self._paginate("/groups", params=params)

    def iter_users(
        self,
        active: bool | None = None,
        blocked: bool | None = None,
    ) -> t.Iterator[dict[str, t.Any]]:
        params: dict[str, t.Any] = {
            "order_by": "id",
            "sort": "asc",
        }
        if active is not None:
            params["active"] = active
        if blocked is not None:
            params["blocked"] = blocked
        return self._paginate("/users", params=params)

    def iter_group_members(
        self,
        group_id: int,
        include_inherited: bool = True,
    ) -> t.Iterator[dict[str, t.Any]]:
        suffix = "/all" if include_inherited else ""
        return self._paginate(f"/groups/{group_id}/members{suffix}")

    def get_user(self, user_id: int) -> dict[str, t.Any]:
        response = self._request("GET", f"/users/{user_id}")
        return response.json()

    def get_current_user(self) -> dict[str, t.Any]:
        response = self._request("GET", "/user")
        return response.json(), response.headers

    def iter_projects(
        self,
        membership_only: bool = True,
        include_statistics: bool = True,
        simple: bool = False,
        search: str | None = None,
    ) -> t.Iterator[dict[str, t.Any]]:
        params: dict[str, t.Any] = {
            "order_by": "path",
            "sort": "asc",
        }
        if membership_only:
            params["membership"] = True
        if include_statistics:
            params["statistics"] = True
        if simple:
            params["simple"] = True
        if search:
            params["search"] = search
        return self._paginate("/projects", params=params)

    def get_project(self, ref: str | int, include_statistics: bool = True) -> dict[str, t.Any]:
        ref_enc = _encode_project_ref(ref)
        params = {"statistics": True} if include_statistics else None
        response = self._request("GET", f"/projects/{ref_enc}", params=params)
        return response.json()

    def iter_project_tree(
        self,
        project_id: int,
        ref: str,
        recursive: bool = True,
    ) -> t.Iterator[dict[str, t.Any]]:
        params = {
            "ref": ref,
            "recursive": recursive,
        }
        return self._paginate(f"/projects/{project_id}/repository/tree", params=params)

    def get_blob(self, project_id: int, blob_sha: str) -> dict[str, t.Any]:
        response = self._request("GET", f"/projects/{project_id}/repository/blobs/{blob_sha}")
        return response.json()

    def iter_project_members(
        self,
        project_id: int,
        include_inherited: bool = True,
    ) -> t.Iterator[dict[str, t.Any]]:
        suffix = "/all" if include_inherited else ""
        return self._paginate(f"/projects/{project_id}/members{suffix}")

    def list_project_hooks(self, project_id: int) -> list[dict[str, t.Any]]:
        response = self._request("GET", f"/projects/{project_id}/hooks")
        payload = response.json()
        if not isinstance(payload, list):
            raise GitLabAPIError("Expected hooks list response")
        return payload

    def list_project_integrations(self, project_id: int) -> list[dict[str, t.Any]]:
        response = self._request("GET", f"/projects/{project_id}/integrations")
        payload = response.json()
        if not isinstance(payload, list):
            raise GitLabAPIError("Expected integrations list response")
        return payload

    def iter_project_pipelines(self, project_id: int) -> t.Iterator[dict[str, t.Any]]:
        return self._paginate(f"/projects/{project_id}/pipelines")

    def iter_pipeline_jobs(self, project_id: int, pipeline_id: int) -> t.Iterator[dict[str, t.Any]]:
        return self._paginate(f"/projects/{project_id}/pipelines/{pipeline_id}/jobs")

    def iter_project_releases(self, project_id: int) -> t.Iterator[dict[str, t.Any]]:
        return self._paginate(f"/projects/{project_id}/releases")

    def iter_project_tags(self, project_id: int) -> t.Iterator[dict[str, t.Any]]:
        return self._paginate(f"/projects/{project_id}/repository/tags")

    def iter_project_protected_branches(self, project_id: int) -> t.Iterator[dict[str, t.Any]]:
        return self._paginate(f"/projects/{project_id}/protected_branches")

    def iter_project_protected_tags(self, project_id: int) -> t.Iterator[dict[str, t.Any]]:
        return self._paginate(f"/projects/{project_id}/protected_tags")

    def iter_project_environments(self, project_id: int) -> t.Iterator[dict[str, t.Any]]:
        return self._paginate(f"/projects/{project_id}/environments")

    def iter_project_deployments(self, project_id: int) -> t.Iterator[dict[str, t.Any]]:
        return self._paginate(f"/projects/{project_id}/deployments")

    def iter_project_packages(self, project_id: int) -> t.Iterator[dict[str, t.Any]]:
        return self._paginate(f"/projects/{project_id}/packages")

    def list_registry_repositories(self, project_id: int) -> list[dict[str, t.Any]]:
        response = self._request("GET", f"/projects/{project_id}/registry/repositories")
        payload = response.json()
        if not isinstance(payload, list):
            raise GitLabAPIError("Expected registry repositories list response")
        return payload

    def list_registry_repository_tags(self, project_id: int, registry_id: int) -> list[dict[str, t.Any]]:
        response = self._request("GET", f"/projects/{project_id}/registry/repositories/{registry_id}/tags")
        payload = response.json()
        if not isinstance(payload, list):
            raise GitLabAPIError("Expected registry tags list response")
        return payload

    def iter_project_variables(self, project_id: int) -> t.Iterator[dict[str, t.Any]]:
        return self._paginate(f"/projects/{project_id}/variables")

    def iter_project_pipeline_schedules(self, project_id: int) -> t.Iterator[dict[str, t.Any]]:
        return self._paginate(f"/projects/{project_id}/pipeline_schedules")
