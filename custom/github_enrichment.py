"""GitHub Enrichment for one Evaluation, telling a missing profile from a failed fetch."""

import os
from dataclasses import dataclass
from typing import Optional

import requests

import github
from custom.llm import LLMChoice, bound_to
from models import JSONResume, OpenAICompatibleProvider
from roles import Role

# Outcomes. Only UNAVAILABLE fails the Evaluation (CONTEXT.md: GitHub Enrichment);
# a profile that does not exist will never be fetched, so it is judged without one.
NO_PROFILE = "no_profile"
FETCHED = "fetched"
NOT_FOUND = "not_found"
UNAVAILABLE = "unavailable"

_GITHUB_API = "https://api.github.com"
_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class GitHubOutcome:
    status: str
    profile_url: Optional[str] = None
    data: Optional[dict] = None  # upstream's {"profile", "projects", ...} when FETCHED


def github_profile_url(resume: JSONResume) -> Optional[str]:
    profiles = (resume.basics.profiles or []) if resume.basics else []
    for profile in profiles:
        if profile.network and profile.network.lower() == "github":
            return profile.url
    return None


def enrich(
    resume: JSONResume, role: Role, choice: LLMChoice, client: OpenAICompatibleProvider
) -> GitHubOutcome:
    """Fetch GitHub Enrichment, selecting projects with the judging model."""
    url = github_profile_url(resume)
    if not url:
        return GitHubOutcome(NO_PROFILE)
    with bound_to(github, choice, client):
        data = github.fetch_and_display_github_info(url, position_title=role.position_title)
    if data and "profile" in data:
        return GitHubOutcome(FETCHED, url, data)
    return GitHubOutcome(_failure_status(url), url)


def _failure_status(url: str) -> str:
    """Upstream returns {} for every failure; ask GitHub whether the user exists at all."""
    username = github.extract_github_username(url)
    if not username:
        return NOT_FOUND
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    try:
        response = requests.get(
            f"{_GITHUB_API}/users/{username}", headers=headers, timeout=_TIMEOUT_SECONDS
        )
    except requests.RequestException:
        return UNAVAILABLE
    return NOT_FOUND if response.status_code == 404 else UNAVAILABLE
