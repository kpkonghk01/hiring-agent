"""GitHub Enrichment for one Evaluation, telling a missing profile from a failed fetch.

Upstream's github.py swallows every API failure and returns empty data, so a rate
limit mid-run would look like "no projects". While enriching, every upstream API call
is observed; any transient failure (rate limit, server error, network error) makes
the outcome UNAVAILABLE, so the Evaluation fails and is retried instead of being
stored without its GitHub Enrichment (CONTEXT.md).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import github
from custom.llm import LLMChoice, bound_to, patched
from models import JSONResume, OpenAICompatibleProvider
from roles import Role

logger = logging.getLogger(__name__)

# Outcomes. Only UNAVAILABLE fails the Evaluation; a linked profile that does not exist
# (or a link with no parsable username) will never be fetched, so it is judged without one.
NO_PROFILE = "no_profile"
FETCHED = "fetched"
NOT_FOUND = "not_found"
UNAVAILABLE = "unavailable"

# Statuses that say "try again later" rather than "this does not exist".
_TRANSIENT_STATUSES = {403, 429}


@dataclass(frozen=True)
class GitHubOutcome:
    status: str
    profile_url: Optional[str] = None
    data: Optional[dict] = None  # upstream's {"profile", "projects", ...} when FETCHED
    reason: Optional[str] = None  # why it is UNAVAILABLE


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
    failures: List[str] = []
    with (
        bound_to(github, choice, client),
        patched(
            github, _fetch_github_api=_observed(github._fetch_github_api, failures)
        ),
    ):
        data = github.fetch_and_display_github_info(
            url, position_title=role.position_title
        )
    if failures:
        logger.warning("GitHub fetch for %s failed: %s", url, "; ".join(failures))
        return GitHubOutcome(UNAVAILABLE, url, reason=failures[0])
    if data and "profile" in data:
        return GitHubOutcome(FETCHED, url, data)
    return GitHubOutcome(NOT_FOUND, url)


def _observed(fetch, failures: List[str]):
    """Wrap upstream's GitHub API call, recording transient failures it would swallow."""

    def observed(api_url, params=None):
        try:
            status, data = fetch(api_url, params)
        except Exception as err:
            failures.append(f"{type(err).__name__} on {api_url}: {err}")
            raise
        if status in _TRANSIENT_STATUSES or status >= 500:
            failures.append(f"HTTP {status} on {api_url}")
        return status, data

    return observed
