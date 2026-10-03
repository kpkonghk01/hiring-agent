"""Batch pipeline: Resume -> Extracted Resume -> Evaluation, reusing the store (ADR 0002).

The side-effecting steps (OCR + extraction, GitHub Enrichment, judging) are injected
as ``Stages`` so the cache rules can be tested with fakes.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

from pydantic import BaseModel

from custom.github_enrichment import UNAVAILABLE, GitHubOutcome
from custom.sources import ResumeRef, ResumeSource
from custom.store import EvaluationKey, Store
from models import JSONResume
from roles import Role


@dataclass(frozen=True)
class Stages:
    extract: Callable[[bytes, str], Optional[JSONResume]]  # (pdf bytes, file name)
    enrich: Callable[[JSONResume, Role], GitHubOutcome]
    evaluate: Callable[[JSONResume, Optional[dict], Role], BaseModel]


@dataclass(frozen=True)
class RunSettings:
    role: Role
    role_hash: str
    judge: str  # provider:model that judges Evaluations
    extracted_by: str  # provider:model that extracts new Resumes
    re_extract: bool = False
    re_evaluate: bool = False


@dataclass(frozen=True)
class ResumeResult:
    refs: Tuple[ResumeRef, ...]  # every file with this Resume's content
    resume: Optional[JSONResume] = None
    evaluation: Optional[dict] = None
    github: Optional[dict] = None
    extraction_reused: bool = False
    evaluation_reused: bool = False
    failed_step: Optional[str] = None  # download | extract | github | evaluate
    failure: Optional[str] = None

    @property
    def resume_hash(self) -> str:
        return self.refs[0].resume_hash

    @property
    def ok(self) -> bool:
        return self.failed_step is None


class _StepFailed(Exception):
    def __init__(self, step: str, reason: str):
        super().__init__(reason)
        self.step = step
        self.reason = reason


def group_by_resume(refs: Iterable[ResumeRef]) -> List[Tuple[ResumeRef, ...]]:
    """Group files with identical content, keeping first-seen order."""
    groups = {}
    for r in refs:
        groups[r.resume_hash] = groups.get(r.resume_hash, ()) + (r,)
    return list(groups.values())


def run(
    source: ResumeSource,
    store: Store,
    settings: RunSettings,
    stages: Stages,
    limit: Optional[int] = None,
    on_start: Optional[Callable[[int, int, Tuple[ResumeRef, ...]], None]] = None,
    on_result: Optional[Callable[[int, int, ResumeResult], None]] = None,
) -> List[ResumeResult]:
    """Process every Resume in ``source`` (at most ``limit``), one at a time."""
    groups = group_by_resume(source.list())[:limit]
    results = []
    for index, refs in enumerate(groups, start=1):
        if on_start:
            on_start(index, len(groups), refs)
        result = process_resume(refs, source, store, settings, stages)
        results.append(result)
        if on_result:
            on_result(index, len(groups), result)
    return results


def process_resume(
    refs: Tuple[ResumeRef, ...],
    source: ResumeSource,
    store: Store,
    settings: RunSettings,
    stages: Stages,
) -> ResumeResult:
    try:
        resume, extraction_reused = _extracted_resume(refs[0], source, store, settings, stages)
        key = EvaluationKey(refs[0].resume_hash, settings.role_hash, settings.judge)
        stored = None if settings.re_evaluate else store.get_evaluation(key)
        if stored:
            return ResumeResult(
                refs, resume, stored.evaluation, stored.github, extraction_reused, True
            )
        evaluation, github = _new_evaluation(resume, key, store, settings, stages)
        return ResumeResult(refs, resume, evaluation, github, extraction_reused, False)
    except _StepFailed as failed:
        return ResumeResult(refs, failed_step=failed.step, failure=failed.reason)


def _extracted_resume(ref, source, store, settings, stages) -> Tuple[JSONResume, bool]:
    if not settings.re_extract:
        cached = store.get_extracted(ref.resume_hash)
        if cached is not None:
            return cached, True
    data = _attempt("download", source.read_bytes, ref)
    resume = _attempt("extract", stages.extract, data, ref.name)
    if not _has_core_content(resume):
        raise _StepFailed("extract", "no usable data extracted (run with --verbose for details)")
    store.put_extracted(ref.resume_hash, resume, settings.extracted_by)
    return resume, False


def _new_evaluation(resume, key, store, settings, stages) -> Tuple[dict, Optional[dict]]:
    outcome = _attempt("github", stages.enrich, resume, settings.role)
    if outcome.status == UNAVAILABLE:
        raise _StepFailed("github", f"could not fetch GitHub profile {outcome.profile_url}")
    evaluation = _attempt("evaluate", stages.evaluate, resume, outcome.data, settings.role)
    evaluation_dict = evaluation.model_dump()
    store.put_evaluation(key, settings.role.name, evaluation_dict, outcome.data)
    return evaluation_dict, outcome.data


def _attempt(step: str, fn: Callable, *args):
    try:
        return fn(*args)
    except Exception as err:  # one Resume's failure must not stop the batch
        raise _StepFailed(step, f"{type(err).__name__}: {err}") from err


def _has_core_content(resume: Optional[JSONResume]) -> bool:
    """Mirrors score.is_valid_resume_data (importing score.py has side effects)."""
    if resume is None:
        return False
    core = (resume.basics, resume.work, resume.education, resume.skills, resume.projects)
    return any(section is not None for section in core)
