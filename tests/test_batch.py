"""Cache invariants of the batch pipeline (docs/adr/0002), with fake OCR/LLM/GitHub."""

from collections import Counter

import pytest

from custom.github_enrichment import (
    FETCHED,
    NO_PROFILE,
    NOT_FOUND,
    UNAVAILABLE,
    GitHubOutcome,
)
from custom.pipeline import RunSettings, Stages, run
from custom.ranking import ranking_rows
from custom.sources import ResumeRef
from custom.store import EvaluationKey, Store
from models import Basics, JSONResume, build_evaluation_model
from roles import load_role

ROLE = load_role("software_engineering_intern")


def make_evaluation(score):
    return build_evaluation_model(ROLE)(
        scores={
            c.key: {"score": score, "max": c.max, "evidence": "e"}
            for c in ROLE.categories
        },
        bonus_points={"total": 0, "breakdown": "-"},
        deductions={"total": 0, "reasons": "-"},
        key_strengths=["s"],
        areas_for_improvement=["a"],
    )


class FakeSource:
    def __init__(self, *refs):
        self.refs = refs
        self.reads = 0

    def list(self):
        return self.refs

    def read_bytes(self, ref):
        self.reads += 1
        return b"%PDF fake"


class FakeStages:
    def __init__(self, github=NO_PROFILE, extract_ok=True, scores=None):
        self.github = github
        self.extract_ok = extract_ok
        self.scores = scores or {}
        self.calls = Counter()

    def extract(self, data, file_name):
        self.calls["extract"] += 1
        return JSONResume(basics=Basics(name=file_name)) if self.extract_ok else None

    def enrich(self, resume, role):
        self.calls["enrich"] += 1
        data = {"profile": {"username": "x"}} if self.github == FETCHED else None
        return GitHubOutcome(self.github, "https://github.com/x", data)

    def evaluate(self, resume, github, role):
        self.calls["evaluate"] += 1
        return make_evaluation(self.scores.get(resume.basics.name, 5))

    def stages(self):
        return Stages(extract=self.extract, enrich=self.enrich, evaluate=self.evaluate)


def ref(name, resume_hash, batch="20261001"):
    return ResumeRef(
        source_id=f"id-{name}", name=name, resume_hash=resume_hash, batch=batch
    )


def settings(**overrides):
    base = dict(
        role=ROLE, role_hash="role-v1", judge="ollama:a", extracted_by="ollama:x"
    )
    return RunSettings(**{**base, **overrides})


@pytest.fixture
def store():
    with Store(":memory:") as s:
        yield s


def test_second_run_reuses_everything_without_downloading(store):
    source = FakeSource(ref("a.pdf", "h1"), ref("b.pdf", "h2"))
    run(source, store, settings(), FakeStages().stages())

    again = FakeStages()
    results = run(source, store, settings(), again.stages())

    assert again.calls == Counter()
    assert source.reads == 2  # only the first run downloaded
    assert all(r.extraction_reused and r.evaluation_reused for r in results)


def test_identical_files_in_two_batches_are_one_resume(store):
    source = FakeSource(ref("a.pdf", "h1", "20261001"), ref("a.pdf", "h1", "20261003"))
    stages = FakeStages()

    results = run(source, store, settings(), stages.stages())

    assert len(results) == 1
    assert [r.batch for r in results[0].refs] == ["20261001", "20261003"]
    assert stages.calls == Counter(extract=1, enrich=1, evaluate=1)


def test_new_judge_or_edited_role_adds_an_evaluation_but_reuses_extraction(store):
    source = FakeSource(ref("a.pdf", "h1"))
    run(source, store, settings(), FakeStages().stages())

    for changed in (settings(judge="openrouter:b"), settings(role_hash="role-v2")):
        stages = FakeStages()
        run(source, store, changed, stages.stages())
        assert stages.calls == Counter(enrich=1, evaluate=1)

    assert store.get_evaluation(EvaluationKey("h1", "role-v1", "ollama:a")) is not None


def test_re_extract_invalidates_every_evaluation_of_the_resume(store):
    source = FakeSource(ref("a.pdf", "h1"))
    run(source, store, settings(), FakeStages().stages())
    run(source, store, settings(judge="openrouter:b"), FakeStages().stages())

    stages = FakeStages()
    run(source, store, settings(re_extract=True), stages.stages())

    assert stages.calls == Counter(extract=1, enrich=1, evaluate=1)
    assert store.get_evaluation(EvaluationKey("h1", "role-v1", "openrouter:b")) is None


def test_unavailable_github_fails_without_storing_and_is_retried(store):
    source = FakeSource(ref("a.pdf", "h1"))
    [failed] = run(source, store, settings(), FakeStages(github=UNAVAILABLE).stages())

    assert failed.failed_step == "github"
    assert store.get_evaluation(EvaluationKey("h1", "role-v1", "ollama:a")) is None

    retry = FakeStages(github=NOT_FOUND)  # a nonexistent profile is judged without one
    [result] = run(source, store, settings(), retry.stages())
    assert result.ok
    assert retry.calls == Counter(enrich=1, evaluate=1)  # extraction was kept


def test_failed_extraction_is_not_stored(store):
    source = FakeSource(ref("a.pdf", "h1"))
    [failed] = run(source, store, settings(), FakeStages(extract_ok=False).stages())

    assert failed.failed_step == "extract"
    assert store.get_extracted("h1") is None


def test_ranking_covers_reused_evaluations_and_lists_failures_last(store):
    source = FakeSource(ref("low.pdf", "h1"), ref("high.pdf", "h2"))
    run(
        source,
        store,
        settings(),
        FakeStages(scores={"low.pdf": 1, "high.pdf": 9}).stages(),
    )
    source = FakeSource(*source.refs, ref("broken.pdf", "h3"))
    results = run(source, store, settings(), FakeStages(extract_ok=False).stages())

    rows = ranking_rows(results, ROLE, judge="ollama:a")

    assert [r["file_name"] for r in rows] == ["high.pdf", "low.pdf", "broken.pdf"]
    assert [r["rank"] for r in rows] == [1, 2, ""]
    assert rows[-1]["failed_step"] == "extract"
