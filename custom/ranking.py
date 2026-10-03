"""Ranking — a run's Evaluations ordered by overall Score (CONTEXT.md)."""

import csv
import unicodedata
from pathlib import Path
from typing import Iterable, List

from custom.pipeline import ResumeResult
from models import build_evaluation_model
from roles import Role
from transform import transform_evaluation_response

# Terminal table column widths (display columns; CJK characters count as two).
_FILE_WIDTH = 32
_NAME_WIDTH = 16


def overall_score(evaluation: dict, role: Role) -> float:
    """Categories (capped at the Role's maxima) + bonus - deductions, capped like the
    OVERALL SCORE that score.py prints."""
    caps = {c.key: c.max for c in role.categories}
    scores = evaluation["scores"]
    categories = sum(
        min(scores[key]["score"], cap) for key, cap in caps.items() if key in scores
    )
    total = (
        categories
        + evaluation["bonus_points"]["total"]
        - evaluation["deductions"]["total"]
    )
    return round(min(total, sum(caps.values()) + role.bonus_max), 1)


def ranking_rows(results: Iterable[ResumeResult], role: Role, judge: str) -> List[dict]:
    """Evaluated Resumes by overall Score (highest first), then failures."""
    results = list(results)
    evaluated = sorted(
        ((r, r.evaluation) for r in results if r.evaluation is not None),
        key=lambda pair: overall_score(pair[1], role),
        reverse=True,
    )
    failed = [r for r in results if r.evaluation is None]
    return [
        _evaluated_row(rank, result, evaluation, role, judge)
        for rank, (result, evaluation) in enumerate(evaluated, start=1)
    ] + [_failed_row(r, judge) for r in failed]


def _evaluated_row(
    rank: int, result: ResumeResult, evaluation_data: dict, role: Role, judge: str
) -> dict:
    evaluation = build_evaluation_model(role).model_validate(evaluation_data)
    upstream_row = transform_evaluation_response(
        file_name=_file_names(result),
        resume_data=result.resume,
        github_data=result.github,
        evaluation=evaluation,
        role=role,
    )
    return {
        "rank": rank,
        "overall_score": overall_score(evaluation_data, role),
        "batch": _batches(result),
        **upstream_row,
        "judge": judge,
        "resume_hash": result.resume_hash,
        "failed_step": "",
        "failure": "",
    }


def _failed_row(result: ResumeResult, judge: str) -> dict:
    return {
        "rank": "",
        "overall_score": "",
        "batch": _batches(result),
        "file_name": _file_names(result),
        "judge": judge,
        "resume_hash": result.resume_hash,
        "failed_step": result.failed_step,
        "failure": result.failure,
    }


def _file_names(result: ResumeResult) -> str:
    return " | ".join(dict.fromkeys(r.name for r in result.refs))


def _batches(result: ResumeResult) -> str:
    return ",".join(sorted({r.batch for r in result.refs if r.batch}))


def write_csv(rows: List[dict], path: Path) -> None:
    """UTF-8 with BOM so Excel shows Chinese correctly."""
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def format_table(rows: List[dict], role: Role) -> str:
    """A terminal table: rank, overall Score, Batch, file, name, category scores."""
    headers = ["#", "Score", "Batch", "File", "Name"] + [c.key for c in role.categories]
    lines = [
        [
            str(row["rank"]),
            str(row["overall_score"]),
            row["batch"],
            _clip(row["file_name"], _FILE_WIDTH),
            _clip(row.get("name", ""), _NAME_WIDTH),
        ]
        + [_cell(row.get(f"{c.key}_score", "")) for c in role.categories]
        for row in rows
        if row["rank"] != ""
    ]
    widths = [max(_width(cell) for cell in column) for column in zip(headers, *lines)]
    rendered = [_render(headers, widths)] + [_render(line, widths) for line in lines]
    failures = [
        f"  ✗ {row['file_name']} [{row['batch']}] {row['failed_step']}: {row['failure']}"
        for row in rows
        if row["rank"] == ""
    ]
    if failures:
        rendered += ["", f"Failed ({len(failures)}):"] + failures
    return "\n".join(rendered)


def _cell(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _clip(text: str, limit: int) -> str:
    text = text or ""
    if _width(text) <= limit:
        return text
    end = len(text)
    while _width(text[:end]) > limit - 1:
        end -= 1
    return text[:end] + "…"


def _width(text: str) -> int:
    """Display width, counting CJK characters as two columns."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _render(cells: List[str], widths: List[int]) -> str:
    return "  ".join(
        cell + " " * (width - _width(cell)) for cell, width in zip(cells, widths)
    )
