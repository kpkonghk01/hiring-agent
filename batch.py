"""Score every Resume in a Resume Source against a Role, reusing earlier work.

Extracted Resumes and Evaluations live in a SQLite store, so re-running only does
new work, and every run ends with a complete Ranking (terminal table + CSV, optionally
a Google Sheet). See docs/custom-usage.md. Upstream's single-PDF score.py is untouched.
"""

import argparse
import contextlib
import io
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

from google.auth.exceptions import GoogleAuthError
from googleapiclient.errors import HttpError

import config
from custom import google_auth
from custom.evaluation import evaluate
from custom.extraction import extract_resume
from custom.github_enrichment import enrich
from custom.hashing import role_hash
from custom.llm import LLMChoice, build_client, resolve_choice
from custom.ocr import MarkerOcr
from custom.pipeline import ResumeResult, RunSettings, Stages, run
from custom.ranking import format_table, overall_score, ranking_rows, write_csv
from custom.sources import GoogleDriveResumeSource, LocalResumeSource, parse_batch_date
from custom.store import Store
from models import build_evaluation_model
from roles import Role, list_available_roles, load_role

REPO_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_STORE_PATH = "data/hiring_agent.sqlite"
DEFAULT_CLIENT_SECRET = "credentials/client_secret.json"
DEFAULT_TOKEN = "credentials/token.json"
# Failures of the environment rather than of one Resume: shown as a message, not a traceback.
RUNTIME_ERRORS = (HttpError, GoogleAuthError, OSError, sqlite3.Error)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score Resumes against a Role, skipping work already in the store."
    )
    parser.add_argument(
        "path", nargs="?", help="--source local: a resume PDF or a folder of PDFs"
    )
    parser.add_argument(
        "--role", required=True, help=f"Available: {', '.join(list_available_roles())}"
    )
    parser.add_argument("--source", choices=("local", "gdrive"), required=True)
    parser.add_argument(
        "--since",
        type=_batch_date,
        metavar="YYYYMMDD",
        help="first Batch (gdrive, inclusive)",
    )
    parser.add_argument(
        "--to",
        type=_batch_date,
        metavar="YYYYMMDD",
        help="last Batch (gdrive, inclusive)",
    )
    parser.add_argument("--provider", help="judging provider (default: LLM_PROVIDER)")
    parser.add_argument("--model", help="judging model (default: DEFAULT_MODEL)")
    parser.add_argument(
        "--re-extract",
        action="store_true",
        help="re-extract; drops those Resumes' Evaluations",
    )
    parser.add_argument(
        "--re-evaluate",
        action="store_true",
        help="re-judge even if an Evaluation exists",
    )
    parser.add_argument(
        "--limit", type=_positive_int, metavar="N", help="process at most N Resumes"
    )
    parser.add_argument(
        "--upload-sheet",
        action="store_true",
        help="also upload the Ranking as a Google Sheet",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="show upstream logs and full reports"
    )
    args = parser.parse_args(argv)

    if args.source == "local" and not args.path:
        parser.error("--source local needs a PDF or folder path")
    if args.source == "gdrive" and args.path:
        parser.error("a path is only used with --source local")
    if args.source == "local" and (args.since or args.to):
        parser.error("--since/--to only apply to --source gdrive")
    if args.since and args.to and args.since > args.to:
        parser.error("--since must not be after --to")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    local_path = Path(args.path).resolve() if args.path else None
    os.chdir(REPO_ROOT)  # upstream reads prompts/ and writes cache/ relative to the CWD
    _configure_output(args.verbose)
    try:
        role = load_role(args.role)
        judge = resolve_choice(args.model or config.DEFAULT_MODEL, args.provider)
        extractor = resolve_choice(
            os.getenv("EXTRACTION_MODEL") or config.DEFAULT_MODEL,
            os.getenv("EXTRACTION_PROVIDER"),
        )
        if args.source == "gdrive":
            _gdrive_folder_id()  # fail on missing config before opening a browser for OAuth
        drive = (
            _drive_service() if args.source == "gdrive" or args.upload_sheet else None
        )
        source = _source(args, local_path, drive)
    except (ValueError, *RUNTIME_ERRORS) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2

    _warn_without_github_token()
    print(f"Role {role.name} · judge {judge.label} · extraction {extractor.label}")
    settings = RunSettings(
        role=role,
        role_hash=role_hash(role.name),
        judge=judge.label,
        extracted_by=extractor.label,
        re_extract=args.re_extract,
        re_evaluate=args.re_evaluate,
    )
    try:
        with Store(os.getenv("STORE_PATH", DEFAULT_STORE_PATH)) as store:
            results = run(
                source,
                store,
                settings,
                _stages(judge, extractor, args.verbose),
                limit=args.limit,
                on_start=partial(_print_start, args.verbose),
                on_result=partial(_print_result, role, args.verbose),
            )
    except RUNTIME_ERRORS as err:
        print(f"Error while processing Resumes: {err}", file=sys.stderr)
        return 2
    if not results:
        print("No Resumes found.")
        return 0
    reported = _report(results, role, judge, args, drive, local_path)
    return 0 if reported and all(r.ok for r in results) else 1


def _report(results, role: Role, judge: LLMChoice, args, drive, local_path) -> bool:
    """Print and save the Ranking; False if the optional Sheet upload failed."""
    rows = ranking_rows(results, role, judge.label)
    print()
    print(format_table(rows, role))
    csv_path = REPORTS_DIR / f"{_report_name(args, role, judge, local_path)}.csv"
    write_csv(rows, csv_path)
    print(f"\nRanking written to {csv_path.relative_to(REPO_ROOT)}")
    if not args.upload_sheet:
        return True
    title = f"{csv_path.stem} ({datetime.now():%Y-%m-%d %H:%M})"
    try:
        link = google_auth.upload_csv_as_sheet(drive, csv_path, title)
    except RUNTIME_ERRORS as err:
        print(
            f"Error: could not upload the Ranking as a Google Sheet: {err}",
            file=sys.stderr,
        )
        return False
    print(f"Ranking uploaded as a Google Sheet: {link}")
    return True


def _stages(judge: LLMChoice, extractor: LLMChoice, verbose: bool) -> Stages:
    judge_client = build_client(judge)
    extractor_client = build_client(extractor)
    ocr = MarkerOcr()
    quiet = (lambda fn: fn) if verbose else _silenced
    return Stages(
        extract=quiet(
            lambda data, name: extract_resume(
                ocr, data, name, extractor, extractor_client
            )
        ),
        enrich=quiet(lambda resume, role: enrich(resume, role, judge, judge_client)),
        evaluate=quiet(
            lambda resume, github, role: evaluate(
                resume, github, role, judge, judge_client
            )
        ),
    )


def _silenced(fn):
    """Hide upstream's print() chatter so each Resume gets one progress line."""

    def wrapper(*args):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args)

    return wrapper


def _source(args, local_path: Optional[Path], drive):
    if args.source == "local":
        if local_path is None or not local_path.exists():
            raise FileNotFoundError(f"'{local_path}' does not exist")
        return LocalResumeSource(local_path)
    return GoogleDriveResumeSource(
        drive, _gdrive_folder_id(), since=args.since, to=args.to
    )


def _gdrive_folder_id() -> str:
    folder_id = os.getenv("GDRIVE_FOLDER_ID")
    if not folder_id:
        raise ValueError("GDRIVE_FOLDER_ID is not set (see docs/custom-usage.md)")
    return folder_id


def _drive_service():
    creds = google_auth.load_credentials(
        Path(os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", DEFAULT_CLIENT_SECRET)),
        Path(os.getenv("GOOGLE_OAUTH_TOKEN", DEFAULT_TOKEN)),
    )
    return google_auth.build_drive_service(creds)


def _configure_output(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not verbose:
        logging.getLogger("evaluator").setLevel(
            logging.CRITICAL
        )  # logs every raw response
        os.environ.setdefault("TQDM_DISABLE", "1")  # Marker's progress bars


def _warn_without_github_token() -> None:
    if not os.getenv("GITHUB_TOKEN"):
        print(
            "⚠️  GITHUB_TOKEN is not set: GitHub allows only 60 requests/hour, which a batch "
            "exhausts quickly (see docs/custom-usage.md).",
            file=sys.stderr,
        )


def _print_start(verbose: bool, index: int, total: int, refs) -> None:
    ref = refs[0]
    label = f"[{index}/{total}] {ref.batch + ' ' if ref.batch else ''}{ref.name}"
    print(label, end="\n" if verbose else " …", flush=True)


def _print_result(
    role: Role, verbose: bool, index: int, total: int, result: ResumeResult
) -> None:
    if result.resume is None or result.evaluation is None:
        print(f" ✗ {result.failed_step}: {result.failure}")
        return
    extracted = "cached" if result.extraction_reused else "new"
    evaluated = "cached" if result.evaluation_reused else "new"
    score = overall_score(result.evaluation, role)
    print(f" extract:{extracted} evaluate:{evaluated} → {score}")
    if verbose:
        name = (
            result.resume.basics.name if result.resume.basics else result.refs[0].name
        )
        _print_full_report(result.evaluation, role, name)


def _print_full_report(evaluation_data: dict, role: Role, candidate_name: str) -> None:
    # Imported late: score.py configures logging and the environment at import time.
    from score import print_evaluation_results

    evaluation = build_evaluation_model(role).model_validate(evaluation_data)
    print_evaluation_results(evaluation, role, candidate_name)


def _report_name(args, role: Role, judge: LLMChoice, local_path: Path) -> str:
    if args.source == "gdrive":
        scope = f"{args.since or 'first'}-{args.to or 'last'}"
    else:
        scope = local_path.stem
    parts = [role.name, judge.label, scope]
    if args.limit:
        parts.append(
            f"limit{args.limit}"
        )  # a partial Ranking must not replace the complete one
    return "__".join(_slug(part) for part in parts)


def _slug(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text).strip("_")


def _batch_date(value: str) -> str:
    try:
        return parse_batch_date(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err))


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


if __name__ == "__main__":
    sys.exit(main())
