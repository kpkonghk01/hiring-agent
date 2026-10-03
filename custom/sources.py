"""Resume Sources — where Resumes are listed and read from (CONTEXT.md)."""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

from custom.hashing import resume_hash

logger = logging.getLogger(__name__)

# A Batch folder is named by the date its Resumes were received.
BATCH_NAME = re.compile(r"^\d{8}$")

_FOLDER_MIME = "application/vnd.google-apps.folder"
_PDF_MIME = "application/pdf"
# Drive ids are URL-safe; checking also keeps the id safe to embed in a search query.
_DRIVE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_DOWNLOAD_ATTEMPTS = 3
# Network stalls (timeouts, resets, broken TLS) are OSErrors worth retrying on a fresh
# request; googleapiclient's HttpError is a real API answer and is not retried.
_TRANSIENT_ERRORS = (OSError,)


@dataclass(frozen=True)
class ResumeRef:
    """One file in a Resume Source. Several refs may share a ``resume_hash``."""

    source_id: str  # Drive file id, or local path
    name: str
    resume_hash: str
    batch: Optional[str] = None  # YYYYMMDD; None for a local source


class ResumeSource(Protocol):
    def list(self) -> Tuple[ResumeRef, ...]: ...

    def read_bytes(self, ref: ResumeRef) -> bytes: ...


def is_batch_date(value: str) -> bool:
    """True for a real calendar date written ``YYYYMMDD`` (rejects e.g. 20261340)."""
    if not BATCH_NAME.match(value):
        return False
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return False
    return True


def parse_batch_date(value: str) -> str:
    """Validate a ``YYYYMMDD`` Batch date given on the command line."""
    if not is_batch_date(value):
        raise ValueError(f"'{value}' is not a valid YYYYMMDD date")
    return value


class LocalResumeSource:
    """A single PDF, or the PDFs directly inside a folder (not recursive)."""

    def __init__(self, path: Path):
        self._path = path

    def list(self) -> Tuple[ResumeRef, ...]:
        if self._path.is_file():
            paths = [self._path]
        elif self._path.is_dir():
            paths = sorted(
                p
                for p in self._path.iterdir()
                if p.is_file() and p.suffix.lower() == ".pdf"
            )
        else:
            raise FileNotFoundError(f"Resume path '{self._path}' does not exist")
        return tuple(
            ResumeRef(
                source_id=str(p), name=p.name, resume_hash=resume_hash(p.read_bytes())
            )
            for p in paths
        )

    def read_bytes(self, ref: ResumeRef) -> bytes:
        return Path(ref.source_id).read_bytes()


class GoogleDriveResumeSource:
    """Batches are date-named subfolders (``YYYYMMDD``) of one parent folder.

    Only Batches within the inclusive ``since``..``to`` range are listed; other
    subfolders and PDFs placed directly in the parent are ignored. Hashes come from
    Drive's ``sha256Checksum``, so already-seen Resumes need no download. The Drive
    ``service`` is injected; see ``custom.google_auth`` for the real one.
    """

    def __init__(
        self,
        service,
        folder_id: str,
        since: Optional[str] = None,
        to: Optional[str] = None,
    ):
        if not _DRIVE_ID.match(folder_id):
            raise ValueError(
                f"'{folder_id}' does not look like a Google Drive folder id"
            )
        self._service = service
        self._folder_id = folder_id
        self._since = since
        self._to = to

    def list(self) -> Tuple[ResumeRef, ...]:
        batches = [
            folder
            for folder in self._children(self._folder_id, _FOLDER_MIME)
            if self._selects(folder["name"])
        ]
        refs = []
        for batch in sorted(batches, key=lambda f: f["name"]):
            # Drive's native order is unstable; sort so --limit picks the same Resumes each run.
            for pdf in sorted(
                self._children(batch["id"], _PDF_MIME), key=lambda f: f["name"]
            ):
                resume_hash = self._hash_of(pdf, batch["name"])
                if resume_hash:
                    refs.append(
                        ResumeRef(
                            source_id=pdf["id"],
                            name=pdf["name"],
                            resume_hash=resume_hash,
                            batch=batch["name"],
                        )
                    )
        return tuple(refs)

    def read_bytes(self, ref: ResumeRef) -> bytes:
        return self._download(ref.source_id)

    def _selects(self, folder_name: str) -> bool:
        if not is_batch_date(folder_name):
            logger.warning(
                "Ignoring Drive folder '%s': not a YYYYMMDD Batch", folder_name
            )
            return False
        if self._since and folder_name < self._since:
            return False
        if self._to and folder_name > self._to:
            return False
        return True

    def _hash_of(self, pdf: dict, batch: str) -> Optional[str]:
        """Drive's checksum, else hash a download. A file that cannot be hashed is
        skipped (and logged) rather than aborting the whole listing."""
        checksum = pdf.get("sha256Checksum")
        if checksum:
            return checksum.lower()
        try:
            return resume_hash(self._download(pdf["id"]))
        except Exception as err:
            logger.error(
                "Skipping %s/%s: could not download it to hash (%s)",
                batch,
                pdf["name"],
                err,
            )
            return None

    def _download(self, file_id: str) -> bytes:
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            request = self._service.files().get_media(
                fileId=file_id, supportsAllDrives=True
            )
            try:
                return request.execute()
            except _TRANSIENT_ERRORS as err:
                if attempt == _DOWNLOAD_ATTEMPTS:
                    raise
                logger.warning(
                    "Drive download of %s failed (%s); retrying", file_id, err
                )
        raise AssertionError("unreachable")

    def _children(self, parent_id: str, mime_type: str) -> List[dict]:
        query = f"'{parent_id}' in parents and mimeType='{mime_type}' and trashed=false"
        files = []
        page_token = None
        while True:
            response = (
                self._service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name, sha256Checksum)",
                    pageSize=1000,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                return files
