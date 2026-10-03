"""Google OAuth and the Drive calls that need real credentials.

Real IO only — exercised by real runs, not unit tests. The client secret file is read
by google-auth at runtime, never by this code.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",  # list and download Resumes
    "https://www.googleapis.com/auth/drive.file",  # create Ranking Sheets in a folder this app owns
]
_HTTP_TIMEOUT_SECONDS = (
    60  # httplib2 has no default timeout; a stalled read would hang forever
)
_FOLDER_MIME = "application/vnd.google-apps.folder"
_SHEET_MIME = "application/vnd.google-apps.spreadsheet"

# drive.file may only write into folders this app created, so Ranking Sheets go into a
# folder the app creates once and finds again by this private app property.
REPORT_FOLDER_NAME = "hiring-agent rankings"
_REPORT_FOLDER_PROPERTY = ("hiringAgentFolder", "rankings")


def load_credentials(client_secret: Path, token_path: Path):
    """Return valid credentials, refreshing or re-running the browser consent as needed."""
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path))
        if not set(SCOPES) <= set(creds.scopes or []):
            logger.warning("Saved Google token lacks required scopes; re-authorising")
            creds = None
    if creds and creds.valid:
        return creds
    if creds and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds, token_path)
            return creds
        except RefreshError as err:
            logger.warning("Google token refresh failed (%s); re-authorising", err)

    if not client_secret.exists():
        raise FileNotFoundError(
            f"Google OAuth client secret not found at '{client_secret}'. "
            "Set GOOGLE_OAUTH_CLIENT_SECRET (see docs/custom-usage.md)."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
    creds = flow.run_local_server(port=0)
    _save_token(creds, token_path)
    return creds


def build_drive_service(creds):
    import httplib2
    from google_auth_httplib2 import AuthorizedHttp
    from googleapiclient.discovery import build

    http = AuthorizedHttp(creds, http=httplib2.Http(timeout=_HTTP_TIMEOUT_SECONDS))
    return build("drive", "v3", http=http, cache_discovery=False)


def upload_csv_as_sheet(service, csv_path: Path, title: str) -> str:
    """Upload a CSV, converted to a Google Sheet, into the app's report folder.

    Returns the Sheet's web link.
    """
    from googleapiclient.http import MediaFileUpload

    created = (
        service.files()
        .create(
            body={
                "name": title,
                "mimeType": _SHEET_MIME,
                "parents": [_report_folder_id(service)],
            },
            media_body=MediaFileUpload(str(csv_path), mimetype="text/csv"),
            fields="id, webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )
    return created["webViewLink"]


def _report_folder_id(service) -> str:
    key, value = _REPORT_FOLDER_PROPERTY
    query = (
        f"mimeType='{_FOLDER_MIME}' and trashed=false and "
        f"appProperties has {{ key='{key}' and value='{value}' }}"
    )
    found = (
        service.files()
        .list(
            q=query,
            fields="files(id)",
            pageSize=1,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
        .get("files", [])
    )
    if found:
        return found[0]["id"]
    created = (
        service.files()
        .create(
            body={
                "name": REPORT_FOLDER_NAME,
                "mimeType": _FOLDER_MIME,
                "appProperties": {key: value},
            },
            fields="id",
        )
        .execute()
    )
    logger.warning(
        "Created Google Drive folder '%s' for Ranking Sheets", REPORT_FOLDER_NAME
    )
    return created["id"]


def _save_token(creds, token_path: Path) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(creds.to_json())
