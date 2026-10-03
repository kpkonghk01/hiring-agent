"""Content identities used as store keys (docs/adr/0002)."""

import hashlib
from pathlib import Path

from roles import ROLES_DIR

# The files that make up a Role; editing any of them makes a new role hash.
ROLE_FILES = ("role.json", "criteria.jinja", "system_message.jinja")


def resume_hash(data: bytes) -> str:
    """SHA-256 hex of a Resume's raw bytes — the same value Drive reports as sha256Checksum."""
    return hashlib.sha256(data).hexdigest()


def role_hash(role_name: str, roles_dir: Path = ROLES_DIR) -> str:
    """SHA-256 hex over the exact content of a Role's files."""
    digest = hashlib.sha256()
    for filename in ROLE_FILES:
        content = (roles_dir / role_name / filename).read_bytes()
        digest.update(filename.encode("utf-8") + b"\0")
        digest.update(len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()
