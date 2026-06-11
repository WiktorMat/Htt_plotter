"""File signatures, cache keys, cache paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cache_root() -> Path:
    return repo_root() / ".cache" / "wham"


def file_signature(path: Path) -> dict[str, Any]:
    st = Path(path).stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def short_key(payload: Any) -> str:
    """Deterministic 12-hex key over any JSON-serializable payload."""
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canon.encode()).hexdigest()[:12]
