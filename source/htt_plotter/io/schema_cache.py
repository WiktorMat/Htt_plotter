from __future__ import annotations

import json
import os
from pathlib import Path


def schema_cache_path(schema_cache_dir: Path, sample: str) -> Path:
    safe = "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in sample)
    return schema_cache_dir / f"{safe}.json"


def file_sig(p: Path) -> dict:
    st = os.stat(p)
    return {"path": str(p), "mtime_ns": st.st_mtime_ns, "size": st.st_size}


def try_load_cached_schema(*, schema_cache_dir: Path, sample: str, files: list[Path]) -> list[str] | None:
    try:
        schema_cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    cache_path = schema_cache_path(schema_cache_dir, sample)
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    cached_sig = payload.get("sig")
    cached_cols = payload.get("columns")
    cached_nfiles = payload.get("nfiles")
    if not isinstance(cached_sig, dict) or not isinstance(cached_cols, list) or not isinstance(cached_nfiles, int):
        return None

    try:
        sig = file_sig(files[0])
    except Exception:
        return None

    if cached_nfiles != len(files):
        return None
    if cached_sig != sig:
        return None
    if not all(isinstance(c, str) for c in cached_cols):
        return None

    return cached_cols


def try_store_cached_schema(*, schema_cache_dir: Path, sample: str, files: list[Path], columns: list[str]) -> None:
    try:
        schema_cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "sig": file_sig(files[0]),
            "nfiles": len(files),
            "columns": list(columns),
        }
        schema_cache_path(schema_cache_dir, sample).write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        return
