from __future__ import annotations

import logging
from pathlib import Path


def resolve_files(project_root: Path, files_cfg) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for f in files_cfg or []:
        p = Path(f)
        if not p.is_absolute():
            p = project_root / p
        p = p.absolute()
        key = p.as_posix()
        if key in seen:
            continue
        seen.add(key)
        files.append(p)
    return files


def scan_dirs(project_root: Path, dirs_cfg, *, logger: logging.Logger) -> list[Path]:
    files: list[Path] = []
    for d in dirs_cfg or []:
        path = (project_root / d).resolve()
        if not path.exists():
            logger.warning("Missing path: %s", path)
            continue

        for file in path.rglob("*"):
            if not file.is_file():
                continue
            if file.suffix.lower() not in {".parquet", ".csv"}:
                continue
            files.append(file.resolve())

    return files
