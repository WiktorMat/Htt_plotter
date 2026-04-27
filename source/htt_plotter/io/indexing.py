from __future__ import annotations

from pathlib import Path
import logging

from rich.console import Console
from rich.live import Live
from rich.table import Table


def build_index(
    *,
    project_root: Path,
    sample_config: dict,
    logger: logging.Logger,
    resolve_files,
    scan_dirs,
    infer_format,
    sample_schema,
) -> list[dict]:
    """Build a per-sample index with a Rich live table.

    This function intentionally mirrors the previous in-method implementation
    to avoid behavior changes (including some redundant work/logging).
    """

    index: list[dict] = []

    root_cfg = sample_config
    sample_config = root_cfg.get("samples", root_cfg)
    process_config = root_cfg.get("process", {})  # kept for backward compatibility
    _ = process_config

    for sample_name, cfg in sample_config.items():
        kind = cfg.get("kind", "mc")
        scale = cfg.get("scale", 1.0)
        color = cfg.get("color", None)

        files_cfg = cfg.get("files")
        dirs_cfg = cfg.get("dirs")

        if files_cfg is not None:
            source_kind = "files"
            files = resolve_files(files_cfg)
        else:
            source_kind = "dirs"
            if dirs_cfg:
                logger.warning(
                    "Sample '%s' uses 'dirs' scanning; generate explicit 'files' list for speed.",
                    sample_name,
                )
            files = scan_dirs(dirs_cfg)

        files = [p for p in files if p.exists()]

        merged = [p for p in files if p.name == "merged.parquet"]
        if merged and len(files) > len(merged):
            logger.warning(
                "Sample '%s' lists merged.parquet together with %d other files; using merged.parquet only to avoid double-counting.",
                sample_name,
                len(files) - len(merged),
            )
            files = merged

        if not files:
            logger.warning("No input files for sample: %s", sample_name)
            continue

        _ = infer_format(files)
        _ = kind, scale, color, source_kind, project_root

        console = Console()
        history = []

        table = Table(title="Indexing samples")
        table.add_column("Sample")
        table.add_column("Kind")
        table.add_column("Source")
        table.add_column("Format")
        table.add_column("Files")

        index = []

        sample_config = sample_config.get("samples", sample_config)

        with Live(table, console=console, refresh_per_second=4) as live:
            for sample_name, cfg in sample_config.items():
                kind = cfg.get("kind", "mc")
                scale = cfg.get("scale", 1.0)
                color = cfg.get("color", None)

                files_cfg = cfg.get("files")
                dirs_cfg = cfg.get("dirs")

                source_kind = "files" if files_cfg else "dirs"

                if files_cfg is not None:
                    files = resolve_files(files_cfg)
                else:
                    if dirs_cfg:
                        logger.warning(
                            "Sample '%s' uses 'dirs' scanning; generate explicit 'files' list for speed.",
                            sample_name,
                        )
                    files = scan_dirs(dirs_cfg)

                files = [p for p in files if p.exists()]

                merged = [p for p in files if p.name == "merged.parquet"]
                if merged and len(files) > len(merged):
                    logger.warning(
                        "Sample '%s' uses merged.parquet only to avoid double counting.",
                        sample_name,
                    )
                    files = merged

                if not files:
                    logger.warning("No input files for sample: %s", sample_name)
                    continue

                fmt = infer_format(files)

                history.append((sample_name, kind, source_kind, fmt, len(files)))

                table = Table(title="Indexing samples")
                table.add_column("Sample")
                table.add_column("Kind")
                table.add_column("Source")
                table.add_column("Format")
                table.add_column("Files")

                for row in history:
                    table.add_row(*map(str, row))

                live.update(table)

                schema = sample_schema(fmt, files)

                index.append(
                    {
                        "sample": sample_name,
                        "kind": kind,
                        "scale": scale,
                        "color": color,
                        "files": files,
                        "format": fmt,
                        "schema": set(schema),
                    }
                )

        return index

    return index
