from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_histograms_parquet(
    *,
    histograms: dict[str, np.ndarray],
    edges: np.ndarray,
    out_path: str | Path,
    plot_type: str,
    variable: str,
    compression: str = "zstd",
) -> Path:
    """Write a {sample -> counts} histogram dict to a sidecar parquet.

    Schema is compatible with the previous Plotter._save_histograms_parquet output.
    Returns the written parquet path.
    """

    if not histograms:
        raise ValueError("histograms is empty")

    out_path = Path(out_path)
    parquet_path = out_path.with_suffix(".parquet")

    edges_list = np.asarray(edges, dtype=float).tolist()
    
    if edges is None or len(edges) == 0:
        raise ValueError("edges is empty")

    samples: list[str] = []
    plot_types: list[str] = []
    variables: list[str] = []
    counts_data: list[list[float]] = []
    edges_data: list[list[float]] = []

    for sample, counts in histograms.items():
        samples.append(sample)
        plot_types.append(plot_type)
        variables.append(variable)
        arr = np.asarray(counts, dtype=float)

        if arr.size == 0:
            arr = np.zeros(len(edges) - 1, dtype=float)

        if arr.ndim == 0:
            arr = np.array([], dtype=float)

        arr = np.nan_to_num(arr, nan=0.0)

        counts_data.append(arr.tolist())
        edges_data.append(edges_list)

    table = pa.Table.from_arrays(
        [
            pa.array(plot_types, type=pa.string()),
            pa.array(variables, type=pa.string()),
            pa.array(samples, type=pa.string()),
            pa.array(counts_data, type=pa.list_(pa.float64())),
            pa.array(edges_data, type=pa.list_(pa.float64())),
        ],
        names=[
            "plot_type",
            "variable",
            "sample",
            "counts",
            "bin_edges",
        ],
    )

    pq.write_table(table, parquet_path, compression=compression)
    return parquet_path


def read_histograms_parquet(
    file_path: str | Path,
) -> tuple[str, str, np.ndarray, dict[str, np.ndarray]] | None:
    """Read histogram parquet written by write_histograms_parquet.

    Returns (plot_type, variable, edges, hist_dict) or None if file is empty.
    """

    try:
        table = pq.read_table(file_path)
    except Exception as e:
        print("[WARN] Failed to read parquet %s → %s", file_path, e)
        return None

    if table.num_rows == 0:
        print("[WARN] Empty histogram parquet: %s", file_path)
        return None

    plot_type = table.column("plot_type")[0].as_py()
    variable = table.column("variable")[0].as_py()

    edges_any: Any = table.column("bin_edges")[0].as_py()
    edges = np.asarray(edges_any, dtype=float)

    samples = table.column("sample").to_pylist()
    counts_list = table.column("counts").to_pylist()

    hist: dict[str, np.ndarray] = {}
    for sample, counts in zip(samples, counts_list, strict=False):
        hist[str(sample)] = np.asarray(counts, dtype=float)

    return str(plot_type), str(variable), edges, hist
