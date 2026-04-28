from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def collect_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    return files


def merge_parquet(files: list[Path], output_file: Path, batch_size: int = 131072) -> None:
    logging.info("Merging %d parquet files → %s", len(files), output_file)

    dataset = ds.dataset([str(f) for f in files], format="parquet")

    scanner = dataset.scanner(
        use_threads=True,
        batch_size=batch_size
    )

    writer = None

    total_rows = 0

    for batch in scanner.to_batches():
        table = pa.Table.from_batches([batch])

        if writer is None:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(
                output_file,
                table.schema,
                compression="zstd"
            )

        writer.write_table(table)
        total_rows += batch.num_rows

    if writer:
        writer.close()

    logging.info("Done. Total rows written: %d", total_rows)


def main():
    parser = argparse.ArgumentParser(description="Fast Parquet merger")
    parser.add_argument("--input", required=True, help="Input directory with parquet files")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--batch-size", type=int, default=131072)

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_file = Path(args.output)

    files = collect_files(input_dir)

    merge_parquet(
        files=files,
        output_file=output_file,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()