"""Process pool tuned for pyarrow on a shared machine.

Without per-worker thread caps every worker spawns a full-machine arrow
thread pool; with N workers that oversubscribes CPUs and the buffering
blows up memory (observed: silent OOM kill of an 8-worker run).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, TypeVar

T = TypeVar("T")


def _init_worker(arrow_threads: int) -> None:
    import pyarrow as pa

    pa.set_cpu_count(max(1, arrow_threads))
    pa.set_io_thread_count(max(2, arrow_threads * 2))


def run_parallel(
    func: Callable[..., T],
    jobs: Iterable[tuple[Any, ...]],
    *,
    workers: int,
    size_of: Callable[[tuple[Any, ...]], int] | None = None,
) -> Iterator[T]:
    """Run func(*job) over jobs, yielding results as they complete.

    Jobs are dispatched largest-first when size_of is given, so the
    biggest inputs never end up as the lone straggler.
    """
    jobs = list(jobs)
    if size_of is not None:
        jobs.sort(key=size_of, reverse=True)

    workers = max(1, min(workers, len(jobs) or 1))
    if workers == 1:
        for job in jobs:
            yield func(*job)
        return

    arrow_threads = max(1, (os.cpu_count() or 8) // workers)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(arrow_threads,)
    ) as pool:
        futures = [pool.submit(func, *job) for job in jobs]
        for future in as_completed(futures):
            yield future.result()
