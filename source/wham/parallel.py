"""Process pool tuned for pyarrow on a shared machine.

Without per-worker thread caps every worker spawns a full-machine arrow
thread pool; with N workers that oversubscribes CPUs and the buffering
blows up memory (observed: silent OOM kill of an 8-worker run).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, TypeVar

T = TypeVar("T")

_PPID_POLL_SECONDS = 2.0


def _die_with_parent() -> None:
    """Self-terminate once orphaned, so a killed run cannot leak its pool.

    ProcessPoolExecutor only reaps workers via its __exit__/shutdown. When the
    parent is SIGKILLed — exactly what the cgroup OOM reaper does — that never
    runs, the workers reparent to init and sit on their pyarrow buffers for
    good. Observed on lxplus: three OOM-killed runs left ~21 orphans holding
    30.6 GB against a 34.3 GB per-user cap, so every subsequent run OOMed on
    arrival and leaked its own pool in turn. Polling getppid() (rather than
    PR_SET_PDEATHSIG, which also fires when the parent *thread* exits and would
    kill live workers) has no false-positive mode; a couple of seconds of
    latency is fine since the point is to stop accumulation across retries.
    """
    orig_ppid = os.getppid()

    def watch() -> None:
        while True:
            time.sleep(_PPID_POLL_SECONDS)
            if os.getppid() != orig_ppid:
                os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def _init_worker(arrow_threads: int) -> None:
    import pyarrow as pa

    pa.set_cpu_count(max(1, arrow_threads))
    pa.set_io_thread_count(max(2, arrow_threads * 2))
    _die_with_parent()


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
