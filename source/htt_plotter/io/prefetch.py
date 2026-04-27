from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Iterable, Iterator


Item = tuple[str, Any]


def iter_batches_from_items(
    items_iter: Iterable[Item],
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    prefetch_batches: int = 0,
) -> Iterator[Any]:
    """Consume a mixed stream of ("event"|"batch", payload).

    - For kind=="event", payload is passed to progress_callback (if provided)
      and not yielded.
    - For kind=="batch", payload is yielded.

    If prefetch_batches > 0, the producer runs in a background thread and items
    are moved through a bounded queue. This overlaps I/O with downstream work
    while keeping callbacks executed in the consumer thread.
    """

    def _iter_direct() -> Iterator[Any]:
        for kind, payload in items_iter:
            if kind == "event":
                if progress_callback is not None:
                    progress_callback(payload)
                continue
            if kind == "batch":
                yield payload
                continue
            raise RuntimeError(f"Unknown item kind: {kind}")

    prefetch_batches = int(prefetch_batches)
    if prefetch_batches <= 0:
        yield from _iter_direct()
        return

    max_queue = max(1, prefetch_batches)
    q: queue.Queue[Item] = queue.Queue(maxsize=max_queue)
    done_sentinel = object()
    err: list[BaseException] = []

    def _producer() -> None:
        try:
            for item in items_iter:
                q.put(item)
        except BaseException as e:
            err.append(e)
        finally:
            q.put(("_done", done_sentinel))

    t = threading.Thread(target=_producer, name="DataAccessPrefetch", daemon=True)
    t.start()

    while True:
        kind, payload = q.get()
        if kind == "_done" and payload is done_sentinel:
            break
        if kind == "event":
            if progress_callback is not None:
                progress_callback(payload)
            continue
        if kind == "batch":
            yield payload
            continue
        raise RuntimeError(f"Unknown item kind: {kind}")

    if err:
        raise err[0]
