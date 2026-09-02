from __future__ import annotations

import queue
import threading
from typing import Any, Callable

_STOP = object()


def decide_scale_action(
    *,
    queue_depth: int,
    downstream_depth: int,
    live_workers: int,
    min_workers: int,
    max_workers: int,
) -> str:
    """Pure scaling decision for a bounded worker pool: 'grow', 'shrink', or 'hold'.

    Grows when this stage's own queue is backed up relative to its worker count *and*
    the next stage isn't itself the bottleneck (no point adding workers that would just
    pile more work in front of an already-saturated downstream queue). Shrinks only
    when the queue is fully drained, so an idle burst between files doesn't thrash the
    pool. Both bounded by [min_workers, max_workers].
    """
    if live_workers < min_workers:
        return "grow"
    if queue_depth > live_workers * 2 and downstream_depth < live_workers and live_workers < max_workers:
        return "grow"
    if queue_depth == 0 and live_workers > min_workers:
        return "shrink"
    return "hold"


class DynamicWorkerPool:
    """A bounded, auto-scaling pool of worker threads all reading from one queue.

    Growing spawns a new thread running `target`. Shrinking pushes exactly one stop
    sentinel into `stop_queue` -- whichever worker happens to dequeue it retires, so
    the "which thread exits" choice is left to the queue, not tracked explicitly. All
    worker-list bookkeeping is lock-protected so `join_all` can always account for
    exactly how many threads are actually still alive, even if some retired themselves
    concurrently with a caller checking `live_count()`.

    Deliberately only used for the extraction stage (CPU-bound ffmpeg subprocesses,
    cheap to start/stop) -- transcription workers stay statically sized because each
    one loads a real model onto a specific GPU device_index at startup, and thrashing
    that is a much bigger cost/risk than an extra ffmpeg thread.
    """

    def __init__(self, *, target: Callable[[], None], name_prefix: str, min_workers: int, max_workers: int) -> None:
        self._target = target
        self._name_prefix = name_prefix
        self.min_workers = max(min_workers, 1)
        self.max_workers = max(max_workers, self.min_workers)
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        self._next_index = 0

    def _spawn_locked(self) -> threading.Thread:
        index = self._next_index
        self._next_index += 1
        worker = threading.Thread(target=self._target, name=f"{self._name_prefix}-{index}", daemon=True)
        worker.start()
        self._workers.append(worker)
        return worker

    def start_initial(self, count: int) -> None:
        with self._lock:
            for _ in range(max(count, self.min_workers)):
                self._spawn_locked()

    def live_count(self) -> int:
        with self._lock:
            self._workers = [worker for worker in self._workers if worker.is_alive()]
            return len(self._workers)

    def grow(self) -> bool:
        with self._lock:
            self._workers = [worker for worker in self._workers if worker.is_alive()]
            if len(self._workers) >= self.max_workers:
                return False
            self._spawn_locked()
            return True

    def join_all(self, stop_queue: "queue.Queue[Any]", stop_sentinel: Any = _STOP) -> None:
        with self._lock:
            self._workers = [worker for worker in self._workers if worker.is_alive()]
            live_workers = list(self._workers)
        for _ in live_workers:
            stop_queue.put(stop_sentinel)
        for worker in live_workers:
            worker.join()
