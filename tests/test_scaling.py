from __future__ import annotations

import queue
import time

from whisper_batch_subtitles.scaling import DynamicWorkerPool, decide_scale_action

STOP = object()


def make_worker(work_queue: "queue.Queue[object]"):
    def worker() -> None:
        while True:
            item = work_queue.get()
            if item is STOP:
                return

    return worker


def wait_until(condition_fn, *, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition_fn():
            return True
        time.sleep(interval)
    return condition_fn()


# --- decide_scale_action: pure decision logic ---


def test_grows_when_backlog_high_and_downstream_not_bottleneck():
    action = decide_scale_action(queue_depth=10, downstream_depth=0, live_workers=2, min_workers=1, max_workers=8)
    assert action == "grow"


def test_holds_when_backlog_high_but_downstream_is_the_bottleneck():
    # No point adding extraction workers if transcription can't keep up anyway.
    action = decide_scale_action(queue_depth=10, downstream_depth=5, live_workers=2, min_workers=1, max_workers=8)
    assert action == "hold"


def test_holds_at_max_workers_even_with_backlog():
    action = decide_scale_action(queue_depth=10, downstream_depth=0, live_workers=8, min_workers=1, max_workers=8)
    assert action == "hold"


def test_shrinks_when_queue_is_empty_and_above_min():
    action = decide_scale_action(queue_depth=0, downstream_depth=0, live_workers=3, min_workers=1, max_workers=8)
    assert action == "shrink"


def test_holds_at_min_workers_even_with_empty_queue():
    action = decide_scale_action(queue_depth=0, downstream_depth=0, live_workers=1, min_workers=1, max_workers=8)
    assert action == "hold"


def test_grows_toward_min_if_somehow_below_it():
    action = decide_scale_action(queue_depth=0, downstream_depth=0, live_workers=0, min_workers=2, max_workers=8)
    assert action == "grow"


def test_holds_with_moderate_backlog():
    action = decide_scale_action(queue_depth=3, downstream_depth=0, live_workers=2, min_workers=1, max_workers=8)
    assert action == "hold"


# --- DynamicWorkerPool: thread lifecycle mechanics ---


def test_start_initial_spawns_min_workers():
    work_queue: "queue.Queue[object]" = queue.Queue()
    pool = DynamicWorkerPool(target=make_worker(work_queue), name_prefix="w", min_workers=2, max_workers=5)
    pool.start_initial(2)
    assert wait_until(lambda: pool.live_count() == 2)
    pool.join_all(work_queue, STOP)


def test_start_initial_respects_min_even_if_count_is_lower():
    work_queue: "queue.Queue[object]" = queue.Queue()
    pool = DynamicWorkerPool(target=make_worker(work_queue), name_prefix="w", min_workers=3, max_workers=5)
    pool.start_initial(1)
    assert wait_until(lambda: pool.live_count() == 3)
    pool.join_all(work_queue, STOP)


def test_grow_adds_a_worker_up_to_max_then_refuses():
    work_queue: "queue.Queue[object]" = queue.Queue()
    pool = DynamicWorkerPool(target=make_worker(work_queue), name_prefix="w", min_workers=1, max_workers=2)
    pool.start_initial(1)
    assert wait_until(lambda: pool.live_count() == 1)

    assert pool.grow() is True
    assert wait_until(lambda: pool.live_count() == 2)

    assert pool.grow() is False
    assert pool.live_count() == 2
    pool.join_all(work_queue, STOP)


def test_single_stop_sentinel_retires_exactly_one_worker():
    work_queue: "queue.Queue[object]" = queue.Queue()
    pool = DynamicWorkerPool(target=make_worker(work_queue), name_prefix="w", min_workers=1, max_workers=3)
    pool.start_initial(3)
    assert wait_until(lambda: pool.live_count() == 3)

    work_queue.put(STOP)
    assert wait_until(lambda: pool.live_count() == 2)

    pool.join_all(work_queue, STOP)
    assert pool.live_count() == 0


def test_join_all_stops_and_joins_every_live_worker_including_grown_ones():
    work_queue: "queue.Queue[object]" = queue.Queue()
    pool = DynamicWorkerPool(target=make_worker(work_queue), name_prefix="w", min_workers=2, max_workers=6)
    pool.start_initial(2)
    assert wait_until(lambda: pool.live_count() == 2)
    pool.grow()
    pool.grow()
    assert wait_until(lambda: pool.live_count() == 4)

    pool.join_all(work_queue, STOP)
    assert pool.live_count() == 0
    assert work_queue.empty()  # exactly 4 sentinels were consumed, none left over


def test_join_all_after_some_workers_already_retired_naturally():
    # Simulates dynamic shrink happening concurrently with final shutdown accounting:
    # push a couple of "natural" retirements first, then join_all must only account
    # for whoever is actually still alive, not the original spawn count.
    work_queue: "queue.Queue[object]" = queue.Queue()
    pool = DynamicWorkerPool(target=make_worker(work_queue), name_prefix="w", min_workers=1, max_workers=5)
    pool.start_initial(4)
    assert wait_until(lambda: pool.live_count() == 4)

    work_queue.put(STOP)
    work_queue.put(STOP)
    assert wait_until(lambda: pool.live_count() == 2)

    pool.join_all(work_queue, STOP)
    assert pool.live_count() == 0
