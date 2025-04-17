import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Union

from loguru import logger

cumulative_time = defaultdict(lambda: 0)
cumulative_count = defaultdict(lambda: 0)

_application_start_time = time.perf_counter_ns()


@contextmanager
def log_time(*args: Any, level: Union[int, str] = "INFO", cumulative_key: str = "", precision: int = 3) -> None:
    start = time.perf_counter_ns()
    yield
    elapsed_time = time.perf_counter_ns() - start

    if cumulative_key:
        cumulative_time[cumulative_key] += elapsed_time
        cumulative_count[cumulative_key] += 1
        extra_args = (
            f"cumulative elapsed: {cumulative_time[cumulative_key] / 1e6} ms elapsed: {elapsed_time / 1e6} ms avg elapsed: {cumulative_time[cumulative_key] / 1e6 / cumulative_count[cumulative_key]:.{precision}f} ms profile: {cumulative_time[cumulative_key] / (time.perf_counter_ns() - _application_start_time) * 100:.1f}%"
            ,)
    else:
        extra_args = (f"elapsed: {elapsed_time / 1e6} ms elapsed",)

    logger.log(level, ' '.join(args + extra_args))
