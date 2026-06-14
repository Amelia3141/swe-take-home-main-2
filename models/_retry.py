import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar
from dataclasses import dataclass

logger = logging.getLogger(__name__)
T = TypeVar('T')


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True
    # Permissive by default — a correct backoff over any failure. Narrowing this
    # to provider-specific transient errors (so auth/4xx fail fast instead of
    # burning the full attempt budget) is a deliberate next step, not yet wired
    # into the adapters.
    exceptions: tuple[type[Exception], ...] = (Exception,)

async def async_retry(
    func: Callable[..., Awaitable[T]],
    *args,
    config: RetryConfig = RetryConfig(),
    **kwargs,
) -> T:
    """Call an awaitable-returning function, retrying with exponential backoff.
    Re-raises the final exception if all attempts fail.
    """
    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except config.exceptions as e:
            if attempt == config.max_attempts - 1:
                raise
            delay = min(config.base_delay * (2**attempt), config.max_delay)
            if config.jitter:
                # Randomise within [0.5x, 1x] so many clients retrying after the
                # same outage don't synchronise into a thundering herd.
                delay *= 0.5 + random.random() / 2
            logger.warning(
                "Call failed (attempt %d/%d), retrying in %.2fs: %r",
                attempt + 1,
                config.max_attempts,
                delay,
                e,
            )
            await asyncio.sleep(delay)
    raise AssertionError("async_retry exhausted its loop without returning")