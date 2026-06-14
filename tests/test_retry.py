import asyncio

import pytest

from models._retry import RetryConfig, async_retry


def run(coro):
    return asyncio.run(coro)


def test_returns_on_success():
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        return "ok"

    assert run(async_retry(func, config=RetryConfig(jitter=False))) == "ok"
    assert calls == 1


def test_retries_then_succeeds():
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("transient")
        return "ok"

    cfg = RetryConfig(max_attempts=5, base_delay=0, jitter=False)
    assert run(async_retry(func, config=cfg)) == "ok"
    assert calls == 3


def test_reraises_after_exhaustion():
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        raise ValueError("always fails")

    cfg = RetryConfig(max_attempts=3, base_delay=0, jitter=False)
    with pytest.raises(ValueError, match="always fails"):
        run(async_retry(func, config=cfg))
    assert calls == 3


def test_non_retryable_passes_through_immediately():
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        raise KeyError("not retryable")

    # Only ValueError is retryable, so KeyError should propagate on first call.
    cfg = RetryConfig(max_attempts=5, base_delay=0, exceptions=(ValueError,))
    with pytest.raises(KeyError):
        run(async_retry(func, config=cfg))
    assert calls == 1


def test_backoff_schedule_is_exponential(monkeypatch):
    delays = []

    async def fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def func():
        raise ValueError("fail")

    cfg = RetryConfig(max_attempts=4, base_delay=1.0, max_delay=30.0, jitter=False)
    with pytest.raises(ValueError):
        run(async_retry(func, config=cfg))

    # 3 sleeps for 4 attempts: 1, 2, 4 (no sleep after the final failure).
    assert delays == [1.0, 2.0, 4.0]


def test_backoff_respects_max_delay(monkeypatch):
    delays = []

    async def fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def func():
        raise ValueError("fail")

    cfg = RetryConfig(max_attempts=6, base_delay=1.0, max_delay=3.0, jitter=False)
    with pytest.raises(ValueError):
        run(async_retry(func, config=cfg))

    # 1, 2, then clamped to max_delay=3 thereafter.
    assert delays == [1.0, 2.0, 3.0, 3.0, 3.0]
