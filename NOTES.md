# Implementation notes

Notes on the changes I made, for the follow-up discussion. Scope: the three
feature requests, the bugs I had to fix to ship them, and tests — then stop.

## Feature requests

### 1. Exponential backoff (`models/_retry.py`)
- Rewrote the retry helper as `async_retry` driven by a `RetryConfig`
  (`max_attempts`, `base_delay`, `max_delay`, `jitter`, retryable `exceptions`).
- Backoff is `min(base_delay * 2**n, max_delay)` with optional jitter to avoid
  synchronised retries.
- The old version silently returned `None` after exhausting retries, which
  surfaced later as a confusing `NoneType` error (e.g. `response.choices[0]`).
  The new one **re-raises** the last exception so failures are explicit; the
  runner then isolates that failure to the single row (see below).
- `exceptions` defaults to `(Exception,)`, which is a correct implementation of
  the literal ask. Narrowing it to provider-specific transient errors (so a
  401/400 fails fast instead of burning ~4 attempts and ~7s) is a **quality**
  improvement to the retry policy, not a correctness fix — so I left
  `RetryConfig.exceptions` as the configurable seam and deferred the
  per-provider sets to the discussion (see next steps).

### 2. Async / concurrent requests
- `BaseLM.generate` is now `async`. OpenAI and Anthropic use the async SDK
  clients (`AsyncOpenAI`, `AsyncAnthropic`). The Mock model is trivially async.
- HuggingFace inference is blocking and compute-bound, so it runs in a worker
  thread via `asyncio.to_thread` — keeping a uniform interface without stalling
  the event loop.
- The `Runner` processes each row end-to-end (answer → grade) concurrently,
  bounded by an `asyncio.Semaphore` (`--max_concurrency`, default 8).
  `main.py` drives it with `asyncio.run`.

### 3. Pause / resume (`evals/checkpoint.py`)
- Completed rows are flushed to a JSONL checkpoint (keyed by row index) as soon
  as they finish, using the `jsonlines` dependency.
- Ctrl+C cancels in-flight work; already-completed rows are durably on disk.
- Re-running with `--resume` loads the checkpoint, skips done rows, and only
  re-issues requests for the rest. The final CSV merges checkpointed + new rows
  in index order.
- A tqdm progress bar reflects this: `initial` is set to the number of
  checkpointed rows, so a resumed run shows true progress rather than 0%.

## Bugs fixed along the way
- `end_index=-1` default silently dropped the **last** dataset row → now `None`
  (evaluate to the end).
- `HuggingfaceLM.generate` returned a `list[str]` (violating `-> str`) and
  required a `model_args["stop"]` the runner never passed (`KeyError`) → returns
  a string, stop sequence is optional.
- `AnthropicLM` embedded the legacy `HUMAN_PROMPT`/`AI_PROMPT` text-completion
  markers in a Messages API call → now a proper `system` + user message.
- The rich results table crashed on non-string cells → explicit columns, values
  coerced to `str`.
- One failed row aborted the whole run → per-row error isolation (records the
  error, grade `eval_error`, continues).

## Tests (`tests/`, run with `uv run pytest`)
- Retry: success, retry-then-succeed, re-raise after exhaustion, exact backoff
  schedule, `max_delay` cap, non-retryable exceptions pass through.
- Runner: grade extraction, mock end-to-end, per-row error isolation,
  concurrency limit is respected.
- Checkpoint/resume: roundtrip, fresh-run truncation, resume skips completed
  rows without re-calling the model.
- Dataset: regression test for the dropped-last-row bug.

## Where I'd take it next (out of scope for the time-box)
- **Provider-specific retryable exception sets** behind `RetryConfig.exceptions`
  (retry `RateLimitError`/`APIConnectionError`/5xx; let auth and 4xx fail fast).
  The seam is already there; only the per-provider wiring is deferred.
- **Run metadata / provenance**: write redacted args, SHA-256 of the dataset and
  prompt templates, and SDK/Python versions per run — the README's "parameters,
  model versions, and data versions logged and easily referenced" consideration.
- Surface the retry knobs (`max_attempts`, `base_delay`) as CLI flags.
- Checkpoint config fingerprint: refuse `--resume` against a different
  dataset/params.
- Typed `GenerateConfig` replacing the `model_args` dict; the seam for
  function-calling / multimodal / structured-output capabilities.
- Provider registry with explicit IDs (`openai/gpt-4o`) instead of
  name-sniffing.
- Swap hand-rolled retry for `tenacity` behind the same `RetryConfig`;
  honour `Retry-After`; disable SDK-internal retries.
- A persistent checkpoint writer (single open handle) if throughput matters;
  current per-row open/close favours durability + simplicity.
