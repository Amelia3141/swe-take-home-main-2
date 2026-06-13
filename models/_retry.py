import time


def retry(func, *args, **kwargs):
    """Call a function, if it fails then retry after 1s. Retry it up to three
    times. This is not a very good retry implementation.

    Args:
        func: Function to call
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        If successful, return the result of the function call. If unsuccessful,
        after all retries, return None.
    """
    for attempt in range(1, 5):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(
                f"Failed to call function {attempt} time(s). Waiting 1s, exception: {e}"
            )
            time.sleep(1)
            continue

    return None
