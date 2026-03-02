import numpy as np

def to_timedelta64(freq: str) -> np.timedelta64:
    """
    Convert a frequency string to a numpy timedelta64 object.
    The frequency string should be in the format of a number followed by a time unit,
    e.g. '1D', '2H', '3M', etc.
    The time unit can be one of the following:
    - 'Y' for years
    - 'M' for months
    - 'W' for weeks
    - 'D' for days
    - 'h' for hours
    - 'm' for minutes
    - 's' for seconds
    - 'ms' for milliseconds
    Parameters
    ----------
    freq : str
        The frequency string to convert.
    
    Returns
    -------
    np.timedelta64
        The converted numpy timedelta64 object.
    """
    value = freq[:-1]
    unit = freq[-1]
    return np.timedelta64(value,unit)


def datetime_batches(start, end, freq, batch_size=-1):
    """
    Yield (start_str, end_str) for each batch using inclusive endpoints.

    Each yielded batch represents a *closed* interval [batch_start, batch_end],
    meaning both start and end are timestamps that should be processed.
    This matches a loop of the form: `while date <= batch_end: ...`.

    Parameters
    ----------
    start : np.datetime64 or str
        Inclusive start timestamp.
    end : np.datetime64 or str
        Inclusive end timestamp.
    freq : np.timedelta64 or compatible
        Step between successive timestamps.
    batch_size : int
        Maximum number of timestamps per batch.
        If -1, yields exactly one batch (start, end).

    Yields
    ------
    (start_str, end_str) : tuple[str, str]
        Datetime strings readable by np.datetime64, defining a closed interval.
    """
    start = np.datetime64(start)
    end = np.datetime64(end)
    freq = to_timedelta64(freq)  # assumed existing in your code

    if start > end:
        return
    if batch_size == 0:
        raise ValueError("batch_size must be -1 or a positive integer")
    if batch_size == -1:
        yield (str(start), str(end))
        return
    if batch_size < -1:
        raise ValueError("batch_size must be -1 or a positive integer")
    if freq <= np.timedelta64(0, "ns"):
        raise ValueError("freq must be a positive timedelta64")

    current = start
    while current <= end:
        batch_start = current
        batch_end = current

        # extend batch_end up to (batch_size - 1) more timestamps, but never past `end`
        for _ in range(batch_size - 1):
            nxt = batch_end + freq
            if nxt > end:
                break
            batch_end = nxt

        yield (str(batch_start), str(batch_end))

        # next batch starts at the next timestamp after batch_end
        current = batch_end + freq