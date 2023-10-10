import warnings
import inspect

import numpy as np


def reverse_diff(orig_vals: np.array, diffed: np.array, periods: int) -> np.array:
    """
    Reverse the first order differencing that was applied to the given
    data.
    Args:
        orig_vals (np.array):
            The last 'd' values of the original series before differencing was
            applied, where 'd' is the number of differencing periods.
        diffed (np.array):
            A series to which first order differencing was applied.
        periods (np.array):
            The number of differencing periods
    Returns:
        res (np.array):
            The given data with no more differencing applied
    """
    length = len(diffed)
    res = np.zeros(length)
    for i in range(periods):
        indices = np.arange(i, length, periods, dtype=int)
        undiffed = orig_vals[i] + np.cumsum(diffed[i::periods])
        res[indices] = undiffed
    return res


def psf_warn(message: str):
    """
    Print a warning using a custom format, without affecting the format of
    warnings issued outside this module.
    """
    caller_frame = inspect.stack()[1][0]
    info = inspect.getframeinfo(caller_frame)
    orig_formatwarning = warnings.formatwarning
    warnings.formatwarning = (lambda message, category,
                                     filename, lineno, _: f"{filename}:{lineno}:{category.__name__}:{message}\n")
    warnings.warn_explicit(message, UserWarning, info.filename, info.lineno)
    warnings.formatwarning = orig_formatwarning