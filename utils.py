import time
from contextlib import contextmanager
import numpy as np
import pandas as pd


@contextmanager
def timer(name):
    start = time.time()
    yield
    seconds = time.time() - start
    minutes = seconds / 60
    if minutes >= 1:
        print("[%s] done in %.3f minutes" % (name, minutes))
    else:
        print("[%s] done in %.3f seconds" % (name, seconds))


def fill_timestamp(timestamps, arrays=None):
    timestamps = np.asarray(timestamps, np.int64)
    arrays = [np.asarray(array) for array in (arrays or ())]

    src_index = np.argsort(timestamps)
    timestamps_sorted = timestamps[src_index]
    intervals = np.unique(np.diff(timestamps_sorted))
    min_interval = np.min(intervals)
    if min_interval == 0:
        raise ValueError("Duplicated values in timestamps")
    for interval in intervals:
        if interval % min_interval != 0:
            raise ValueError("Not all intervals in timestamps are multiples"
                             "of the minimum interval")

    length = (timestamps_sorted[-1] - timestamps_sorted[0]) // min_interval + 1
    ret_timestamps = np.arange(timestamps_sorted[0],
                               timestamps_sorted[-1] + min_interval,
                               min_interval, dtype=np.int64)

    ret_missings = np.ones([length], dtype=np.bool)
    ret_arrays = [np.zeros([length], dtype=array.dtype) for array in arrays]
    dst_index = np.asarray((timestamps_sorted - timestamps_sorted[0]) // min_interval,
                           dtype=np.int)

    ret_missings[dst_index] = False
    for ret_array, array in zip(ret_arrays, arrays):
        ret_array[dst_index] = array[src_index]

    if len(arrays) == 0:
        return ret_timestamps, ret_missings
    else:
        return ret_timestamps, ret_missings, ret_arrays


def standardize_values(values, excludes = None):
    if excludes is None:
        excludes = np.zeros_like(values, dtype=np.bool)
    val = values[np.logical_not(excludes)]
    mean = val.mean()
    std = val.std()
    return (values - mean) / std, mean, std


def smooth_errors(y_true, y_pred, smoothing_window=90):
    e = np.abs(y_true-y_pred)
    e_s = pd.DataFrame(e).ewm(span=smoothing_window).mean().values.flatten()
    return e_s
