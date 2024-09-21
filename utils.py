import itertools
import warnings

import pandas as pd
from scipy.interpolate import interp1d
import numpy as np


def replace_outliers_1d(arr):
    # Calculate the mean and standard deviation
    mean = np.mean(arr)
    std = np.std(arr)

    # Define the lower and upper bounds for outlier detection
    lower_bound = 0
    upper_bound = 100

    # Create a copy of the original array
    new_arr = arr.copy()

    # Iterate over the array and replace outliers
    for i in range(len(arr)):
        if arr[i] < lower_bound or arr[i] > upper_bound:
            if i == 0:
                new_arr[i] = arr[i + 1]  # Replace the first element with the next value
            elif i == len(arr) - 1:
                new_arr[i] = arr[i - 1]  # Replace the last element with the previous value
            else:
                new_arr[i] = (arr[i - 1] + arr[i + 1]) / 2  # Replace with the average of neighbors

    return new_arr


def replace_outliers_2d(data):
    arr = np.array(data)
    rows, cols = arr.shape

    lb, ub = 0, 1000

    # Loop through each column
    for col in range(cols):
        # Loop through each row in the column
        for row in range(rows):
            # Check if the current element is an outlier
            if arr[row, col] < lb or arr[row, col] > ub:
                # If the outlier is the first element in the column
                if row == 0:
                    arr[row, col] = arr[row + 1, col]
                # If the outlier is the last element in the column
                elif row == rows - 1:
                    arr[row, col] = arr[row - 1, col]
                # For all other cases, replace with the average of before and after values
                else:
                    arr[row, col] = (arr[row - 1, col] + arr[row + 1, col]) / 2

    return arr


def count_violations(arr, ub, lb):
    arr = np.array(arr)
    within_bounds = np.sum((arr >= lb) & (arr <= ub))
    count = len(arr) - within_bounds

    return count


def extend_array_to_n_values(arr, n):
    # Check if the input is a numpy array
    is_numpy = isinstance(arr, np.ndarray)

    # Convert to list if it's a numpy array
    if is_numpy:
        arr = arr.tolist()

    # Extend the list
    repeated = arr * (n // len(arr))
    remaining = arr[:n % len(arr)]
    extended = repeated + remaining

    # Convert back to numpy array if needed
    if is_numpy:
        extended = np.array(extended)

    return extended


def signals_generator(mu, sigma, freq, noise, n, times_values=None, m=1, signal_type="random"):
    if signal_type == "random":
        return np.hstack([np.random.normal(loc=mu[_], scale=sigma[_], size=(n, 1)) for _ in range(len(mu))])

    elif signal_type == "periodic":
        s = np.zeros((n, len(mu)))
        for _ in range(len(mu)):
            phase = 2 * np.pi * np.random.rand()
            time = np.linspace(0, 4 * np.pi, n)
            sine = sigma[_] * np.sin(freq[_] * time + phase) + mu[_]
            noise = np.random.normal(0, noise[_], n)
            s[:, _] = sine + noise
        return s

    elif signal_type == "sequence":
        s = np.zeros((n, len(mu)))
        for _ in range(len(mu)):
            pattern = []
            for value, repetitions in times_values[_]:
                pattern.extend([value] * repetitions)

            repeat_pattern_times = n // len(pattern) + 1
            s[:, _] = np.tile(pattern, repeat_pattern_times)[:n]

        return s

    elif signal_type == "const":
        return np.hstack([np.tile(mu[_], n).reshape(-1, 1) for _ in range(len(mu))])

    else:
        warnings.warn(f"signal_type not matching valid types, try: random, periodic, sequence, or const")
        return None


def split_label(label, max_length):
    words = label.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_length:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return '\n'.join(lines)


def flip(items, ncol):
    """transpose legend labels"""
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def calculate_cross_correlation(ts1, ts2):
    """
    Calculate the cross-correlation between a 1D time series and a 1D or 2D time series.

    Args:
        ts1 (numpy.array): 1D time series
        ts2 (numpy.array): 1D or 2D time series

    Returns:
        numpy.array: Cross-correlation coefficients for different lags
        numpy.array: Corresponding lags
    """
    # Prepare ts1
    if isinstance(ts1, list):
        ts1 = np.array(ts1)
    if ts1.ndim == 2 and ts1.shape[1] == 1:
        ts1 = ts1.squeeze(axis=1)
    assert ts1.ndim == 1 and ts1.shape[0] > 1, "ts1 must be a 1D array with at least 2 elements"

    # Prepare ts2
    if isinstance(ts2, list):
        ts2 = np.array(ts2)
    if ts2.ndim == 2 and ts2.shape[1] == 1:
        ts2 = ts2.squeeze(axis=1)
    assert ts2.ndim in [1, 2], "ts2 must be a 1D or 2D array"

    # Ensure the time series have the same length
    assert len(ts1) == ts2.shape[-1], "Time series must have the same length"

    if ts2.ndim == 1:
        ts1_normalized = (ts1 - np.mean(ts1)) / np.std(ts1)
        ts2_normalized = (ts2 - np.mean(ts2)) / np.std(ts2)

        correlation = np.correlate(ts1_normalized, ts2_normalized, mode='full')
        lags = np.arange(-len(ts2) + 1, len(ts1))
        norm_correlation = correlation / (len(ts1) - np.abs(lags))

        no_lag_index, no_lag_correlation = len(ts2) - 1, norm_correlation[len(ts2) - 1]
        max_lag, max_correlation = lags[np.argmax(norm_correlation)], np.max(norm_correlation)

        return no_lag_index, no_lag_correlation, max_lag, max_correlation
    else:
        # 2D time series case
        pearson_correlations = []
        corrs = []
        lags = np.array([-len(ts1) + 1])
        for i in range(ts2.shape[0]):
            # Ensure the 2D time series has at least 2 elements
            assert ts2[i].ndim == 1 and ts2[i].shape[
                0] > 1, f"Time series {i} must be a 1D array with at least 2 elements"
            corr = np.correlate(ts1, ts2[i], mode='valid')
            corr /= np.sqrt(np.sum(ts1 ** 2) * np.sum(ts2[i] ** 2))
            corrs.append(corr)
            pearson_correlations.append(np.corrcoef(ts1, ts2[i])[0, 1])
        return np.array(pearson_correlations), np.array(corrs), lags