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