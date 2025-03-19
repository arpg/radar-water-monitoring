import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from scipy import stats as scipy_stats


def filter_points(
        sensor_data,
        intensity_threshold=0.0, intensity_threshold_percent=0.0,
        x_limit=100.0, y_limit=100.0, y_threshold=0.0
):
    """
    Filters radar point cloud data based on intensity and positional constraints.

    Parameters:
        sensor_data (dict): Dictionary containing lists of lists of NumPy arrays
            with keys: ['x', 'y', 'z', 'intensity'].
        intensity_threshold (float): Minimum absolute intensity value to keep a point.
        intensity_threshold_percent (float): Percentage threshold of max intensity in a measurement.
        x_limit (float): Maximum absolute value for x-coordinates.
        y_limit (float): Maximum y-coordinate value.
        y_threshold (float): Minimum y-coordinate value.

    Returns:
        dict: Filtered sensor data, with empty measurements removed.
    """
    filtered_data = {key: [] for key in sensor_data.keys()}

    for run_idx in range(len(sensor_data['x'])):
        filtered_run = {key: [] for key in sensor_data.keys()}

        for measurement_idx in range(len(sensor_data['x'][run_idx])):
            x_vals = sensor_data['x'][run_idx][measurement_idx]
            y_vals = sensor_data['y'][run_idx][measurement_idx]
            z_vals = sensor_data['z'][run_idx][measurement_idx]
            intensities = sensor_data['intensity'][run_idx][measurement_idx]
            if x_vals.size == 0:
                continue

            if intensity_threshold_percent > 0:
                intensity_threshold = np.max(intensities) * intensity_threshold_percent / 100

            valid_indices = (
                    (intensities >= intensity_threshold) &
                    (np.abs(x_vals) <= x_limit) &
                    (y_vals >= y_threshold) &
                    (y_vals <= y_limit)
            )
            x_filtered = x_vals[valid_indices]
            y_filtered = y_vals[valid_indices]
            z_filtered = z_vals[valid_indices]
            intensity_filtered = intensities[valid_indices]

            if x_filtered.size > 0:
                filtered_run['x'].append(x_filtered)
                filtered_run['y'].append(y_filtered)
                filtered_run['z'].append(z_filtered)
                filtered_run['intensity'].append(intensity_filtered)

        for key in filtered_data:
            filtered_data[key].append(filtered_run[key])

    return filtered_data


def compute_statistics(sensor_data, window_size=1):
    """
    Compute statistical metrics for each measurement and apply a rolling window to measurements.

    Parameters:
        sensor_data (dict): Dictionary containing 'x', 'y', 'z', and 'intensity' values.
            Each entry is a list of lists of numpy arrays (shape: (runs, measurements, points)).
        window_size (int): Size of the rolling window applied to measurements.

    Returns:
        dict: Processed statistics with the same number of runs, and windowed statistics
              with (measurements - window_size + 1) values.
    """
    num_runs = len(sensor_data["y"])
    results = {
        "max_intensity": [],
        "min_distance": [],
        "max_intensity_w": [],
        "min_distance_w": [],
        "mean_distance_w": [],
        "median_distance_w": []
    }
    for run_idx in range(num_runs):
        y_values_run = sensor_data["y"][run_idx]
        intensity_run = sensor_data["intensity"][run_idx]
        num_measurements = len(y_values_run)
        y_max_intens, y_min = [], []

        for meas_idx in range(num_measurements):
            y_meas = y_values_run[meas_idx]
            int_meas = intensity_run[meas_idx]
            if y_meas.size:
                y_max_intens.append(y_meas[np.argmax(int_meas)])
                y_min.append(y_meas.min())
            else:
                y_max_intens.append(np.nan)
                y_min.append(np.nan)
        results["max_intensity"].append(np.array(y_max_intens))
        results["min_distance"].append(np.array(y_min))

        y_max_intens_windowed, y_min_windowed, y_mean_windowed, y_median_windowed = [], [], [], []
        for i in range(num_measurements - window_size + 1):
            y_window = np.concatenate(y_values_run[i: i + window_size])
            int_window = np.concatenate(intensity_run[i: i + window_size])
            y_max_intens_windowed.append(y_window[np.argmax(int_window)])
            y_min_windowed.append(y_window.min())
            y_mean_windowed.append(y_window.mean())
            y_median_windowed.append(np.median(y_window))

        results["max_intensity_w"].append(np.array(y_max_intens_windowed))
        results["min_distance_w"].append(np.array(y_min_windowed))
        results["mean_distance_w"].append(np.array(y_mean_windowed))
        results["median_distance_w"].append(np.array(y_median_windowed))

    return results


def filter_gt(timestamps, depths, min_depth_threshold=0.0):
    if len(timestamps) != len(depths):
        raise ValueError("Timestamps and depths must have the same length.")
    valid_indices = depths >= min_depth_threshold
    return timestamps[valid_indices], depths[valid_indices]


def match_gt_to_timestamps(sensor_dates, gt_dates, gt_depths):
    matched_depths = np.zeros_like(sensor_dates, dtype=float)
    for i, sensor_time in enumerate(sensor_dates):
        closest_idx = np.argmin(np.abs(gt_dates - sensor_time))
        matched_depths[i] = gt_depths[closest_idx]
    return matched_depths


def calc_distance_per_run(sensor_stat, max_measurements=0):
    max_measurements = max_measurements or 99999999
    sensor_modes = np.array([scipy_stats.mode(y[:max_measurements], keepdims=False).mode for y in sensor_stat])
    return sensor_modes


def calc_deltas(sensor_measurements, gt_depths):
    sensor_deltas = sensor_measurements - sensor_measurements[0]
    gt_deltas = -(gt_depths - gt_depths[0])
    return sensor_deltas, gt_deltas


def calc_delta_mse(sensor_y, gt_depths, max_measurements=0):
    sensor_modes = calc_distance_per_run(sensor_y, max_measurements)
    sensor_deltas, gt_deltas = calc_deltas(sensor_modes, gt_depths)
    mse = np.mean((sensor_deltas - gt_deltas) ** 2)
    return mse


def param_search_smallw(sensor_data, gt_depths, sensor_name):
    scores = []
    for y_threshold in (0.0, 0.1, 0.5):
        print('y_threshold', y_threshold)
        for y_limit in tqdm((10, 15, 20, 50)):
            for x_limit in (0.1, 0.2, 0.5, 1, 2, 5, 10):
                for intensity_threshold_percent in (0, 5, 10, 25, 50, 75, 90):
                    for window_size in (2, 5, 10, 15, 20):
                        data_filtered = filter_points(sensor_data, y_threshold=y_threshold, y_limit=y_limit, x_limit=x_limit, intensity_threshold_percent=intensity_threshold_percent)
                        sensor_stats = compute_statistics(data_filtered, window_size=window_size)
                        for stat_name in sensor_stats:
                            mse = calc_delta_mse(sensor_stats[stat_name], gt_depths)
                            score = {
                                'y_threshold': y_threshold,
                                'y_limit': y_limit,
                                'x_limit': x_limit,
                                'intensity_threshold_percent': intensity_threshold_percent,
                                'window_size': window_size,
                                'stat_name': stat_name,
                                'mse': mse
                            }
                            scores.append(score)
    with open(f'{sensor_name}_params_smallw.pkl', 'wb') as f:
        pickle.dump(scores, f)
    return scores

# def param_search_limited_measurements(sensor_data, gt_depths, sensor_name):
#     scores = []
#     for y_threshold in (0.0, 0.1, 0.5):
#         print('y_threshold', y_threshold)
#         for y_limit in tqdm((10, 15, 20, 50)):
#             for x_limit in (0.1, 0.2, 0.5, 1, 2, 5, 10):
#                 for intensity_threshold_percent in (0, 5, 10, 25, 50, 75, 90):
#                     for window_size in (2, 5, 10, 15, 20):
#                         data_filtered = filter_points(sensor_data, y_threshold=y_threshold, y_limit=y_limit, x_limit=x_limit, intensity_threshold_percent=intensity_threshold_percent)
#                         sensor_stats = compute_statistics(data_filtered, window_size=window_size)
#                         for stat_name in sensor_stats:
#                             for max_measurements in (1, 2, 5, 10, 25, 50, 100):
#                                 mse = calc_delta_mse(sensor_stats[stat_name], gt_depths, max_measurements=max_measurements)
#                                 score = {
#                                     'y_threshold': y_threshold,
#                                     'y_limit': y_limit,
#                                     'x_limit': x_limit,
#                                     'intensity_threshold_percent': intensity_threshold_percent,
#                                     'window_size': window_size,
#                                     'stat_name': stat_name,
#                                     'mse': mse
#                                 }
#                                 scores.append(score)
#
#     with open(f'{sensor_name}_params_limited_measurements.pkl', 'wb') as f:
#         pickle.dump(scores, f)
#     return scores

def window_size_search(sensor_data, gt_depths, sensor_name):
    scores = []
    for window_size in (2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200):
        sensor_stats = compute_statistics(sensor_data, window_size=window_size)
        for stat_name in sensor_stats:
            for max_measurements in (1, 2, 5, 10, 25, 50, 100):
                mse = calc_delta_mse(sensor_stats[stat_name], gt_depths, max_measurements=max_measurements)
                score = {
                    'window_size': window_size,
                    'max_measurements': max_measurements,
                    'stat_name': stat_name,
                    'mse': mse
                }
                scores.append(score)
    with open(f'{sensor_name}_window_size.pkl', 'wb') as f:
        pickle.dump(scores, f)
    return scores


def param_search_limited_measurements(sensor_data, gt_depths, sensor_name):
    scores = []
    for y_threshold in (0.0, 0.1):
        print('y_threshold', y_threshold)
        for y_limit in tqdm((25, 50, 100)):
            for x_limit in (1, 2, 5, 10):
                for intensity_threshold_percent in (50, 75, 90):
                    for window_size in (2, 5, 10, 15, 20):
                        data_filtered = filter_points(sensor_data, y_threshold=y_threshold, y_limit=y_limit, x_limit=x_limit, intensity_threshold_percent=intensity_threshold_percent)
                        sensor_stats = compute_statistics(data_filtered, window_size=window_size)
                        for stat_name in sensor_stats:
                            for max_measurements in (1, 2, 5, 10, 25, 50, 100):
                                mse = calc_delta_mse(sensor_stats[stat_name], gt_depths, max_measurements=max_measurements)
                                score = {
                                    'y_threshold': y_threshold,
                                    'y_limit': y_limit,
                                    'x_limit': x_limit,
                                    'intensity_threshold_percent': intensity_threshold_percent,
                                    'window_size': window_size,
                                    'max_measurements': max_measurements,
                                    'stat_name': stat_name,
                                    'mse': mse
                                }
                                scores.append(score)
    with open(f'{sensor_name}_params_limited_measurements.pkl', 'wb') as f:
        pickle.dump(scores, f)
    return scores


def get_best_scores(scores):
    best_mse = min(scores, key=lambda x: x['mse'])['mse']
    smallest_scores = list(filter(lambda x: x['mse'] == best_mse, scores))
    return smallest_scores


def print_unique_score_params(scores):
    params = defaultdict(set)
    for score in scores:
        for stat_name in score:
            params[stat_name].add(score[stat_name])
    for stat_name in params:
        print(f'{stat_name}: {sorted(list(params[stat_name]))}')


def compute_outlier_idx(data, multiplier=1.5):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return (data >= lower_bound) & (data <= upper_bound)


def compute_metrics(measurements, autocorrelation_lag=1):
    standard_deviation = np.std(measurements)
    total_variation = np.sum(np.abs(np.diff(measurements)))

    if len(measurements) < autocorrelation_lag + 1:
        raise ValueError("Lag is too large for the given data.")
    mean_measurements = np.mean(measurements)
    numerator = np.sum((measurements[:-autocorrelation_lag] - mean_measurements) * (measurements[autocorrelation_lag:] - mean_measurements))
    denominator = np.sum((measurements - mean_measurements) ** 2)
    autocorrelation = numerator / denominator if denominator != 0 else 0.0

    signal_variance = np.var(measurements)
    noise_variance = np.var(np.diff(measurements))
    snr = 10 * np.log10(signal_variance / noise_variance) if noise_variance != 0 else float('inf')

    return {
        'standard_deviation': standard_deviation,
        'total_variation': total_variation,
        'autocorrelation': autocorrelation,
        'snr': snr
    }
