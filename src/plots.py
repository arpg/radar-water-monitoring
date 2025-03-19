import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from src import data_utils


def points(sensor_data, run_index):
    if run_index >= len(sensor_data['x']) or run_index < 0:
        raise ValueError(f"Invalid run index: {run_index}. Must be between 0 and {len(sensor_data['x']) - 1}.")
    x_vals = np.concatenate(sensor_data['x'][run_index])
    y_vals = np.concatenate(sensor_data['y'][run_index])
    intensities = np.concatenate(sensor_data['intensity'][run_index])
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x_vals, y_vals, c=intensities, cmap='viridis', alpha=0.7, s=5)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Intensity")
    ax.set_xlabel("X Coordinate (m)")
    ax.set_ylabel("Y Coordinate (m)")
    ax.set_title(f"Run {run_index}")
    plt.show()


def plot_statistic(x_values, y_values, x_label, y_label, title):
    result_value = scipy_stats.mode(y_values, keepdims=False).mode
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_values, y_values)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{title}, {result_value:.3f} m')
    ax.grid(True)
    plt.show()


def stats(sensor_stats, run_index):
    max_intensities = sensor_stats['max_intensity'][run_index]
    min_distances = sensor_stats['min_distance'][run_index]
    max_intens_windowed = sensor_stats['max_intensity_w'][run_index]
    min_dist_windowed = sensor_stats['min_distance_w'][run_index]
    mean_distances = sensor_stats['mean_distance_w'][run_index]
    median_distances = sensor_stats['median_distance_w'][run_index]

    plot_statistic(np.arange(len(max_intensities)), max_intensities, "Measurement Index", "Max Intensity", f"Run {run_index}: Max Intensity per Measurement")
    plot_statistic(np.arange(len(min_distances)), min_distances, "Measurement Index", "Min Distance", f"Run {run_index}: Min Distance per Measurement")
    plot_statistic(np.arange(len(max_intens_windowed)), max_intens_windowed, "Measurement Index", "Max Intensity (Windowed)", f"Run {run_index}: Windowed Max Intensity")
    plot_statistic(np.arange(len(min_dist_windowed)), min_dist_windowed, "Measurement Index", "Min Distance (Windowed)", f"Run {run_index}: Windowed Min Distance")
    plot_statistic(np.arange(len(mean_distances)), mean_distances, "Measurement Index", "Mean Distance", f"Run {run_index}: Mean Distance per Measurement")
    plot_statistic(np.arange(len(median_distances)), median_distances, "Measurement Index", "Median Distance", f"Run {run_index}: Median Distance per Measurement")


def plot_deltas(sensor_t, sensor_y, gt_depths, stat_name, sensor_name, max_measurements=0):
    max_measurements = max_measurements or 99999999
    sensor_modes = np.array([scipy_stats.mode(y[:max_measurements], keepdims=False).mode for y in sensor_y])
    sensor_deltas = sensor_modes - sensor_modes[0]
    gt_deltas = -(gt_depths - gt_depths[0])
    mse = np.mean((sensor_deltas - gt_deltas) ** 2)

    plt.figure(figsize=(12, 6))
    plt.plot(sensor_t, sensor_deltas, label="Distance[i] - Distance[0]: Increase in Distance", color="blue")
    plt.plot(sensor_t, gt_deltas, label="Depth[0] - Depth[i]: Decrease in Depth", color="orange")
    plt.xlabel("Time")
    plt.ylabel("Water Level Delta (m)")
    plt.title(f"{sensor_name.upper()} {stat_name.replace('_', ' ').capitalize()}, MSE: {mse:.6f} m^2")
    plt.legend()
    plt.savefig(f"plots/{sensor_name.replace(' ', '_').lower()}/{stat_name}_deltas_mse{mse:.6f}.png", dpi=600, bbox_inches="tight")
    plt.show()


def deltas(dates, sensor_stats, gt_depths, sensor_name):
    for stat_name in sensor_stats:
        plot_deltas(dates, sensor_stats[stat_name], gt_depths, stat_name, sensor_name)


def plot_stat_all_runs(sensor_t, sensor_stat, x_label, y_label, stat_name, sensor_name):
    sensor_modes = np.array([scipy_stats.mode(run_stat, keepdims=False).mode for run_stat in sensor_stat])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sensor_t, sensor_modes)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(sensor_name.upper() + ' ' + stat_name.replace(' ', '_').capitalize())
    ax.grid(True)

    plt.savefig(f"plots/{sensor_name.replace(' ', '_').lower()}/{stat_name}.png", dpi=600, bbox_inches="tight")
    plt.show()


def stat_all_runs(dates, sensor_stats, sensor_name):
    for stat_name in sensor_stats:
        plot_stat_all_runs(dates, sensor_stats[stat_name], 'Time', 'Distance (m)', stat_name, sensor_name)


def compare_metrics(data_arrays, labels, autocorrelation_lag=1):
    if len(data_arrays) != len(labels):
        raise ValueError("The number of data arrays must match the number of labels.")

    metric_results = {label: data_utils.compute_metrics(data, autocorrelation_lag) for data, label in zip(data_arrays, labels)}
    metrics = ['standard_deviation', 'total_variation', 'autocorrelation', 'snr']
    num_metrics = len(metrics)
    bar_width = 0.15
    x = np.arange(num_metrics)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        values = [metric_results[label][metric] for metric in metrics]
        ax.bar(x + i * bar_width, values, bar_width, label=label)

    ax.set_xticks(x + (bar_width * (len(labels) - 1)) / 2)
    ax.set_xticklabels(metrics, rotation=20)
    ax.set_ylabel("Metric Value")
    ax.set_title("Comparison of Measurement Metrics")
    ax.legend()

    plt.show()
    return metric_results


def compare_metrics_specific(sensor_data, gt_depths, autocorrelation_lag=1):
    scores = []
    for y_threshold in (0.0, 0.1):
        for y_limit in (10, 100):
            for x_limit in (0.1, 0.5, 1, 2, 5):
                for intensity_threshold_percent in (0, 50, 90):
                    for window_size in (5, 10, 15, 20):
                        data_filtered = data_utils.filter_points(sensor_data, y_threshold=y_threshold, y_limit=y_limit, x_limit=x_limit, intensity_threshold_percent=intensity_threshold_percent)
                        sensor_stats = data_utils.compute_statistics(data_filtered, window_size=window_size)
                        for max_measurements in (10, 20, 50, 100, 10000):
                            for stat_name in sensor_stats:
                                mse = data_utils.calc_delta_mse(sensor_stats[stat_name], gt_depths)
                                metrics = data_utils.compute_metrics(sensor_stats[stat_name], autocorrelation_lag)
                                metrics['mse'] = -mse
                                score = {
                                    'y_threshold': y_threshold,
                                    'y_limit': y_limit,
                                    'x_limit': x_limit,
                                    'intensity_threshold_percent': intensity_threshold_percent,
                                    'window_size': window_size,
                                    'max_measurements': max_measurements,
                                    'stat_name': stat_name,
                                    'metrics': metrics
                                }
                                scores.append(score)
    num_metrics = len(metrics)
    bar_width = 0.15
    x = np.arange(num_metrics)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        values = [metric_results[label][metric] for metric in metrics]
        ax.bar(x + i * bar_width, values, bar_width, label=label)

    ax.set_xticks(x + (bar_width * (len(labels) - 1)) / 2)
    ax.set_xticklabels(metrics, rotation=20)
    ax.set_ylabel("Metric Value")
    ax.set_title("Comparison of Measurement Metrics")
    ax.legend()

    plt.show()
    return metric_results
