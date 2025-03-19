import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


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


def stats(stats, run_index):    
    max_intensities = stats['max_intensity'][run_index]
    min_distances = stats['min_distance'][run_index]
    max_intens_windowed = stats['max_intensity_w'][run_index]
    min_dist_windowed = stats['min_distance_w'][run_index]
    mean_distances = stats['mean_distance_w'][run_index] 
    median_distances = stats['median_distance_w'][run_index]

    plot_statistic(np.arange(len(max_intensities)), max_intensities, "Measurement Index", "Max Intensity", f"Run {run_index}: Max Intensity per Measurement")
    plot_statistic(np.arange(len(min_distances)), min_distances, "Measurement Index", "Min Distance", f"Run {run_index}: Min Distance per Measurement")
    plot_statistic(np.arange(len(max_intens_windowed)), max_intens_windowed, "Measurement Index", "Max Intensity (Windowed)", f"Run {run_index}: Windowed Max Intensity")
    plot_statistic(np.arange(len(min_dist_windowed)), min_dist_windowed, "Measurement Index", "Min Distance (Windowed)", f"Run {run_index}: Windowed Min Distance")
    plot_statistic(np.arange(len(mean_distances)), mean_distances, "Measurement Index", "Mean Distance", f"Run {run_index}: Mean Distance per Measurement")
    plot_statistic(np.arange(len(median_distances)), median_distances, "Measurement Index", "Median Distance", f"Run {run_index}: Median Distance per Measurement")


def plot_deltas(sensor_t, sensor_y, gt_depths, label):
    sensor_modes = np.array([scipy_stats.mode(y, keepdims=False).mode if y.size > 0 else np.nan for y in sensor_y])
    sensor_deltas = sensor_modes - sensor_modes[0]
    gt_deltas = -(gt_depths - gt_depths[0])
    mse = np.mean((sensor_deltas - gt_deltas) ** 2)

    plt.figure(figsize=(12, 6))
    plt.plot(sensor_t, sensor_deltas, label="Distance(i) vs Distance(0) - Increase in Distance", color="blue")
    plt.plot(sensor_t, gt_deltas, label="Depth(i) vs Depth(0) - Decrease in Depth", color="orange")
    plt.xlabel("Time")
    plt.ylabel("Delta (m)")
    plt.title(f"{label}, MSE: {mse:.6f} m")
    plt.legend()
    plt.show()


def deltas(dates, sensor_stats, gt_depths):
    stats_labels = {
        "max_intensity": "Max Intensity",
        "min_distance": "Min Distance",
        "max_intensity_w": "Max Intensity (Windowed)",
        "min_distance_w": "Min Distance (Windowed)",
        "mean_distance_w": "Mean Distance (Windowed)",
        "median_distance_w": "Median Distance (Windowed)",
    }
    for key, label in stats_labels.items():
        if key in sensor_stats:
            plot_deltas(dates, sensor_stats[key], gt_depths, label=label)
