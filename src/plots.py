import numpy as np
import matplotlib.pyplot as plt


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
    ax.set_title(f"Run {run_index}: All Points from All Measurements (Colored by Intensity)")
    plt.show()


def radar_measurements(dates, y_values, title="Radar Measurements", ylabel="Distance (m)"):
    """
    Plot radar measurements over time.

    Parameters:
        dates (array-like): Timestamps of the measurements.
        y_values (array-like): Measured distances or intensities.
        title (str): Title of the plot.
        ylabel (str): Label for the Y-axis.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_values, marker="o", linestyle="-", label="Measured Data")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def statistics(dates, max_intens, min_values, mean_values, median_values, max_values, title="Radar Measurement Statistics"):
    """
    Plot multiple statistical measures over time.

    Parameters:
        dates (array-like): Timestamps of the measurements.
        max_intens (array-like): Max intensity distances.
        min_values (array-like): Minimum distances.
        mean_values (array-like): Mean distances.
        median_values (array-like): Median distances.
        max_values (array-like): Maximum distances.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, max_intens, marker="o", linestyle="-", label="Max Intensity")
    plt.plot(dates, min_values, marker="s", linestyle="-", label="Min Distance")
    plt.plot(dates, mean_values, marker="^", linestyle="-", label="Mean Distance")
    plt.plot(dates, median_values, marker="x", linestyle="-", label="Median Distance")
    plt.plot(dates, max_values, marker="d", linestyle="-", label="Max Distance")

    plt.xlabel("Time")
    plt.ylabel("Distance (m)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def ground_truth_comparison(dates, measured_deltas, gt_dates, gt_deltas, title="Comparison with Ground Truth"):
    """
    Compare radar-measured water level changes with ground truth.

    Parameters:
        dates (array-like): Timestamps of the radar measurements.
        measured_deltas (array-like): Computed changes in measured distance.
        gt_dates (array-like): Timestamps of the ground truth data.
        gt_deltas (array-like): Ground truth water level changes.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, measured_deltas, marker="o", linestyle="-", label="Measured Delta")
    plt.plot(gt_dates, gt_deltas, marker="s", linestyle="--", label="Ground Truth Delta")

    plt.xlabel("Time")
    plt.ylabel("Change in Distance (m)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def final_error(dates, measured_deltas, gt_dates, gt_deltas, interpolated_gt, mse_error):
    """
    Plot the final computed error between measured deltas and ground truth.

    Parameters:
        dates (array-like): Timestamps of the radar measurements.
        measured_deltas (array-like): Computed changes in measured distance.
        gt_dates (array-like): Timestamps of the ground truth data.
        gt_deltas (array-like): Ground truth water level changes.
        interpolated_gt (array-like): Interpolated ground truth values.
        mse_error (float): Mean Squared Error between measured and GT deltas.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, measured_deltas, marker="o", linestyle="-", label="Measured Delta")
    plt.plot(gt_dates, gt_deltas, marker="s", linestyle="--", label="Ground Truth Delta")
    plt.plot(dates, interpolated_gt, linestyle="dotted", label="Interpolated Ground Truth")

    plt.xlabel("Time")
    plt.ylabel("Change in Distance (m)")
    plt.title(f"Final Error Comparison (MSE: {mse_error:.4f})")
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Final MSE Error: {mse_error:.4f}")
