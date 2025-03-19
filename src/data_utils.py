import numpy as np
from tqdm import tqdm


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

    for run_idx in tqdm(range(len(sensor_data['x'])), desc="Filtering points"):
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


def compute_statistics(sensor_data, window_size=20):
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
    for run_idx in tqdm(range(num_runs)):
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
