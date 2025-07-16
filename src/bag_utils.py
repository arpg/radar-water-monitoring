import csv
import datetime
import os
import numpy as np
from tqdm import tqdm

import rosbag
from sensor_msgs import point_cloud2 as pc2
import tf.transformations as tf


def read_bag_messages(bag_path):
    imu_topic, r1843_topic, r1443_topic, cascade_cube_topic = (
        "/gx5/imu/data",
        "/1843_demo/mmWaveDataHdl/RScan",
        "/1443_hals/mmWaveDataHdl/RScan",
        "/dca_node/data_cube"
    )
    imu, r1843, r1443 = [], [], []
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, _ in bag.read_messages(topics=[imu_topic, r1843_topic, r1443_topic]):
            if topic == imu_topic:
                imu.append(msg)
            elif topic == r1843_topic:
                r1843.append(msg)
            else:
                r1443.append(msg)
    return imu, r1843, r1443


def process_1843(radar_msgs, imu_msgs=None, apply_rotation=False):
    """Process AWR 1843 radar data with optional IMU-based rotation correction and threshold filtering."""
    angle = []
    if imu_msgs:
        for imu_msg in imu_msgs:
            quaternion = [imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w]
            _, pitch, _ = tf.euler_from_quaternion(quaternion)
            angle.append(pitch)
        angle_avg = np.mean(angle)
    elif apply_rotation:
        raise ValueError("Cannot apply rotation if imu_msgs is None")

    x_values, y_values, z_values, intensities = [], [], [], []
    for r_msg in radar_msgs:
        cloud_gen = pc2.read_points(r_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        frame_x, frame_y, frame_z, frame_int = [], [], [], []
        for point in cloud_gen:
            frame_x.append(point[2])
            frame_y.append(point[0])
            frame_z.append(point[1])
            frame_int.append(point[3])
        if apply_rotation:
            cos_theta = np.cos(angle_avg)
            frame_y = np.array(frame_y) * cos_theta
        x_values.append(np.array(frame_x))
        y_values.append(np.array(frame_y))
        z_values.append(np.array(frame_z))
        intensities.append(np.array(frame_int))
    return x_values, y_values, z_values, intensities


def process_1443(radar_msgs):
    """Process IWR 1443 radar data, applying y_threshold filtering."""
    x_values, y_values, z_values, intensities = [], [], [], []
    for r_msg in radar_msgs:
        cloud_gen = pc2.read_points(r_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        frame_x, frame_y, frame_z, frame_int = [], [], [], []
        for point in cloud_gen:
            frame_x.append(point[0])
            frame_y.append(-point[1])
            frame_z.append(point[2])
            frame_int.append(point[3])
        x_values.append(np.array(frame_x))
        y_values.append(np.array(frame_y))
        z_values.append(np.array(frame_z))
        intensities.append(np.array(frame_int))
    return x_values, y_values, z_values, intensities


def read_gt(path):
    """
    Read ground truth water depth measurements from a CSV file.

    Automatically detects the header row instead of using a hardcoded row number.

    Parameters:
        path (str): Path to the ground truth CSV file.

    Returns:
        gt_dates (np.array): Array of datetime objects.
        gt_depths (np.array): Array of water depth measurements in meters.
    """
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)

        # Find the header row dynamically
        for row in reader:
            if row and "Date" in row:
                header_row = row  # Capture the header for potential debugging
                break  # Stop at the header row

        # Read data after the detected header row
        gt_dates, gt_depths = [], []
        for row in reader:
            if not row or len(row) < 4:  # Skip empty rows or malformed ones
                continue
            try:
                # Extract date & time and convert to datetime object
                month, day, year = map(int, row[0].split("/"))
                time_str, am_pm = row[1].split(" ")
                hour, minute = map(int, time_str.split(":")[:2])

                # Convert 12-hour time format to 24-hour
                if am_pm.lower() == "pm" and hour != 12:
                    hour += 12
                elif am_pm.lower() == "am" and hour == 12:
                    hour = 0
                timestamp = datetime.datetime(year, month, day, hour, minute)
                gt_dates.append(timestamp)

                # Read the LEVEL column (assuming it's always the 4th column)
                gt_depths.append(float(row[3]))

            except (ValueError, IndexError) as e:
                print(f"Skipping malformed row: {row} | Error: {e}")

    return np.array(gt_dates), np.array(gt_depths)


def process_bags(bags_folder, gt_path=None):
    """Process ROS bags and return extracted radar measurements structured in a dictionary."""
    filename_formats = ("%d.%m.%Y_%H.%M", "%H.%M.%S_%d.%m.%Y")
    bag_paths = sorted(
        os.path.join(bags_folder, f)
        for f in os.listdir(bags_folder)
        if f.endswith(".bag")
    )
    print(f"Processing {len(bag_paths)} files in directory: {bags_folder}")

    results = {
        "dates": [],
        "AWR1843": {"x": [], "y": [], "z": [], "intensity": []},
        "IWR1443": {"x": [], "y": [], "z": [], "intensity": []}
    }
    if gt_path:
        results["ground_truth"] = read_gt(gt_path)

    for path in tqdm(bag_paths):
        filename = os.path.basename(path).replace(".bag", "")
        dt = None
        for fmt in filename_formats:
            try:
                dt = datetime.datetime.strptime(filename, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            raise ValueError(f"Invalid bag filename format: {filename}")
        results["dates"].append(dt)

        imu, r1843, r1443 = read_bag_messages(path)
        x_1843, y_1843, z_1843, intens_1843 = process_1843(r1843, imu, apply_rotation=True)
        x_1443, y_1443, z_1443, intens_1443 = process_1443(r1443)

        results["AWR1843"]["x"].append(x_1843)
        results["AWR1843"]["y"].append(y_1843)
        results["AWR1843"]["z"].append(z_1843)
        results["AWR1843"]["intensity"].append(intens_1843)

        results["IWR1443"]["x"].append(x_1443)
        results["IWR1443"]["y"].append(y_1443)
        results["IWR1443"]["z"].append(z_1443)
        results["IWR1443"]["intensity"].append(intens_1443)

    for k in results:
        if not isinstance(results[k], dict):
            results[k] = np.array(results[k])

    return results


def process_single_bag(bag_path):

