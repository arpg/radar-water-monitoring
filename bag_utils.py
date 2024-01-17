import csv
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from tqdm import tqdm

import rosbag
from sensor_msgs import point_cloud2 as pc2
import tf.transformations as tf


def read_bag_messages(bag_path):
    imu_topic, r1843_topic, r1443_topic = (
        '/gx5/imu/data', '/1843_demo/mmWaveDataHdl/RScan', '/1443_hals/mmWaveDataHdl/RScan'
    )
    imu, r1843, r1443 = [], [], []
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, t in bag.read_messages(topics=[imu_topic, r1843_topic, r1443_topic]):                    
            if topic == imu_topic:
                imu.append(msg)
            elif topic == r1843_topic:
                r1843.append(msg)
            else:
                r1443.append(msg)
    
    return imu, r1843, r1443


def read_old_messages(bag_path):
    imu_topic, r1843_topic, r1443_topic = (
        '/gx5/imu/data', '/mmWaveDataHdl/RScan', '/mmWaveDataHdl/RScan'
    )
    imu, r1843, r1443 = [], [], []
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, t in bag.read_messages(topics=[imu_topic, r1843_topic, r1443_topic]):
            if topic == imu_topic:
                imu.append(msg)
            elif topic == r1843_topic:
                r1843.append(msg)
            else:
                r1443.append(msg)

    return imu, r1843, r1443


def process_1843(radar_msgs, imu_msgs=None, apply_rotation=False):
    angle = []
    if imu_msgs is not None:
        for imu_msg in imu_msgs:
            quaternion = [imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w]
            roll, pitch, yaw = tf.euler_from_quaternion(quaternion)
            angle.append(pitch)
        angle_avg = np.mean(angle)
    elif apply_rotation:
        raise ValueError('Cannot apply rotation if imu_msgs is None')
            
    x, y, z, intensity, rng, doppler = [], [], [], [], [], []
    for r_msg in radar_msgs:
        frame_x, frame_y, frame_z, frame_int, frame_range, frame_doppler = [], [], [], [], [], []
        cloud_gen = pc2.read_points(r_msg, field_names=(
            "x", "y", "z",'intensity', 'range', 'doppler'
        ), skip_nans=True)
        for point in cloud_gen:
            frame_x.append(point[2])
            frame_y.append(point[0])
            frame_z.append(point[1])
            frame_int.append(point[3])
            frame_range.append(point[4])
            frame_doppler.append(point[5])
            
        frame_x = np.array(frame_x)
        frame_y = np.array(frame_y)
        frame_z = np.array(frame_z)
        if apply_rotation:
            cos_theta, sin_theta = np.cos(angle_avg), np.sin(angle_avg)
            y_norot = frame_y.copy()
            frame_y = y_norot * cos_theta - frame_z * sin_theta
            frame_z = y_norot * sin_theta + frame_z * cos_theta
            
        x.append(frame_x)
        y.append(frame_y)
        z.append(frame_z)
        intensity.append(np.array(frame_int))
        rng.append(np.array(frame_range))
        doppler.append(np.array(frame_doppler))

    # x, y, z, intensity, rng, doppler = np.array(x), np.array(y), np.array(z), np.array(intensity), np.array(rng), np.array(doppler)
    return x, y, z, angle, intensity, rng, doppler


def process_1443(radar_msgs, angle=None):
    x, y, z, intensity, rng, doppler = [], [], [], [], [], []
    for r_msg in radar_msgs:
        cloud_gen = pc2.read_points(r_msg, field_names=(
            "x", "y", "z",'intensity', 'range', 'doppler'
        ), skip_nans=True)
        for point in cloud_gen:
            x.append(point[0])
            y.append(point[1] * -1)
            z.append(point[2])
            intensity.append(point[3])
            rng.append(point[4])
            doppler.append(point[5])

    x, y, z, intensity, rng, doppler = np.array(x), np.array(y), np.array(z), np.array(intensity), np.array(rng), np.array(doppler)

    # if angle is not None:
    #     print(angle)
    #     print(np.cos(angle))
    #     cos_theta, sin_theta = np.cos(angle), np.sin(angle)
    #     print(cos_theta, sin_theta)
    #     y_norot = y.copy()
    #     y = y_norot * cos_theta - z * sin_theta
    #     z = y_norot * sin_theta + z * cos_theta
            
    return x, y, z, intensity, rng, doppler


def process_mimo(mimo_msgs):
    x, y, z, intensity, rng, doppler = [], [], [], [], [], []
    for r_msg in mimo_msgs:
        frame_x, frame_y, frame_z, frame_int, frame_range, frame_doppler = [], [], [], [], [], []
        cloud_gen = pc2.read_points(r_msg, field_names=(
            "x", "y", "z", 'intensity', 'range', 'doppler'
        ), skip_nans=True)
        for point in cloud_gen:
            frame_x.append(point[1])
            frame_y.append(point[0])
            frame_z.append(point[2])
            frame_int.append(point[3])
            frame_range.append(point[4])
            frame_doppler.append(point[5])

        frame_x = np.array(frame_x)
        frame_y = np.array(frame_y)
        frame_z = np.array(frame_z)
        x.append(frame_x)
        y.append(frame_y)
        z.append(frame_z)
        intensity.append(np.array(frame_int) / 1000)
        rng.append(np.array(frame_range))
        doppler.append(np.array(frame_doppler))

    return np.array(x), np.array(y), np.array(z), np.array(intensity), np.array(rng), np.array(doppler)


def plot_mimo(x, y, intens, rng, dop, int_threshold=0, x_limit=100, y_limit=100, y_threshold=0):
    int_total = np.concatenate(intens)
    x_max, y_max = 2, 2
    max_int_per_frame, max_int_y_per_frame = [], []

    fig, ax = plt.subplots()

    for frame_x, frame_y, frame_int in zip(x, y, intens):
        target_idx = frame_int >= int_threshold
        target_x, target_y, target_int = frame_x[target_idx], frame_y[target_idx], frame_int[target_idx]

        target_idx = np.abs(target_x) <= x_limit
        target_x, target_y, target_int = target_x[target_idx], target_y[target_idx], target_int[target_idx]
        target_idx = target_y <= y_limit
        target_x, target_y, target_int = target_x[target_idx], target_y[target_idx], target_int[target_idx]
        target_idx = target_y >= y_threshold
        target_x, target_y, target_int = target_x[target_idx], target_y[target_idx], target_int[target_idx]

        if target_y.size:
            x_max = max(x_max, target_x.max())
            y_max = max(y_max, target_y.max())
            max_int_idx = np.argmax(target_int)
            max_int_per_frame.append(target_int[max_int_idx])
            max_int_y_per_frame.append(target_y[max_int_idx])

        ax.scatter(target_x, target_y, c=target_int, vmin=np.min(int_total), vmax=np.max(int_total))

    max_int_per_frame, max_int_y_per_frame = np.array(max_int_per_frame), np.array(max_int_y_per_frame)
    max_int_idx = np.argwhere(max_int_per_frame == max_int_per_frame)
    max_int_y = max_int_y_per_frame[max_int_idx][0]

    fig.set_figheight(y_max * 2), fig.set_figwidth(x_max * 4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.title.set_text(f'Max intensity distance: {", ".join(max_int_y.round(3).astype(str))} m.')
    plt.show()


def plot_1843(
    x, y, z, ang, intens, rng, dop, 
    x_limit=100, y_limit=100, y_threshold=0.,
    int_threshold=0, int_threshold_percent=False,
    show_plots=True
):
    if int_threshold_percent:
        if int_threshold < 0 or int_threshold > 100:
            raise ValueError('int_threshold cannot be outside the range from 0 to 100 when int_threshold_percent is True')
        int_threshold = int_threshold / 100
    
    int_total = np.concatenate(intens)
    x_max, y_max = 2, 2
    max_int_per_frame, max_int_y_per_frame = [], []
    # int_total = int_total[int_total >= int_threshold]

    if show_plots:
        fig, ax = plt.subplots()
        # fig.set_figheight(13), fig.set_figwidth(16)

    for frame_x, frame_y, frame_z, frame_int in zip(x, y, z, intens):
        if int_threshold_percent:
            target_idx = frame_int >= (np.max(frame_int) * int_threshold)
        else:
            target_idx = frame_int >= int_threshold
        target_x, target_y, target_z, target_int = frame_x[target_idx], frame_y[target_idx], frame_z[target_idx], frame_int[target_idx]
        
        target_idx = np.abs(target_x) <= x_limit
        target_x, target_y, target_z, target_int = target_x[target_idx], target_y[target_idx], target_z[target_idx], target_int[target_idx]
        target_idx = target_y <= y_limit
        target_x, target_y, target_z, target_int = target_x[target_idx], target_y[target_idx], target_z[target_idx], target_int[target_idx]
        target_idx = target_y >= y_threshold
        target_x, target_y, target_z, target_int = target_x[target_idx], target_y[target_idx], target_z[target_idx], target_int[target_idx]

        if target_y.size:
            x_max = max(x_max, target_x.max())
            y_max = max(y_max, target_y.max())
            max_int_idx = np.argmax(target_int)
            max_int_per_frame.append(target_int[max_int_idx])
            max_int_y_per_frame.append(target_y[max_int_idx])

        if show_plots:
            ax.scatter(target_x, target_y, c=target_int, vmin=np.min(int_total), vmax=np.max(int_total))
    
    max_int_per_frame, max_int_y_per_frame = np.array(max_int_per_frame), np.array(max_int_y_per_frame)
    max_int_idx = np.argwhere(max_int_per_frame == np.max(max_int_per_frame))
    max_int_y = max_int_y_per_frame[max_int_idx][0]

    max_distance = max_int_y.round(3)
    if show_plots:
        fig.set_figheight(y_max * 2), fig.set_figwidth(x_max * 4)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.title.set_text(f'Max intensity distance: {", ".join(max_int_y.round(3).astype(str))} m.')
        plt.show()

    return max_int_y.round(3)[0]


def plot_1443(y, intens, y_limit=100, y_threshold=0., int_threshold=0, int_threshold_percent=False, show_plots=True):
    target_idx = intens >= int_threshold
    target_y, target_int = y[target_idx], intens[target_idx]
    
    target_idx = target_y <= y_limit
    target_y, target_int = target_y[target_idx], target_int[target_idx]
    target_idx = target_y >= y_threshold
    target_y, target_int = target_y[target_idx], target_int[target_idx]

    if target_y.size:
        y_max = target_y.max()
        max_int_idx = np.argmax(target_int)
        # max_int = target_int[max_int_idx]
        max_int_y = target_y.round(3)[max_int_idx]

        min_distance = np.min(target_y.round(3))
        if show_plots:
            fig, ax = plt.subplots()
            ax.scatter(np.zeros(len(target_y)), target_y, c=target_int, vmin=np.min(intens), vmax=np.max(intens))
            fig.set_figheight(y_max * 4), fig.set_figwidth(6)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.title.set_text(f'Min distance: {min_distance} m.')
            plt.show()

        return min_distance



def plot_stats(
    x, y, intens, 
    y_limit=100, x_limit=100, y_threshold=0., window_size=10,
    int_threshold=0, int_threshold_percent=False,
    show_plots=True
): 
    if int_threshold_percent:
        if int_threshold < 0 or int_threshold > 100:
            raise ValueError('int_threshold cannot be outside the range from 0 to 100 when int_threshold_percent is True')
        int_threshold = int_threshold / 100
        
    y_max_intens, y_min, y_mean, y_min_mean, y_median, y_max = [], [], [], [], [], []
    
    for i in range(len(y) - window_size):        
        x_frames, y_frames, int_frames = x[i : i + window_size], y[i : i + window_size], intens[i : i + window_size]
        x_flat, y_flat = np.concatenate(x_frames), np.concatenate(y_frames)
        int_flat = np.concatenate(int_frames)

        if int_threshold_percent:
            target_idx = int_flat >= (np.max(int_flat) * int_threshold)  
        else:
            target_idx = int_flat >= int_threshold
        target_x, target_y, target_int = x_flat[target_idx], y_flat[target_idx], int_flat[target_idx]
        
        target_idx = target_y >= y_threshold
        target_x, target_y, target_int = target_x[target_idx], target_y[target_idx], target_int[target_idx]
        target_idx = target_y <= y_limit
        target_x, target_y, target_int = target_x[target_idx], target_y[target_idx], target_int[target_idx]
        target_idx = np.abs(target_x) <= x_limit
        target_x, target_y, target_int = target_x[target_idx], target_y[target_idx], target_int[target_idx]
        
        if target_y.size:
            y_max_intens.append(target_y[np.argmax(target_int)])
            y_min.append(target_y.min())
            y_mean.append(target_y.mean())
            y_median.append(np.median(target_y.round(3)))
            y_max.append(target_y.max())
            
    y_max_intens, y_min, y_mean, y_min_mean, y_median, y_max = np.array(y_max_intens), np.array(y_min), np.array(y_mean), np.array(y_min_mean), np.array(y_median), np.array(y_max)
    y_max_intens_mode = stats.mode(y_max_intens.round(3)).mode[0]
    y_min_mode = stats.mode(y_min.round(3)).mode[0]
    y_mean_mode = stats.mode(y_mean.round(3)).mode[0]
    y_median_mode = stats.mode(y_median).mode[0]
    y_max_mode = stats.mode(y_max.round(3)).mode[0]

    if show_plots:
        fig, ax = plt.subplots(5)
        fig.set_figheight(22), fig.set_figwidth(14)
        
        ax[0].plot(np.arange(len(y_max_intens)), y_max_intens)
        ax[0].title.set_text(f'Max Intensity over Window: {y_max_intens_mode} m.')
    
        ax[1].plot(np.arange(len(y_min)), y_min)
        ax[1].title.set_text(f'Min over Window: {y_min_mode} m.')
        
        ax[2].plot(np.arange(len(y_mean)), y_mean)
        ax[2].title.set_text(f'Mean over Window: {y_mean_mode} m.')
        
        ax[3].plot(np.arange(len(y_median)), y_median)
        ax[3].title.set_text(f'Median over Window: {y_median_mode} m.')
        
        ax[4].plot(np.arange(len(y_max)), y_max)
        ax[4].title.set_text(f'Max over Window: {y_max_mode} m.')
        
        plt.show()

    return y_max_intens_mode, y_min_mode, y_mean_mode, y_median_mode, y_max_mode


def plot_stats_1443(
    x, y, intens, 
    y_limit=100, x_limit=100, y_threshold=0, window_size=10, 
    int_threshold=0., int_threshold_percent=False,
    show_plots=True
): 
    if int_threshold_percent:
        if int_threshold < 0 or int_threshold > 100:
            raise ValueError('int_threshold cannot be outside the range from 0 to 100 when int_threshold_percent is True')
        int_threshold = int_threshold / 100
        
    y_max_intens, y_min, y_mean, y_min_mean, y_median, y_max = [], [], [], [], [], []
    
    for i in range(len(y) - window_size):        
        x_flat, y_flat, int_flat = x[i : i + window_size], y[i : i + window_size], intens[i : i + window_size]

        if int_threshold_percent:
            target_idx = int_flat >= (np.max(int_flat) * int_threshold)  
        else:
            target_idx = int_flat >= int_threshold
        target_x, target_y, target_int = x_flat[target_idx], y_flat[target_idx], int_flat[target_idx]
        
        target_idx = target_y >= y_threshold
        target_x, target_y, target_int = target_x[target_idx], target_y[target_idx], target_int[target_idx]
        target_idx = target_y <= y_limit
        target_x, target_y, target_int = target_x[target_idx], target_y[target_idx], target_int[target_idx]
        target_idx = np.abs(target_x) <= x_limit
        target_x, target_y, target_int = target_x[target_idx], target_y[target_idx], target_int[target_idx]
        
        if target_y.size:
            y_max_intens.append(target_y[np.argmax(target_int)])
            y_min.append(target_y.min())
            y_mean.append(target_y.mean())
            y_median.append(np.median(target_y.round(3)))
            y_max.append(target_y.max())
            
    y_max_intens, y_min, y_mean, y_min_mean, y_median, y_max = np.array(y_max_intens), np.array(y_min), np.array(y_mean), np.array(y_min_mean), np.array(y_median), np.array(y_max)
    y_max_intens_mode = stats.mode(y_max_intens.round(3)).mode[0]
    y_min_mode = stats.mode(y_min.round(3)).mode[0]
    y_mean_mode = stats.mode(y_mean.round(3)).mode[0]
    y_median_mode = stats.mode(y_median).mode[0]
    y_max_mode = stats.mode(y_max.round(3)).mode[0]

    if show_plots:
        fig, ax = plt.subplots(5)
        fig.set_figheight(22), fig.set_figwidth(14)
        
        ax[0].plot(np.arange(len(y_max_intens)), y_max_intens)
        ax[0].title.set_text(f'Max Intensity over Window: {y_max_intens_mode} m.')
    
        ax[1].plot(np.arange(len(y_min)), y_min)
        ax[1].title.set_text(f'Min over Window: {y_min_mode} m.')
        
        ax[2].plot(np.arange(len(y_mean)), y_mean)
        ax[2].title.set_text(f'Mean over Window: {y_mean_mode} m.')
        
        ax[3].plot(np.arange(len(y_median)), y_median)
        ax[3].title.set_text(f'Median over Window: {y_median_mode} m.')
        
        ax[4].plot(np.arange(len(y_max)), y_max)
        ax[4].title.set_text(f'Max over Window: {y_max_mode} m.')
        
        plt.show()

    return y_max_intens_mode, y_min_mode, y_mean_mode, y_median_mode, y_max_mode


def process_bags(
    bags_folder, return_dates=False, show_plots=True,
    y_limit=6, x_limit=6, y_threshold=0.5, int_threshold=60, int_threshold_percent=True, window_size=20
):
    filename_formats = '%d.%m.%Y_%H.%M', '%H.%M.%S_%d.%m.%Y'
    paths = tuple(
        os.path.join(bags_folder, filename) for filename in os.listdir(bags_folder)
        if filename.endswith('.bag')
    )
    paths = sorted(paths)
    print('Processing', len(paths), 'files')
    
    max_int_y_1843, min_y_1443 = [], []
    y_max_intens_mode_1843, y_min_mode_1843, y_mean_mode_1843, y_median_mode_1843, y_max_mode_1843 = [], [], [], [], []
    y_max_intens_mode_1443, y_min_mode_1443, y_mean_mode_1443, y_median_mode_1443, y_max_mode_1443 = [], [], [], [], []
    dates = []
    
    for path in tqdm(paths):
        filename = os.path.basename(path).replace('.bag', '')
        dt = None
        for format in filename_formats:
            try:
                dt = datetime.datetime.strptime(filename, format)
            except ValueError:
                continue
        if not dt:    
            raise ValueError(f"Invalid bag filename format: {filename}. Acceptable formats are: {', '.join(filename_formats)}")
        dates.append(dt)
        
        imu, r1843, r1443 = read_bag_messages(path)
        x1843, y1843, z1843, ang1843, int1843, rng1843, dop1843 = process_1843(r1843, imu, apply_rotation=True)
        angle_avg = np.mean(ang1843)
        x1443, y1443, z1443, int1443, rng1443, dop1443 = process_1443(r1443, angle=angle_avg)
        
        max_int_y = plot_1843(
            x1843, y1843, z1843, ang1843, int1843, rng1843, dop1843, 
            y_limit=y_limit, x_limit=x_limit, y_threshold=y_threshold,
            int_threshold=int_threshold, int_threshold_percent=int_threshold_percent,
            show_plots=show_plots
        )
        max_int_y_1843.append(max_int_y)

        min_y = plot_1443(
            y1443, int1443,
            y_limit=y_limit, y_threshold=y_threshold,
            show_plots=show_plots
        )
        min_y_1443.append(min_y)

        y_max_intens_mode, y_min_mode, y_mean_mode, y_median_mode, y_max_mode = plot_stats(
            x1843, y1843, int1843, 
            y_limit=y_limit, x_limit=x_limit, y_threshold=y_threshold,
            int_threshold=int_threshold, int_threshold_percent=int_threshold_percent,
            show_plots=False
        )
        y_max_intens_mode_1843.append(y_max_intens_mode), y_min_mode_1843.append(y_min_mode), y_mean_mode_1843.append(y_mean_mode)
        y_median_mode_1843.append(y_median_mode), y_max_mode_1843.append(y_max_mode)

        y_max_intens_mode, y_min_mode, y_mean_mode, y_median_mode, y_max_mode = plot_stats_1443(
            x1443, y1443, int1443, 
            y_limit=y_limit, x_limit=x_limit, y_threshold=y_threshold,
            int_threshold=int_threshold, int_threshold_percent=int_threshold_percent,
            window_size=window_size, show_plots=False
        )
        y_max_intens_mode_1443.append(y_max_intens_mode), y_min_mode_1443.append(y_min_mode), y_mean_mode_1443.append(y_mean_mode)
        y_median_mode_1443.append(y_median_mode), y_max_mode_1443.append(y_max_mode)

    if not return_dates:
        return (
            max_int_y_1843, min_y_1443, 
            y_max_intens_mode_1843, y_min_mode_1843, y_mean_mode_1843, y_median_mode_1843, y_max_mode_1843,
            y_max_intens_mode_1443, y_min_mode_1443, y_mean_mode_1443, y_median_mode_1443, y_max_mode_1443
        )
        
    return (
            dates, max_int_y_1843, min_y_1443, 
            y_max_intens_mode_1843, y_min_mode_1843, y_mean_mode_1843, y_median_mode_1843, y_max_mode_1843,
            y_max_intens_mode_1443, y_min_mode_1443, y_mean_mode_1443, y_median_mode_1443, y_max_mode_1443
        )


def read_gt(path):
    with open(path, newline='') as csvfile:
        rows = [row for row in csv.reader(csvfile)]
    dates, depths = [], []
    for row in rows[14:163]:
        month_str, day_str, year_str = row[0].split('/')
        time_str, diff = row[1].split(' ')
        hour_str, minute_str, _ = time_str.split(':')
        dates.append(datetime.datetime(
            year=int(year_str), month=int(month_str), day=int(day_str), 
            hour=int(hour_str) + (12 if diff == 'pm' and hour_str != '12' else 0) - (12 if diff == 'am' and hour_str == '12' else 0), minute=int(minute_str)
        ))
        depths.append(float(row[3]))
    return dates, depths


def plot_total(data, gt, label_dates=False):
    if label_dates:
        (
            dates, max_int_y_1843, min_y_1443, 
            y_max_intens_mode_1843, y_min_mode_1843, y_mean_mode_1843, y_median_mode_1843, y_max_mode_1843, 
            y_max_intens_mode_1443, y_min_mode_1443, y_mean_mode_1443, y_median_mode_1443, y_max_mode_1443
        ) = data
    else:
        (
            max_int_y_1843, min_y_1443, 
            y_max_intens_mode_1843, y_min_mode_1843, y_mean_mode_1843, y_median_mode_1843, y_max_mode_1843, 
            y_max_intens_mode_1443, y_min_mode_1443, y_mean_mode_1443, y_median_mode_1443, y_max_mode_1443
        ) = data

    depth_diffs, min_y_1443_diffs, y_max_intens_mode_1443_diffs, y_min_mode_1443_diffs = [], [], [], []
    for depth in gt[1][1:]:
        depth_diffs.append(depth - gt[1][0])
    for min_y, y_max, y_min in zip(min_y_1443[1:], y_max_intens_mode_1443[1:], y_min_mode_1443[1:]):
        min_y_1443_diffs.append(min_y_1443[0] - min_y)
        y_max_intens_mode_1443_diffs.append(y_max_intens_mode_1443[0] - y_max)
        y_min_mode_1443_diffs.append(y_min_mode_1443[0] - y_min)

    depth_diffs, min_y_1443_diffs, y_max_intens_mode_1443_diffs, y_min_mode_1443_diffs = np.array(depth_diffs), np.array(min_y_1443_diffs), np.array(y_max_intens_mode_1443_diffs), np.array(y_min_mode_1443_diffs)
        
    fig, ax = plt.subplots(6)
    fig.set_figheight(40), fig.set_figwidth(14)

    ax[0].plot(dates, min_y_1443)
    # ax[0].plot(gt[0], gt[1])
    ax[0].title.set_text(f'Min Distance from 1443, m.')
    ax[0].set_xlabel('Measurement Time')
    ax[0].set_ylabel('Distance, m.')

    ax[1].plot(dates[1:], min_y_1443_diffs)
    ax[1].plot(gt[0][1:], depth_diffs)
    ax[1].set_xlabel('Measurement Time')
    ax[1].set_ylabel('Decrease in Depth / Increase in Distance, m.')

    ax[2].plot(dates, y_max_intens_mode_1443)
    # ax[2].plot(gt[0], gt[1])
    ax[2].title.set_text(f'Windowed Mode of Max Intensity from 1443, m.')
    ax[2].set_xlabel('Measurement Time')
    ax[2].set_ylabel('Distance, m.')

    ax[3].plot(dates[1:], y_max_intens_mode_1443_diffs)
    ax[3].plot(gt[0][1:], depth_diffs)
    ax[3].set_xlabel('Measurement Time')
    ax[3].set_ylabel('Decrease in Depth / Increase in Distance, m.')

    ax[4].plot(dates, y_min_mode_1443)
    # ax[4].plot(gt[0], gt[1])
    ax[4].title.set_text(f'Windowed Mode of Min Distance from 1443, m.')
    ax[4].set_xlabel('Measurement Time')
    ax[4].set_ylabel('Distance, m.')

    ax[5].plot(dates[1:], y_min_mode_1443_diffs)
    ax[5].plot(gt[0][1:], depth_diffs)
    ax[5].set_xlabel('Measurement Time')
    ax[5].set_ylabel('Decrease in Depth / Increase in Distance, m.')

    indices_diffs = np.linspace(0, 1, len(min_y_1443_diffs))
    indices_data = np.linspace(0, 1, len(depth_diffs))
    f = interp1d(indices_data, depth_diffs, kind='quadratic')
    interpolated_data = f(indices_diffs)
    print('Min distance MSE:', np.mean((min_y_1443_diffs - interpolated_data)**2))

    # ax[6].plot(dates[1:], min_y_1443_diffs)
    # ax[6].plot(gt[0][1:], depth_diffs)
    # ax[6].plot(dates[1:], interpolated_data)

    indices_diffs = np.linspace(0, 1, len(y_max_intens_mode_1443_diffs))
    interpolated_data = f(indices_diffs)
    print('Max Intensity MSE:', np.mean((y_max_intens_mode_1443_diffs - interpolated_data) ** 2))

    indices_diffs = np.linspace(0, 1, len(y_min_mode_1443_diffs))
    interpolated_data = f(indices_diffs)
    print('Windowed min distance MSE:', np.mean((y_min_mode_1443_diffs - interpolated_data) ** 2))

    # ax[3].plot(dates, y_mean_mode_1443)
    # ax[3].title.set_text(f'Windowed Mode of Mean Distance from 1443, m.')
    # ax[3].set_xlabel('Measurement Number')
    # ax[3].set_ylabel('Distance, m.')

    # ax[4].plot(dates, y_median_mode_1443)
    # ax[4].title.set_text(f'Windowed Mode of Median Distance from 1443, m.')
    # ax[4].set_xlabel('Measurement Number')
    # ax[4].set_ylabel('Distance, m.')

    # ax[5].plot(dates, y_max_mode_1443)
    # ax[5].title.set_text(f'Windowed Mode of Max Distance from 1443, m.')
    # ax[5].set_xlabel('Measurement Number')
    # ax[5].set_ylabel('Distance, m.')

    # ax[6].plot(dates, max_int_y_1843)
    # ax[6].title.set_text(f'Max Intensity Distance from 1843, m.')
    # ax[6].set_xlabel('Measurement Number')
    # ax[6].set_ylabel('Distance, m.')
    
    # ax[7].plot(dates, y_max_intens_mode_1843)
    # ax[7].title.set_text(f'Windowed Mode of Max Intensity from 1843, m.')
    # ax[7].set_xlabel('Measurement Number')
    # ax[7].set_ylabel('Distance, m.')
    
    # ax[8].plot(dates, y_min_mode_1843)
    # ax[8].title.set_text(f'Windowed Mode of Min Distance from 1843, m.')
    # ax[8].set_xlabel('Measurement Number')
    # ax[8].set_ylabel('Distance, m.')

    # ax[9].plot(dates, y_mean_mode_1843)
    # ax[9].title.set_text(f'Windowed Mode of Mean Distance from 1843, m.')
    # ax[9].set_xlabel('Measurement Number')
    # ax[9].set_ylabel('Distance, m.')

    # ax[10].plot(dates, y_median_mode_1843)
    # ax[10].title.set_text(f'Windowed Mode of Median Distance from 1843, m.')
    # ax[10].set_xlabel('Measurement Number')
    # ax[10].set_ylabel('Distance, m.')

    # ax[11].plot(dates, y_max_mode_1843)
    # ax[11].title.set_text(f'Windowed Mode of Max Distance from 1843, m.')
    # ax[11].set_xlabel('Measurement Number')
    # ax[11].set_ylabel('Distance, m.')

    plt.show()