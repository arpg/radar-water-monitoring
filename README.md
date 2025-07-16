# Radar Water Level Monitoring


## 1. Data Info

### [Zenodo Link](https://zenodo.org/records/15941467)


### Auto Deployment Summary

#### Storage File
 `deployment_auto_nov2023.zip`

#### File Contents
- `GT_Nov2023.csv`: groundtruth relative depth measurements.
- `*.bag`: original ROS message recordings with radar measurements.
- `processed_data.pkl`: radar measurements extracted.

#### Main ROS Message Types used
- [sensor_msgs/Imu](https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html)
- [sensor_msgs/PointCloud2](https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointCloud2.html)

#### Deployment Statistics

|                       |       |
|-----------------------|-------|
| N runs                | 114   |
| Avg run duration, s   | 173   |
| AWR1843 frequency, Hz | 10    |
| IWR1443 frequency, Hz | 1.6   |



## 2. Processing Steps

### 2.1 Bag To Pkl

Set the extracted data location in `docker-compose.yaml` under `services.radar-processing.volumes` and run processing:

```bash
docker compose up --build auto
```


### 2.2 Analysis

Follow `plot.ipynb`.