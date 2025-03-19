FROM ros:noetic


RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-noetic-rosbag \
    ros-noetic-sensor-msgs \
    ros-noetic-tf-conversions \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade numpy matplotlib scipy tqdm


WORKDIR /app
COPY src src
COPY process_bags.py ./


CMD ["python3", "process_bags.py"]
