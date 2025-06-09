# User container for the Origin

This container provides a development environment for the Avular Origin One robot. It comes pre-installed with ROS 2 and all necessary dependencies, allowing deployment and testing of your code inside the same environment used on the actual robot.

This guide explains how to build, run, and develop using the `rl-user` Docker container.

## Launch the container
Navigate to the container directory and start the container with:

```bash
cd /avular_ws/src/RL_container
docker compose up -d
```

## Using the user container for development
To enter the container, you can run the following command:
```bash
docker exec -it rl-user /bin/bash
```

When entering the container, you will be in the `/home/user/ws` directory, your ROS 2 workspace. This workspace directory is also mounted from the host OS, so your code changes persist even if the container is stopped or rebuilt.

### Installing packages
All the packages you need are in the `Dockerfile`.
After adding new packages to the `Dockerfile` you need to rebuild the container. You can do this by running
the following command:
```bash
docker compose up -d --build
```

### Building the workspace
If you make changes to the packages, you need to rebuild the workspace:
```bash
cd /home/user/ws
colcon build --symlink-install
source install/setup.bash
```

You will need to source the setup file every time you open a new terminal in the container.

ROS 2 supports multiple DDS implementations via the ROS Middleware.

The line:
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```
forces ROS 2 to use CycloneDDS, ensuring compatibility with Avular’s internal DDS setup and the Zenoh bridge used to connect to the internal ROS 2 network of the Origin One.

#### Running the ROS 2 node
If you want to run or modify the velocity control node, edit the following file:
```bash
avular_ws/src/RL_container/ws/src/rl_controller/rl_controller/send_velocity.py
```

To run the node:
```bash
ros2 run rl_controller send_velocity
```

### Stop and restart the container:
To stop the container:
```bash
docker compose down
```

To restart it:
```bash
docker compose up -d
```

