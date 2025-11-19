1. Building the Package

Before building, set your TurtleBot3 model:
export TURTLEBOT3_MODEL=burger

Build the workspace:
cd ~/cs_39536/ros2_ws
colcon build --symlink-install

Source your workspace:
source install/setup.bash


Maps are located in:
lab5_nav/maps/

The main launch file is:
launch/nav_bringup.launch.py


General launch command:
MAP="$(ros2 pkg prefix lab5_nav)/share/lab5_nav/maps/<map_file>.yaml"
ros2 launch lab5_nav nav_bringup.launch.py map_path:=$MAP

Examples:

Launch navigation with the world map:
MAP="$(ros2 pkg prefix lab5_nav)/share/lab5_nav/maps/tb3_world_map.yaml"
ros2 launch lab5_nav nav_bringup.launch.py map_path:=$MAP

Launch with the good house map:
MAP="$(ros2 pkg prefix lab5_nav)/share/lab5_nav/maps/tb3_house_map_good.yaml"
ros2 launch lab5_nav nav_bringup.launch.py map_path:=$MAP

Launch with the bad house map:
MAP="$(ros2 pkg prefix lab5_nav)/share/lab5_nav/maps/tb3_house_map_bad.yaml"
ros2 launch lab5_nav nav_bringup.launch.py map_path:=$MAP


4. Using RViz to Navigate
When RViz opens:

1. Click "2D Pose Estimate" and click your robot's location to initialize AMCL.
2 .Click "2D Goal Pose" to send a navigation goal.


5. Running the Custom Planner Plugin
lab5_nav/obstacle_aware_waypoint_nav.py

ros2 launch lab5_nav obstacle_nav.launch.py map_path:=$MAP

Selecting Planner Plugins

Planners are configured in:

lab5_nav/config/nav2_params.yaml

planner_plugins: ["GridBased", "ObstacleAware"]

To run only your plugin:
planner_plugins: ["ObstacleAware"]

If your launch file exposes a planner argument, you can run:
ros2 launch lab5_nav nav_bringup.launch.py map_path:=$MAP planner:=ObstacleAware

6. Launching the Robot Model Only
ros2 launch lab5_nav robot_state_publisher_tb3.launch.py

7. Summary of Commands:
Build:
colcon build --symlink-install
source install/setup.bash

Run Nav2:
ros2 launch lab5_nav nav_bringup.launch.py map_path:=$MAP

Run custom planner:
ros2 launch lab5_nav obstacle_nav.launch.py map_path:=$MAP

Set TB3 model:
export TURTLEBOT3_MODEL=burger
