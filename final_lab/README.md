# Final Lab - Color Following TurtleBot3

This ROS2 package enables a TurtleBot3 to detect and follow colored objects using its camera.

## Features
- Detects red, green, and blue objects using HSV 
- Proportional control for smooth object following
- Configurable parameters for different scenarios
- Works with TurtleBot3 simulation in Gazebo

## Quick Start

1. **Build the package:**
   ```bash
   cd /path/to/your/ros2_ws
   colcon build --packages-select final_lab
   source install/setup.bash
   ```

2. **Launch TurtleBot3 with camera in Gazebo:**
   ```bash
   # Use waffle model (has camera):
   export TURTLEBOT3_MODEL=waffle
   ros2 launch turtlebot3_gazebo empty_world.launch.py
   ```

3. **Add colored objects in Gazebo:**
   - In Gazebo GUI, click the "Insert" tab
   - Add a cube from the shapes menu
   - Right-click the cube → "Edit Model" 
   - Select the cube's visual → Change material color (red, green, or blue)
   - Place the cube where the robot can see it
   - Save the model and exit edit mode

4. **Run the color follower:**
   ```bash
   ros2 run final_lab color_follower
   ```

## Usage Examples

**Follow any colored object:**
```bash
ros2 run final_lab color_follower
```

**Follow only red objects:**
```bash
ros2 run final_lab color_follower --ros-args -p color:=red
```

**Adjust speed settings:**
```bash
ros2 run final_lab color_follower --ros-args -p max_linear:=0.3 -p max_angular:=1.0
```

**Use different camera topic:**
```bash
ros2 run final_lab color_follower --ros-args -p image_topic:=/camera/rgb/image_raw
```

## Parameters

- `image_topic` (string, default: '/camera/image_raw'): Camera image topic
- `max_linear` (double, default: 0.4): Maximum forward speed (m/s)
- `max_angular` (double, default: 1.2): Maximum rotation speed (rad/s)
- `color` (string, default: ''): Specific color to follow ('red', 'green', 'blue', or '' for any)
- `min_area` (int, default: 500): Minimum blob area to consider for following

## Color Detection Ranges

The node detects colors in HSV space with these ranges:
- **Red**: 0-10° and 170-180° hue, high saturation/value
- **Green**: 40-80° hue, medium-high saturation/value  
- **Blue**: 100-130° hue, medium-high saturation/value



