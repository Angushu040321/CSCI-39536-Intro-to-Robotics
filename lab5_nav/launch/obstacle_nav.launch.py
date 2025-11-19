from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    lab5_share = FindPackageShare('lab5_nav')

    # Arguments to control Gazebo + Nav2 bringup
    map_arg = DeclareLaunchArgument(
        'map_path',
        default_value=PathJoinSubstitution([lab5_share, 'maps', 'tb3_world_map.yaml']),
        description='Absolute path to the map yaml produced by SLAM'
    )
    model_arg = DeclareLaunchArgument(
        'turtlebot3_model',
        default_value='burger',
        description='Model exported to TURTLEBOT3_MODEL for Gazebo spawn'
    )
    rviz_arg = DeclareLaunchArgument(
        'include_rviz',
        default_value='false',
        description='Start RViz with Nav2 (true/false)'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Propagate simulated clock to all nodes'
    )
    waypoints_arg = DeclareLaunchArgument(
        'waypoints',
        default_value='[[0.8, 0.0], [0.8, 0.8], [0.0, 0.8], [0.0, 0.0]]',
        description='List of map-frame waypoints. Accepts [[x,y,yaw?], ...] or flat [x1,y1,...]'
    )
    initial_pose_arg = DeclareLaunchArgument(
        'initial_pose',
        default_value='[]',
        description='Optional [x, y, yaw] map-frame initial pose to seed AMCL'
    )

    map_cfg = LaunchConfiguration('map_path')
    model_cfg = LaunchConfiguration('turtlebot3_model')
    rviz_cfg = LaunchConfiguration('include_rviz')
    use_sim_time_cfg = LaunchConfiguration('use_sim_time')
    waypoints_cfg = LaunchConfiguration('waypoints')
    init_pose_cfg = LaunchConfiguration('initial_pose')

    nav_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([lab5_share, 'launch', 'nav_bringup.launch.py'])
        ),
        launch_arguments={
            'map_path': map_cfg,
            'turtlebot3_model': model_cfg,
            'include_rviz': rviz_cfg,
            'use_sim_time': use_sim_time_cfg,
        }.items()
    )

    waypoint_nav = Node(
        package='lab5_nav',
        executable='obstacle_aware_waypoint_nav',
        name='obstacle_aware_waypoint_nav',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time_cfg,
            'waypoints': waypoints_cfg,
            'initial_pose': init_pose_cfg,
        }]
    )

    return LaunchDescription([
        map_arg,
        model_arg,
        rviz_arg,
        use_sim_time_arg,
        waypoints_arg,
        initial_pose_arg,
        nav_bringup,
        waypoint_nav,
    ])
