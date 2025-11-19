import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory # <- CRITICAL FIX

def generate_launch_description():
    
    # 1. FIND PACKAGE SHARED DIRECTORIES
    lab5_share = FindPackageShare('lab5_nav')
    tb3_share  = FindPackageShare('turtlebot3_gazebo')
    nav2_share = FindPackageShare('nav2_bringup')
    
    # 2. DECLARE LAUNCH ARGUMENTS
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock',
    )
    map_arg = DeclareLaunchArgument(
        'map_path',
        default_value=PathJoinSubstitution([lab5_share, 'maps', 'tb3_world_map.yaml']),
        description='Full path to map YAML'
    )
    world_arg = DeclareLaunchArgument(
        'world_launch',
        default_value='turtlebot3_world.launch.py',
        description='TB3 world launcher filename in turtlebot3_gazebo/launch'
    )
    model_arg = DeclareLaunchArgument(
        'turtlebot3_model',
        default_value='burger',
        description='Model name exported to TURTLEBOT3_MODEL'
    )
    rviz_arg = DeclareLaunchArgument(
        'include_rviz',
        default_value='true',
        description='Launch RViz with the Nav2 configuration'
    )

    # 3. CONFIGURE LAUNCH SUBSTITUTIONS (read argument values)
    use_sim_time_cfg = LaunchConfiguration('use_sim_time')
    map_cfg   = LaunchConfiguration('map_path')
    world_cfg = LaunchConfiguration('world_launch')
    model_cfg = LaunchConfiguration('turtlebot3_model')
    rviz_cfg  = LaunchConfiguration('include_rviz')

    # 4. DEFINE FILE PATH SUBSTITUTIONS
    params_file = PathJoinSubstitution([lab5_share, 'config', 'nav2_params.yaml'])
    tb3_world_launch_path = PathJoinSubstitution([tb3_share, 'launch', world_cfg])
    nav_stack_launch_path = PathJoinSubstitution([lab5_share, 'launch', 'nav_stack.launch.py'])
    
    # 5. RETURN LAUNCH DESCRIPTION (ACTIONS)
    return LaunchDescription([
        # Declare all arguments first
        map_arg, world_arg, model_arg, rviz_arg, use_sim_time_arg,

        # 5a. Set Environment Variable
        SetEnvironmentVariable('TURTLEBOT3_MODEL', model_cfg),

        # 5b. Gazebo world (TB3 only)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(tb3_world_launch_path),
            launch_arguments={'use_sim_time': use_sim_time_cfg}.items()
        ),

        # 5c. Nav2 stack (localization + navigation servers)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav_stack_launch_path),
            launch_arguments={
                'map': map_cfg,
                'params_file': params_file,
                'use_sim_time': use_sim_time_cfg,
                'autostart': 'true',
                # REMOVE the 'remappings' key from here
            }.items()
           
        ),

        # 5d. RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', PathJoinSubstitution([nav2_share, 'rviz', 'nav2_default_view.rviz'])],
            parameters=[{'use_sim_time': use_sim_time_cfg}],
            output='screen',
            condition=IfCondition(rviz_cfg),
        ),
    ])