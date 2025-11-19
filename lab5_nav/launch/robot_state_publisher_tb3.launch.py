import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']
    # model from env, default to burger
    model = os.environ.get("TURTLEBOT3_MODEL", "burger")
    desc_share = get_package_share_directory('turtlebot3_description')

    # Prefer URDF; fall back to Xacro if needed
    urdf_path = os.path.join(desc_share, 'urdf', f'turtlebot3_{model}.urdf')
    if not os.path.exists(urdf_path):
        urdf_path = os.path.join(desc_share, 'urdf', f'turtlebot3_{model}.urdf.xacro')
        # Render xacro -> xml
        import xacro
        robot_desc = xacro.process_file(urdf_path).toxml()
    else:
        with open(urdf_path, 'r') as f:
            robot_desc = f.read()

    use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true')
    # use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    # urdf_file_name = 'turtlebot3_' + model + '.urdf'
    # frame_prefix = LaunchConfiguration('frame_prefix', default='')

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time', default='true'),
            'robot_description': robot_desc
        }]
    )

    return LaunchDescription([use_sim_time, rsp])
