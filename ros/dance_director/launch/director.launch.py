"""Start the director with the parameter file.

    ros2 launch dance_director director.launch.py
    ros2 launch dance_director director.launch.py use_sim_time:=true      # Gazebo
    ros2 launch dance_director director.launch.py params_file:=/path/my.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('dance_director'), 'config', 'director.yaml')
    return LaunchDescription([  
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='Director parameter file'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true when running against the simulator clock'),
        Node(
            package='dance_director',
            executable='director',
            name='dance_director',   # must match the top-level key in the YAML file
            output='screen',
            parameters=[LaunchConfiguration('params_file'),
                        {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
    ])