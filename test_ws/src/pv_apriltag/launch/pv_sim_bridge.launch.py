from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os, yaml

def generate_launch_description():
    pkg_share = get_package_share_directory('pv_apriltag')
    params_file = os.path.join(pkg_share, 'config', 'pv_sim_bridge.params.yaml')
    layout_file = os.path.join(pkg_share, 'config', 'field_layout.params.yaml')

    with open(layout_file, 'r') as f:
        data = yaml.safe_load(f) or {}
    field_layout_dict = data.get('field_layout', data)

    node = Node(
        package='pv_apriltag',
        executable='pv_sim_bridge',
        name='pv_sim_bridge',
        output='screen',
        parameters=[params_file, {'field_layout': field_layout_dict}],
    )
    return LaunchDescription([node])
