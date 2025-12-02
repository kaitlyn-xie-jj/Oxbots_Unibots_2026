from setuptools import find_packages, setup

package_name = 'pv_apriltag'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='verity-xie',
    maintainer_email='verity-xie@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pv_node = pv_apriltag.pv_node:main',           # 实机
            'pv_sim_bridge = pv_apriltag.pv_sim_bridge:main', # 仿真
            'epuck_camera_info = pv_apriltag.epuck_camera_info_node:main',
            'publish_tags_and_markers = pv_apriltag.publish_tags_and_markers:main',
            'apriltag_map_localizer = pv_apriltag.apriltag_map_localizer:main',
            'apriltag_pnp_localizer = pv_apriltag.apriltag_pnp_localizer:main',
            'apriltag_reproj_checker = pv_apriltag.apriltag_reproj_checker:main',
            'apriltag_iterative_map_pose = pv_apriltag.apriltag_iterative_map_pose:main'
        ],
    },
)
