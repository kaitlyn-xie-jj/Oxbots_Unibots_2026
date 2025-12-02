ros2 launch webots_ros2_epuck robot_launch.py world:=Run.wbt

ros2 run apriltag_ros apriltag_node --ros-args   -p use_sim_time:=true   -r image_rect:=/camera/image_color   -r camera_info:=/camera/camera_info

colcon build --symlink-install
source install/setup.bash


ros2 run pv_apriltag pv_sim_bridge   --ros-args   -p use_sim_time:=true   --params-file src/pv_apriltag/pv_apriltag/config/pv_sim_bridge.params.yaml

ros2 topic echo /pv/estimated_robot_pose