#!/usr/bin/env python3
"""
publish_tags_and_markers.py

功能:
- 读取 YAML 格式的 field layout（map->tag poses）
- 使用 tf2_ros.StaticTransformBroadcaster 发布静态 transforms: map -> tag_frame
- 发布 visualization_msgs/MarkerArray: 立方体 + 文本标签，便于 RViz 可视化

使用:
  export FIELD_LAYOUT=~/field_layout.yaml
  ros2 run <your_pkg> publish_tags_and_markers.py
  或 python3 publish_tags_and_markers.py
"""
import rclpy
from rclpy.node import Node
import os
import yaml
import tf_transformations as tft
from geometry_msgs.msg import TransformStamped, Vector3
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
import math
import time

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_frame_name(entry_key, entry):
    if isinstance(entry, dict) and 'frame' in entry:
        return entry['frame']
    try:
        return f"tag36h11:{int(entry_key)}"
    except Exception:
        return str(entry_key)

class TagVizNode(Node):
    def __init__(self, yaml_path, map_frame='map'):
        super().__init__('tag_viz_node')
        self.map_frame = map_frame
        self.yaml_path = yaml_path
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.marker_pub = self.create_publisher(MarkerArray, 'tag_markers', 10)
        self.declare_parameter('publish_rate', 20.0)
        self.timer = self.create_timer(1.0 / self.get_parameter('publish_rate').value, self.timer_cb)
        self.tags = {}
        self.load_field_layout()
        self.get_logger().info(f"Loaded {len(self.tags)} tags; publishing static TF and markers")

    def load_field_layout(self):
        data = load_yaml(self.yaml_path)
        tags = {}
        for k, v in data.items():
            # support both simple dict or frame-keyed entries
            if isinstance(v, dict):
                frame = ensure_frame_name(k, v)
                pos = v.get('position', [0.0, 0.0, 0.0])
                ori = v.get('orientation', [0.0, 0.0, 0.0, 1.0])
                size = float(v.get('size', 0.15))
            else:
                # fallback if value is list etc (not expected)
                frame = ensure_frame_name(k, {})
                pos = [0.0, 0.0, 0.0]
                ori = [0.0, 0.0, 0.0, 1.0]
                size = 0.15
            tags[str(k)] = {
                'frame': frame,
                'position': pos,
                'orientation': ori,
                'size': size
            }
        self.tags = tags

    def timer_cb(self):
        # publish static transforms once (tf2_static persists; but we call sendTransform repeatedly is ok)
        tf_msgs = []
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        idx = 0
        for tag_id, info in self.tags.items():
            frame = info['frame']
            px, py, pz = info['position']
            qx, qy, qz, qw = info['orientation']
            size = info['size']
            ts = TransformStamped()
            ts.header.stamp = now
            ts.header.frame_id = self.map_frame
            ts.child_frame_id = frame
            ts.transform.translation.x = float(px)
            ts.transform.translation.y = float(py)
            ts.transform.translation.z = float(pz)
            ts.transform.rotation.x = float(qx)
            ts.transform.rotation.y = float(qy)
            ts.transform.rotation.z = float(qz)
            ts.transform.rotation.w = float(qw)
            tf_msgs.append(ts)

            # Cube marker centered at tag pose (slightly above ground)
            cube = Marker()
            cube.header.stamp = now
            cube.header.frame_id = self.map_frame
            cube.ns = "apriltag_cubes"
            cube.id = idx*2
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position.x = px
            cube.pose.position.y = py
            cube.pose.position.z = pz + size/2.0  # raise so cube sits on ground if z is tag bottom
            cube.pose.orientation.x = qx
            cube.pose.orientation.y = qy
            cube.pose.orientation.z = qz
            cube.pose.orientation.w = qw
            cube.scale = Vector3(x=size, y=size, z=size)
            cube.color.r = 0.0
            cube.color.g = 0.6
            cube.color.b = 1.0
            cube.color.a = 0.9
            cube.lifetime = Duration(sec=2)  # keep persistent if re-publishing
            marker_array.markers.append(cube)

            # Text marker for label
            text = Marker()
            text.header.stamp = now
            text.header.frame_id = self.map_frame
            text.ns = "apriltag_labels"
            text.id = idx*2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = px
            text.pose.position.y = py
            text.pose.position.z = pz + size + 0.06
            text.scale.z = 0.06  # text height
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"{frame}"
            text.lifetime = Duration(sec=2)
            marker_array.markers.append(text)
            idx += 1

        # publish static transforms and marker array
        if len(tf_msgs) > 0:
            self.tf_broadcaster.sendTransform(tf_msgs)
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    yaml_path = os.environ.get('FIELD_LAYOUT', os.path.expanduser('src/pv_apriltag/pv_apriltag/config/field_layout.yaml'))
    if not os.path.exists(yaml_path):
        print("ERROR: FIELD_LAYOUT file not found:", yaml_path)
        return
    node = TagVizNode(yaml_path=yaml_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
