#!/usr/bin/env python3
"""
apriltag_map_localizer.py (fixed, robust)

功能:
- 读取 field/layout yaml (map->tag poses)
- 订阅 /tf (tf2_msgs/msg/TFMessage) 中 apriltag 发布的 tag transforms
- 基于每个 tag 的已知 map->tag 和 apriltag 发布的 cam->tag 反推 camera 在 map 下的 pose:
    T_map_cam = T_map_tag * inv(T_cam_tag)
- 多个 tag 同时可用时做简单融合（位置平均，四元数平均）
- 发布 map->camera Transform（tf broadcaster）
- 发布 geometry_msgs/PoseStamped -> topic /camera_in_map
- 发布 visualization Marker -> topic /camera_in_map_marker

用法:
  python3 apriltag_map_localizer.py --layout ~/field_layout_inward.yaml
或在 ROS2 package 中:
  ros2 run pv_apriltag apriltag_map_localizer -- --layout ~/field_layout_inward.yaml

注意:
- 需要依赖: rclpy, tf_transformations, numpy, pyyaml, visualization_msgs
"""

import rclpy
from rclpy.node import Node
import yaml
import os
import math
import numpy as np
import sys
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker

# tf_transformations 常用函数
try:
    from tf_transformations import (
        quaternion_matrix,
        quaternion_from_matrix,
        quaternion_multiply,
        quaternion_inverse,
    )
except Exception as e:
    # 如果没有 tf_transformations，退出并提示
    print("ERROR: tf_transformations is required. Install python3-tf-transformations or the package that provides it.")
    raise

# -----------------------------
# Math helpers (quaternions/matrices)
# -----------------------------
def q_to_matrix(q):
    """q: [x,y,z,w] -> 4x4 matrix"""
    return quaternion_matrix(q)

def matrix_to_q_safe(M):
    """
    返回 [x,y,z,w]，兼容 quaternion_from_matrix 返回 tuple 或 ndarray
    """
    q_raw = quaternion_from_matrix(M)
    # q_raw may be tuple or ndarray; convert to list of floats
    return [float(q_raw[0]), float(q_raw[1]), float(q_raw[2]), float(q_raw[3])]

def q_inverse(q):
    return quaternion_inverse(q)

def q_mul(q1, q2):
    return quaternion_multiply(q1, q2)

def transform_to_matrix(translation, quat):
    """
    translation: [x,y,z], quat: [x,y,z,w]
    return 4x4 numpy matrix
    """
    M = q_to_matrix(quat)
    M = np.array(M, dtype=float)
    M[0:3, 3] = np.array(translation, dtype=float)
    return M

def matrix_to_transform_safe(M):
    """
    输入 4x4 numpy 矩阵，返回 (translation_list, quat_list)
    quat_list 保证为 [x,y,z,w] floats
    """
    M = np.array(M, dtype=float)
    t = M[0:3, 3].astype(float).tolist()
    q = matrix_to_q_safe(M)
    return t, q

def normalize_quat_sum(quats):
    """
    简单平均四元数：把四元数向量求和再归一化
    quats: list of [x,y,z,w]
    """
    arr = np.array(quats, dtype=float)
    s = arr.sum(axis=0)
    n = np.linalg.norm(s)
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return (s / n).tolist()

# -----------------------------
# Layout loader
# -----------------------------
def load_layout(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Layout file not found: {path}")
    data = yaml.safe_load(open(path, 'r'))
    out = {}
    for k, v in data.items():
        if isinstance(v, dict):
            frame = v.get('frame', f"tag{k}")
            pos = v.get('position', [0.0, 0.0, 0.0])
            ori = v.get('orientation', [0.0, 0.0, 0.0, 1.0])
            out[str(frame)] = {
                'position': [float(pos[0]), float(pos[1]), float(pos[2] if len(pos) > 2 else 0.0)],
                'orientation': [float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])]
            }
        else:
            frame = f"tag{k}"
            out[str(frame)] = {
                'position': [0.0, 0.0, 0.0],
                'orientation': [0.0, 0.0, 0.0, 1.0]
            }
    return out

# -----------------------------
# Node
# -----------------------------
class ApriltagMapLocalizer(Node):
    def __init__(self, layout_path, camera_frame_override=None, publish_rate=10.0):
        super().__init__('apriltag_map_localizer')
        try:
            self.layout = load_layout(layout_path)
        except Exception as e:
            self.get_logger().error(f"Failed to load layout: {e}")
            raise

        self.get_logger().info(f"Loaded {len(self.layout)} tags from {layout_path}")
        self.camera_frame_override = camera_frame_override

        # tf broadcaster
        try:
            from tf2_ros import TransformBroadcaster
            self.tf_broadcaster = TransformBroadcaster(self)
        except Exception:
            self.get_logger().warning("tf2_ros.TransformBroadcaster not available; transform broadcasting will fail if required.")
            self.tf_broadcaster = None

        self.pose_pub = self.create_publisher(PoseStamped, 'camera_in_map', 10)
        self.marker_pub = self.create_publisher(Marker, 'camera_in_map_marker', 10)
        self.tf_sub = self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)

        self.last_camera_transform = None
        self.camera_frame_name = None
        self.timer = self.create_timer(1.0 / float(publish_rate), self.timer_cb)

    def timer_cb(self):
        if self.last_camera_transform is None:
            return
        ts = self.last_camera_transform
        # broadcast
        if self.tf_broadcaster is not None:
            try:
                self.tf_broadcaster.sendTransform(ts)
            except Exception as e:
                self.get_logger().warning(f"Failed to broadcast transform: {e}")
        # publish PoseStamped
        pose = PoseStamped()
        pose.header.stamp = ts.header.stamp
        pose.header.frame_id = ts.header.frame_id
        pose.pose.position.x = ts.transform.translation.x
        pose.pose.position.y = ts.transform.translation.y
        pose.pose.position.z = ts.transform.translation.z
        pose.pose.orientation = ts.transform.rotation
        self.pose_pub.publish(pose)
        # publish marker
        m = Marker()
        m.header = pose.header
        m.ns = "camera_in_map"
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose = pose.pose
        m.scale.x = 0.25
        m.scale.y = 0.06
        m.scale.z = 0.06
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        self.marker_pub.publish(m)

    def tf_callback(self, msg: TFMessage):
        """
        For each TransformStamped in /tf published by apriltag detector:
          - expect parent = camera_frame, child = tag frame (e.g. tag36h11:5)
          - if child in layout, compute candidate T_map_cam = T_map_tag * inv(T_cam_tag)
        Then fuse candidates and publish map->camera transform.
        """
        candidates_trans = []
        candidates_quat = []
        camera_frame_seen = None
        count = 0

        for t in msg.transforms:
            # t: geometry_msgs/TransformStamped
            child = getattr(t, 'child_frame_id', '') or ''
            parent = getattr(t, 'header', None)
            if parent is not None:
                parent = getattr(t.header, 'frame_id', None)
            # normalize string types
            child = str(child)
            parent = str(parent) if parent is not None else None

            # Check child against layout keys; some systems use 'tagXX' or 'tag36h11:ID'
            if child not in self.layout:
                # try common alternative (if child is 'tag36h11:5' but layout keys are 'tag36h11:5' it's fine)
                # else maybe layout used only numeric keys -> try 'tag36h11:<id>'
                # try parse numeric id from child
                try:
                    import re
                    m = re.search(r'(\d+)', child)
                    if m:
                        cid = m.group(1)
                        alt = f"tag36h11:{cid}"
                        if alt in self.layout:
                            child = alt
                        else:
                            # maybe layout keys use just numeric string
                            if cid in self.layout:
                                child = cid
                            else:
                                continue
                    else:
                        continue
                except Exception:
                    continue

            # use camera frame if not overridden
            if self.camera_frame_override is None:
                camera_frame_seen = parent

            # get transform values
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            tz = t.transform.translation.z
            qx = t.transform.rotation.x
            qy = t.transform.rotation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w

            # Build matrices
            try:
                M_map_tag = transform_to_matrix(self.layout[child]['position'], self.layout[child]['orientation'])
            except Exception as e:
                self.get_logger().warn(f"Failed to build M_map_tag for {child}: {e}")
                continue

            M_cam_tag = transform_to_matrix([tx, ty, tz], [qx, qy, qz, qw])

            # Invert M_cam_tag safely
            try:
                M_cam_tag_inv = np.linalg.inv(M_cam_tag)
            except Exception as e:
                self.get_logger().warn(f"Failed to invert M_cam_tag for {child}: {e}")
                continue

            # Compute candidate
            M_map_cam = M_map_tag.dot(M_cam_tag_inv)
            try:
                t_map_cam, q_map_cam = matrix_to_transform_safe(M_map_cam)
            except Exception as e:
                self.get_logger().warn(f"Failed to convert matrix->transform for {child}: {e}")
                continue

            candidates_trans.append(t_map_cam)
            candidates_quat.append(q_map_cam)
            count += 1

        if count == 0:
            return

        # fuse: simple average of translation and quaternion
        trans_avg = np.mean(np.array(candidates_trans), axis=0).tolist()
        quat_avg = normalize_quat_sum(candidates_quat)

        cam_frame = self.camera_frame_override if self.camera_frame_override is not None else (camera_frame_seen or 'camera')

        # build TransformStamped
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = 'map'
        ts.child_frame_id = cam_frame
        ts.transform.translation.x = float(trans_avg[0])
        ts.transform.translation.y = float(trans_avg[1])
        ts.transform.translation.z = float(trans_avg[2])
        ts.transform.rotation.x = float(quat_avg[0])
        ts.transform.rotation.y = float(quat_avg[1])
        ts.transform.rotation.z = float(quat_avg[2])
        ts.transform.rotation.w = float(quat_avg[3])

        # store and immediate publish
        self.last_camera_transform = ts
        if self.tf_broadcaster is not None:
            try:
                self.tf_broadcaster.sendTransform(ts)
            except Exception as e:
                self.get_logger().warn(f"Broadcast failed: {e}")

        # publish PoseStamped
        pose = PoseStamped()
        pose.header.stamp = ts.header.stamp
        pose.header.frame_id = 'map'
        pose.pose.position.x = ts.transform.translation.x
        pose.pose.position.y = ts.transform.translation.y
        pose.pose.position.z = ts.transform.translation.z
        pose.pose.orientation = ts.transform.rotation
        self.pose_pub.publish(pose)

        # publish marker
        m = Marker()
        m.header = pose.header
        m.ns = "camera_in_map"
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose = pose.pose
        m.scale.x = 0.25
        m.scale.y = 0.06
        m.scale.z = 0.06
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        self.marker_pub.publish(m)

        self.get_logger().info(f"Published camera_in_map using {count} tags, frame: {cam_frame}")

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', default='src/pv_apriltag/pv_apriltag/config/field_layout.yaml', help='path to layout yaml')
    parser.add_argument('--camera-frame', default=None, help='override camera frame name (optional)')
    parser.add_argument('--rate', default=10.0, type=float, help='publish rate for RViz republish')
    parsed, unknown = parser.parse_known_args()

    rclpy.init(args=sys.argv)
    node = ApriltagMapLocalizer(parsed.layout, camera_frame_override=parsed.camera_frame, publish_rate=parsed.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
