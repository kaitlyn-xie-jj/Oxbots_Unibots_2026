#!/usr/bin/env python3
"""
apriltag_map_localizer.py (per-tag visualization version)

功能:
- 读取 field/layout yaml (map->tag poses)
- 订阅 /tf 中 apriltag 发布的 tag transforms
- 对每个检测到并在 layout 中的 tag 单独计算:
    T_map_cam = T_map_tag * inv(T_cam_tag)
- 为每个 tag 的结果分别:
    - 广播一个 map->camera_by_tag TF (child_frame_id = "<camera>_by_<tag>")
    - 发布 PoseStamped 到 topic /camera_in_map_by_tag
    - 发布 visualization Marker 到 /camera_in_map_marker_by_tag
- 不再对多个 tag 做平均/融合

用法:
  python3 apriltag_map_localizer.py --layout ~/field_layout.yaml
"""
import rclpy
from rclpy.node import Node
import yaml
import os
import math
import numpy as np
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_msgs.msg import TFMessage
from tf_transformations import quaternion_matrix, quaternion_from_matrix
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

# -----------------------------
# Math helpers (quaternions)
# -----------------------------
def q_to_matrix(q):
    # q: [x,y,z,w]
    M4 = quaternion_matrix(q)  # returns 4x4
    return M4

def transform_to_matrix(translation, quat):
    # translation: [x,y,z], quat: [x,y,z,w]
    M = q_to_matrix(quat)
    M[0:3,3] = np.array(translation, dtype=float)
    return M

def matrix_to_transform(M):
    """
    Convert 4x4 matrix to (translation_list, quaternion_list).
    quaternion_from_matrix may return tuple/array; normalize to list safely.
    """
    t = M[0:3, 3].tolist()
    q_raw = quaternion_from_matrix(M)
    # quaternion_from_matrix may return tuple, list or numpy array.
    # convert robustly to list of floats.
    try:
        q = list(map(float, q_raw))
    except Exception:
        # fallback: try np.asarray
        import numpy as _np
        q = _np.asarray(q_raw, dtype=float).tolist()
    return t, q

# -----------------------------
# Layout loader
# -----------------------------
def load_layout(path):
    data = yaml.safe_load(open(os.path.expanduser(path), 'r'))
    out = {}
    for k,v in data.items():
        if isinstance(v, dict):
            frame = v.get('frame', f"tag{k}")
            pos = v.get('position', [0.0,0.0,0.0])
            ori = v.get('orientation', [0.0,0.0,0.0,1.0])
            out[str(frame)] = {
                'position': [float(pos[0]), float(pos[1]), float(pos[2] if len(pos)>2 else 0.0)],
                'orientation': [float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])]
            }
        else:
            frame = f"tag{k}"
            out[str(frame)] = {
                'position': [0.0,0.0,0.0],
                'orientation': [0.0,0.0,0.0,1.0]
            }
    return out

# -----------------------------
# Node
# -----------------------------
class ApriltagMapLocalizer(Node):
    def __init__(self, layout_path, camera_frame_override=None, publish_rate=10.0):
        super().__init__('apriltag_map_localizer')
        self.layout = load_layout(os.path.expanduser(layout_path))
        self.get_logger().info(f"Loaded {len(self.layout)} tags from {layout_path}")
        self.camera_frame_override = camera_frame_override

        # Tf publisher
        self.tf_broadcaster = TransformBroadcaster(self)

        # single publisher topics for per-tag results
        self.pose_pub = self.create_publisher(PoseStamped, 'camera_in_map_by_tag', 10)
        # we will publish marker per tag with unique ns "camera_in_map_by_tag"
        self.marker_pub = self.create_publisher(Marker, 'camera_in_map_marker_by_tag', 10)

        # subscribe to /tf
        self.tf_sub = self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)

        # store last per-tag transforms for timer republish
        # dict: child_tag_frame -> TransformStamped
        self.last_camera_transforms = {}

        # timer to republish last poses at given rate (helps RViz)
        self.timer = self.create_timer(1.0/float(publish_rate), self.timer_cb)

    def timer_cb(self):
        # republish all last camera transforms (one per tag) so RViz keeps them visible
        for tag_frame, ts in self.last_camera_transforms.items():
            # broadcast TF
            self.tf_broadcaster.sendTransform(ts)
            # publish PoseStamped
            pose = PoseStamped()
            pose.header = ts.header
            pose.header.frame_id = 'map'
            pose.pose.position.x = ts.transform.translation.x
            pose.pose.position.y = ts.transform.translation.y
            pose.pose.position.z = ts.transform.translation.z
            pose.pose.orientation = ts.transform.rotation
            # include tag frame in header stamp? keep as map
            self.pose_pub.publish(pose)
            # Marker
            m = Marker()
            m.header = pose.header
            m.ns = "camera_in_map_by_tag"
            # choose id deterministically from tag_frame string
            m.id = abs(hash(tag_frame)) % 10000
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose = pose.pose
            m.scale.x = 0.2
            m.scale.y = 0.06
            m.scale.z = 0.06
            # color by tag (hash to rgb)
            h = abs(hash(tag_frame)) % 360
            # simple hue->rgb mapping (approx)
            r = ((h % 180) / 180.0)
            g = (((h+60) % 180) / 180.0)
            b = (((h+120) % 180) / 180.0)
            m.color.r = float(r)
            m.color.g = float(g)
            m.color.b = float(b)
            m.color.a = 1.0
            self.marker_pub.publish(m)

    def sanitize_tag_frame_for_name(self, s: str):
        # make a safe name for child_frame_id by replacing characters like ':' with '_'
        return s.replace(':','_').replace('/','_')

    def tf_callback(self, msg: TFMessage):
        """
        For each TransformStamped in /tf where child_frame_id is a tag in layout,
        compute per-tag camera pose in map and publish it separately.
        """
        for t in msg.transforms:
            child = t.child_frame_id
            parent = t.header.frame_id  # expected camera frame (as published by apriltag)
            if child not in self.layout:
                continue

            # If camera frame name overridden, use that for naming child transforms
            cam_frame = self.camera_frame_override if self.camera_frame_override is not None else (parent or 'camera')

            # read transform parent(frame=cam)->child(frame=tag) => this is T_cam_tag
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            tz = t.transform.translation.z
            qx = t.transform.rotation.x
            qy = t.transform.rotation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w

            # build matrices
            M_map_tag = transform_to_matrix(self.layout[child]['position'], self.layout[child]['orientation'])
            M_cam_tag = transform_to_matrix([tx,ty,tz], [qx,qy,qz,qw])

            # invert M_cam_tag
            try:
                M_cam_tag_inv = np.linalg.inv(M_cam_tag)
            except Exception as e:
                self.get_logger().warn(f"inv failed for tag {child}: {e}")
                continue

            # compute M_map_cam = M_map_tag * inv(M_cam_tag)
            M_map_cam = M_map_tag.dot(M_cam_tag_inv)
            t_map_cam, q_map_cam = matrix_to_transform(M_map_cam)

            # build a unique child frame name per tag so we can publish all of them:
            safe_child = f"{cam_frame}_by_{self.sanitize_tag_frame_for_name(child)}"

            ts = TransformStamped()
            ts.header.stamp = self.get_clock().now().to_msg()
            ts.header.frame_id = 'map'
            ts.child_frame_id = safe_child
            ts.transform.translation.x = float(t_map_cam[0])
            ts.transform.translation.y = float(t_map_cam[1])
            ts.transform.translation.z = float(t_map_cam[2])
            ts.transform.rotation.x = float(q_map_cam[0])
            ts.transform.rotation.y = float(q_map_cam[1])
            ts.transform.rotation.z = float(q_map_cam[2])
            ts.transform.rotation.w = float(q_map_cam[3])

            # store last transform for this tag and publish immediately
            self.last_camera_transforms[child] = ts
            self.tf_broadcaster.sendTransform(ts)

            # publish PoseStamped (single topic; consumers can read header.stamp and child key mapping)
            pose = PoseStamped()
            pose.header.stamp = ts.header.stamp
            pose.header.frame_id = 'map'
            # include which tag produced this in the text? can't in header; user can infer via marker color/id
            pose.pose.position.x = ts.transform.translation.x
            pose.pose.position.y = ts.transform.translation.y
            pose.pose.position.z = ts.transform.translation.z
            pose.pose.orientation = ts.transform.rotation
            self.pose_pub.publish(pose)

            # publish Marker with deterministic id/color
            m = Marker()
            m.header = pose.header
            m.ns = "camera_in_map_by_tag"
            m.id = abs(hash(child)) % 10000
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose = pose.pose
            m.scale.x = 0.2
            m.scale.y = 0.06
            m.scale.z = 0.06
            # color from tag string hash
            h = abs(hash(child)) % 360
            r = ((h % 180) / 180.0)
            g = (((h+60) % 180) / 180.0)
            b = (((h+120) % 180) / 180.0)
            m.color.r = float(r)
            m.color.g = float(g)
            m.color.b = float(b)
            m.color.a = 1.0
            self.marker_pub.publish(m)

            # continue loop (we handle all tags independently)

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', default='src/pv_apriltag/pv_apriltag/config/field_layout.yaml', help='path to layout yaml')
    parser.add_argument('--camera-frame', default=None, help='override camera frame name (optional)')
    parser.add_argument('--rate', default=10.0, type=float, help='publish rate for RViz republish')
    parsed, unknown = parser.parse_known_args()

    rclpy.init()
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
