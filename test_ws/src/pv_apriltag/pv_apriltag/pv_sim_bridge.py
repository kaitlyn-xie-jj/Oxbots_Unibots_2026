#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pv_sim_bridge.py
Compute camera pose in map from apriltag TF and field layout, publish for RViz check.

Publishes:
  /pv/camera_pose    : geometry_msgs/PoseStamped (frame_id='map')
  /pv/has_target     : std_msgs/Bool
  /pv/target_yaw_deg : std_msgs/Float32

Params (ros2 param or --ros-args -p ...):
  camera_frame         (str)  default "camera_color_optical_frame"
  field_layout         (str)  JSON string: { "tag_frame": [x,y,z,roll,pitch,yaw], ... }
  field_layout_type    (str)  "map_to_tag" (default) or "tag_to_map"
  min_tags_for_fusion  (int)  default 1
  rate_hz              (int)  default 20
  publish_tf           (bool) default True  # publish map->camera_frame
  publish_pose         (bool) default True  # publish /pv/camera_pose
  yaw_lpf_alpha        (float) default 0.3
"""
from typing import Dict, List, Optional, Tuple
import math, json, sys
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped, TransformStamped

import tf_transformations
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

# ---------- helpers ----------
def quat_to_mat(qx, qy, qz, qw):
    return tf_transformations.quaternion_matrix([qx, qy, qz, qw])

def mat_to_quat(T):
    q = tf_transformations.quaternion_from_matrix(T)
    return q[0], q[1], q[2], q[3]

def xyzrpy_to_mat(x, y, z, rr, rp, ry):
    Rx = tf_transformations.euler_matrix(rr, 0, 0)
    Ry = tf_transformations.euler_matrix(0, rp, 0)
    Rz = tf_transformations.euler_matrix(0, 0, ry)
    R = tf_transformations.concatenate_matrices(Rz, Ry, Rx)
    T = R.copy()
    T[0,3], T[1,3], T[2,3] = x, y, z
    return T

def invert(T):
    return tf_transformations.inverse_matrix(T)

def yaw_from_T(T):
    R = T[:3,:3]
    return math.atan2(R[1,0], R[0,0])

def quat_markley_average(quats: List[np.ndarray], weights: List[float]) -> np.ndarray:
    w = np.array(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(quats), dtype=float)
    w = w / (w.sum() + 1e-12)
    M = np.zeros((4,4), dtype=float)
    for qi, wi in zip(quats, w):
        q = np.array(qi, dtype=float)
        if q[3] < 0:
            q = -q
        q = q / (np.linalg.norm(q) + 1e-12)
        M += wi * np.outer(q, q)
    eigvals, eigvecs = np.linalg.eigh(M)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    if q_avg[3] < 0:
        q_avg = -q_avg
    q_avg = q_avg / (np.linalg.norm(q_avg) + 1e-12)
    return q_avg

# ---------- node ----------
class PVSimBridge(Node):
    def __init__(self):
        super().__init__('pv_sim_bridge')

        # params
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('field_layout', '')
        self.declare_parameter('field_layout_type', 'map_to_tag')  # or 'tag_to_map'
        self.declare_parameter('min_tags_for_fusion', 1)
        self.declare_parameter('rate_hz', 20)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('publish_pose', True)
        self.declare_parameter('yaw_lpf_alpha', 0.3)

        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.field_layout_raw = self.get_parameter('field_layout').get_parameter_value().string_value
        self.field_layout_type = self.get_parameter('field_layout_type').get_parameter_value().string_value
        self.min_tags = int(self.get_parameter('min_tags_for_fusion').get_parameter_value().integer_value)
        self.rate_hz = int(self.get_parameter('rate_hz').get_parameter_value().integer_value)
        self.publish_tf = bool(self.get_parameter('publish_tf').get_parameter_value().bool_value)
        self.publish_pose = bool(self.get_parameter('publish_pose').get_parameter_value().bool_value)
        self.alpha = float(self.get_parameter('yaw_lpf_alpha').get_parameter_value().double_value)

        # parse layout into map->tag matrices (np.array 4x4)
        self.field_layout = self._parse_field_layout(self.field_layout_raw, self.field_layout_type)

        # tf
        self.tf_buffer = Buffer(cache_time=Duration(seconds=4.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_br = TransformBroadcaster(self) if self.publish_tf else None

        # pubs
        self.pub_pose = self.create_publisher(PoseStamped, '/pv/camera_pose', 10) if self.publish_pose else None
        self.pub_has = self.create_publisher(Bool, '/pv/has_target', 10)
        self.pub_yaw = self.create_publisher(Float32, '/pv/target_yaw_deg', 10)

        self.prev_yaw = None

        period = 1.0 / max(1, self.rate_hz)
        self.create_timer(period, self.tick)

        self.get_logger().info(f"[pv_sim_bridge] ready. camera_frame={self.camera_frame}, tags_in_layout={len(self.field_layout)}")

    def _parse_field_layout(self, raw: str, layout_type: str) -> Dict[str, np.ndarray]:
        out = {}
        if not raw or not raw.strip():
            self.get_logger().warn("[pv_sim_bridge] empty field_layout")
            return out
        try:
            d = json.loads(raw)
        except Exception as e:
            self.get_logger().error(f"[pv_sim_bridge] failed parse field_layout JSON: {e}")
            return out
        for tag_name, vals in d.items():
            try:
                if not isinstance(vals, (list, tuple)) or len(vals) != 6:
                    self.get_logger().warn(f"[pv_sim_bridge] invalid layout for {tag_name}")
                    continue
                x,y,z,rr,rp,ry = [float(v) for v in vals]
                # auto-detect degrees
                if any(abs(a) > 2*math.pi for a in (rr,rp,ry)):
                    self.get_logger().warn(f"[pv_sim_bridge] detected degree angles for {tag_name}; converting to radians")
                    rr, rp, ry = math.radians(rr), math.radians(rp), math.radians(ry)
                T = xyzrpy_to_mat(x,y,z,rr,rp,ry)
                if layout_type == 'tag_to_map':
                    # input is tag->map, invert to map->tag
                    T = invert(T)
                out[tag_name] = T
            except Exception as e:
                self.get_logger().warn(f"[pv_sim_bridge] error parsing {tag_name}: {e}")
        return out

    def _lookup_cam_T_tag(self, tag_frame: str) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(self.camera_frame, tag_frame, rclpy.time.Time())
        except Exception:
            return None

    def _tfmsg_to_mat(self, tfmsg: TransformStamped) -> np.ndarray:
        t = tfmsg.transform.translation
        q = tfmsg.transform.rotation
        T = quat_to_mat(q.x, q.y, q.z, q.w)
        T[0,3], T[1,3], T[2,3] = t.x, t.y, t.z
        return T

    def tick(self):
        visible = []
        for tag_frame in self.field_layout.keys():
            tfmsg = self._lookup_cam_T_tag(tag_frame)
            if tfmsg is not None:
                visible.append((tag_frame, tfmsg))
        self.pub_has.publish(Bool(data=bool(visible)))
        if not visible:
            return

        # choose nearest for target yaw
        def cam_to_tag_dist(tfmsg):
            T = self._tfmsg_to_mat(tfmsg)
            return float(np.linalg.norm(T[:3,3]))
        best_tag, best_tf = min(visible, key=lambda it: cam_to_tag_dist(it[1]))
        cam_T_tag_best = self._tfmsg_to_mat(best_tf)
        yaw_cam_deg = math.degrees(math.atan2(cam_T_tag_best[0,3], cam_T_tag_best[2,3] + 1e-9))
        # smoothing
        if self.prev_yaw is None:
            yaw_sm = yaw_cam_deg
        else:
            err = ((yaw_cam_deg - self.prev_yaw + 180.0) % 360.0) - 180.0
            yaw_sm = self.prev_yaw + self.alpha * err
        self.prev_yaw = yaw_sm
        yaw_out = (yaw_sm + 360.0) % 360.0
        self.pub_yaw.publish(Float32(data=float(yaw_out)))

        # compute map->camera candidates
        candidates = []
        weights = []
        for tag_frame, tfmsg in visible:
            cam_T_tag = self._tfmsg_to_mat(tfmsg)   # camera -> tag
            tag_T_cam = invert(cam_T_tag)           # tag -> camera
            map_T_tag = self.field_layout.get(tag_frame, None)  # map -> tag
            if map_T_tag is None:
                continue
            map_T_cam = map_T_tag @ tag_T_cam
            candidates.append(map_T_cam)
            r = cam_to_tag_dist(tfmsg) + 1e-9
            weights.append(1.0/(r*r))
        if not candidates:
            return

        # fusion
        if len(candidates) < self.min_tags:
            T_cam = candidates[0]
        else:
            ps = np.stack([T[:3,3] for T in candidates], axis=0)
            w = np.array(weights, dtype=float)
            w = w / (w.sum() + 1e-12)
            p_avg = (w[:,None]*ps).sum(axis=0)
            qs = []
            for T in candidates:
                qx,qy,qz,qw = mat_to_quat(T)
                qs.append(np.array([qx,qy,qz,qw], dtype=float))
            q_avg = quat_markley_average(qs, list(w))
            T_cam = tf_transformations.identity_matrix()
            T_cam[:3,:3] = quat_to_mat(q_avg[0], q_avg[1], q_avg[2], q_avg[3])[:3,:3]
            T_cam[0,3], T_cam[1,3], T_cam[2,3] = float(p_avg[0]), float(p_avg[1]), float(p_avg[2])

        # publish PoseStamped
        now = self.get_clock().now().to_msg()
        qx,qy,qz,qw = mat_to_quat(T_cam)
        if self.pub_pose is not None:
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.header.stamp = now
            ps.pose.position.x = float(T_cam[0,3])
            ps.pose.position.y = float(T_cam[1,3])
            ps.pose.position.z = float(T_cam[2,3])
            ps.pose.orientation.x = float(qx)
            ps.pose.orientation.y = float(qy)
            ps.pose.orientation.z = float(qz)
            ps.pose.orientation.w = float(qw)
            self.pub_pose.publish(ps)

        # optionally publish TF map->camera_frame (for RViz check)
        if self.tf_br is not None and self.publish_tf:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = 'map'
            t.child_frame_id = self.camera_frame
            t.transform.translation.x = float(T_cam[0,3])
            t.transform.translation.y = float(T_cam[1,3])
            t.transform.translation.z = float(T_cam[2,3])
            t.transform.rotation.x = float(qx)
            t.transform.rotation.y = float(qy)
            t.transform.rotation.z = float(qz)
            t.transform.rotation.w = float(qw)
            self.tf_br.sendTransform(t)

        # debug compare with ground-truth if exists: map->camera_frame and map->base_link
        try:
            real_cam = self.tf_buffer.lookup_transform('map', self.camera_frame, rclpy.time.Time())
            realT = self._tfmsg_to_mat(real_cam)
            dx,dy,dz = T_cam[0,3]-realT[0,3], T_cam[1,3]-realT[1,3], T_cam[2,3]-realT[2,3]
            dpos = math.sqrt(dx*dx + dy*dy + dz*dz)
            y1 = math.degrees(yaw_from_T(T_cam)); y2 = math.degrees(yaw_from_T(realT))
            ydiff = ((y1 - y2 + 180) % 360) - 180
            self.get_logger().info(f"[pv_sim_bridge debug] cam pos diff to TF {self.camera_frame}: {dpos:.3f} m; yaw diff {ydiff:.2f} deg")
        except Exception:
            self.get_logger().debug("[pv_sim_bridge debug] no ground-truth map->camera_frame for comparison")

        try:
            real_base = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            realB = self._tfmsg_to_mat(real_base)
            # if you want to compare base, you'd need base->cam transform (external)
            self.get_logger().debug("[pv_sim_bridge debug] found ground-truth map->base_link (you may compare separately)")
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = PVSimBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
