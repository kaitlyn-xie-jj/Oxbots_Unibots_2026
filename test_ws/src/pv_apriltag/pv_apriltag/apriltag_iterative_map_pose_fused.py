#!/usr/bin/env python3
"""
apriltag_iterative_map_pose_fused.py

每帧对多 tag 的 map->camera 解做加权 SE(3) 融合，并对融合结果做指数平滑（low-pass）。
同时发布 TF map->camera_iterative、PoseStamped，以及 RViz 用的 MarkerArray 和 PoseArray。

参数（可通过 ROS2 参数覆盖）：
 - cam_info_topic (string) default: /camera/camera_info
 - detections_topic (string) default: /detections
 - tag_size_m (double) default: 0.08
 - field_layout (string) default: src/pv_apriltag/pv_apriltag/config/field_layout.yaml
 - reproj_thresh_px (double) default: 3.0
 - cam_frame (string) default: camera_iterative
 - map_frame (string) default: map
 - fuse_by_distance (bool) default: True (在权重中加入距衰减)
 - alpha (double) default: 0.3 (指数平滑系数，0-1，越小越平滑)
 - min_weight (double) default: 1e-6
 - marker_lifetime (double) default: 0.6 (秒)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped, PoseStamped, PoseArray, Pose
import tf2_ros
import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as R
import os
import math
from visualization_msgs.msg import Marker, MarkerArray
from apriltag_msgs.msg import AprilTagDetectionArray

# -------------------- 默认配置（可在 node 参数里覆盖） --------------------
CAM_INFO_TOPIC = '/camera/camera_info'
AP_DETECTIONS_TOPIC = '/detections'
TAG_SIZE_M = 0.08
DEFAULT_FIELD_LAYOUT = 'src/pv_apriltag/pv_apriltag/config/field_layout.yaml'
REPROJ_THRESH_PX = 3.0
CAM_FRAME = 'camera_iterative'
MAP_FRAME = 'map'
FUSE_BY_DISTANCE = True
ALPHA = 0.3
MIN_WEIGHT = 1e-6
MARKER_LIFETIME = 0.6
# ---------------------------------------------------------------------------

def obj_pts_for_tag(size_m):
    s = float(size_m)
    return np.array([
        [-s/2, -s/2, 0.0],
        [ s/2, -s/2, 0.0],
        [ s/2,  s/2, 0.0],
        [-s/2,  s/2, 0.0],
    ], dtype=np.float64)

def load_field_layout(path):
    """
    读取 field_layout.yaml，返回 dict: {tag_id: 4x4 T_map_tag}
    支持顶层 numeric keys 格式（你给出的样例）。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"field_layout not found: {path}")
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    tag_poses = {}
    if isinstance(data, dict):
        # numeric keys?
        for k, v in data.items():
            try:
                tid = int(k)
            except:
                continue
            if not isinstance(v, dict):
                continue
            pos = v.get('position', None)
            ori = v.get('orientation', None)
            if pos is None or ori is None:
                continue
            try:
                tx, ty, tz = float(pos[0]), float(pos[1]), float(pos[2])
                qx, qy, qz, qw = float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])
            except Exception:
                continue
            # normalize quaternion and convert
            norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
            if norm == 0:
                qx,qy,qz,qw = 0.0,0.0,0.0,1.0
            else:
                qx,qy,qz,qw = qx/norm, qy/norm, qz/norm, qw/norm
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3,:3] = rot
            T[0,3] = tx; T[1,3] = ty; T[2,3] = tz
            tag_poses[tid] = T
    return tag_poses

def quat_avg_weighted(quats, weights):
    """
    简单的加权四元数平均：
    - quats: Nx4 array in (x,y,z,w) format
    - weights: N weights (sum not necessarily 1)
    返回归一化后的 (x,y,z,w)
    方法：加权线性和然后归一化（对近似旋转有效）
    """
    q = np.array(quats, dtype=np.float64)
    w = np.array(weights, dtype=np.float64).reshape(-1,1)
    qw = (q * w).sum(axis=0)
    norm = np.linalg.norm(qw)
    if norm < 1e-12:
        return np.array([0,0,0,1], dtype=np.float64)
    return (qw / norm)

def transform_from_rt(Rm, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = Rm
    T[:3,3] = t
    return T

def decompose_T(T):
    t = T[:3,3]
    rotm = T[:3,:3]
    q = R.from_matrix(rotm).as_quat()  # x,y,z,w
    return t, q

def compose_T_from_tq(t, q):
    rotm = R.from_quat(q).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = rotm
    T[:3,3] = t
    return T

class ApriltagIterativeMapPoseFused(Node):
    def __init__(self):
        super().__init__('apriltag_iterative_map_pose_fused')
        # parameters
        self.declare_parameter('cam_info_topic', CAM_INFO_TOPIC)
        self.declare_parameter('detections_topic', AP_DETECTIONS_TOPIC)
        self.declare_parameter('tag_size_m', float(TAG_SIZE_M))
        self.declare_parameter('field_layout', DEFAULT_FIELD_LAYOUT)
        self.declare_parameter('reproj_thresh_px', float(REPROJ_THRESH_PX))
        self.declare_parameter('cam_frame', CAM_FRAME)
        self.declare_parameter('map_frame', MAP_FRAME)
        self.declare_parameter('fuse_by_distance', bool(FUSE_BY_DISTANCE))
        self.declare_parameter('alpha', float(ALPHA))
        self.declare_parameter('min_weight', float(MIN_WEIGHT))
        self.declare_parameter('marker_lifetime', float(MARKER_LIFETIME))

        self.cam_info_topic = self.get_parameter('cam_info_topic').get_parameter_value().string_value
        self.detections_topic = self.get_parameter('detections_topic').get_parameter_value().string_value
        self.tag_size = self.get_parameter('tag_size_m').get_parameter_value().double_value
        self.field_layout_path = self.get_parameter('field_layout').get_parameter_value().string_value
        self.reproj_thresh = self.get_parameter('reproj_thresh_px').get_parameter_value().double_value
        self.cam_frame = self.get_parameter('cam_frame').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.fuse_by_distance = self.get_parameter('fuse_by_distance').get_parameter_value().bool_value
        self.alpha = self.get_parameter('alpha').get_parameter_value().double_value
        self.min_weight = self.get_parameter('min_weight').get_parameter_value().double_value
        self.marker_lifetime = self.get_parameter('marker_lifetime').get_parameter_value().double_value

        self.get_logger().info(f'Using field_layout: {self.field_layout_path}')
        try:
            self.tag_map = load_field_layout(self.field_layout_path)
            self.get_logger().info(f'Loaded {len(self.tag_map)} tags from field_layout.')
        except Exception as e:
            self.get_logger().error(f'Failed to load field_layout: {e}')
            self.tag_map = {}

        self.K = None
        self.D = None
        self.objp = obj_pts_for_tag(self.tag_size)

        # subscribers & pubs
        self.sub_ci = self.create_subscription(CameraInfo, self.cam_info_topic, self.cb_caminfo, 10)
        try:
            self.sub_det = self.create_subscription(AprilTagDetectionArray, self.detections_topic, self.cb_detections, 10)
        except Exception:
            self.get_logger().error('apriltag_msgs AprilTagDetectionArray topic type not available; ensure package installed.')
            raise

        self.tf_b = tf2_ros.TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, '/apriltag_iterative/pose_stamped', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/apriltag_iterative/markers', 10)
        self.posearray_pub = self.create_publisher(PoseArray, '/apriltag_iterative/per_tag_poses', 10)

        # smoothing state (smoothed pose in map frame)
        self.smoothed_t = None   # 3-vector
        self.smoothed_q = None   # quat x,y,z,w

    def cb_caminfo(self, msg: CameraInfo):
        if self.K is None:
            K = np.array(msg.k).reshape(3,3).astype(np.float64)
            D = np.array(msg.d).astype(np.float64) if len(msg.d)>0 else np.zeros((5,), dtype=np.float64)
            self.K = K
            self.D = D
            self.get_logger().info(f'Got CameraInfo: fx={K[0,0]:.3f}, fy={K[1,1]:.3f}, cx={K[0,2]:.3f}, cy={K[1,2]:.3f}')

    def robust_parse_detection(self, det):
        tid = None
        image_pts = None
        if hasattr(det, 'id'):
            try:
                tid = int(det.id) if isinstance(det.id, int) else int(det.id[0])
            except Exception:
                try:
                    tid = int(det.tag_id)
                except Exception:
                    tid = None
        elif hasattr(det, 'tag_id'):
            try:
                tid = int(det.tag_id)
            except:
                tid = None
        if hasattr(det, 'corners'):
            corners = det.corners
            try:
                image_pts = np.array([[float(c.x), float(c.y)] for c in corners], dtype=np.float64)
            except Exception:
                try:
                    image_pts = np.array([[float(c[0]), float(c[1])] for c in corners], dtype=np.float64)
                except Exception:
                    try:
                        flat = [float(x) for x in corners]
                        image_pts = np.array(flat, dtype=np.float64).reshape(-1,2)
                    except Exception:
                        image_pts = None
        return tid, image_pts

    def cb_detections(self, msg):
        if self.K is None:
            self.get_logger().warn('No CameraInfo yet; skipping detections')
            return
        detections = msg.detections if hasattr(msg, 'detections') else []
        per_tag_mapcams = []  # list of dicts {id, T_map_cam, mean_err, dist}
        for det in detections:
            tid, image_pts = self.robust_parse_detection(det)
            if image_pts is None or image_pts.shape[0] < 4:
                continue

            # solvePnP iterative
            try:
                ok, rvec, tvec = cv2.solvePnP(self.objp, image_pts, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
            except Exception as e:
                self.get_logger().error(f'solvePnP exception for tag {tid}: {e}')
                continue
            if not ok:
                continue

            proj, _ = cv2.projectPoints(self.objp, rvec, tvec, self.K, self.D)
            proj = proj.reshape(-1,2)
            errs = np.linalg.norm(proj - image_pts.reshape(-1,2), axis=1)
            mean_err = float(errs.mean())
            if mean_err > self.reproj_thresh:
                self.get_logger().warn(f'Tag {tid}: large reproj error {mean_err:.3f}px')
                # still keep but with low weight? here we drop
                continue

            Rm, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3)
            T_tag_cam = transform_from_rt(Rm, t)

            if tid not in self.tag_map:
                self.get_logger().debug(f'Tag {tid} not in field_layout; skipping for map fusion')
                continue
            T_map_tag = self.tag_map[tid]
            # correct composition
            T_map_cam = T_map_tag.dot(np.linalg.inv(T_tag_cam))

            # distance (camera to tag) in tag frame
            cam_pos_tag = -Rm.T.dot(t)
            dist = float(np.linalg.norm(cam_pos_tag))

            per_tag_mapcams.append({'id': tid, 'T': T_map_cam, 'mean_err': mean_err, 'dist': dist})

        if len(per_tag_mapcams) == 0:
            self.get_logger().debug('No usable tag detections this frame')
            return

        # compute weights
        weights = []
        quats = []
        trans = []
        for p in per_tag_mapcams:
            # base weight: inverse reproj error
            w = 1.0 / (p['mean_err'] + 1e-6)
            # optional distance penalty (farther tags slightly penalized)
            if self.fuse_by_distance:
                w = w / (1.0 + 0.05 * p['dist'])
            w = max(w, self.min_weight)
            weights.append(w)
            t_p, q_p = decompose_T(p['T'])
            trans.append(t_p)
            quats.append(q_p)
        weights = np.array(weights, dtype=np.float64)
        weights = weights / (weights.sum() + 1e-12)

        # weighted translation
        trans_arr = np.array(trans)  # N x 3
        fused_t = (trans_arr.T * weights).sum(axis=1)

        # weighted quaternion average
        fused_q = quat_avg_weighted(np.array(quats), weights)

        # compose fused transform
        T_fused = compose_T_from_tq(fused_t, fused_q)

        # smoothing with exponential moving average
        if self.smoothed_t is None:
            self.smoothed_t = fused_t.copy()
            self.smoothed_q = fused_q.copy()
        else:
            # translation EMA
            self.smoothed_t = self.alpha * fused_t + (1.0 - self.alpha) * self.smoothed_t
            # quaternion EMA: simple slerp-like approx by linear interp then normalize to avoid flips
            # but better: ensure unified sign
            q_prev = self.smoothed_q.copy()
            # make sure same hemisphere
            if np.dot(q_prev, fused_q) < 0:
                fused_q = -fused_q
            q_lin = self.alpha * fused_q + (1.0 - self.alpha) * q_prev
            q_lin = q_lin / (np.linalg.norm(q_lin) + 1e-12)
            self.smoothed_q = q_lin

        T_smoothed = compose_T_from_tq(self.smoothed_t, self.smoothed_q)

        # publish TF map -> cam_frame
        tstamp = TransformStamped()
        tstamp.header.stamp = self.get_clock().now().to_msg()
        tstamp.header.frame_id = self.map_frame
        tstamp.child_frame_id = self.cam_frame
        tstamp.transform.translation.x = float(self.smoothed_t[0])
        tstamp.transform.translation.y = float(self.smoothed_t[1])
        tstamp.transform.translation.z = float(self.smoothed_t[2])
        tstamp.transform.rotation.x = float(self.smoothed_q[0])
        tstamp.transform.rotation.y = float(self.smoothed_q[1])
        tstamp.transform.rotation.z = float(self.smoothed_q[2])
        tstamp.transform.rotation.w = float(self.smoothed_q[3])
        self.tf_b.sendTransform(tstamp)

        # publish PoseStamped (map frame)
        pose_msg = PoseStamped()
        pose_msg.header = tstamp.header
        pose_msg.header.frame_id = self.map_frame
        pose_msg.pose.position.x = float(self.smoothed_t[0])
        pose_msg.pose.position.y = float(self.smoothed_t[1])
        pose_msg.pose.position.z = float(self.smoothed_t[2])
        pose_msg.pose.orientation.x = float(self.smoothed_q[0])
        pose_msg.pose.orientation.y = float(self.smoothed_q[1])
        pose_msg.pose.orientation.z = float(self.smoothed_q[2])
        pose_msg.pose.orientation.w = float(self.smoothed_q[3])
        self.pose_pub.publish(pose_msg)

        # publish RViz markers & per-tag pose array
        self.publish_markers_and_posearray(per_tag_mapcams, T_fused, T_smoothed, tstamp.header)

        self.get_logger().info(f'Published fused map->{self.cam_frame}: reproj_mean={float((np.array([p["mean_err"] for p in per_tag_mapcams])*weights).sum()):.3f}px tags={len(per_tag_mapcams)}')

    def publish_markers_and_posearray(self, per_tag_mapcams, T_fused, T_smoothed, header):
        """
        发布 MarkerArray 与 PoseArray: 
         - camera cube & axes at smoothed pose
         - arrows tag->camera for each tag
         - PoseArray containing per-tag map->camera (useful for debug)
        """
        ma = MarkerArray()
        pa = PoseArray()
        pa.header = header
        pa.header.frame_id = self.map_frame

        # camera cube marker
        cam_marker = Marker()
        cam_marker.header = header
        cam_marker.ns = 'camera_fused'
        cam_marker.id = 0
        cam_marker.type = Marker.CUBE
        cam_marker.action = Marker.ADD
        cam_marker.pose.position.x = float(self.smoothed_t[0])
        cam_marker.pose.position.y = float(self.smoothed_t[1])
        cam_marker.pose.position.z = float(self.smoothed_t[2])
        cam_marker.pose.orientation.x = float(self.smoothed_q[0])
        cam_marker.pose.orientation.y = float(self.smoothed_q[1])
        cam_marker.pose.orientation.z = float(self.smoothed_q[2])
        cam_marker.pose.orientation.w = float(self.smoothed_q[3])
        cam_marker.scale.x = 0.08
        cam_marker.scale.y = 0.05
        cam_marker.scale.z = 0.04
        cam_marker.color.r = 0.8; cam_marker.color.g = 0.2; cam_marker.color.b = 0.2; cam_marker.color.a = 0.9
        cam_marker.lifetime.sec = int(self.marker_lifetime)
        cam_marker.lifetime.nanosec = int((self.marker_lifetime - int(self.marker_lifetime)) * 1e9)
        ma.markers.append(cam_marker)

        # per-tag arrows & pose entries
        mid = 10
        for p in per_tag_mapcams:
            T = p['T']
            # arrow from tag position (from tag_map) to camera
            tid = p['id']
            if tid not in self.tag_map:
                continue
            T_map_tag = self.tag_map[tid]
            tag_pos = T_map_tag[:3,3]
            cam_pos = T[:3,3]

            # arrow
            m = Marker()
            m.header = header
            m.ns = 'tag_to_cam'
            m.id = mid; mid += 1
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.points = []
            from geometry_msgs.msg import Point
            m.points.append(Point(x=float(tag_pos[0]), y=float(tag_pos[1]), z=float(tag_pos[2])))
            m.points.append(Point(x=float(cam_pos[0]), y=float(cam_pos[1]), z=float(cam_pos[2])))
            m.scale.x = 0.01; m.scale.y = 0.02; m.scale.z = 0.02
            m.color.r = 0.1; m.color.g = 0.6; m.color.b = 0.9; m.color.a = 0.9
            m.lifetime.sec = int(self.marker_lifetime)
            m.lifetime.nanosec = int((self.marker_lifetime - int(self.marker_lifetime)) * 1e9)
            ma.markers.append(m)

            # small sphere at tag
            s = Marker()
            s.header = header
            s.ns = 'tag_pos'
            s.id = mid; mid += 1
            s.type = Marker.SPHERE
            s.action = Marker.ADD
            s.pose.position.x = float(tag_pos[0]); s.pose.position.y = float(tag_pos[1]); s.pose.position.z = float(tag_pos[2]) + 0.02
            s.scale.x = 0.03; s.scale.y = 0.03; s.scale.z = 0.03
            s.color.r = 0.9; s.color.g = 0.4; s.color.b = 0.1; s.color.a = 0.9
            s.lifetime.sec = int(self.marker_lifetime)
            s.lifetime.nanosec = int((self.marker_lifetime - int(self.marker_lifetime)) * 1e9)
            ma.markers.append(s)

            # per-tag pose into PoseArray
            pose = Pose()
            pose.position.x = float(cam_pos[0]); pose.position.y = float(cam_pos[1]); pose.position.z = float(cam_pos[2])
            q = R.from_matrix(T[:3,:3]).as_quat()
            pose.orientation.x = float(q[0]); pose.orientation.y = float(q[1]); pose.orientation.z = float(q[2]); pose.orientation.w = float(q[3])
            pa.poses.append(pose)

        # publish
        try:
            self.marker_pub.publish(ma)
            self.posearray_pub.publish(pa)
        except Exception as e:
            self.get_logger().warn(f'Failed publish markers/posearray: {e}')

def main(args=None):
    rclpy.init(args=args)
    try:
        node = ApriltagIterativeMapPoseFused()
    except Exception as e:
        print(f'Failed to start node: {e}')
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
