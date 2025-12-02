#!/usr/bin/env python3
"""
apriltag_iterative_map_pose.py

订阅 apriltag detection + camera_info，使用 OpenCV SOLVEPNP_ITERATIVE 计算 tag->camera，
读取 field_layout.yaml (map->tag) 并发布 map->camera_iterative TF + PoseStamped.

配置参数（可在脚本中修改或用 ROS2 参数覆盖）：
 - CAM_INFO_TOPIC
 - AP_DETECTIONS_TOPIC
 - TAG_SIZE_M
 - FIELD_LAYOUT_PATH
 - REPROJ_THRESH_PX
 - CAM_FRAME
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped, PoseStamped
import tf2_ros
import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as R
import os
import math
import time

# -------------------- 默认配置（如需，可在 node 参数里覆盖） --------------------
CAM_INFO_TOPIC = '/camera/camera_info'
AP_DETECTIONS_TOPIC = '/detections'
TAG_SIZE_M = 0.08
# 默认 field_layout 文件路径（按你的 workspace 相对路径，如果你有不同路径请改）
DEFAULT_FIELD_LAYOUT = 'src/pv_apriltag/pv_apriltag/config/field_layout.yaml'
REPROJ_THRESH_PX = 3.0
CAM_FRAME = 'camera_iterative'
MAP_FRAME = 'map'
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
    更强健的 field_layout 解析：
    支持你的格式（顶层为数字键，value 包含 frame, position, orientation, size）
    以及其它常见形式（list of entries, mapping id->pose）。
    返回 dict: { tag_id (int) : 4x4 numpy T_map_tag }
    """
    import math
    from scipy.spatial.transform import Rotation as R

    if not os.path.exists(path):
        raise FileNotFoundError(f"field_layout not found: {path}")
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    tag_poses = {}

    # Case 1: top-level dict with numeric keys (your file)
    if isinstance(data, dict):
        # Try parse numeric keys first
        all_keys_numeric = all(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in data.keys())
        if all_keys_numeric:
            for k, v in data.items():
                try:
                    tid = int(k)
                except:
                    continue
                # v expected to be dict with position & orientation (or frame etc.)
                pos = v.get('position') if isinstance(v, dict) else None
                ori = v.get('orientation') if isinstance(v, dict) else None
                if pos is None or ori is None:
                    # also allow flatten with keys 'position' as list or 'pose'
                    if isinstance(v.get('position', None), list) and isinstance(v.get('orientation', None), list):
                        pos = v.get('position'); ori = v.get('orientation')
                if pos is None or ori is None:
                    continue
                try:
                    tx,ty,tz = float(pos[0]), float(pos[1]), float(pos[2])
                    qx,qy,qz,qw = float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])
                except Exception:
                    continue
                # normalize quat
                norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
                if norm == 0:
                    qx,qy,qz,qw = 0,0,0,1
                else:
                    qx,qy,qz,qw = qx/norm, qy/norm, qz/norm, qw/norm
                rot = R.from_quat([qx,qy,qz,qw]).as_matrix()
                T = np.eye(4, dtype=np.float64)
                T[:3,:3] = rot
                T[0,3] = tx; T[1,3] = ty; T[2,3] = tz
                tag_poses[tid] = T
            return tag_poses

    # Case 2: list of entries (fallback to previous logic)
    # Try to find entries as list
    entries = []
    if isinstance(data, dict) and 'tags' in data and isinstance(data['tags'], list):
        entries = data['tags']
    elif isinstance(data, list):
        entries = data
    else:
        # try to collect any dict-values that look like tag entries
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and ('position' in v or 'pose' in v or 'orientation' in v):
                    # assume this is an entry
                    e = v.copy()
                    # if key is numeric, try add id
                    try:
                        if isinstance(k, int) or (isinstance(k,str) and k.isdigit()):
                            e['id'] = int(k)
                    except:
                        pass
                    entries.append(e)
    # parse entries list
    for e in entries:
        tid = None
        pose = None
        if isinstance(e, dict):
            if 'id' in e:
                try:
                    tid = int(e['id'])
                except:
                    tid = None
            if 'position' in e and 'orientation' in e:
                pose = {'position': e['position'], 'orientation': e['orientation']}
            elif 'pose' in e and isinstance(e['pose'], list) and len(e['pose'])>=7:
                arr = e['pose']; tid = tid if tid is not None else None
                try:
                    tx,ty,tz = float(arr[0]), float(arr[1]), float(arr[2])
                    qx,qy,qz,qw = float(arr[3]), float(arr[4]), float(arr[5]), float(arr[6])
                except:
                    continue
                rot = R.from_quat([qx,qy,qz,qw]).as_matrix()
                T = np.eye(4, dtype=np.float64)
                T[:3,:3] = rot
                T[0,3] = tx; T[1,3] = ty; T[2,3] = tz
                if tid is not None:
                    tag_poses[tid] = T
                continue
        if tid is None:
            continue
        pos = pose.get('position', {})
        ori = pose.get('orientation', {})
        try:
            tx = float(pos.get('x', pos.get(0,0)))
            ty = float(pos.get('y', pos.get(1,0)))
            tz = float(pos.get('z', pos.get(2,0)))
            qx = float(ori.get('x', 0.0))
            qy = float(ori.get('y', 0.0))
            qz = float(ori.get('z', 0.0))
            qw = float(ori.get('w', 1.0))
        except Exception:
            continue
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm == 0:
            qx,qy,qz,qw = 0,0,0,1
        else:
            qx,qy,qz,qw = qx/norm, qy/norm, qz/norm, qw/norm
        rot = R.from_quat([qx,qy,qz,qw]).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3,:3] = rot
        T[0,3] = tx; T[1,3] = ty; T[2,3] = tz
        tag_poses[tid] = T

    return tag_poses


class ApriltagIterativeMapPose(Node):
    def __init__(self):
        super().__init__('apriltag_iterative_map_pose')
        # parameters
        self.declare_parameter('cam_info_topic', CAM_INFO_TOPIC)
        self.declare_parameter('detections_topic', AP_DETECTIONS_TOPIC)
        self.declare_parameter('tag_size_m', float(TAG_SIZE_M))
        self.declare_parameter('field_layout', DEFAULT_FIELD_LAYOUT)
        self.declare_parameter('reproj_thresh_px', float(REPROJ_THRESH_PX))
        self.declare_parameter('cam_frame', CAM_FRAME)
        self.declare_parameter('map_frame', MAP_FRAME)

        self.cam_info_topic = self.get_parameter('cam_info_topic').get_parameter_value().string_value
        self.detections_topic = self.get_parameter('detections_topic').get_parameter_value().string_value
        self.tag_size = self.get_parameter('tag_size_m').get_parameter_value().double_value
        self.field_layout_path = self.get_parameter('field_layout').get_parameter_value().string_value
        self.reproj_thresh = self.get_parameter('reproj_thresh_px').get_parameter_value().double_value
        self.cam_frame = self.get_parameter('cam_frame').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value

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

        # subs
        self.sub_ci = self.create_subscription(CameraInfo, self.cam_info_topic, self.cb_caminfo, 10)
        # try import apriltag_msgs
        try:
            from apriltag_msgs.msg import AprilTagDetectionArray
            self.using_apriltag_msgs = True
            self.sub_det = self.create_subscription(AprilTagDetectionArray, self.detections_topic, self.cb_detections, 10)
            self.get_logger().info('Subscribed to apriltag_msgs.AprilTagDetectionArray')
        except Exception:
            self.get_logger().error('apriltag_msgs not available; please install or change topic/message type')
            raise

        self.tf_b = tf2_ros.TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, '/apriltag_iterative/pose_stamped', 10)

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
        # id
        if hasattr(det, 'id'):
            try:
                if isinstance(det.id, int):
                    tid = int(det.id)
                else:
                    tid = int(det.id[0])
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
        # corners
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
        for det in detections:
            tid, image_pts = self.robust_parse_detection(det)
            if image_pts is None or image_pts.shape[0] < 4:
                self.get_logger().debug(f'Bad corners for tag {tid}; skipping')
                continue

            # solvePnP iterative (robust)
            try:
                ok, rvec, tvec = cv2.solvePnP(self.objp, image_pts, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
            except Exception as e:
                self.get_logger().error(f'solvePnP exception for tag {tid}: {e}')
                continue
            if not ok:
                self.get_logger().warn(f'solvePnP returned false for tag {tid}')
                continue

            # reprojection check
            proj, _ = cv2.projectPoints(self.objp, rvec, tvec, self.K, self.D)
            proj = proj.reshape(-1,2)
            errs = np.linalg.norm(proj - image_pts.reshape(-1,2), axis=1)
            mean_err = float(errs.mean())
            if mean_err > self.reproj_thresh:
                self.get_logger().warn(f'Tag {tid}: large reproj error {mean_err:.3f}px')

            # construct T_tag_cam from rvec,tvec (OpenCV: object->camera = tag->cam)
            Rm, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3)
            T_tag_cam = np.eye(4, dtype=np.float64)
            T_tag_cam[:3,:3] = Rm
            T_tag_cam[:3,3] = t

            # need map->tag (from field_layout), compute map->camera = map->tag * tag->cam
            if tid not in self.tag_map:
                self.get_logger().warn(f'Tag {tid} not in field_layout; cannot compute map->camera. (Publish skipped)')
                continue
            T_map_tag = self.tag_map[tid]
            T_map_cam = T_map_tag.dot(np.linalg.inv(T_tag_cam))  # 4x4

            # extract translation and quaternion
            trans = T_map_cam[:3,3]
            rotm = T_map_cam[:3,:3]
            quat = R.from_matrix(rotm).as_quat()  # x,y,z,w

            # publish TransformStamped map -> camera_frame
            tstamp = TransformStamped()
            tstamp.header.stamp = self.get_clock().now().to_msg()
            tstamp.header.frame_id = self.map_frame
            tstamp.child_frame_id = self.cam_frame
            tstamp.transform.translation.x = float(trans[0])
            tstamp.transform.translation.y = float(trans[1])
            tstamp.transform.translation.z = float(trans[2])
            tstamp.transform.rotation.x = float(quat[0])
            tstamp.transform.rotation.y = float(quat[1])
            tstamp.transform.rotation.z = float(quat[2])
            tstamp.transform.rotation.w = float(quat[3])
            # broadcast
            self.tf_b.sendTransform(tstamp)

            # also publish a PoseStamped for debugging
            pose_msg = PoseStamped()
            pose_msg.header = tstamp.header
            pose_msg.header.frame_id = self.map_frame
            pose_msg.pose.position.x = float(trans[0])
            pose_msg.pose.position.y = float(trans[1])
            pose_msg.pose.position.z = float(trans[2])
            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])
            self.pose_pub.publish(pose_msg)

            self.get_logger().info(f'Published map->{self.cam_frame} from tag {tid}: reproj={mean_err:.3f}px')

def main(args=None):
    rclpy.init(args=args)
    try:
        node = ApriltagIterativeMapPose()
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
