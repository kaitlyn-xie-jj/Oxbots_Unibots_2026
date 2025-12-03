#!/usr/bin/env python3
"""
apriltag_pnp_localizer.py

- Subscribes to:
    /camera/image_raw       (sensor_msgs/Image)
    /camera/camera_info     (sensor_msgs/CameraInfo)
    /tag_detections         (apriltag_msgs/AprilTagDetectionArray)  <- topic name from ros-apriltag, adjust if different
- Loads layout.yaml (map -> tag poses)
- For each detection of configured tag (or any tag if not specified), does:
    - extract pixel corners
    - subpixel refine (cv2.cornerSubPix)
    - solvePnP (cv2.solvePnP) + refine (cv2.solvePnPRefineLM)
    - build T_cam_tag, invert to get T_tag_cam if needed, compute T_map_cam = T_map_tag * inv(T_cam_tag)
- Publishes:
    - map -> camera transform (tf broadcaster)
    - /camera_in_map (geometry_msgs/PoseStamped)
    - /camera_in_map_marker (visualization_msgs/Marker) for RViz
Usage:
    chmod +x apriltag_pnp_localizer.py
    python3 apriltag_pnp_localizer.py --layout ~/field_layout_inward.yaml --tag-frame tag36h11:5

Notes:
- Ensure apriltag node publishes AprilTagDetectionArray with 'corners' and 'id' fields.
- Replace topic names if your ros-apriltag uses different names.
- Requires opencv (cv2) and cv_bridge.
"""
import rclpy
from rclpy.node import Node
import os, yaml, math, time
import numpy as np
import cv2

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, PoseStamped
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_matrix, quaternion_matrix
from cv_bridge import CvBridge

# Try import apriltag detection message (name may vary)
try:
    from apriltag_msgs.msg import AprilTagDetectionArray
except Exception:
    # fallback name: some distros/packages use different msg paths
    try:
        from apriltag_ros.msg import AprilTagDetectionArray as AprilTagDetectionArray
    except Exception:
        AprilTagDetectionArray = None

# -------- helpers ----------
def load_layout(path):
    data = yaml.safe_load(open(os.path.expanduser(path),'r'))
    out = {}
    for k,v in data.items():
        if isinstance(v, dict):
            frame = v.get('frame', f"tag{k}")
            pos = v.get('position', [0.0,0.0,0.0])
            ori = v.get('orientation', [0.0,0.0,0.0,1.0])
            out[str(frame)] = {'position':[float(pos[0]),float(pos[1]),float(pos[2] if len(pos)>2 else 0.0)],
                               'orientation':[float(ori[0]),float(ori[1]),float(ori[2]),float(ori[3])]}
    return out

def transform_to_matrix(translation, quat):
    M = quaternion_matrix(quat)  # 4x4
    M[0:3,3] = np.array(translation, dtype=float)
    return M

def matrix_to_transform(M):
    t = M[0:3,3].tolist()
    q = quaternion_from_matrix(M).tolist()
    return t, q

# -------- node ----------
class ApriltagPnPLocalizer(Node):
    def __init__(self, layout_path, tag_frame=None, tag_size=0.1,
                 img_topic='/camera/image_raw', info_topic='/camera/camera_info',
                 det_topic='/tag_detections', publish_rate=10.0):
        super().__init__('apriltag_pnp_localizer')
        self.get_logger().info("Starting apriltag_pnp_localizer...")
        self.layout = load_layout(layout_path)
        self.get_logger().info(f"Loaded {len(self.layout)} tags from {layout_path}")
        self.tag_frame = tag_frame  # if None, process first detection found
        self.tag_size = float(tag_size)

        # cv bridge
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_image_time = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # tf broadcaster + publishers
        self.tf_b = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, 'camera_in_map', 10)
        self.marker_pub = self.create_publisher(Marker, 'camera_in_map_marker', 10)

        # subs
        self.create_subscription(Image, img_topic, self.image_cb, 5)
        self.create_subscription(CameraInfo, info_topic, self.caminfo_cb, 5)

        if AprilTagDetectionArray is None:
            self.get_logger().error("Cannot find AprilTagDetectionArray message type. Install apriltag_msgs or adjust import.")
        else:
            self.create_subscription(AprilTagDetectionArray, det_topic, self.detections_cb, 10)
            self.get_logger().info(f"Subscribed to detection topic: {det_topic}")

        self.last_tf = None
        self.timer = self.create_timer(1.0/float(publish_rate), self.timer_cb)

    def caminfo_cb(self, msg: CameraInfo):
        # store intrinsic
        K = np.array(msg.k, dtype=float).reshape(3,3)
        D = np.array(msg.d, dtype=float) if len(msg.d)>0 else np.zeros((5,))
        self.camera_matrix = K
        self.dist_coeffs = D
        # only need to set once normally
        #self.get_logger().info("CameraInfo received and stored.")

    def image_cb(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failure: {e}")
            return
        self.latest_image = cv_img
        self.latest_image_time = msg.header.stamp

    def timer_cb(self):
        # republish last computed tf/pose to keep RViz up to date
        if self.last_tf is None:
            return
        ts = self.last_tf
        self.tf_b.sendTransform(ts)
        # PoseStamped
        pose = PoseStamped()
        pose.header = ts.header
        pose.header.frame_id = 'map'
        pose.pose.position.x = ts.transform.translation.x
        pose.pose.position.y = ts.transform.translation.y
        pose.pose.position.z = ts.transform.translation.z
        pose.pose.orientation = ts.transform.rotation
        self.pose_pub.publish(pose)
        # marker
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
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        self.marker_pub.publish(m)

    def detections_cb(self, msg):
        if self.camera_matrix is None or self.latest_image is None:
            # not ready
            return

        # loop detections and find target
        for det in msg.detections:
            # try to read id - different msg definitions exist
            tag_id = None
            # many apriltag messages contain det.id as list or array
            if hasattr(det, 'id') and det.id:
                try:
                    # id could be list [id]
                    tag_id = det.id[0] if isinstance(det.id, (list, tuple)) else int(det.id)
                except Exception:
                    tag_id = None
            # build frame name
            frame_name = None
            if tag_id is not None:
                frame_name = f"tag36h11:{tag_id}"
            else:
                # some msgs include det.frame_id
                frame_name = getattr(det, 'frame_id', None)

            # if user requested a specific tag_frame but this isn't it -> continue
            if self.tag_frame is not None and frame_name != self.tag_frame:
                continue

            # find pixel corners; support different field types
            img_pts = []
            if hasattr(det, 'corners') and det.corners:
                # corners could be list of Point32 or list of lists
                for c in det.corners:
                    if hasattr(c, 'x'):
                        img_pts.append([float(c.x), float(c.y)])
                    else:
                        # assume iterable [x,y]
                        img_pts.append([float(c[0]), float(c[1])])
            else:
                # try homography fallback: some msgs store homography (3x3) - not implemented here
                self.get_logger().warn("Detection has no corners; skipping (need corners for PnP).")
                continue

            if len(img_pts) != 4:
                self.get_logger().warn(f"Detection corners length !=4 ({len(img_pts)}) - skipping")
                continue

            img_pts = np.array(img_pts, dtype=np.float32).reshape(-1,2)

            # subpixel refine
            gray = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)
            # ensure img_pts are int for cornerSubPix initial locations
            win = (5,5)
            zero = (-1,-1)
            try:
                cv2.cornerSubPix(gray, img_pts, win, zero,
                                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            except Exception as e:
                # fallback: continue with raw points
                self.get_logger().debug(f"cornerSubPix failed: {e}")

            # model points: order MUST match detector's corner order; common order: (tl,tr,br,bl)
            s = float(self.tag_size)
            model_pts = np.array([[-s/2, -s/2, 0],
                                  [ s/2, -s/2, 0],
                                  [ s/2,  s/2, 0],
                                  [-s/2,  s/2, 0]], dtype=np.float64)

            # Solve PnP
            try:
                ok, rvec, tvec = cv2.solvePnP(model_pts, img_pts, self.camera_matrix, self.dist_coeffs,
                                              flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    self.get_logger().warn("solvePnP returned failure for tag {}".format(frame_name))
                    continue
            except Exception as e:
                self.get_logger().warn(f"solvePnP exception: {e}")
                continue

            # refine with LM if available
            try:
                cv2.solvePnPRefineLM(model_pts, img_pts, self.camera_matrix, self.dist_coeffs, rvec, tvec)
            except Exception:
                # not fatal
                pass

            # compute reprojection error (debug)
            proj, _ = cv2.projectPoints(model_pts, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            proj = proj.reshape(-1,2)
            repro = np.linalg.norm(proj - img_pts, axis=1).mean()

            # build M_cam_tag (tag pose in camera frame)
            R_cam_tag, _ = cv2.Rodrigues(rvec)
            M_cam_tag = np.eye(4, dtype=float)
            M_cam_tag[0:3,0:3] = R_cam_tag
            M_cam_tag[0:3,3] = tvec.flatten()

            # invert: we want T_cam_tag -> inv to get T_tag_cam if required by formula
            try:
                M_cam_tag_inv = np.linalg.inv(M_cam_tag)
            except Exception as e:
                self.get_logger().warn(f"matrix invert fail: {e}")
                continue

            # find T_map_tag from layout
            if frame_name not in self.layout:
                self.get_logger().warn(f"{frame_name} not in layout.yaml; skipping")
                continue
            M_map_tag = transform_to_matrix(self.layout[frame_name]['position'],
                                           self.layout[frame_name]['orientation'])

            # compute T_map_cam = T_map_tag * inv(T_cam_tag)
            M_map_cam = M_map_tag.dot(M_cam_tag_inv)
            t_map_cam, q_map_cam = matrix_to_transform(M_map_cam)

            # create TransformStamped
            ts = TransformStamped()
            ts.header.stamp = self.get_clock().now().to_msg()
            ts.header.frame_id = 'map'
            # child_frame: set to camera frame seen by apriltag node (header.frame_id), fallback 'camera'
            cam_frame = getattr(det, 'pose', None)
            # more reliable: apriltag TF header.frame_id is camera frame; here we can't access it from detection msg,
            # so use a generic name or let user set override if needed
            cam_frame_name = 'camera'  # user can remap later if needed
            ts.child_frame_id = cam_frame_name
            ts.transform.translation.x = float(t_map_cam[0])
            ts.transform.translation.y = float(t_map_cam[1])
            ts.transform.translation.z = float(t_map_cam[2])
            ts.transform.rotation.x = float(q_map_cam[0])
            ts.transform.rotation.y = float(q_map_cam[1])
            ts.transform.rotation.z = float(q_map_cam[2])
            ts.transform.rotation.w = float(q_map_cam[3])

            # store and publish
            self.last_tf = ts
            self.tf_b.sendTransform(ts)

            # publish PoseStamped
            pose = PoseStamped()
            pose.header = ts.header
            pose.header.frame_id = 'map'
            pose.pose.position.x = ts.transform.translation.x
            pose.pose.position.y = ts.transform.translation.y
            pose.pose.position.z = ts.transform.translation.z
            pose.pose.orientation = ts.transform.rotation
            self.pose_pub.publish(pose)

            # marker
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
            m.color.a = 1.0
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            self.marker_pub.publish(m)

            # log reprojection error for debugging
            self.get_logger().debug(f"Tag {frame_name} PnP reproj error: {repro:.3f} px")

            # single-tag mode: process first matching detection and return
            return

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', default='src/pv_apriltag/pv_apriltag/config/field_layout.yaml', help='layout yaml path')
    parser.add_argument('--tag-frame', default=None, help='only localize using this tag frame e.g. tag36h11:5')
    parser.add_argument('--tag-size', default=0.10, type=float, help='tag size in meters (edge length)')
    parser.add_argument('--image-topic', default='/camera/image_raw')
    parser.add_argument('--info-topic', default='/camera/camera_info')
    parser.add_argument('--det-topic', default='/detections')
    parser.add_argument('--rate', default=10.0, type=float)
    args, unknown = parser.parse_known_args()

    rclpy.init()
    node = ApriltagPnPLocalizer(args.layout, tag_frame=args.tag_frame, tag_size=args.tag_size,
                                img_topic=args.image_topic, info_topic=args.info_topic,
                                det_topic=args.det_topic, publish_rate=args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
