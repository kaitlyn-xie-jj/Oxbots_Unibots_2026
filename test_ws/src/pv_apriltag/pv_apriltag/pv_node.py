#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PhotonVision + PhotonLib (Python) ROS 2 node
# Publishes:
#   /pv/estimated_robot_pose   geometry_msgs/PoseStamped (frame_id=map)
#   /pv/has_target             std_msgs/Bool
#   /pv/target_yaw_deg         std_msgs/Float32  (best target yaw in degrees)
#
# Params:
#   camera_name (string)         : PV UI 相机昵称
#   nt_server (string)           : NT4 服务器地址，默认 127.0.0.1
#   use_estimator (bool)         : 是否启用 PhotonPoseEstimator（多标签融合）
#   pose_strategy (string)       : PoseStrategy，默认 MULTI_TAG_PNP_ON_COPROCESSOR
#   publish_tf (bool)            : 是否把 map->base_link 写到 /tf，默认 False
#   robot_to_cam_xyz_rpy (double[6]) : 机器人到相机外参 [x y z roll pitch yaw] (m, rad)
#   field_layout (string)        : （可选）自定义赛场布局

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

# --- PhotonLib (Python) ---
HAS_ESTIMATOR = True
try:
    from photonlibpy import PhotonCamera, PhotonPoseEstimator, PoseStrategy
except Exception:
    from photonlibpy import PhotonCamera
    HAS_ESTIMATOR = False

# --- NT4 connection (PhotonLib uses default instance) ---
from ntcore import NetworkTableInstance


def rpy_to_quat(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5);  sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5);   sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


class PVNode(Node):
    def __init__(self):
        super().__init__("pv_node")

        # ---------------- Parameters ----------------
        self.declare_parameter("camera_name", "USB_Cam")
        self.declare_parameter("nt_server", "127.0.0.1")
        self.declare_parameter("use_estimator", True)
        self.declare_parameter("pose_strategy", "MULTI_TAG_PNP_ON_COPROCESSOR")
        self.declare_parameter("publish_tf", False)
        self.declare_parameter("robot_to_cam_xyz_rpy", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("field_layout", "")

        self._camera_name = self.get_parameter("camera_name").get_parameter_value().string_value
        self._nt_server = self.get_parameter("nt_server").get_parameter_value().string_value
        self._use_estimator = self.get_parameter("use_estimator").get_parameter_value().bool_value
        self._pose_strategy = self.get_parameter("pose_strategy").get_parameter_value().string_value
        self._publish_tf = self.get_parameter("publish_tf").get_parameter_value().bool_value
        self._robot_to_cam = self.get_parameter("robot_to_cam_xyz_rpy").get_parameter_value().double_array_value

        # ---------------- NT4 ----------------
        self._nt = NetworkTableInstance.getDefault()
        self._nt.startClient4("ros2_photonlib_client")
        self._nt.setServer(self._nt_server)

        # ---------------- PhotonLib camera ----------------
        self._cam = PhotonCamera(self._camera_name)

        # ---------------- Estimator (optional) ----------------
        self._estimator = None
        if self._use_estimator and HAS_ESTIMATOR:
            strategy = getattr(PoseStrategy, self._pose_strategy, PoseStrategy.MULTI_TAG_PNP_ON_COPROCESSOR)
            if len(self._robot_to_cam) == 6:
                tx, ty, tz, rr, rp, ry = self._robot_to_cam
            else:
                tx = ty = tz = rr = rp = ry = 0.0

            try:
                from wpimath.geometry import Transform3d, Translation3d, Rotation3d
                robot_to_cam = Transform3d(Translation3d(tx, ty, tz), Rotation3d(rr, rp, ry))
            except Exception:
                robot_to_cam = None

            field_layout = None  # 如需自定义赛场布局，可在此构造
            try:
                self._estimator = PhotonPoseEstimator(
                    field_layout=field_layout,
                    pose_strategy=strategy,
                    camera=self._cam,
                    robot_to_cam=robot_to_cam
                )
                self.get_logger().info("[pv_node] PhotonPoseEstimator enabled.")
            except Exception as e:
                self.get_logger().warn(f"[pv_node] Estimator init failed, fallback to MultiTag result. {e}")
                self._estimator = None
        elif self._use_estimator and not HAS_ESTIMATOR:
            self.get_logger().warn("[pv_node] photonlibpy without PhotonPoseEstimator; will fallback to MultiTag result.")

        # ---------------- Publishers & TF ----------------
        self.pub_has = self.create_publisher(Bool, "/pv/has_target", 10)
        self.pub_yaw = self.create_publisher(Float32, "/pv/target_yaw_deg", 10)
        self.pub_pose = self.create_publisher(PoseStamped, "/pv/estimated_robot_pose", 10)
        self.tf_br = TransformBroadcaster(self) if self._publish_tf else None

        # 定时器：50 Hz
        self.timer = self.create_timer(0.02, self.tick)

    def _publish_pose(self, x, y, z, qx, qy, qz, qw, stamp):
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = stamp
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = float(z)
        ps.pose.orientation.x = float(qx)
        ps.pose.orientation.y = float(qy)
        ps.pose.orientation.z = float(qz)
        ps.pose.orientation.w = float(qw)
        self.pub_pose.publish(ps)

        if self.tf_br:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = "map"
            t.child_frame_id = "base_link"
            t.transform.translation.x = float(x)
            t.transform.translation.y = float(y)
            t.transform.translation.z = float(z)
            t.transform.rotation.x = float(qx)
            t.transform.rotation.y = float(qy)
            t.transform.rotation.z = float(qz)
            t.transform.rotation.w = float(qw)
            self.tf_br.sendTransform(t)

    def tick(self):
        # 批量消费未读结果（避免丢帧/重复）
        for res in self._cam.getAllUnreadResults():
            # has_target
            has = Bool()
            try:
                has.data = bool(res.hasTargets())
            except Exception:
                has.data = bool(getattr(res, "hasTargets", False))
            self.pub_has.publish(has)

            # best target yaw (deg)
            yaw_msg = Float32()
            try:
                best = res.getBestTarget() if res.hasTargets() else None
                yaw_msg.data = float(best.getYaw()) if best else 0.0
            except Exception:
                yaw_msg.data = 0.0
            self.pub_yaw.publish(yaw_msg)

            # 位姿：优先 Estimator；否则 PV MultiTag
            stamp = self.get_clock().now().to_msg()

            if self._estimator is not None:
                try:
                    est = self._estimator.update(res)
                    if est and getattr(est, "estimatedPose", None) is not None:
                        p3 = est.estimatedPose
                        x = getattr(p3, "x", getattr(getattr(p3, "translation", None), "x", 0.0))
                        y = getattr(p3, "y", getattr(getattr(p3, "translation", None), "y", 0.0))
                        z = getattr(p3, "z", getattr(getattr(p3, "translation", None), "z", 0.0))
                        qw = getattr(p3, "qw", getattr(getattr(p3, "rotation", None), "w", 1.0))
                        qx = getattr(p3, "qx", getattr(getattr(p3, "rotation", None), "x", 0.0))
                        qy = getattr(p3, "qy", getattr(getattr(p3, "rotation", None), "y", 0.0))
                        qz = getattr(p3, "qz", getattr(getattr(p3, "rotation", None), "z", 0.0))
                        self._publish_pose(x, y, z, qx, qy, qz, qw, stamp)
                        continue
                except Exception as e:
                    self.get_logger().warn(f"[pv_node] Estimator update failed: {e}")

            # fallback: PV Multi-Tag
            try:
                mt = res.getMultiTagResult()
            except Exception:
                mt = None

            if mt and getattr(mt, "estimatedPose", None) is not None:
                p3 = mt.estimatedPose
                x = getattr(p3, "x", getattr(getattr(p3, "translation", None), "x", 0.0))
                y = getattr(p3, "y", getattr(getattr(p3, "translation", None), "y", 0.0))
                z = getattr(p3, "z", getattr(getattr(p3, "translation", None), "z", 0.0))
                qw = getattr(p3, "qw", getattr(getattr(p3, "rotation", None), "w", 1.0))
                qx = getattr(p3, "qx", getattr(getattr(p3, "rotation", None), "x", 0.0))
                qy = getattr(p3, "qy", getattr(getattr(p3, "rotation", None), "y", 0.0))
                qz = getattr(p3, "qz", getattr(getattr(p3, "rotation", None), "z", 0.0))
                self._publish_pose(x, y, z, qx, qy, qz, qw, stamp)
                return

def main():
    rclpy.init()
    node = PVNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
