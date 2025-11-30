#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo


class EpuckCamSync(Node):
    """
    功能：
      订阅 /epuck/camera/image_raw （Webots 出来的原始图像）
      发布：
        /epuck/camera_sync/image_raw
        /epuck/camera_sync/camera_info

      - CameraInfo 内参固定配置（用参数或默认值）
      - 每帧 image 到时，复制 header 给 CameraInfo
      - image 和 camera_info 的时间戳、frame_id 完全一致
      - 所有 pub/sub 都用 sensor_data QoS，和相机、apriltag_ros 一致
    """

    def __init__(self):
        super().__init__("epuck_cam_sync")

        # -------- 参数：分辨率 & 坐标系 --------
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("frame_id", "epuck_camera_optical_frame")

        # 畸变模型 & 参数（先设为 0）
        self.declare_parameter("distortion_model", "plumb_bob")
        self.declare_parameter("distortion_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])

        # 方式一：直接提供 fx, fy, cx, cy
        self.declare_parameter("fx", 0.0)
        self.declare_parameter("fy", 0.0)
        self.declare_parameter("cx", 0.0)
        self.declare_parameter("cy", 0.0)

        # 方式二：用水平视场角估算焦距
        self.declare_parameter("horizontal_fov_deg", 60.0)

        # 读参数
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.frame_id = self.get_parameter("frame_id").value

        self.distortion_model = self.get_parameter("distortion_model").value
        self.distortion_coeffs = list(self.get_parameter("distortion_coeffs").value)

        fx = float(self.get_parameter("fx").value)
        fy = float(self.get_parameter("fy").value)
        cx = float(self.get_parameter("cx").value)
        cy = float(self.get_parameter("cy").value)
        fov_deg = float(self.get_parameter("horizontal_fov_deg").value)

        if fx <= 0.0:
            fov_rad = math.radians(fov_deg)
            fx = self.width / (2.0 * math.tan(fov_rad / 2.0))
            fy = fx

        if cx <= 0.0:
            cx = self.width / 2.0
        if cy <= 0.0:
            cy = self.height / 2.0

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # -------- 构造 CameraInfo 模板 --------
        self.cam_info_template = CameraInfo()
        self.cam_info_template.width = self.width
        self.cam_info_template.height = self.height
        self.cam_info_template.distortion_model = self.distortion_model
        self.cam_info_template.d = self.distortion_coeffs

        # K 内参矩阵
        self.cam_info_template.k[0] = self.fx
        self.cam_info_template.k[2] = self.cx
        self.cam_info_template.k[4] = self.fy
        self.cam_info_template.k[5] = self.cy
        self.cam_info_template.k[8] = 1.0

        # R 单位阵
        self.cam_info_template.r[0] = 1.0
        self.cam_info_template.r[4] = 1.0
        self.cam_info_template.r[8] = 1.0

        # P 投影矩阵（简单 pinhole）
        self.cam_info_template.p[0] = self.fx
        self.cam_info_template.p[2] = self.cx
        self.cam_info_template.p[5] = self.fy
        self.cam_info_template.p[6] = self.cy
        self.cam_info_template.p[10] = 1.0

        # -------- QoS：统一用 sensor_data --------
        qos = qos_profile_sensor_data

        # 订阅原始图像（Webots）
        self.sub_img = self.create_subscription(
            Image,
            "/epuck/camera/image_raw",
            self.image_callback,
            qos
        )

        # 发布同步后的图像和 CameraInfo
        self.pub_img = self.create_publisher(
            Image,
            "/epuck/camera_sync/image_raw",
            qos
        )
        self.pub_info = self.create_publisher(
            CameraInfo,
            "/epuck/camera_sync/camera_info",
            qos
        )

        self.get_logger().info(
            f"[epuck_cam_sync] started, width={self.width}, height={self.height}, "
            f"fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}"
        )

    def image_callback(self, img: Image):
        # 确保 frame_id 有值
        if not img.header.frame_id:
            img.header.frame_id = self.frame_id

        # 用模板拷贝出一个 CameraInfo，并同步 header
        cam_info = CameraInfo()
        cam_info = self.cam_info_template
        cam_info.header = img.header  # 时间戳 & frame_id 完全一致 ✅

        # 同时发布 image + camera_info
        self.pub_img.publish(img)
        self.pub_info.publish(cam_info)


def main(args=None):
    rclpy.init(args=args)
    node = EpuckCamSync()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
