#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pv_sim_bridge.py（增强版）
用途：
  在 Webots 仿真里，不用 PhotonVision，直接用 apriltag_ros 的 TF（相机->标签）、
  配合场地布局（map->tag）和相机外参（base->cam），计算机器人在 map 下的位姿，
  并以统一接口发布：

  发布话题：
    /pv/estimated_robot_pose : geometry_msgs/PoseStamped (frame_id=map)
    /pv/has_target           : std_msgs/Bool
    /pv/target_yaw_deg       : std_msgs/Float32  （相机系到“最佳标签”的水平偏航角，便于对准）
    /pv/robot_yaw_deg        : std_msgs/Float32  （全局 yaw，map 坐标系下，单位°）

  可选：
    /tf  : map -> base_link （将估计位姿写入 TF 树，便于 RViz 查看；默认关闭）

数学关系（核心公式）：
  cam_T_tag   ← apriltag_ros 在 /tf 中发布（父：相机光学坐标系；子：tag36h11:<id>）
  tag_T_cam   = (cam_T_tag)^(-1)
  base_T_cam  ← 参数 robot_to_cam_xyz_rpy（base->cam）
  cam_T_base  = (base_T_cam)^(-1)
  map_T_tag   ← 参数/布局文件（场地里每个标签的绝对位姿）
  map_T_base  = map_T_tag · tag_T_cam · cam_T_base

增强点：
  - 多标签融合：同一帧可见多张标签时，对所有候选 map_T_base 做加权平均（平移线性、旋转四元数 Markley 平均）
  - 选择“最佳标签”用于 /pv/target_yaw_deg（默认：相机距离最近）
  - yaw 一阶低通平滑（避免轻微抖动）
  - 可选发布 TF（map->base_link）
  - 更健壮的 field_layout 参数读取（支持直接字典或 JSON 字符串；保持向后兼容）

注意：
  - 该桥接用于仿真。上实机请使用 PhotonVision + pv_node（接口一致）；
    下游（规划/控制）始终只订阅 /pv/* 即可。
"""

from typing import Dict, Tuple, Optional, List
import math
import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.parameter import Parameter

from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped, TransformStamped

import tf_transformations  # 若系统无：sudo apt update && sudo apt install ros-${ROS_DISTRO}-tf-transformations
from tf2_ros import Buffer, TransformListener, TransformBroadcaster


# -------------------- 线性代数工具（基于 tf_transformations + numpy） --------------------

def quat_to_mat(qx, qy, qz, qw):
    """四元数 -> 4x4 齐次旋转矩阵（平移为 0）。"""
    return tf_transformations.quaternion_matrix([qx, qy, qz, qw])

def mat_to_quat(T):
    """4x4 齐次矩阵 -> 四元数 (qx,qy,qz,qw)。"""
    q = tf_transformations.quaternion_from_matrix(T)
    return q[0], q[1], q[2], q[3]

def xyzrpy_to_mat(x, y, z, rr, rp, ry):
    """位姿参数 -> 4x4 齐次变换；R = Rz(ry) * Ry(rp) * Rx(rr)。"""
    Rx = tf_transformations.euler_matrix(rr, 0, 0)
    Ry = tf_transformations.euler_matrix(0, rp, 0)
    Rz = tf_transformations.euler_matrix(0, 0, ry)
    R = tf_transformations.identity_matrix()
    R = tf_transformations.concatenate_matrices(R, Rz, Ry, Rx)
    T = R.copy()
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T

def invert(T):
    """4x4 齐次变换求逆。"""
    return tf_transformations.inverse_matrix(T)

def yaw_from_T(T):
    """从 4x4 旋转矩阵提取 Z 轴 yaw（弧度）。"""
    R = T[:3, :3]
    return math.atan2(R[1, 0], R[0, 0])

def quat_markley_average(quats: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """
    四元数 Markley 加权平均（参考常用姿态平均方法）。
    输入：
      quats  : N×4 数组列表，每项为 [qx, qy, qz, qw]
      weights: N 权重（非负），函数内部会归一化
    返回：
      q_avg : 归一化后的平均四元数 [qx,qy,qz,qw]，规范化为 qw>=0
    """
    w = np.array(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(quats), dtype=float)
    w = w / (w.sum() + 1e-12)

    # 累积 M = Σ w_i * q_i * q_i^T
    M = np.zeros((4, 4), dtype=float)
    for qi, wi in zip(quats, w):
        q = np.array(qi, dtype=float)
        # 统一符号（避免 q 与 -q 抵消）：强制 qw>=0
        if q[3] < 0:
            q = -q
        q = q / (np.linalg.norm(q) + 1e-12)
        M += wi * np.outer(q, q)

    # 最大特征值对应的特征向量即平均四元数
    eigvals, eigvecs = np.linalg.eigh(M)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    if q_avg[3] < 0:
        q_avg = -q_avg
    q_avg = q_avg / (np.linalg.norm(q_avg) + 1e-12)
    return q_avg  # [qx,qy,qz,qw]


# -------------------- 主节点 --------------------

class PVSimBridge(Node):
    def __init__(self):
        super().__init__("pv_sim_bridge")

        # ========== 参数（可在 launch / params.yaml 设置） ==========
        self.declare_parameter("camera_optical_frame", "camera_color_optical_frame")
        self.declare_parameter("robot_to_cam_xyz_rpy", [0., 0., 0., 0., 0., 0.])  # base->cam
        self.declare_parameter("field_layout", "")          # { "tag36h11:ID": [x,y,z,roll,pitch,yaw], ... }
        self.declare_parameter("min_tags_for_fusion", 1)    # N>=1；小于 N 则退化为“使用最佳标签”
        self.declare_parameter("yaw_lpf_alpha", 0.3)        # 0~1；越小越平滑
        self.declare_parameter("publish_tf", False)         # 是否广播 map->base_link

        # 读取基础参数
        self.cam_frame: str = self.get_parameter("camera_optical_frame").get_parameter_value().string_value
        self.robot_to_cam = list(self.get_parameter("robot_to_cam_xyz_rpy").get_parameter_value().double_array_value)
        self.min_tags = int(self.get_parameter("min_tags_for_fusion").get_parameter_value().integer_value)
        self.alpha = float(self.get_parameter("yaw_lpf_alpha").get_parameter_value().double_value)
        self.publish_tf = bool(self.get_parameter("publish_tf").get_parameter_value().bool_value)

        # 读取场地布局（兼容多种入参形式）
        self.field_layout: Dict[str, list] = self._load_field_layout_param()

        # 外参 base->cam 转矩阵；随后取逆得到 cam->base
        if len(self.robot_to_cam) != 6:
            self.get_logger().warn("[pv_sim_bridge] robot_to_cam_xyz_rpy 长度非法，重置为 0。")
            self.robot_to_cam = [0, 0, 0, 0, 0, 0]
        B_T_C = xyzrpy_to_mat(*self.robot_to_cam)  # base->cam
        self.C_T_B = invert(B_T_C)                 # cam->base

        # TF 监听器（用于查 cam->tag）
        self.tf_buffer = Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # 可选 TF 广播器（map->base_link）
        self.tf_br = TransformBroadcaster(self) if self.publish_tf else None

        # 发布者
        self.pub_has  = self.create_publisher(Bool, "/pv/has_target", 10)
        self.pub_yaw  = self.create_publisher(Float32, "/pv/target_yaw_deg", 10)
        self.pub_pose = self.create_publisher(PoseStamped, "/pv/estimated_robot_pose", 10)
        self.pub_ryaw = self.create_publisher(Float32, "/pv/robot_yaw_deg", 10)

        # 定时器：50 Hz
        self.prev_yaw_deg = None  # yaw 低通滤波器内部状态
        self.timer = self.create_timer(0.02, self.tick)

        self.get_logger().info(f"[pv_sim_bridge] ready. cam_frame={self.cam_frame}, tags={len(self.field_layout)}")

    # ---------- 参数解析：支持 dict 或 JSON 字符串 ----------
    def _load_field_layout_param(self) -> Dict[str, list]:
        """
        从参数 field_layout 读取 JSON 字符串并解析为字典：
        field_layout: '{"tag36h11:0":[x,y,z,rr,rp,ry], ...}'
        """
        p: Parameter = self.get_parameter("field_layout")
        s = p.get_parameter_value().string_value  # 一定按字符串读

        if not s or not s.strip():
            self.get_logger().warn("[pv_sim_bridge] field_layout 为空字符串，将使用空字典。")
            return {}

        try:
            d = json.loads(s)
            if isinstance(d, dict):
                return d
            else:
                self.get_logger().warn("[pv_sim_bridge] field_layout JSON 解析结果不是 dict，将使用空字典。")
                return {}
        except Exception as e:
            self.get_logger().warn(f"[pv_sim_bridge] field_layout JSON 解析失败：{e}，将使用空字典。")
            return {}

    # ---------- TF 查询与矩阵转换 ----------
    def _lookup_cam_T_tag(self, tag_frame: str) -> Optional[TransformStamped]:
        """查询 cam->tag 的 TF；apriltag_ros 会发布该变换。"""
        try:
            return self.tf_buffer.lookup_transform(
                self.cam_frame, tag_frame, rclpy.time.Time())
        except Exception:
            return None

    def _msg_to_mat(self, tfmsg: TransformStamped):
        """TransformStamped → 4x4 齐次矩阵。"""
        t = tfmsg.transform.translation
        q = tfmsg.transform.rotation
        T = quat_to_mat(q.x, q.y, q.z, q.w)
        T[0, 3], T[1, 3], T[2, 3] = t.x, t.y, t.z
        return T

    # ---------- 主循环 ----------
    def tick(self):
        # 1) 收集“当前可见”的标签（在 TF 中能查到 cam->tag 的）
        visible: List[Tuple[str, TransformStamped]] = []
        for tag_name in self.field_layout.keys():
            tfmsg = self._lookup_cam_T_tag(tag_name)
            if tfmsg is not None:
                visible.append((tag_name, tfmsg))

        # 发布是否有目标
        self.pub_has.publish(Bool(data=bool(visible)))
        if not visible:
            self.get_logger().debug("[pv_sim_bridge] no visible tags this tick")
            return

        # 2) 选择一个“最佳标签”（用于 /pv/target_yaw_deg）
        #    策略：相机到标签的距离越近越好
        def cam_to_tag_distance(tfmsg: TransformStamped) -> float:
            T = self._msg_to_mat(tfmsg)
            t = T[:3, 3]
            return float(np.linalg.norm(t))
        best_name, best_tf = min(visible, key=lambda it: cam_to_tag_distance(it[1]))
        cam_T_tag_best = self._msg_to_mat(best_tf)
        # 相机系水平偏航近似：atan2(X, Z)
        yaw_cam_deg = math.degrees(math.atan2(cam_T_tag_best[0, 3], cam_T_tag_best[2, 3] + 1e-9))
        self.pub_yaw.publish(Float32(data=float(yaw_cam_deg)))

        # 3) 对每个可见标签，计算一个候选的 map_T_base
        map_T_bases: List[np.ndarray] = []
        weights: List[float] = []
        for tag_name, tfmsg in visible:
            cam_T_tag = self._msg_to_mat(tfmsg)
            tag_T_cam = invert(cam_T_tag)

            layout = self.field_layout.get(tag_name, None)
            if layout is None or len(layout) != 6:
                self.get_logger().warn(f"[pv_sim_bridge] 布局缺失或格式错误：{tag_name}")
                continue
            x, y, z, rr, rp, ry = [float(v) for v in layout]
            map_T_tag = xyzrpy_to_mat(x, y, z, rr, rp, ry)

            # map_T_base = map_T_tag · tag_T_cam · cam_T_base
            map_T_base = map_T_tag @ tag_T_cam @ self.C_T_B
            map_T_bases.append(map_T_base)

            # 权重：距离越近权重越大（~ 1/r^2）
            r = cam_to_tag_distance(tfmsg) + 1e-6
            weights.append(1.0 / (r * r))

        if not map_T_bases:
            # 罕见：可见但都不在布局中
            return

        # 4) 多标签融合（不足 min_tags 则退回用“最佳标签”）
        if len(map_T_bases) < self.min_tags:
            T_avg = map_T_bases[0]
        else:
            # 4.1 平移加权平均
            ps = np.stack([T[:3, 3] for T in map_T_bases], axis=0)  # Nx3
            w = np.array(weights, dtype=float)
            w = w / (w.sum() + 1e-12)
            p_avg = (w[:, None] * ps).sum(axis=0)

            # 4.2 旋转四元数 Markley 平均
            qs = []
            for T in map_T_bases:
                qx, qy, qz, qw = mat_to_quat(T)
                qs.append(np.array([qx, qy, qz, qw], dtype=float))
            q_avg = quat_markley_average(qs, list(w))  # [qx,qy,qz,qw]

            # 4.3 组装平均变换
            T_avg = tf_transformations.identity_matrix()
            T_avg[:3, :3] = quat_to_mat(q_avg[0], q_avg[1], q_avg[2], q_avg[3])[:3, :3]
            T_avg[0, 3], T_avg[1, 3], T_avg[2, 3] = float(p_avg[0]), float(p_avg[1]), float(p_avg[2])

        # 5) 发布 PoseStamped（map）
        now = self.get_clock().now().to_msg()
        qx, qy, qz, qw = mat_to_quat(T_avg)
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = now
        ps.pose.position.x = float(T_avg[0, 3])
        ps.pose.position.y = float(T_avg[1, 3])
        ps.pose.position.z = float(T_avg[2, 3])
        ps.pose.orientation.x = float(qx)
        ps.pose.orientation.y = float(qy)
        ps.pose.orientation.z = float(qz)
        ps.pose.orientation.w = float(qw)
        self.pub_pose.publish(ps)

        # 6) 发布全局 yaw（°），先平滑再转换到 0~360
        yaw_deg = math.degrees(yaw_from_T(T_avg))  # -180 ~ 180

        if self.prev_yaw_deg is None:
            yaw_smoothed = yaw_deg
        else:
            # 先算和上一帧的包角误差，避免跨 ±180 时突变
            err = ((yaw_deg - self.prev_yaw_deg + 180.0) % 360.0) - 180.0
            yaw_smoothed = self.prev_yaw_deg + self.alpha * err

        self.prev_yaw_deg = yaw_smoothed

        # 对外统一输出 0~360°
        yaw_0_360 = yaw_smoothed % 360.0
        if yaw_0_360 < 0.0:
            yaw_0_360 += 360.0

        self.pub_ryaw.publish(Float32(data=float(yaw_0_360)))


        # 7) 可选：把 map->base_link 写入 TF（用于 RViz 直接看 TF 树）
        if self.tf_br is not None:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = "map"
            t.child_frame_id = "base_link"
            t.transform.translation.x = ps.pose.position.x
            t.transform.translation.y = ps.pose.position.y
            t.transform.translation.z = ps.pose.position.z
            t.transform.rotation.x = ps.pose.orientation.x
            t.transform.rotation.y = ps.pose.orientation.y
            t.transform.rotation.z = ps.pose.orientation.z
            t.transform.rotation.w = ps.pose.orientation.w
            self.tf_br.sendTransform(t)


def main():
    rclpy.init()
    node = PVSimBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
