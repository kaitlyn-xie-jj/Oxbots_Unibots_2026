#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pv_sim_bridge.py — Webots 仿真桥（不依赖 PhotonVision）
功能：
  从 apriltag_ros 的 TF 里读取 “相机→标签 (cam_T_tag)”；
  用场地布局 “map→标签 (map_T_tag)” 和相机外参 “base→相机 (base_T_cam)”，
  计算机器人 “map→base (map_T_base)” 并发布到 /pv/*。
  - /pv/estimated_robot_pose  PoseStamped(frame=map)
  - /pv/has_target            Bool
  - /pv/target_yaw_deg        Float32  （相机系：对准用）
  - /pv/robot_yaw_deg         Float32  （全局 yaw：你要观察的朝向）
  - （可选）/tf               map→base_link

数学：
  tag_T_cam = (cam_T_tag)^(-1)
  cam_T_base = (base_T_cam)^(-1)
  map_T_base = map_T_tag · tag_T_cam · cam_T_base

增强：
  - 支持多标签融合（加权平均：平移线性、旋转四元数 Markley 平均）
  - 简单低通平滑 yaw（alpha 参数）
"""
from typing import Dict, Tuple, Optional, List
import math, json, os, yaml
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

# ---------- 线性代数工具（纯 numpy，无外部依赖） ----------
def xyzrpy_to_mat(x,y,z, rr,rp,ry) -> np.ndarray:
    """位姿参数 -> 4x4 齐次变换；R = Rz(ry)*Ry(rp)*Rx(rr)"""
    cr,sr = math.cos(rr), math.sin(rr)
    cp,sp = math.cos(rp), math.sin(rp)
    cy,sy = math.cos(ry), math.sin(ry)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], float)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], float)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], float)
    R = Rz @ Ry @ Rx
    T = np.eye(4, float); T[:3,:3] = R; T[:3,3] = [x,y,z]; return T

def quat_to_mat(qx,qy,qz,qw) -> np.ndarray:
    n = math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw)+1e-12
    x,y,z,w = qx/n, qy/n, qz/n, qw/n
    xx,yy,zz = x*x,y*y,z*z; xy,xz,yz = x*y,x*z,y*z; wx,wy,wz = w*x,w*y,w*z
    R = np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]], float)
    T = np.eye(4, float); T[:3,:3] = R; return T

def mat_to_quat(T: np.ndarray) -> tuple:
    """返回 (qx,qy,qz,qw)"""
    R = T[:3,:3]; tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr+1.0)*2.0
        w = 0.25*s
        x = (R[2,1]-R[1,2])/s; y = (R[0,2]-R[2,0])/s; z = (R[1,0]-R[0,1])/s
    else:
        i = int(np.argmax([R[0,0],R[1,1],R[2,2]]))
        if i==0:
            s=math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2.0
            w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
        elif i==1:
            s=math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2.0
            w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
        else:
            s=math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2.0
            w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    return (x,y,z,w)

def invert(T: np.ndarray) -> np.ndarray:
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4, float); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t; return Ti

def yaw_from_T(T: np.ndarray) -> float:
    """提取 Z 轴 yaw (rad)，map 右手系：atan2(R[1,0], R[0,0])"""
    R = T[:3,:3]; return math.atan2(R[1,0], R[0,0])

def average_quaternions(quats: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """Markley 四元数平均（抗 ±q），quats: Nx4 [x,y,z,w]"""
    M = np.zeros((4,4), float)
    for q, w in zip(quats, weights):
        q = q / (np.linalg.norm(q)+1e-12)
        M += w * np.outer(q, q)
    eigvals, eigvecs = np.linalg.eigh(M)
    q = eigvecs[:, np.argmax(eigvals)]
    # 归一化与 w>=0 规范
    if q[3] < 0: q = -q
    return q / (np.linalg.norm(q)+1e-12)

# ---------- ROS 节点 ----------
class PVSimBridge(Node):
    def __init__(self):
        super().__init__('pv_sim_bridge')
        # 参数
        self.declare_parameter('camera_optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('robot_to_cam_xyz_rpy', [0.12, 0.0, 0.18, 0.0, 0.0, 0.0])  # base->cam
        self.declare_parameter('field_layout_yaml', 'config/field_layout.yaml')
        self.declare_parameter('min_tags_for_fusion', 1)
        self.declare_parameter('yaw_lpf_alpha', 0.3)      # 低通滤波系数（0~1，越小越平滑）
        self.declare_parameter('publish_tf', False)

        self.cam_frame = self.get_parameter('camera_optical_frame').value
        self.B_T_C = xyzrpy_to_mat(*self.get_parameter('robot_to_cam_xyz_rpy').value)  # base->cam
        self.C_T_B = invert(self.B_T_C)                                                # cam->base
        self.min_tags = int(self.get_parameter('min_tags_for_fusion').value)
        self.alpha = float(self.get_parameter('yaw_lpf_alpha').value)
        self.publish_tf = bool(self.get_parameter('publish_tf').value)

        # 读取布局 YAML（map->tag）
        yaml_path = self.get_parameter('field_layout_yaml').value
        if not os.path.isabs(yaml_path):
            # 相对路径相对于这个脚本所在目录
            base_dir = os.path.dirname(os.path.realpath(__file__))
            yaml_path = os.path.join(base_dir, yaml_path)
        with open(yaml_path, 'r') as f:
            y = yaml.safe_load(f)
        self.layout: Dict[str, list] = y.get('field_layout', {})
        self.tag_names = list(self.layout.keys())
        if not self.tag_names:
            self.get_logger().error("field_layout is empty!")
        else:
            self.get_logger().info(f"Loaded {len(self.tag_names)} tags from {yaml_path}")

        # TF & pubs
        self.tf_buf = Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.tf_br = TransformBroadcaster(self) if self.publish_tf else None

        self.pub_has  = self.create_publisher(Bool, '/pv/has_target', 10)
        self.pub_pose = self.create_publisher(PoseStamped, '/pv/estimated_robot_pose', 10)
        self.pub_tyaw = self.create_publisher(Float32, '/pv/target_yaw_deg', 10)  # camera->best tag
        self.pub_ryaw = self.create_publisher(Float32, '/pv/robot_yaw_deg', 10)   # global yaw in map

        self.prev_yaw = None  # for LPF
        self.timer = self.create_timer(0.02, self.tick)  # 50 Hz

    def tf_to_mat(self, tfmsg: TransformStamped) -> np.ndarray:
        t = tfmsg.transform.translation; q = tfmsg.transform.rotation
        T = quat_to_mat(q.x, q.y, q.z, q.w)
        T[0,3], T[1,3], T[2,3] = t.x, t.y, t.z
        return T

    def lookup_cam_T_tag(self, tag_frame: str) -> Optional[TransformStamped]:
        try:
            # 查询 cam->tag；apriltag_ros 发布的是 parent=cam, child=tag
            return self.tf_buf.lookup_transform(self.cam_frame, tag_frame, rclpy.time.Time())
        except Exception:
            return None

    def tick(self):
        # 收集可见标签
        vis = []
        for name in self.tag_names:
            tfmsg = self.lookup_cam_T_tag(name)
            if tfmsg is not None:
                vis.append((name, tfmsg))
        self.pub_has.publish(Bool(data=bool(vis)))
        if not vis:
            return

        # 选择一个“最佳标签”用于 /pv/target_yaw_deg（对准用）
        # 这里选距离最近的（cam_T_tag 平移的范数最小）
        best_name, best_tf = min(
            vis, key=lambda it: np.linalg.norm(self.tf_to_mat(it[1])[:3,3])
        )
        cam_T_tag_best = self.tf_to_mat(best_tf)
        yaw_cam = math.degrees(math.atan2(cam_T_tag_best[0,3], cam_T_tag_best[2,3] + 1e-9))
        self.pub_tyaw.publish(Float32(data=float(yaw_cam)))

        # 多标签融合：对每个可见标签计算 map_T_base，再加权平均
        poses_T = []
        weights = []
        for name, tfmsg in vis:
            cam_T_tag = self.tf_to_mat(tfmsg)
            tag_T_cam = invert(cam_T_tag)
            # map_T_tag from layout
            x,y,z, rr,rp,ry = [float(v) for v in self.layout[name]]
            map_T_tag = xyzrpy_to_mat(x,y,z, rr,rp,ry)
            map_T_base = map_T_tag @ tag_T_cam @ self.C_T_B
            poses_T.append(map_T_base)
            # 权重：距离越近权重越大（~ 1/r^2）
            r = np.linalg.norm(cam_T_tag[:3,3]) + 1e-6
            weights.append(1.0/(r*r))

        if len(poses_T) < self.min_tags:
            # 仅用最佳标签
            poses_T = [poses_T[0]]
            weights = [1.0]

        # 平移加权平均
        ps = np.stack([T[:3,3] for T in poses_T], axis=0)  # Nx3
        w = np.array(weights, float); w /= (w.sum()+1e-12)
        p_avg = (w[:,None] * ps).sum(axis=0)

        # 旋转四元数平均
        qs = [np.array(mat_to_quat(T)) for T in poses_T]   # (qx,qy,qz,qw)
        q_avg = average_quaternions([q[[0,1,2,3]] for q in qs], list(w))
        # 组装 4x4
        T_avg = np.eye(4, float)
        T_avg[:3,:3] = quat_to_mat(q_avg[0], q_avg[1], q_avg[2], q_avg[3])[:3,:3]
        T_avg[:3,3]  = p_avg

        # 发布 PoseStamped
        now = self.get_clock().now().to_msg()
        qx,qy,qz,qw = mat_to_quat(T_avg)
        ps_msg = PoseStamped()
        ps_msg.header.frame_id = 'map'
        ps_msg.header.stamp = now
        ps_msg.pose.position.x, ps_msg.pose.position.y, ps_msg.pose.position.z = map(float, p_avg.tolist())
        ps_msg.pose.orientation.x, ps_msg.pose.orientation.y = float(qx), float(qy)
        ps_msg.pose.orientation.z, ps_msg.pose.orientation.w = float(qz), float(qw)
        self.pub_pose.publish(ps_msg)

        # 全局 yaw（deg），LPF 平滑
        yaw = yaw_from_T(T_avg)                # rad
        yaw_deg = math.degrees(yaw)            # -180..180
        if self.prev_yaw is None:
            y_sm = yaw_deg
        else:
            # 简单角度 LPF，处理 ±180° 跨越
            e = ((yaw_deg - self.prev_yaw + 180) % 360) - 180
            y_sm = self.prev_yaw + self.alpha * e
        self.prev_yaw = y_sm
        self.pub_ryaw.publish(Float32(data=float(y_sm)))

        # 可选 TF
        if self.tf_br:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = 'map'
            t.child_frame_id = 'base_link'
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, p_avg.tolist())
            t.transform.rotation.x, t.transform.rotation.y = float(qx), float(qy)
            t.transform.rotation.z, t.transform.rotation.w = float(qz), float(qw)
            self.tf_br.sendTransform(t)

def main():
    rclpy.init()
    node = PVSimBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pv_sim_bridge.py
将 apriltag_ros 的输出（相机->标签 TF）转换为统一接口：
  /pv/estimated_robot_pose   geometry_msgs/PoseStamped (frame_id=map)
  /pv/has_target             std_msgs/Bool
  /pv/target_yaw_deg         std_msgs/Float32  (以最佳标签为准，近似相机坐标系下的水平偏航)

原理：
  apriltag_ros 在 /tf 发布 cam_T_tag（父：相机光学坐标系；子：tag36h11:ID）。        [docs: publishes TF + DetectionArray]
  我们从配置文件得到 map_T_tag（标签在世界坐标 map 中的固定位姿）。
  我们从参数得到 base_T_cam（机器人到底盘的外参）。
  由 cam_T_tag 可逆出 tag_T_cam，再乘以 map_T_tag 与 cam_T_base 得到 map_T_base。

注意：
  - 只要有任意一个标签可见，就能估计位姿；多标签可做择优（此处先取“最近/置信最高”的一个）。
  - 该桥接仅用于“仿真替身”，实机请使用 PhotonVision + pv_node（PhotonLib）。

"""
from typing import Dict, Tuple, Optional
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf_transformations   # 如果没有：sudo apt update
                            #sudo apt install ros-humble-tf-transformations

from tf2_ros import Buffer, TransformListener

def quat_to_mat(qx, qy, qz, qw):
    return tf_transformations.quaternion_matrix([qx, qy, qz, qw])  # 4x4

def mat_to_quat(T):
    q = tf_transformations.quaternion_from_matrix(T)
    return q[0], q[1], q[2], q[3]

def xyzrpy_to_mat(x, y, z, rr, rp, ry):
    Rx = tf_transformations.euler_matrix(rr, 0, 0)
    Ry = tf_transformations.euler_matrix(0, rp, 0)
    Rz = tf_transformations.euler_matrix(0, 0, ry)
    R = tf_transformations.identity_matrix()
    R = tf_transformations.concatenate_matrices(R, Rz, Ry, Rx)
    T = R.copy()
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T

def invert(T):
    return tf_transformations.inverse_matrix(T)

class PVSimBridge(Node):
    def __init__(self):
        super().__init__("pv_sim_bridge")

        # ---- 参数（可通过 launch/yaml 配置）----
        self.declare_parameter("camera_optical_frame", "camera_color_optical_frame")
        self.declare_parameter("robot_to_cam_xyz_rpy", [0., 0., 0., 0., 0., 0.])  # base->cam
        self.declare_parameter("tag_frame_prefix", "tag36h11:")  # apriltag_ros child_frame_id 前缀
        self.declare_parameter("field_layout", {})               # { "tag36h11:0": [x,y,z,roll,pitch,yaw], ... }

        self.cam_frame = self.get_parameter("camera_optical_frame").get_parameter_value().string_value
        self.robot_to_cam = list(self.get_parameter("robot_to_cam_xyz_rpy").get_parameter_value().double_array_value)
        self.tag_prefix = self.get_parameter("tag_frame_prefix").get_parameter_value().string_value
        self.field_layout: Dict[str, list] = self.get_parameter("field_layout").get_parameter_value().string_array_value \
            if self.get_parameter("field_layout").type_ == rclpy.Parameter.Type.STRING_ARRAY else \
            self.get_parameter("field_layout").get_parameter_value().string_value

        # 兼容：field_layout 既可用 yaml 加载成参数字典，也可直接在 launch 里传
        if isinstance(self.field_layout, str):
            # 如果是字符串（例如空），退回成空 dict
            self.field_layout = {}

        # 将 base->cam 外参转为矩阵；后续需要 cam->base，用逆矩阵
        if len(self.robot_to_cam) != 6:
            self.robot_to_cam = [0, 0, 0, 0, 0, 0]
        B_T_C = xyzrpy_to_mat(*self.robot_to_cam)
        self.C_T_B = invert(B_T_C)

        # TF 监听器（用来读取 cam->tag）
        self.tf_buffer = Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 发布者
        self.pub_has = self.create_publisher(Bool, "/pv/has_target", 10)
        self.pub_yaw = self.create_publisher(Float32, "/pv/target_yaw_deg", 10)
        self.pub_pose = self.create_publisher(PoseStamped, "/pv/estimated_robot_pose", 10)

        # 定时器：50 Hz
        self.timer = self.create_timer(0.02, self.tick)

        self.get_logger().info("[pv_sim_bridge] ready. Waiting for TF from apriltag_ros...")

    def _lookup_cam_T_tag(self, tag_frame: str) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(
                self.cam_frame, tag_frame, rclpy.time.Time())
        except Exception:
            return None

    def _choose_best_tag(self) -> Optional[Tuple[str, TransformStamped]]:
        """
        策略：遍历 field_layout 里列出的 tag（即你场地上存在的 tag），
              谁能在 TF 里查到，就选第一个；也可改成“距离最近/置信最高”。
        """
        for tag_name in self.field_layout.keys():
            tag_tf = self._lookup_cam_T_tag(tag_name)
            if tag_tf is not None:
                return tag_name, tag_tf
        return None

    def _msg_to_mat(self, tfmsg: TransformStamped):
        t = tfmsg.transform.translation
        q = tfmsg.transform.rotation
        T = quat_to_mat(q.x, q.y, q.z, q.w)
        T[0, 3], T[1, 3], T[2, 3] = t.x, t.y, t.z
        return T

    def tick(self):
        best = self._choose_best_tag()
        has = Bool(); has.data = best is not None
        self.pub_has.publish(has)

        if not best:
            return

        tag_name, cam_T_tag_msg = best
        cam_T_tag = self._msg_to_mat(cam_T_tag_msg)

        # 读取 map_T_tag（来自参数/布局）
        layout = self.field_layout.get(tag_name, None)
        if layout is None:
            self.get_logger().warn(f"[pv_sim_bridge] tag {tag_name} not in field_layout; skipping.")
            return
        x, y, z, rr, rp, ry = [float(v) for v in layout]
        map_T_tag = xyzrpy_to_mat(x, y, z, rr, rp, ry)

        # 计算 map_T_base = map_T_tag * inv(cam_T_tag) * cam_T_base
        tag_T_cam = invert(cam_T_tag)
        map_T_base = map_T_tag @ tag_T_cam @ self.C_T_B

        # 发布位姿（map）
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = map_T_base[0, 3]
        ps.pose.position.y = map_T_base[1, 3]
        ps.pose.position.z = map_T_base[2, 3]
        qx, qy, qz, qw = mat_to_quat(map_T_base)
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        self.pub_pose.publish(ps)

        # 近似 yaw：取相机坐标系下 best target 的 yaw（投影到水平）
        # 这里做一个简单近似：在 cam_T_tag 中，取相机水平面上的偏转角（仅用于调试对准）
        yaw_msg = Float32()
        yaw_msg.data = math.degrees(math.atan2(cam_T_tag[0, 3], cam_T_tag[2, 3] + 1e-9))
        self.pub_yaw.publish(yaw_msg)

def main():
    rclpy.init()
    node = PVSimBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()



