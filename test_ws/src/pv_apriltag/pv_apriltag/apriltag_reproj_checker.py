#!/usr/bin/env python3
# apriltag_reproj_checker.py
# 依赖: opencv-python, numpy, scipy, rclpy, tf_transformations (或 scipy.spatial.transform)
# 用法: 放在 ROS2 python package 或直接用 ros2 run python3 执行（根据环境调整）

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
# apriltag detection msg 名称可能根据你用的包不同，请按实际替换
from apriltag_msgs.msg import AprilTagDetectionArray   # 替换为你实际的 message type
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import math, time

# 配置项：根据你实际情况改 topic 名与 tag_size
CAM_INFO_TOPIC = '/camera/camera_info'
AP_DETECTIONS_TOPIC = '/detections'   # 改成你实际的 topic
TAG_SIZE_M = 0.1  # 例如 0.1 m = 100 mm

def obj_pts_for_tag(size_m):
    s = size_m
    # 注意：顺序必须和检测返回的 corners 顺序一致（一般为 ccw 或 cw，需验证）
    return np.array([
        [-s/2, -s/2, 0.0],
        [ s/2, -s/2, 0.0],
        [ s/2,  s/2, 0.0],
        [-s/2,  s/2, 0.0],
    ], dtype=np.float32)

class ReprojChecker(Node):
    def __init__(self):
        super().__init__('apriltag_reproj_checker')
        self.get_logger().info('Starting reprojection checker...')
        self.cam_info = None
        self.K = None
        self.D = None
        self.sub_ci = self.create_subscription(CameraInfo, CAM_INFO_TOPIC, self.cb_caminfo, 10)
        self.sub_det = self.create_subscription(AprilTagDetectionArray, AP_DETECTIONS_TOPIC, self.cb_detections, 10)

    def cb_caminfo(self, msg: CameraInfo):
        if self.cam_info is None:
            self.cam_info = msg
            K = np.array(msg.k).reshape(3,3)
            D = np.array(msg.d)
            self.K = K
            self.D = D
            self.get_logger().info(f'Got CameraInfo: fx={K[0,0]:.3f}, fy={K[1,1]:.3f}, cx={K[0,2]:.3f}, cy={K[1,2]:.3f}, D len={len(D)}')

    def cb_detections(self, msg: AprilTagDetectionArray):
        if self.K is None:
            self.get_logger().warn('No CameraInfo yet; skipping detection')
            return
        detections = msg.detections if hasattr(msg, 'detections') else getattr(msg, 'detections', [])
        if len(detections) == 0:
            return
        cams = []
        for det in detections:
            tid = det.id[0] if hasattr(det, 'id') else (det.tag_id if hasattr(det, 'tag_id') else None)
            # corners: find how your message gives corners; common: det.corners (4) each with x,y
            # 下面按 det.corners[i].x/y 假设，请根据实际 msg 类型修改
            try:
                image_pts = np.array([[c.x, c.y] for c in det.corners], dtype=np.float32)
            except Exception as e:
                # 尝试另一种字段名
                try:
                    image_pts = np.array([[c[0], c[1]] for c in det.corners], dtype=np.float32)
                except:
                    self.get_logger().error(f'Cannot parse corners for detection {tid}: {e}')
                    continue

            objp = obj_pts_for_tag(TAG_SIZE_M)
            # 求解 PnP
            try:
                ok, rvec, tvec = cv2.solvePnP(objp, image_pts, self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            except Exception as e:
                self.get_logger().error(f'solvePnP exception: {e}')
                continue
            if not ok:
                self.get_logger().warn(f'solvePnP failed for tag {tid}')
                continue

            proj, _ = cv2.projectPoints(objp, rvec, tvec, self.K, self.D)
            proj = proj.reshape(-1,2)
            errs = np.linalg.norm(proj - image_pts, axis=1)
            mean_err = float(errs.mean())
            self.get_logger().info(f'Tag {tid}: mean reproj err = {mean_err:.3f} px, per-corner: {errs.tolist()}')

            # 组装 tag->cam 的变换（OpenCV 的 rvec,tvec 表示 object->camera）
            Rm, _ = cv2.Rodrigues(rvec)
            T_tag_cam = np.eye(4)
            T_tag_cam[:3,:3] = Rm
            T_tag_cam[:3,3] = tvec.reshape(3)

            cams.append({'id': tid, 'T_tag_cam': T_tag_cam, 'mean_err': mean_err})

        # 如果你有 field_layout(tag->map)，可以把 T_map_cam = T_map_tag * T_tag_cam 并做融合
        # 这里只做简单的基于误差倒数加权的相机位姿融合（在 tag frame 下）
        if len(cams) > 0:
            weights = np.array([1.0/(c['mean_err']+1e-6) for c in cams])
            weights /= weights.sum()
            # positions (transform tag->cam origin to tag frame then average camera pos in tag frames is not directly meaningful
            # 一般要先把 tag->map 提供进来，以下仅示例：把每个 tag 的 tvec 转为 camera 在 tag frame 的坐标 (-R^T * t)
            cam_positions = []
            quats = []
            for c in cams:
                T = c['T_tag_cam']
                # Camera in tag frame:
                Rm = T[:3,:3]
                t = T[:3,3]
                # camera position expressed in tag frame:
                campos = -Rm.T.dot(t)
                cam_positions.append(campos)
                q = R.from_matrix(Rm).as_quat()  # [x,y,z,w]
                quats.append(q)
            cam_positions = np.array(cam_positions)
            pos_avg = (cam_positions.T * weights).sum(axis=1)
            q_arr = np.array(quats)
            q_avg = (q_arr.T * weights).sum(axis=1)
            q_avg /= np.linalg.norm(q_avg)
            R_avg = R.from_quat(q_avg).as_matrix()
            self.get_logger().info(f'Weighted avg camera pos (tag-frame mix): {pos_avg}, orientation quat {q_avg}')

def main(args=None):
    rclpy.init(args=args)
    node = ReprojChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
