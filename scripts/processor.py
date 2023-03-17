#!/root/mambaforge/envs/grcnn/bin/python

import time
import yaml

import torch
import numpy as np
from scipy.stats import mode
from skimage.transform import resize

import rospy
import ros_numpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

from grcnn.msg import GraspCandidate
from grcnn.srv import PredictGrasp, PredictGraspResponse
from utils import *


class GraspPlanner:
    def __init__(self):
        modelPath = "src/grcnn/models/jac_rgbd_epoch_48_iou_0.93"
        rospy.logdebug(f"加载模型 {modelPath}")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(modelPath, map_location=self.device)
        self.model.eval()

        # TODO 从sensor_msgs.msg.CameraInfo加载内参
        intrinsicPath = rospy.get_param(
            "~intrinsicPath", "src/grcnn/config/ir_camera.yaml"
        )
        with open(intrinsicPath, "r", encoding="utf-8") as f:
            config = yaml.load(f, yaml.FullLoader)
        intrinsic = config["camera_matrix"]["data"]
        intrinsic = np.array(intrinsic).reshape(3, 3)
        # 相机内参
        self.cx = intrinsic[0, 2]
        self.cy = intrinsic[1, 2]
        self.fx = intrinsic[0, 0]
        self.fy = intrinsic[1, 1]
        # 深度图微调参数
        self.depthScale = rospy.get_param("~depthScale", 1)
        self.depthBias = rospy.get_param("~depthBias", 0)
        rospy.logdebug(
            "cx=%s, cy=%s, fx=%s, fy=%s, ds=%s, db=%s",
            self.cx,
            self.cy,
            self.fx,
            self.fy,
            self.depthScale,
            self.depthBias,
        )

        width = config["image_width"]
        height = config["image_height"]
        self.useCrop = rospy.get_param("~useCrop", False)
        """裁剪图像至目标尺寸，否则将图像压缩至目标尺寸"""
        size = 300
        self.size = size
        left = (width - size) // 2
        top = (height - size) // 2
        right = (width + size) // 2
        bottom = (height + size) // 2
        self.bottom_right = (bottom, right)
        self.top_left = (top, left)

        self.zeroDepthPolicy = rospy.get_param("~zeroDepthPolicy", "mean")
        """mean:将0值替换为深度图其他值的均值
        mode:将0值替换为深度图其他值的众数"""
        self.applyGaussian = rospy.get_param("~applyGaussian", True)
        """是否对网络输出进行高斯滤波"""
        self.pubInterData = rospy.get_param("~pubInterData", True)
        """是否发布修复的深度图和原始网络输出"""
        self.pubVis = rospy.get_param("~pubVis", True)
        """是否发布绘制的抓取结果"""
        self.visType = rospy.get_param("~visType", "depth")
        """抓取结果绘制在何种图像上,depth:深度图,rgb:rgb图,q:预测的抓取质量图"""

        if self.pubInterData:
            self.dpub = rospy.Publisher("fixed_depth", Image, queue_size=10)
            self.qpub = rospy.Publisher("img_q", Image, queue_size=10)
            self.apub = rospy.Publisher("img_a", Image, queue_size=10)
            self.wpub = rospy.Publisher("img_w", Image, queue_size=10)
        if self.pubVis:
            self.vpub = rospy.Publisher("plotted_grabs", Image, queue_size=10)
        rospy.Service("plan_grasp", PredictGrasp, self.callback)
        rospy.loginfo("抓取规划器就绪")

    def crop(self, img, channelFirst=False):
        if len(img.shape) == 2:
            return img[
                self.top_left[0] : self.bottom_right[0],
                self.top_left[1] : self.bottom_right[1],
            ]
        elif len(img.shape) == 3 and channelFirst:
            return img[
                :,
                self.top_left[0] : self.bottom_right[0],
                self.top_left[1] : self.bottom_right[1],
            ]
        elif len(img.shape) == 3:
            return img[
                self.top_left[0] : self.bottom_right[0],
                self.top_left[1] : self.bottom_right[1],
                :,
            ]
        else:
            raise NotImplementedError

    def callback(self, data):
        rospy.loginfo("开始推理")
        t_start = time.perf_counter()

        # 获取并预处理图像
        rgb_raw: np.ndarray = ros_numpy.numpify(data.rgb)  # (480, 640, 3) uint8
        depth_raw: np.ndarray = ros_numpy.numpify(data.depth)  # (480, 640) uint16 单位为毫米
        if self.zeroDepthPolicy == "mean":
            meanVal = depth_raw[depth_raw != 0].mean()
            rospy.logdebug(f"使用平均数 = {meanVal} 替换深度图中的0值")
            depth_raw[depth_raw == 0] = meanVal
        elif self.zeroDepthPolicy == "mode":
            modeVal = mode(depth_raw[depth_raw != 0], axis=None, keepdims=False)[0]
            rospy.logdebug(f"使用众数 = {modeVal} 替换深度图中的0值")
            depth_raw[depth_raw == 0] = modeVal
        if self.useCrop:
            rgb_raw = self.crop(rgb_raw)
            depth_raw = self.crop(depth_raw)
        else:
            rgb_raw = resize(
                rgb_raw, (self.size, self.size, 3), preserve_range=True
            ).astype(np.uint8)
            depth_raw = resize(
                depth_raw, (self.size, self.size), preserve_range=True
            ).astype(np.uint16)
        if self.pubInterData:
            self.dpub.publish(
                ros_numpy.msgify(Image, to_u8(depth_raw), encoding="mono8")
            )
        rgb = rgb_raw.transpose((2, 0, 1))  # (3,size,size)
        depth = np.expand_dims(depth_raw, 0)  # (1,size,size)
        rgb = normalise(rgb)
        depth = normalise(depth, 1000)
        x = np.concatenate((depth, rgb), 0)
        x = np.expand_dims(x, 0)  # (1, 4, size, size)

        # 推理
        x = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            pred = self.model.predict(x)

        # 后处理
        q_img, ang_img, width_img = post_process_output(
            pred["pos"], pred["cos"], pred["sin"], pred["width"], self.applyGaussian
        )
        """尺寸与输入相同,典型区间:
        -0.01~0.96
        -0.90~1.36
        -3.05~60.57"""
        if self.pubInterData:
            self.qpub.publish(ros_numpy.msgify(Image, to_u8(q_img), encoding="mono8"))
            self.apub.publish(ros_numpy.msgify(Image, to_u8(ang_img), encoding="mono8"))
            self.wpub.publish(
                ros_numpy.msgify(Image, to_u8(width_img), encoding="mono8")
            )
        minWidth = width_img.min()

        grasps = detect_grasps(q_img, ang_img, width_img, 5)
        t_end = time.perf_counter()
        rospy.loginfo(f"推理完成,耗时: {(t_end - t_start)*1000:.2f}ms")
        if len(grasps) == 0:
            raise rospy.ServiceException("未检测到抓取")
        else:
            rospy.loginfo(f"检测到 {len(grasps)} 个抓取候选: {[str(g) for g in grasps]}")

        # 绘制抓取线
        if self.pubVis:
            for g in grasps:
                line, val = g.draw(minWidth, self.size)
                if self.visType == "rgb":
                    rgb_raw[line] += (
                        ((255, 0, 0) - rgb_raw[line]) * np.expand_dims(val, 1)
                    ).astype(np.uint8)
                elif self.visType == "depth":
                    depth_raw[line] += (val * 200).astype(np.uint16)
                elif self.visType == "q":
                    q_img[line] += val
            if self.visType == "rgb":
                self.vpub.publish(ros_numpy.msgify(Image, rgb_raw, encoding="rgb8"))
            elif self.visType == "depth":
                self.vpub.publish(
                    ros_numpy.msgify(Image, to_u8(depth_raw), encoding="mono8")
                )
            elif self.visType == "q":
                self.vpub.publish(
                    ros_numpy.msgify(Image, to_u8(q_img), encoding="mono8")
                )

        # 计算相机坐标系下的坐标
        for g in grasps:
            y = g.center[0]
            x = g.center[1]
            a = g.angle
            pos_z = depth_raw[x + 0, y + 0] * self.depthScale + self.depthBias
            pos_x = np.multiply(x - self.cx, pos_z / self.fx)
            pos_y = np.multiply(y - self.cy, pos_z / self.fy)

            if pos_z < 200:
                rospy.logwarn(f"深度值过小: {pos_z:.3f},丢弃")
                continue

            rospy.loginfo(f"抓取方案: [{pos_x:.3f},{pos_y:.3f},{pos_z:.3f}]")
            res = PredictGraspResponse()
            res.grasps = [GraspCandidate(Point(pos_x, pos_y, pos_z), a)]
            return res

        raise rospy.ServiceException("无法找到有效的抓取方案")


if __name__ == "__main__":
    rospy.init_node("grasp_planner_2d")
    p = GraspPlanner()
    rospy.spin()
