#!/root/mambaforge/envs/grcnn/bin/python

import rospy
import ros_numpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

import time
import torch
import numpy as np
from scipy.stats import mode
from skimage.transform import resize

from grcnn.srv import GetGrasp
from utils import *


class processor:
    def __init__(self):
        rospy.init_node("processor")

        rospy.logdebug("load model")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(
            "/root/grasp/src/grcnn/models/jac_rgbd_epoch_48_iou_0.93",
            map_location=self.device,
        )
        self.model.eval()

        # 相机内参
        self.cx = rospy.get_param("~cx", 311)
        self.cy = rospy.get_param("~cy", 237)
        self.fx = rospy.get_param("~fx", 517)
        self.fy = rospy.get_param("~fy", 517)
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

        width = rospy.get_param("~width", 640)
        height = rospy.get_param("~height", 480)
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
        self.publishIntermediateData = rospy.get_param("~publishIntermediateData", True)
        """是否发布修复的深度图和网络输出"""
        self.publishVisualisation = rospy.get_param("~publishVisualisation", True)
        """是否发布绘制的抓取结果"""

        if self.publishIntermediateData:
            self.dpub = rospy.Publisher("fixed_depth", Image, queue_size=10)
            self.qpub = rospy.Publisher("img_q", Image, queue_size=10)
            self.apub = rospy.Publisher("img_a", Image, queue_size=10)
            self.wpub = rospy.Publisher("img_w", Image, queue_size=10)
        if self.publishVisualisation:
            self.vpub = rospy.Publisher("plotted_grabs", Image, queue_size=10)
        rospy.Service("plan_grasp", GetGrasp, self.callback)
        rospy.spin()

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
        rospy.loginfo("start inferencing")
        t_start = time.perf_counter()

        # 获取并预处理图像
        rgb_raw: np.ndarray = ros_numpy.numpify(data.rgb)  # (480, 640, 3) uint8
        depth_raw: np.ndarray = ros_numpy.numpify(data.depth)  # (480, 640) uint16 单位为毫米
        if self.zeroDepthPolicy == "mean":
            meanVal = depth_raw[depth_raw != 0].mean()
            rospy.logdebug(f"replace 0 by mean = {meanVal}")
            depth_raw[depth_raw == 0] = meanVal
        elif self.zeroDepthPolicy == "mode":
            modeVal = mode(depth_raw[depth_raw != 0], axis=None, keepdims=False)[0]
            rospy.logdebug(f"replace 0 by mode = {modeVal}")
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
        if self.publishIntermediateData:
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
        if self.publishIntermediateData:
            self.qpub.publish(ros_numpy.msgify(Image, to_u8(q_img), encoding="mono8"))
            self.apub.publish(ros_numpy.msgify(Image, to_u8(ang_img), encoding="mono8"))
            self.wpub.publish(
                ros_numpy.msgify(Image, to_u8(width_img), encoding="mono8")
            )
        minWidth = width_img.min()

        grasps = detect_grasps(q_img, ang_img, width_img, 5)
        t_end = time.perf_counter()
        rospy.loginfo(f"End inferencing, time cost: {(t_end - t_start)*1000:.2f}ms")
        if len(grasps) == 0:
            raise rospy.ServiceException("No grasp detected.")
        else:
            rospy.loginfo(
                f"{len(grasps)} grasp(s) detected: {[str(g) for g in grasps]}"
            )

        # 绘制抓取线
        if self.publishVisualisation:
            for g in grasps:
                line, val = g.draw(minWidth, self.size)
                # rgb_raw[line] += (
                #     ((255, 0, 0) - rgb_raw[line]) * np.expand_dims(val, 1)
                # ).astype(np.uint8)
                depth_raw[line] += (val * 200).astype(np.uint16)
                # q_img[line] += val
            # self.vpub.publish(ros_numpy.msgify(Image, rgb_raw, encoding="rgb8"))
            self.vpub.publish(
                ros_numpy.msgify(Image, to_u8(depth_raw), encoding="mono8")
            )
            # self.vpub.publish(ros_numpy.msgify(Image, to_u8(q_img), encoding="mono8"))

        # 计算相机坐标系下的坐标
        for g in grasps:
            y = g.center[0]
            x = g.center[1]
            a = g.angle
            pos_z = depth_raw[x + 0, y + 0] * self.depthScale + self.depthBias
            pos_x = np.multiply(x - self.cx, pos_z / self.fx)
            pos_y = np.multiply(y - self.cy, pos_z / self.fy)

            if pos_z < 200:
                rospy.logwarn(f"Depth too small: {pos_z:.3f}, ignored")
                continue

            rospy.loginfo(f"target: [{pos_x:.3f},{pos_y:.3f},{pos_z:.3f}]")
            return (Point(pos_x, pos_y, pos_z), a)

        raise rospy.ServiceException("Can't find valid grasp.")


if __name__ == "__main__":
    try:
        p = processor()
    except rospy.ServiceException:
        pass
