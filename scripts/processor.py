#!/root/mambaforge/envs/grcnn/bin/python

import rospy
import ros_numpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.transform import resize
from grcnn.srv import GetGrasp

from utils import *


def to_u8(array: np.ndarray, max=255):
    array = array - array.min()
    return array * (max / array.max())


def normalise(img: np.ndarray, factor=255):
    img = img.astype(np.float32) / factor
    img -= img.mean()
    return img


class processor:
    def __init__(self):
        rospy.init_node("processor")

        rospy.loginfo("load model")
        self.device = torch.device("cpu")
        self.model = torch.load(
            "/root/grasp/src/grcnn/models/jac_rgbd_epoch_48_iou_0.93",
            map_location=self.device,
        )
        self.model.eval()

        self.rgbInputSize = (3, 300, 300)
        self.depthInputSize = (1, 300, 300)
        # 相机内参 考虑深度为空间点到相机平面的垂直距离
        self.cx = rospy.get_param("~cx", 311)
        self.cy = rospy.get_param("~cy", 237)
        self.fx = rospy.get_param("~fx", 517)
        self.fy = rospy.get_param("~fy", 517)
        self.depthScale = rospy.get_param("~depthScale", 1)
        self.depthBias = rospy.get_param("~depthBias", 0)
        rospy.loginfo(
            "cx=%s,cy=%s,fx=%s,fy=%s,ds=%s,db=%s",
            self.cx,
            self.cy,
            self.fx,
            self.fy,
            self.depthScale,
            self.depthBias,
        )

        self.useCrop = rospy.get_param("~useCrop", False)
        width = 640
        height = 480
        output_size = 300
        left = (width - output_size) // 2
        top = (height - output_size) // 2
        right = (width + output_size) // 2
        bottom = (height + output_size) // 2
        self.bottom_right = (bottom, right)
        self.top_left = (top, left)

        self.pub = rospy.Publisher("processed_img", Image, queue_size=10)
        self.qpub = rospy.Publisher("img_q", Image, queue_size=10)
        self.apub = rospy.Publisher("img_a", Image, queue_size=10)
        self.wpub = rospy.Publisher("img_w", Image, queue_size=10)
        rospy.Service("plan_grasp", GetGrasp, self.callback)
        rospy.spin()

    def crop(self, img, channelAhead=False):
        if len(img.shape) == 2:
            return img[
                self.top_left[0] : self.bottom_right[0],
                self.top_left[1] : self.bottom_right[1],
            ]
        elif len(img.shape) == 3 and channelAhead:
            return img[
                self.top_left[0] : self.bottom_right[0],
                self.top_left[1] : self.bottom_right[1],
                :,
            ]
        elif len(img.shape) == 3:
            return img[
                :,
                self.top_left[0] : self.bottom_right[0],
                self.top_left[1] : self.bottom_right[1],
            ]
        else:
            raise NotImplementedError

    def callback(self, data):
        rospy.loginfo("start inferencing")
        rgb_raw: np.ndarray = ros_numpy.numpify(data.rgb)  # (480, 640, 3) uint8
        rgb = rgb_raw.transpose((2, 0, 1))  # (3,480,640)
        depth_raw: np.ndarray = ros_numpy.numpify(data.depth)  # (480, 640) uint16 单位为毫米
        depth = np.expand_dims(depth_raw, 0)
        if self.useCrop:
            rgb = self.crop(rgb)
            depth = self.crop(depth)
        else:
            rgb = resize(rgb, self.rgbInputSize, preserve_range=True).astype(np.float32)
            depth = resize(depth, self.depthInputSize, preserve_range=True).astype(
                np.float32
            )
        # normalise
        # rgb = np.clip((rgb - rgb.mean()), -1, 1)
        # depth = np.clip((depth - depth.mean()), -1, 1)
        rgb = normalise(rgb)
        depth = normalise(depth, 1000)
        x = np.concatenate((depth, rgb), 0)
        x = np.expand_dims(x, 0)  # (1, 4, 300, 300)
        x = torch.from_numpy(x)
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)

        # np.ndarray (300,300) dtype=float32
        q_img, ang_img, width_img = post_process_output(
            pred["pos"], pred["cos"], pred["sin"], pred["width"]
        )
        # print(q_img.min(),q_img.max())#-0.47500402 1.3665676
        # print(ang_img.min(),ang_img.max())#-1.5282992 1.5262172
        # print(width_img.min(),width_img.max())#-65.58831 63.643776
        self.qpub.publish(
            ros_numpy.msgify(Image, to_u8(q_img).astype(np.uint8), encoding="mono8")
        )
        self.apub.publish(
            ros_numpy.msgify(Image, to_u8(ang_img).astype(np.uint8), encoding="mono8")
        )
        self.wpub.publish(
            ros_numpy.msgify(Image, to_u8(width_img).astype(np.uint8), encoding="mono8")
        )
        grasps = detect_grasps(q_img, ang_img, width_img, 5)

        if len(grasps) == 0:
            raise rospy.ServiceException("no grasp detected")
        else:
            rospy.loginfo("grasp detected:%s", [str(g) for g in grasps])

        if self.useCrop:
            rgb_raw = self.crop(rgb_raw, True)
            depth_raw = self.crop(depth_raw)
        else:
            rgb_raw = resize(rgb_raw, (300, 300, 3), preserve_range=True).astype(
                np.uint8
            )
            depth_raw = resize(depth_raw, (300, 300), preserve_range=True)

        # 绘制抓取线
        for g in grasps:
            center = g.center
            # for x in range(-2,3):
            #     for y in range(-2,3):
            #         q_img[center[0]+y,center[1]+x]=2
            angle = g.angle
            length = abs(g.length)
            xo = np.cos(angle)
            yo = np.sin(angle)
            y1 = center[0] + length / 2 * yo
            x1 = center[1] - length / 2 * xo
            y2 = center[0] - length / 2 * yo
            x2 = center[1] + length / 2 * xo

            x1, y1, x2, y2 = list(map((lambda x: x.astype(np.uint8)), [x1, y1, x2, y2]))
            line = skimage.draw.line(y1, x1, y2, x2)
            # rgb_raw[line]=[255,255,255]
            # depth_raw[line]=[1000]
            q_img[line] = 2
        # self.pub.publish(ros_numpy.msgify(Image, rgb_raw, encoding='rgb8'))
        # self.pub.publish(ros_numpy.msgify(Image, to_u8(depth_raw).astype(np.uint8), encoding='mono8'))
        self.pub.publish(
            ros_numpy.msgify(Image, to_u8(q_img).astype(np.uint8), encoding="mono8")
        )

        # 计算相机坐标系下的坐标
        for g in grasps:
            x = g.center[0]
            y = g.center[1]
            a = g.angle
            pos_z = depth_raw[x + 0, y + 0] * self.depthScale + self.depthBias
            if pos_z < 100:
                print(f"got z={pos_z} at {x},{y}")
            pos_x = np.multiply(x - self.cx, pos_z / self.fx)
            pos_y = np.multiply(y - self.cy, pos_z / self.fy)

            if pos_z < 100:
                continue

            rospy.loginfo("target: [%.3f,%.3f,%.3f]", pos_x, pos_y, pos_z)
            return (Point(pos_x, pos_y, pos_z), a)

        raise rospy.ServiceException("No depth pixel")


if __name__ == "__main__":
    try:
        p = processor()
    except rospy.ServiceException:
        pass
