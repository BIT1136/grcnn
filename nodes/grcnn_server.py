#!/usr/bin/env python

import time
import math
from typing import Union

import torch
import numpy as np
from scipy import stats
from scipy.spatial.transform import Rotation
from skimage import measure, morphology, transform, feature, color, draw

import rospy
import ros_numpy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Quaternion

from grcnn.msg import GraspCandidate
from grcnn.srv import PredictGrasps, PredictGraspsResponse
from grasp import Grasp
from gr_convnet import GRConvNet
from type import *


class ImagePublisher:
    def __init__(self, name, format):
        self.format = format
        self.pub = rospy.Publisher(name, Image, queue_size=1)

    def __call__(self, data, format=None):
        self.pub.publish(
            ros_numpy.msgify(
                Image, data, encoding=(self.format if format == None else format)
            )
        )


class GraspPlanner:
    def __init__(self):
        self.get_ros_param()

        model_path = "../models/jac_rgbd_epoch_48_iou_0.93"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.logdebug(f"加载抓取规划网络 {model_path} 至 {device}")
        self.gr_convnet = GRConvNet(model_path, device)

        if self.apply_seg:
            from maskrcnn_ros.srv import InstanceSeg

            self.seg = rospy.ServiceProxy(
                "/maskrcnn_ros_server/seg_instance", InstanceSeg
            )
            self.colors = [
                [255, 165, 0],
                [255, 255, 0],
                [255, 0, 255],
                [0, 255, 255],
                [255, 0, 0],
                [0, 255, 0],
                [0, 128, 128],
                [128, 0, 128],
                [128, 0, 0],
                [139, 69, 19],
            ]

        try:
            msg: CameraInfo = rospy.wait_for_message(self.info_topic, CameraInfo, 1)
        except rospy.ROSException as e:
            msg = CameraInfo()
            msg.K = [
                554.382,
                0.0,
                320.0,
                0.0,
                554.382,
                240.0,
                0.0,
                0.0,
                1.0,
            ]
            msg.width, msg.height = 640, 480
            rospy.logwarn(f"{e}, 使用默认相机参数")
        self.fx, self.fy, self.cx, self.cy = msg.K[0], msg.K[4], msg.K[2], msg.K[5]
        rospy.logdebug("cx=%s, cy=%s, fx=%s, fy=%s", self.cx, self.cy, self.fx, self.fy)

        width, height = msg.width, msg.height
        size = self.gr_convnet.size
        self.size = size
        left = (width - size) // 2
        top = (height - size) // 2
        right = (width + size) // 2
        bottom = (height + size) // 2
        self.bottom_right = (bottom, right)
        self.top_left = (top, left)
        self.y_scale = size / height
        self.x_scale = size / width

        if self.pub_inter_data:
            if self.apply_seg:
                self.labelpub = ImagePublisher("img_label", "rgb8")
                self.linepub = ImagePublisher("img_line", "rgb8")
            self.rdpub=ImagePublisher("raw_depth","mono8")
            self.dpub = ImagePublisher("fixed_depth", "mono8")
            self.qpub = ImagePublisher("img_q", "mono8")
            self.apub = ImagePublisher("img_a", "mono8")
            self.wpub = ImagePublisher("img_w", "mono8")
        if self.pub_vis:
            self.vpub = ImagePublisher("plotted_grabs", "rgb8")

        rospy.Service(
            f"{rospy.get_name()}/predict_grasps", PredictGrasps, self.callback
        )

        rospy.loginfo(f"{rospy.get_name()}节点就绪")

    def get_ros_param(self):
        self.apply_seg = rospy.get_param("~apply_seg", True)
        """是否使用额外的实例分割网络以增强效果"""
        self.apply_line_detect = rospy.get_param("~apply_line_detect", True)
        """是否检测直线以增强效果.针对矩形物体抓取方向不正确做出的修正,某些情况下可能增加误判.apply_seg需为True"""
        self.apply_line_detect = self.apply_seg and self.apply_line_detect
        self.angle_policy = rospy.get_param("~angle_policy", "neighborhood")
        """strict:将推理出的抓取角度改为最接近的直线角度
        neighborhood:将直线角度一个邻域内的抓取角度改为直线角度"""

        self.invalid_depth_policy = rospy.get_param("~invalid_depth_policy", "mean")
        """mean:将无效值替换为深度图其他值的均值
        mode:将无效值替换为深度图其他值的众数"""
        self.pub_inter_data = rospy.get_param("~pub_inter_data", True)
        """是否发布中间数据,包括实例分割和grcnn原始输出等"""
        self.pub_vis = rospy.get_param("~pub_vis", True)
        """是否发布绘制的抓取结果"""
        self.vis_type = rospy.get_param("~vis_type", "rgb")
        """抓取结果绘制在何种图像上,depth:深度图,rgb:rgb图,q:预测的抓取质量图"""
        self.use_crop = rospy.get_param("~use_crop", False)
        """裁剪图像至目标尺寸,否则将图像压缩至目标尺寸"""

        self.info_topic = rospy.get_param(
            "~info_topic", "/d435/camera/depth/camera_info"
        )

    def resize(self, img, is_label=False) -> np.ndarray:
        if self.use_crop:
            if len(img.shape) == 2:
                return img[
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
        else:
            dtype = img.dtype
            if len(img.shape) == 2 and is_label:
                return transform.resize(
                    img,
                    (self.size, self.size),
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False,
                )
            elif len(img.shape) == 2:
                return transform.resize(
                    img, (self.size, self.size), preserve_range=True
                ).astype(dtype)
            elif len(img.shape) == 3:
                return transform.resize(
                    img, (self.size, self.size, 3), preserve_range=True
                ).astype(dtype)
            else:
                raise NotImplementedError

    def centroid_scale(self, img: imgf32, index, centroid):
        for y, x in zip(index[0], index[1]):
            img[y, x] = 1 - 0.01 * np.sqrt(
                (y - centroid[0]) ** 2 + (x - centroid[1]) ** 2
            )
        return img

    def to_u8(self, array: np.ndarray, max=255) -> npt.NDArray[np.uint8]:
        array = array - array.min()
        if array.max() == 0:
            return array.astype(np.uint8)
        return (array * (max / array.max())).astype(np.uint8)

    def normalise(self, img: np.ndarray, factor=255) -> imgf32:
        img = img.astype(np.float32) / factor
        return np.clip(img - img.mean(), -1, 1)

    def handle_seg(self, rgb_raw, depth_raw, seg=None):
        l_img = np.zeros(list(depth_raw.shape) + [3], dtype=np.uint8)  # 彩色标签显示
        scale_img = np.zeros_like(depth_raw, dtype=np.float32)  # q_img 缩放系数
        num_instances = 0
        if self.apply_seg:
            if seg == None:
                try:
                    seg = self.seg(rgb_raw)
                except rospy.ServiceException as e:
                    rospy.logerr(f"分割服务调用失败: {e} 本次推理不使用分割结果")
                    return None, 1, []
        label_mask: npt.NDArray = ros_numpy.numpify(seg)
        num_instances = label_mask.max()
        mask_indexes = []
        for i in range(num_instances):
            index = np.where(label_mask == i + 1)
            mask_indexes.append(index)

        # 根据实例重心计算q_img缩放系数
        props = measure.regionprops_table(label_mask, properties=("centroid",))
        for i in range(num_instances):
            centroid = (props["centroid-0"][i], props["centroid-1"][i])
            scale_img = self.centroid_scale(scale_img, mask_indexes[i], centroid)
            l_img[mask_indexes[i]] = self.colors[i]

        l_img = self.resize(l_img)
        if self.apply_seg:
            self.labelpub(l_img)

        # 每个实例mask膨胀后进行直线检测
        lines_angle = []
        if self.apply_line_detect:
            line_img = rgb_raw.copy()
            gray_image = color.rgb2gray(rgb_raw)
            edges = feature.canny(gray_image, sigma=3,low_threshold=0,high_threshold=0.1)
            for i in range(num_instances):
                inst_mask = label_mask == i + 1
                inst_mask = morphology.binary_dilation(inst_mask, morphology.disk(7))
                masked_edges = np.zeros_like(edges)
                masked_edges[inst_mask] = edges[inst_mask]
                lines = transform.probabilistic_hough_line(
                    masked_edges, threshold=20, line_length=20, line_gap=0, seed=0
                )
                rospy.logdebug(f"检测第 {i} 个实例")
                rospy.logdebug(f"检测到 {len(lines)} 条直线")
                angles = []
                for line in lines:
                    (x0, y0), (x1, y1) = line
                    rr, cc, val = draw.line_aa(y0, x0, y1, x1)
                    mask = (-1 < rr) & (rr < 480) & (-1 < cc) & (cc < 640)
                    draw_line = (rr[mask], cc[mask])
                    line_img[draw_line] += (
                        ((0, 255, 0) - line_img[draw_line]) * np.expand_dims(val, 1)
                    ).astype(np.uint8)
                    angle = math.atan2(y1 - y0, x1 - x0)
                    angle += math.pi / 2  # 夹爪应垂直于检测到的线段
                    if angle > math.pi / 2:
                        angle -= math.pi
                    elif angle < -math.pi / 2:
                        angle += math.pi
                    angles.append(angle)
                lines_angle.append(angles)
                rospy.logdebug(f"直线角度: {angles}")
            if self.pub_inter_data:
                self.linepub(line_img)

        label_mask = self.resize(label_mask, is_label=True)
        scale_img = self.resize(scale_img)

        return label_mask, scale_img, lines_angle

    def detect_grasps(
        self,
        q_img: imgf32,
        ang_img: imgf32,
        width_img: imgf32,
        label_mask: Union[npt.NDArray[np.int_], None] = None,
        lines_angle=[[]],
    ) -> list[Grasp]:
        min_distance = 10
        threshold = 0.75
        if label_mask is not None:
            local_max = feature.peak_local_max(
                q_img,
                min_distance,
                threshold_rel=threshold,
                labels=label_mask,
            )
        else:
            # 行,列像素坐标
            local_max = feature.peak_local_max(
                q_img, min_distance, threshold_rel=threshold
            )

        grasps = []
        for grasp_point_array in local_max:
            grasp_point = tuple(grasp_point_array)
            grasp_angle = ang_img[grasp_point]
            grasp_width = width_img[grasp_point]
            grasp_quality = q_img[grasp_point]
            grasp_inst_id = label_mask[grasp_point] if label_mask is not None else 0
            if label_mask is not None:
                inst = grasp_inst_id - 1
                if lines_angle[inst]:
                    diff = np.abs(np.array(lines_angle[inst]) - grasp_angle)
                    min_index = np.argmin(diff)
                    if self.angle_policy == "strict":
                        rospy.logdebug(f"角度{grasp_angle}修正为{lines_angle[inst][min_index]}")
                        grasp_angle = lines_angle[inst][min_index]
                    elif (
                        self.angle_policy == "neighborhood"
                        and diff[min_index] < np.pi / 8
                    ):
                        grasp_angle = lines_angle[inst][min_index]
            g = Grasp(
                grasp_point, grasp_angle, grasp_width, grasp_quality, grasp_inst_id
            )
            grasps.append(g)
        grasps.sort(key=lambda x: x.quality, reverse=True)
        return grasps

    def callback(self, data):
        rospy.loginfo("开始推理")
        t_start = time.perf_counter()

        # 获取并预处理图像
        rgb_raw: npt.NDArray[np.uint8] = ros_numpy.numpify(data.rgb)
        """(480, 640, 3)"""
        depth_raw: npt.NDArray[np.uint16] = ros_numpy.numpify(data.depth)
        """(480, 640) 单位为毫米"""
        depth_fixed=depth_raw.copy()
        valid_mask = (depth_fixed != 0) & (depth_fixed < 1000)
        invalid_mask = (depth_fixed == 0) | (depth_fixed > 1000)
        if self.invalid_depth_policy == "mean":
            mean_val = depth_fixed[valid_mask].mean()
            rospy.logdebug(f"使用平均数 {mean_val:.3f} 替换深度图中的 {invalid_mask.sum()} 个无效值")
            depth_fixed[invalid_mask] = mean_val
        elif self.invalid_depth_policy == "mode":
            mode_val = stats.mode(depth_fixed[valid_mask], axis=None, keepdims=False)[0]
            rospy.logdebug(f"使用众数 {mode_val:.3f} 替换深度图中的 {invalid_mask.sum()} 个无效值")
            depth_fixed[invalid_mask] = mode_val
        # TODO use nearest depth value as desktop_depth
        desktop_depth = stats.mode(depth_fixed, axis=None, keepdims=False)[0]
        if self.pub_inter_data:
            self.rdpub(self.to_u8(depth_raw))
            self.dpub(self.to_u8(depth_fixed))

        label_mask, scale_img, lines_angle = None, 1, []
        if self.apply_seg:
            label_mask, scale_img, lines_angle = self.handle_seg(
                rgb_raw, depth_fixed, data.seg
            )

        # 调整图像到网络输入大小
        rgb_raw = self.resize(rgb_raw)
        depth_fixed = self.resize(depth_fixed)

        rgb = rgb_raw.transpose((2, 0, 1))  # (3,size,size)
        depth = np.expand_dims(depth_fixed, 0)  # (1,size,size)
        rgb = self.normalise(rgb)
        depth = self.normalise(depth, 1000)

        # 抓取网络推理
        q_img, ang_img, width_img = self.gr_convnet.predict(rgb, depth)
        """尺寸与输入相同,典型区间:
        -0.01~0.96
        -0.90~1.36
        -3.05~60.57"""
        if self.apply_seg:
            q_img *= scale_img
        if self.pub_inter_data:
            self.qpub(self.to_u8(q_img))
            self.apub(self.to_u8(ang_img))
            self.wpub(self.to_u8(width_img))

        # 检测抓取
        grasps = self.detect_grasps(q_img, ang_img, width_img, label_mask, lines_angle)

        t_end = time.perf_counter()
        rospy.loginfo(f"推理完成,耗时: {(t_end - t_start)*1000:.2f}ms")
        if len(grasps) == 0:
            rospy.logerr("未检测到抓取")
            return PredictGraspsResponse()
        else:
            rospy.loginfo(f"检测到 {len(grasps)} 个抓取候选: {[str(g) for g in grasps]}")

        # 绘制抓取线
        if self.pub_vis:
            min_width = width_img.min() if width_img.min() < 0 else 0
            for g in grasps:
                line, val = g.draw(min_width, (self.size, self.size))
                if self.vis_type == "rgb":
                    rgb_raw[line] += (
                        ((255, 0, 0) - rgb_raw[line]) * np.expand_dims(val, 1)
                    ).astype(np.uint8)
                elif self.vis_type == "depth":
                    y = g.center[0]
                    x = g.center[1]
                    depth_fixed[line] += (val * 200).astype(np.uint16)
                    depth_fixed[y - 1 : y + 2, x - 1 : x + 2] = depth_fixed.min()
                elif self.vis_type == "q":
                    q_img[line] += val
            if self.vis_type == "rgb":
                self.vpub(rgb_raw)
            elif self.vis_type == "depth":
                self.vpub(self.to_u8(depth_fixed), "mono8")
            elif self.vis_type == "q":
                self.vpub(self.to_u8(q_img), "mono8")

        # 计算相机坐标系下的坐标
        predict_grasps = []
        for g in grasps:
            y = g.center[0]
            x = g.center[1]
            a = g.angle
            pos_z = depth_fixed[y, x]
            pos_z = (pos_z + desktop_depth) / 2
            if not self.use_crop:
                y = y / self.y_scale
                x = x / self.x_scale
            pos_x = (x - self.cx) * (pos_z / self.fx)
            pos_y = (y - self.cy) * (pos_z / self.fy)

            if pos_z < 200:
                rospy.logwarn(f"深度值过小: {pos_z:.3f}mm, 丢弃")
                continue

            pos_z /= 1000
            pos_x /= 1000
            pos_y /= 1000

            rospy.loginfo(f"抓取方案: [{pos_x:.3f},{pos_y:.3f},{pos_z:.3f}], {a:.3f}")

            grasp = GraspCandidate()
            grasp.pose.position = Point(pos_x, pos_y, pos_z)
            grasp.pose.orientation = Quaternion(
                *Rotation.from_rotvec([0, 0, a]).as_quat().tolist()
            )
            grasp.inst_id = g.inst_id
            grasp.quality = g.quality
            predict_grasps.append(grasp)

        if len(predict_grasps) == 0:
            raise rospy.ServiceException("无法找到有效的抓取方案")

        res = PredictGraspsResponse()
        res.grasps = predict_grasps
        return res


if __name__ == "__main__":
    rospy.init_node("grcnn_server", log_level=rospy.DEBUG)
    p = GraspPlanner()
    rospy.spin()