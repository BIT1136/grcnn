#!/root/mambaforge/envs/grcnn/bin/python

import time
import math

import torch
import numpy as np
import numpy.typing as npt
from scipy import stats
from skimage import measure, morphology, transform, feature, color, draw

import rospy
import ros_numpy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point

from grcnn.msg import GraspCandidate
from grcnn.srv import PredictGrasps, PredictGraspsResponse
from grasp import Grasp
from gr_convnet import GRConvNet
from type import imgf32


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

        model_path = "src/grcnn/models/jac_rgbd_epoch_48_iou_0.93"
        rospy.logdebug(f"加载抓取规划网络 {model_path}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gr_convnet = GRConvNet(model_path, device, self.apply_gaussian)

        if self.apply_seg:
            from segmentation import Segmentation

            model_path = "src/grcnn/models/model_19.pth"
            rospy.logdebug(f"加载实例分割网络 {model_path}")
            self.seg = Segmentation(model_path, device)
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

        info_topic = rospy.get_param("~info_topic", "/d435/camera/depth/camera_info")
        try:
            msg: CameraInfo = rospy.wait_for_message(info_topic, CameraInfo, 1)
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
            self.dpub = ImagePublisher("fixed_depth", "mono8")
            self.qpub = ImagePublisher("img_q", "mono8")
            self.apub = ImagePublisher("img_a", "mono8")
            self.wpub = ImagePublisher("img_w", "mono8")
        if self.pub_vis:
            self.vpub = ImagePublisher("plotted_grabs", "rgb8")
        rospy.Service("plan_grasp", PredictGrasps, self.callback)
        rospy.loginfo("抓取规划器就绪")

    def get_ros_param(self):
        self.apply_seg = rospy.get_param("~apply_seg", True)
        """是否使用额外的实例分割网络以增强效果"""
        self.apply_line_detect = rospy.get_param("~apply_line_detect", True)
        """是否检测直线以增强效果.针对矩形物体抓取方向不正确做出的修正,某些情况下可能增加误判.apply_seg需为True"""
        self.apply_line_detect = self.apply_seg and self.apply_line_detect
        self.angle_policy = rospy.get_param("~angle_policy", "strict")
        """strict:将推理出的抓取角度改为最接近的直线角度
        neighborhood:将直线角度一个邻域内的抓取角度改为直线角度"""

        self.output_meter = rospy.get_param("~output_matric", True)
        """设置输出单位为米,否则为毫米"""
        self.invalid_depth_policy = rospy.get_param("~invalid_depth_policy", "mean")
        """mean:将无效值替换为深度图其他值的均值
        mode:将无效值替换为深度图其他值的众数"""
        self.apply_gaussian = rospy.get_param("~apply_gaussian", True)
        """是否对网络输出进行高斯滤波"""
        self.pub_inter_data = rospy.get_param("~pub_inter_data", True)
        """是否发布修复的深度图和原始网络输出"""
        self.pub_vis = rospy.get_param("~pub_vis", True)
        """是否发布绘制的抓取结果"""
        self.vis_type = rospy.get_param("~vis_type", "rgb")
        """抓取结果绘制在何种图像上,depth:深度图,rgb:rgb图,q:预测的抓取质量图"""
        self.use_crop = rospy.get_param("~use_crop", False)
        """裁剪图像至目标尺寸，否则将图像压缩至目标尺寸"""

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

    def centroid_scale(self, x, y) -> np.float32:
        return 1 - 0.01 * np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    def to_u8(self, array: np.ndarray, max=255) -> npt.NDArray[np.uint8]:
        array = array - array.min()
        if array.max() == 0:
            return array.astype(np.uint8)
        return (array * (max / array.max())).astype(np.uint8)

    def normalise(self, img: np.ndarray, factor=255) -> imgf32:
        img = img.astype(np.float32) / factor
        return np.clip(img - img.mean(), -1, 1)

    def detect_grasps(
        self,
        q_img: imgf32,
        ang_img: imgf32,
        width_img: imgf32,
        num_grasps=1,
        label_mask=None,
        lines_angle=[],
    ) -> list[Grasp]:
        if label_mask is not None:
            local_max = feature.peak_local_max(
                q_img,
                min_distance=50,
                threshold_abs=0.5,
                num_peaks=num_grasps,
                labels=label_mask,
            )
        else:
            local_max = feature.peak_local_max(
                q_img, min_distance=50, threshold_abs=0.5, num_peaks=num_grasps
            )

        grasps = []
        for grasp_point_array in local_max:
            grasp_point = tuple(grasp_point_array)
            grasp_angle = ang_img[grasp_point]
            grasp_width = width_img[grasp_point]
            grasp_quality= q_img[grasp_point]
            if label_mask is not None:
                inst = label_mask[grasp_point] - 1
                if lines_angle:
                    diff = np.abs(np.array(lines_angle[inst]) - grasp_angle)
                    min_index = np.argmin(diff)
                    if self.angle_policy == "strict":
                        grasp_angle = lines_angle[inst][min_index]
                    elif (
                        self.angle_policy == "neighborhood"
                        and diff[min_index] < np.pi / 8
                    ):
                        grasp_angle = lines_angle[inst][min_index]
            g = Grasp(grasp_point, grasp_angle, grasp_width,grasp_quality)
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
        valid_mask = (depth_raw != 0) & (depth_raw < 1000)
        invalid_mask = (depth_raw == 0) | (depth_raw > 1000)
        if self.invalid_depth_policy == "mean":
            mean_val = depth_raw[valid_mask].mean()
            rospy.logdebug(f"使用平均数 {mean_val} 替换深度图中的 {invalid_mask.sum()} 个无效值")
            depth_raw[invalid_mask] = mean_val
        elif self.invalid_depth_policy == "mode":
            mode_val = stats.mode(depth_raw[valid_mask], axis=None, keepdims=False)[0]
            rospy.logdebug(f"使用众数 {mode_val} 替换深度图中的 {invalid_mask.sum()} 个无效值")
            depth_raw[invalid_mask] = mode_val
        # TODO use nearest depth value as desktop_depth
        desktop_depth = stats.mode(depth_raw, axis=None, keepdims=False)[0]

        # 处理实例分割
        label_mask = None
        l_img = np.zeros(list(depth_raw.shape) + [3], dtype=np.uint8)  # 彩色标签显示
        scale_img = np.zeros_like(depth_raw, dtype=np.float32)
        num_instances = 0
        if self.apply_seg:
            img = np.transpose(rgb_raw, (2, 0, 1))
            img = self.normalise(img)
            _, predict_classes, _, predict_masks = self.seg.predict(img)
            label_mask = np.zeros_like(depth_raw, dtype=np.uint8)
            mask_threshold = 0.5
            num_instances = len(predict_classes)
            mask_indexes = []
            for i in range(num_instances):
                index = np.where(predict_masks[i] > mask_threshold)
                label_mask[index] = i + 1
                mask_indexes.append(index)
            # https://scikit-image.org/docs/0.20.x/api/skimage.measure.html?highlight=label#regionprops-table
            props = measure.regionprops_table(label_mask, properties=("centroid",))

            for i in range(num_instances):
                centroid = (props["centroid-0"][i], props["centroid-1"][i])
                for y, x in zip(mask_indexes[i][0], mask_indexes[i][1]):
                    scale_img[y, x] = self.centroid_scale((y, x), centroid)
                    l_img[y, x] = self.colors[predict_classes[i]]

        # 直线检测
        lines_angle = []
        line_img = rgb_raw.copy()
        if self.apply_line_detect:
            gray_image = color.rgb2gray(rgb_raw)
            edges = feature.canny(gray_image, sigma=2)
            for i in range(num_instances):
                inst_mask = label_mask == i + 1
                inst_mask = morphology.binary_dilation(inst_mask, morphology.disk(5))
                masked_edges = np.zeros_like(edges)
                masked_edges[inst_mask] = edges[inst_mask]
                lines = transform.probabilistic_hough_line(
                    masked_edges, threshold=10, line_length=25, line_gap=10, seed=0
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
                rospy.logdebug(f"直线角度 {angles}")

        # 调整图像到网络输入大小
        rgb_raw = self.resize(rgb_raw)
        depth_raw = self.resize(depth_raw)
        if self.apply_seg:
            label_mask = self.resize(label_mask, is_label=True)
            scale_img = self.resize(scale_img)
            l_img = self.resize(l_img)
        if self.pub_inter_data:
            self.dpub(self.to_u8(depth_raw))
        rgb = rgb_raw.transpose((2, 0, 1))  # (3,size,size)
        depth = np.expand_dims(depth_raw, 0)  # (1,size,size)
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
            if self.apply_seg:
                self.labelpub(l_img)
                if self.apply_line_detect:
                    self.linepub(line_img)
            self.qpub(self.to_u8(q_img))
            self.apub(self.to_u8(ang_img))
            self.wpub(self.to_u8(width_img))

        # 检测抓取
        grasps = self.detect_grasps(
            q_img, ang_img, width_img, 5, label_mask, lines_angle
        )

        t_end = time.perf_counter()
        rospy.loginfo(f"推理完成,耗时: {(t_end - t_start)*1000:.2f}ms")
        if len(grasps) == 0:
            raise rospy.ServiceException("未检测到抓取")
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
                    depth_raw[line] += (val * 200).astype(np.uint16)
                    depth_raw[y - 1 : y + 2, x - 1 : x + 2] = depth_raw.min()
                elif self.vis_type == "q":
                    q_img[line] += val
            if self.vis_type == "rgb":
                self.vpub(rgb_raw)
            elif self.vis_type == "depth":
                self.vpub(self.to_u8(depth_raw), "mono8")
            elif self.vis_type == "q":
                self.vpub(self.to_u8(q_img), "mono8")

        # 计算相机坐标系下的坐标
        predict_grasps = []
        for g in grasps:
            y = g.center[0]
            x = g.center[1]
            a = g.angle
            pos_z = depth_raw[y, x]
            pos_z = (pos_z + desktop_depth) / 2
            if not self.use_crop:
                y = y / self.y_scale
                x = x / self.x_scale
            pos_x = -(x - self.cx) * (pos_z / self.fx)
            pos_y = -(y - self.cy) * (pos_z / self.fy)

            if pos_z < 200:
                rospy.logwarn(f"深度值过小: {pos_z:.3f}mm, 丢弃")
                continue

            if self.output_meter:
                pos_z = pos_z / 1000
                pos_x = pos_x / 1000
                pos_y = pos_y / 1000

            rospy.loginfo(f"抓取方案: [{pos_x:.3f},{pos_y:.3f},{pos_z:.3f}], {a:.3f}")
            predict_grasps.append(GraspCandidate(Point(pos_x, pos_y, pos_z), -a))

        if len(predict_grasps) == 0:
            raise rospy.ServiceException("无法找到有效的抓取方案")

        res = PredictGraspsResponse()
        res.grasps = predict_grasps
        return res


if __name__ == "__main__":
    rospy.init_node("grasp_planner_2d", log_level=rospy.DEBUG)
    p = GraspPlanner()
    rospy.spin()
