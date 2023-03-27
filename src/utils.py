import torch
import numpy as np
from math import sin, cos
from skimage.draw import line_aa
from skimage.filters import gaussian
from skimage.feature import peak_local_max

# TODO 重构:函数移动到抓取检测类中

def post_process_output(q_img, cos_img, sin_img, width_img, apply_gaussian=True):
    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0

    if apply_gaussian:
        q_img = gaussian(q_img, 2.0, preserve_range=True)
        ang_img = gaussian(ang_img, 2.0, preserve_range=True)
        width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img


class Grasp:
    def __init__(self, center, angle, width):
        self.center = center
        """像素位置(行,列)即(y,x)"""
        self.angle = angle
        """极轴指向右侧，逆时针为正方向，单位为弧度"""
        self.width = width

    def __str__(self) -> str:
        return f"center:{self.center}, angle:{self.angle:.3f}, width:{self.width:.3f}"

    def draw(self, minWidth, size):
        y = self.center[0]
        x = self.center[1]
        length = self.width - minWidth
        xo = cos(self.angle)
        yo = sin(self.angle)
        y1 = y + length / 2 * yo
        x1 = x - length / 2 * xo
        y2 = y - length / 2 * yo
        x2 = x + length / 2 * xo

        x1, y1, x2, y2 = list(map((lambda x: x.astype(np.int16)), [x1, y1, x2, y2]))
        rr, cc, val = line_aa(y1, x1, y2, x2)
        mask = (-1 < rr) & (rr < size) & (-1 < cc) & (cc < size)
        line = (rr[mask], cc[mask])
        return line, val[mask]


def detect_grasps(q_img, ang_img, width_img, num_grasps=1,label_img=None) -> list[Grasp]:
    local_max = peak_local_max(
        q_img, min_distance=50, threshold_abs=0.5, num_peaks=num_grasps,labels=label_img
    )

    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)
        grasp_angle = ang_img[grasp_point]
        grasp_width = width_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle, grasp_width)
        grasps.append(g)

    return grasps


def to_u8(array: np.ndarray, max=255):
    array = array - array.min()
    return (array * (max / array.max())).astype(np.uint8)


def normalise(img: np.ndarray, factor=255):
    img = img.astype(np.float32) / factor
    return np.clip(img - img.mean(), -1, 1)
