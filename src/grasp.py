import math
import numpy as np
from skimage import draw


class Grasp:
    def __init__(self, center, angle, width, quality,inst_id):
        self.center = center
        """像素位置(行,列)即(y,x)"""
        self.angle = angle
        """极轴指向右侧，逆时针为正方向，单位为弧度"""
        self.width = width
        self.quality = quality
        self.inst_id=inst_id

    def __str__(self) -> str:
        return f"center:{self.center}, angle:{self.angle:.3f}, width:{self.width:.3f}"

    def draw(self, minWidth, size):
        y = self.center[0]
        x = self.center[1]
        length = self.width - minWidth
        xo = math.cos(self.angle)
        yo = math.sin(self.angle)
        y1 = y + length / 2 * yo
        x1 = x - length / 2 * xo
        y2 = y - length / 2 * yo
        x2 = x + length / 2 * xo

        x1, y1, x2, y2 = list(map((lambda x: x.astype(np.int16)), [x1, y1, x2, y2]))
        rr, cc, val = draw.line_aa(y1, x1, y2, x2)
        mask = (-1 < rr) & (rr < size[0]) & (-1 < cc) & (cc < size[1])
        line = (rr[mask], cc[mask])
        return line, val[mask]
