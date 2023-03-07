import torch
import numpy as np
from skimage.filters import gaussian
from skimage.feature import peak_local_max


def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the network, convert to numpy arrays, apply filtering.
    :param q_img: Q output of network (as torch Tensors)
    :param cos_img: cos output of network
    :param sin_img: sin output of network
    :param width_img: Width output of network
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img

class Grasp:
    """
    A Grasp represented by a center pixel, rotation angle and gripper width (length)
    """

    def __init__(self, center, angle, length=60, width=30):
        self.center = center
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.
        self.length = length
        self.width = width

    # @property
    # def as_gr(self):
    #     """
    #     Convert to GraspRectangle
    #     :return: GraspRectangle representation of grasp.
    #     """
    #     xo = np.cos(self.angle)
    #     yo = np.sin(self.angle)

    #     y1 = self.center[0] + self.length / 2 * yo
    #     x1 = self.center[1] - self.length / 2 * xo
    #     y2 = self.center[0] - self.length / 2 * yo
    #     x2 = self.center[1] + self.length / 2 * xo

    #     return GraspRectangle(np.array(
    #         [
    #             [y1 - self.width / 2 * xo, x1 - self.width / 2 * yo],
    #             [y2 - self.width / 2 * xo, x2 - self.width / 2 * yo],
    #             [y2 + self.width / 2 * xo, x2 + self.width / 2 * yo],
    #             [y1 + self.width / 2 * xo, x1 + self.width / 2 * yo],
    #         ]
    #     ).astype(np.float))

    # def max_iou(self, grs):
    #     """
    #     Return maximum IoU between self and a list of GraspRectangles
    #     :param grs: List of GraspRectangles
    #     :return: Maximum IoU with any of the GraspRectangles
    #     """
    #     self_gr = self.as_gr
    #     max_iou = 0
    #     for gr in grs:
    #         iou = self_gr.iou(gr)
    #         max_iou = max(max_iou, iou)
    #     return max_iou

    # def plot(self, ax, color=None):
    #     """
    #     Plot Grasp
    #     :param ax: Existing matplotlib axis
    #     :param color: (optional) color
    #     """
    #     self.as_gr.plot(ax, color)

    def to_jacquard(self, scale=1):
        """
        Output grasp in "Jacquard Dataset Format" (https://jacquard.liris.cnrs.fr/database.php)
        :param scale: (optional) scale to apply to grasp
        :return: string in Jacquard format
        """
        # Output in jacquard format.
        return '%0.2f;%0.2f;%0.2f;%0.2f;%0.2f' % (
            self.center[1] * scale, self.center[0] * scale, -1 * self.angle * 180 / np.pi, self.length * scale,
            self.width * scale)

def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1):
    """
    Detect grasps in a network output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)

    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp_angle = ang_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length / 2

        grasps.append(g)

    return grasps