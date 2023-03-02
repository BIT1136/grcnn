#!/home/yangzhuo/mambaforge/envs/grcnn/bin/python

import rospy
import ros_numpy

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os
from io import BytesIO
from PIL import Image as pImage
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from grcnn.msg import Rgbd

from utils import *

def callback(rgbd):
    rgb=ros_numpy.numpify(rgbd.rgb)
    depth=ros_numpy.numpify(rgbd.depth)
    x=np.concatenate((np.expand_dims(rgb, 0),np.expand_dims(depth, 0)),1)
    x=torch.from_numpy(np.expand_dims(x, 0).astype(np.float32))
    with torch.no_grad():
        xc = x.to(device)
        pred = model.predict(xc)

    q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
    grasps = detect_grasps(q_img, ang_img, width_img)

    # 原始方法：直接根据xy坐标从深度图获取z
    pos_z = depth[grasps[0].center[0] + self.cam_data.top_left[0], grasps[0].center[1] + self.cam_data.top_left[1]] * self.cam_depth_scale - 0.04
    pos_x = np.multiply(grasps[0].center[1] + self.cam_data.top_left[1] - self.camera.intrinsics.ppx,
                        pos_z / self.camera.intrinsics.fx)
    pos_y = np.multiply(grasps[0].center[0] + self.cam_data.top_left[0] - self.camera.intrinsics.ppy,
                        pos_z / self.camera.intrinsics.fy)

    if pos_z == 0:
        return

    target = np.asarray([pos_x, pos_y, pos_z])
    target.shape = (3, 1)
    print('target: ', target)

def listener():
    global device
    device=torch.device("cpu")
    global model
    model = torch.load("models/xxx")
    model.to(device)

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("imagePair", Rgbd, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()