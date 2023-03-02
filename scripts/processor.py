#!/root/mambaforge/envs/grcnn/bin/python

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
from grcnn.srv import GetGraspLocal,GetGraspLocalResponse

from utils import *

class processor:
    def __init__(self):
        self.model = torch.load("/root/2d_grasp/src/grcnn/models/jac_rgbd_epoch_48_iou_0.93",
                                map_location=torch.device('cpu'))
        rospy.init_node('processor')

        #相机内参 考虑深度为空间点到相机平面的垂直距离
        self.cx=rospy.get_param('cx', 1)
        self.cy=rospy.get_param('~cy', '1')
        rospy.logwarn("%s %s",self.cx,self.cy)
        rospy.logerr("%s %s",type(self.cx),type(self.cy))
        self.fx=rospy.get_param('~fx', '')
        self.fy=rospy.get_param('~fx', '')
        self.depthScale=rospy.get_param('~depthScale', '')
        self.depthBias=rospy.get_param('~depthBias', '')

        rospy.Service('plan_grasp', GetGraspLocal, self.callback)
        rospy.spin()
    
    def callback(self,data):
        rgb=ros_numpy.numpify(data.rgb)
        depth=ros_numpy.numpify(data.depth)
        x=np.concatenate((np.expand_dims(rgb, 0),np.expand_dims(depth, 0)),1)
        x=torch.from_numpy(np.expand_dims(x, 0).astype(np.float32))
        with torch.no_grad():
            xc = x.to(device)
            pred = model.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        grasps = detect_grasps(q_img, ang_img, width_img)

        if len(grasps)==0:
            print("no grasp detected!")
            return
        
        # 原始方法：直接根据xy坐标从深度图获取z
        # pos_z = depth[grasps[0].center[0] + self.cam_data.top_left[0], grasps[0].center[1] + self.cam_data.top_left[1]] * self.cam_depth_scale - 0.04
        # pos_x = np.multiply(grasps[0].center[1] + self.cam_data.top_left[1] - self.camera.intrinsics.ppx,
        #                     pos_z / self.camera.intrinsics.fx)
        # pos_y = np.multiply(grasps[0].center[0] + self.cam_data.top_left[0] - self.camera.intrinsics.ppy,
        #                     pos_z / self.camera.intrinsics.fy)
        # if pos_z == 0:
        #     return

        pos_z=depth[grasps[0].cneter[0]+0,grasps[0].center[1]+0]*self.depthScale+self.depthBias
        pos_x=np.multiply(grasps[0].cneter[0]-self.cx,pos_z/self.fx)
        pos_x=np.multiply(grasps[0].cneter[1]-self.cy,pos_z/self.fy)

        target = np.asarray([pos_x, pos_y, pos_z])
        target.shape = (3, 1)
        print('target: ', target)

        #相机外参
        T=ros_numpy.numpify(data.T)
        worldTarget=T*target
        print('worldTarget: ', worldTarget)

        return GetGraspLocalResponse()

if __name__ == '__main__':
    p=processor()