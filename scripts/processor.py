#!/root/mambaforge/envs/grcnn/bin/python

import rospy
import ros_numpy

import torch
import numpy as np
from skimage.transform import resize
from grcnn.srv import GetGraspLocal,GetGraspLocalResponse

from utils import *

class processor:
    def __init__(self):
        rospy.loginfo("load ")
        self.model = torch.load("/root/grasp/src/grcnn/models/jac_rgbd_epoch_48_iou_0.93",
                                map_location=torch.device('cpu'))
        rospy.init_node('processor')

        self.inputSize=(300,300)

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

    def numpy_to_torch(self,s):
        if len(s.shape) == 2:
            rospy.loginfo("len(x.shape) == 2")
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            rospy.loginfo("len(x.shape) != 2")
            return torch.from_numpy(s.astype(np.float32))
    
    def callback(self,data):
        rgb=ros_numpy.numpify(data.rgb)
        depth=ros_numpy.numpify(data.depth)
        rgb=resize(rgb, self.inputSize, preserve_range=True).astype(np.uint8)
        depth=resize(depth, self.inputSize, preserve_range=True).astype(np.float32)
        # normalise
        rgb = np.clip((rgb - rgb.mean()), -1, 1)
        depth = np.clip((depth - depth.mean()), -1, 1)
        x=np.concatenate((np.expand_dims(depth, 0),rgb),0)#[4, 300, 300]
        rospy.loginfo("x.shape=%s",x.shape)
        x=self.numpy_to_torch(x)
        with torch.no_grad():
            xc = x.to(device)
            pred = model.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        grasps = detect_grasps(q_img, ang_img, width_img)

        if len(grasps)==0:
            rospy.logwarn("no grasp detected!")
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