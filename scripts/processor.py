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

def norm(array:np.ndarray,max=255):
    array=array-array.min()
    return array*(max/array.max())

class processor:
    def __init__(self):
        rospy.init_node('processor')

        rospy.loginfo("load model")
        self.device=torch.device('cpu')
        self.model = torch.load("/root/grasp/src/grcnn/models/jac_rgbd_epoch_48_iou_0.93",
                                map_location=self.device)
        self.model.eval()

        self.rgbInputSize=(3,300,300)
        self.depthInputSize=(1,300,300)
        #相机内参 考虑深度为空间点到相机平面的垂直距离
        self.cx=rospy.get_param('~cx', 311)
        self.cy=rospy.get_param('~cy', 237)
        self.fx=rospy.get_param('~fx', 517)
        self.fy=rospy.get_param('~fy', 517)
        self.depthScale=rospy.get_param('~depthScale', 1)
        self.depthBias=rospy.get_param('~depthBias', 0)
        rospy.loginfo("cx=%s,cy=%s,fx=%s,fy=%s,ds=%s,db=%s",
                      self.cx,self.cy,self.fx,self.fy,self.depthScale,self.depthBias)

        self.pub = rospy.Publisher('processed_img', Image, queue_size=10)
        self.qpub = rospy.Publisher('img_q', Image, queue_size=10)
        self.apub = rospy.Publisher('img_a', Image, queue_size=10)
        self.wpub = rospy.Publisher('img_w', Image, queue_size=10)
        rospy.Service('plan_grasp', GetGrasp, self.callback)
        rospy.spin()
    
    def callback(self,data):
        rgb_raw:np.ndarray=ros_numpy.numpify(data.rgb) # (480, 640, 3) uint8
        rgb=rgb_raw.reshape((3,480,640))
        depth_raw:np.ndarray=ros_numpy.numpify(data.depth) # (480, 640) uint16
        np.savetxt("depth.txt",depth_raw,"%d")
        depth=np.expand_dims(depth_raw, 0)
        rgb=resize(rgb, self.rgbInputSize, preserve_range=True).astype(np.float32)
        depth=resize(depth, self.depthInputSize, preserve_range=True).astype(np.float32)
        print(rgb.min(),rgb.max())#0.0 226.95447
        print(depth.min(),depth.max())#0.0 776.0
        # normalise
        rgb = np.clip((rgb - rgb.mean()), -1, 1)
        depth = np.clip((depth - depth.mean()), -1, 1)
        # rgb = np.clip((rgb/127.5 - 1), -1, 1)
        # depth = np.clip((depth/388 - 1), -1, 1)
        x=np.concatenate((depth,rgb),0)
        x=np.expand_dims(x,0)# (1, 4, 300, 300)
        x=torch.from_numpy(x)
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)

        # np.ndarray (300,300) dtype=float32
        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        print(q_img.min(),q_img.max())#-0.47500402 1.3665676
        print(ang_img.min(),ang_img.max())#-1.5282992 1.5262172
        print(width_img.min(),width_img.max())#-65.58831 63.643776
        self.qpub.publish(ros_numpy.msgify(Image,norm(q_img).astype(np.uint8),encoding='mono8'))
        self.apub.publish(ros_numpy.msgify(Image,norm(ang_img).astype(np.uint8),encoding='mono8'))
        self.wpub.publish(ros_numpy.msgify(Image,norm(width_img).astype(np.uint8),encoding='mono8'))
        grasps = detect_grasps(q_img, ang_img, width_img,5)

        if len(grasps)==0:
            rospy.logwarn("no grasp detected!")
            return
        else:
            rospy.loginfo("grasp detected:%s",[str(g) for g in grasps])

        # 绘制抓取线
        rgb_raw=resize(rgb_raw,(300,300,3), preserve_range=True).astype(np.uint8)
        for g in grasps:
            center=g.center
            rgb_raw[center]=[255,0,0]
            # angle=g.angle
            # length=abs(g.length)
            # startX=int(np.cos(angle)*length/2)
            # startY=int(np.sin(angle)*length/2)
            # endX=-startX
            # endY=-startY
            # x1,x2=startX+center[0],endX+center[0]
            # y1,y2=startY+center[1],endY+center[1]
            # line=skimage.draw.line(x1,y1,x2,y2)
            # rgb_raw[line]=[255,255,255]
        rospy.loginfo(rgb_raw.shape)
        self.pub.publish(ros_numpy.msgify(Image, rgb_raw, encoding='rgb8'))

        # 计算相机坐标系下的坐标
        x = grasps[0].center[0]
        y = grasps[0].center[1]
        a=grasps[0].angle
        depth_raw=resize(depth_raw,(300,300), preserve_range=True)
        pos_z=depth_raw[x+0,y+0]*self.depthScale+self.depthBias
        pos_x=np.multiply(x-self.cx,pos_z/self.fx)
        pos_y=np.multiply(y-self.cy,pos_z/self.fy)

        target = np.asarray([pos_x, pos_y, pos_z])
        target.shape = (3, 1)
        print('target: ', target)

        if pos_z==0:
            raise rospy.ServiceException("No depth pixel")

        return (Point(pos_x,pos_y,pos_z),a)

if __name__ == '__main__':
    try:
        p=processor()
    except rospy.ServiceException:
        pass