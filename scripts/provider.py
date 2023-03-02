#!/home/yangzhuo/mambaforge/envs/grcnn/bin/python
# 按键时从orbbec话题截取RGB和深度图发送到话题

import rospy
import numpy as np
import ros_numpy
import os
from PIL import Image as pImage
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from grcnn.msg import Rgbd
import torch

IMAGE_HEIGHT=224
IMAGE_WIDTH=224

def publish_image(rgb,depth,publisher):
    rgbd=Rgbd()
    header = Header()
    rgbd.header=header
    image_temp=Image()
    header = Header()
    image_temp.height=IMAGE_HEIGHT
    image_temp.width=IMAGE_HEIGHT
    image_temp.encoding='rgb8'
    image_temp.data=rgb.tostring()
    image_temp.header=header
    image_temp.step=IMAGE_WIDTH*3
    rgbd.rgb=image_temp
    image_temp=Image()
    header = Header()
    image_temp.height=IMAGE_HEIGHT
    image_temp.width=IMAGE_HEIGHT
    image_temp.encoding='mono8'
    image_temp.data=depth.tostring()
    image_temp.header=header
    image_temp.step=IMAGE_WIDTH*1
    rgbd.rgb=image_temp
    publisher.publish(rgbd)

def provider():
    rospy.init_node('provider', anonymous=True)
    pub = rospy.Publisher('imagePair', Rgbd, queue_size=10)

    while input()=="":
        rgb = rospy.wait_for_message("",Image,10)
        print("get rgb image")
        depth = rospy.wait_for_message("",Image,10)
        print("get depth image")
        rgb=ros_numpy.numpify(rgb)
        depth=ros_numpy.numpify(depth)
        print("np images:\n",rgb,depth)
        publish_image(rgb,depth,pub)

if __name__ == '__main__':
    try:
        provider()
    except rospy.ROSInterruptException:
        pass
