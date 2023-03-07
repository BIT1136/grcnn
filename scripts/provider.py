#!/root/mambaforge/envs/grcnn/bin/python
# 按键时从orbbec话题截取RGB和深度图发送到话题

import rospy
import ros_numpy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from grcnn.msg import Rgbd
from grcnn.srv import GetGraspLocal

IMAGE_HEIGHT=224
IMAGE_WIDTH=224

def callProcessor(rgb,depth):
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
    rospy.wait_for_service('plan_grasp')
    try:
        handle = rospy.ServiceProxy('plan_grasp', GetGraspLocal)
        data = handle(rgb, depth)
        return data
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def provider():
    while input()=="":
        rgb = rospy.wait_for_message("",Image,10)
        print("get rgb image")
        depth = rospy.wait_for_message("",Image,10)
        print("get depth image")
        rgb=ros_numpy.numpify(rgb)
        depth=ros_numpy.numpify(depth)
        print("np images:\n",rgb,depth)
        callProcessor(rgb,depth)

if __name__ == '__main__':
    try:
        provider()
    except rospy.ROSInterruptException:
        pass
