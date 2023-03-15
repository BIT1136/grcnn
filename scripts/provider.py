#!/root/mambaforge/envs/grcnn/bin/python
# 按键时从orbbec话题截取RGB和深度图发送到话题

import rospy
from sensor_msgs.msg import Image
from grcnn.srv import GetGrasp


def provider():
    while input() == "":
        try:
            rgb = rospy.wait_for_message("/camera/color/image_raw", Image, 10)
        except rospy.ROSException as e:
            rospy.logwarn(e)
            continue
        else:
            rospy.loginfo("get rgb image")
        try:
            depth = rospy.wait_for_message("/camera/depth/image_raw", Image, 10)
        except rospy.ROSException as e:
            rospy.logwarn(e)
            continue
        else:
            rospy.loginfo("get depth image")
        # rospy.loginfo("np images:\n%s\n%s\n",rgb[0],depth[0])
        try:
            rospy.wait_for_service("plan_grasp", 1)
        except rospy.ROSException as e:
            rospy.logwarn("Wait for service failed: %s", e)
            continue
        try:
            handle = rospy.ServiceProxy("plan_grasp", GetGrasp)
            data = handle(rgb, depth)
            rospy.loginfo("get grasp:\n%s", data)
            # return data
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s", e)


if __name__ == "__main__":
    try:
        rospy.init_node("provider", disable_signals=True)
        provider()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
