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
            rospy.loginfo("获取RGB图像")
        try:
            depth = rospy.wait_for_message("/camera/depth/image_raw", Image, 10)
        except rospy.ROSException as e:
            rospy.logwarn(e)
            continue
        else:
            rospy.loginfo("获取深度图像")
        # rospy.loginfo("np images:\n%s\n%s\n",rgb[0],depth[0])
        try:
            rospy.wait_for_service("plan_grasp", 1)
        except rospy.ROSException as e:
            rospy.logwarn("等待服务失败: %s", e)
            continue
        try:
            handle = rospy.ServiceProxy("plan_grasp", GetGrasp)
            data = handle(rgb, depth)
            rospy.loginfo("得到抓取规划:\n%s", data)
        except rospy.ServiceException as e:
            rospy.logwarn("服务调用失败: %s", e)


if __name__ == "__main__":
    rospy.init_node("provider", disable_signals=True)
    # 调用服务不需要节点，这里是为了输出日志；disable_signals=True使得可以用ctrl+c退出
    provider()
