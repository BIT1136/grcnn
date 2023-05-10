#!/usr/bin/env python
# 按键时从orbbec话题截取RGB和深度图发送到话题

import rospy
from sensor_msgs.msg import Image
from grcnn.srv import PredictGrasps
from maskrcnn_ros.srv import InstanceSeg


def parse_exp(exp):
    exp = (
        str(exp)
        .encode("utf-8")
        .decode("unicode_escape")
        .encode("latin1")
        .decode("utf-8")
    )
    return exp


if __name__ == "__main__":
    seg = rospy.ServiceProxy("/maskrcnn_ros_server/seg_instance", InstanceSeg)
    predict = rospy.ServiceProxy("process_rgbd/predict_grasps", PredictGrasps)
    rospy.init_node("provider", disable_signals=True)
    # 调用服务不需要节点，这里是为了输出日志；disable_signals=True使得可以用ctrl+c退出
    while input() == "":
        try:
            rgb = rospy.wait_for_message("/d435/camera/color/image_raw", Image, 1)
        except rospy.ROSException as e:
            rospy.logwarn(e)
            continue
        else:
            rospy.loginfo("获取RGB图")

        try:
            depth = rospy.wait_for_message("/d435/camera/depth/image_raw", Image, 1)
        except rospy.ROSException as e:
            rospy.logwarn(e)
            continue
        else:
            rospy.loginfo("获取深度图")

        try:
            resp = seg(rgb)
            data = predict(rgb, depth,resp.seg)
            rospy.loginfo("返回的抓取规划:\n%s", data)
        except rospy.ServiceException as e:
            rospy.logwarn("服务调用失败: %s", parse_exp(e))
