cmake_minimum_required(VERSION 3.0.2)
project(grcnn)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  Rgbd.msg
)

add_service_files(
  FILES
  GetGrasp.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
