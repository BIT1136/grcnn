cmake_minimum_required(VERSION 3.0.2)
project(grcnn)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  GraspCandidate.msg
)

add_service_files(
  FILES
  PredictGrasps.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  geometry_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
