cmake_minimum_required(VERSION 3.0.2)
project(simod_rolls_detection)

add_subdirectory(pallet_detection)

find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  roscpp
  tf2_ros
  visualization_msgs
  geometry_msgs
  cv_bridge
  tf2_eigen
  eigen_conversions
  actionlib
  actionlib_msgs
  message_generation
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

###################################
## catkin specific configuration ##
###################################
add_action_files(
  DIRECTORY action
  FILES DetectPallet.action
)

generate_messages(
  DEPENDENCIES actionlib_msgs std_msgs geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES simod_rolls_detection
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  pallet_detection/src
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

add_executable(pallet_detection_node 
               src/pallet_detection_node.cpp
               )

add_dependencies(pallet_detection_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(pallet_detection_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
  pallet_detection_lib
)

add_executable(pallet_detection_test 
               src/pallet_detection_test.cpp
               )

add_dependencies(pallet_detection_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(pallet_detection_test
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(save_images 
               src/save_images.cpp
               )

add_dependencies(save_images ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(save_images
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

#############
## Install ##
#############

install(TARGETS pallet_detection_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS save_images
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

