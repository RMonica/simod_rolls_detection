#ifndef SIMOD_VISION_CTRL_H
#define SIMOD_VISION_CTRL_H

#include "eigen_conversions/eigen_msg.h"
#include "geometry_msgs/WrenchStamped.h"
#include "geometry_msgs/Pose.h"
#include "ros/ros.h"
#include "sensor_msgs/JointState.h"
#include "std_msgs/Float64MultiArray.h"
#include "tf/transform_listener.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "trajectory_msgs/MultiDOFJointTrajectory.h"
#include "ur_dashboard_msgs/RobotMode.h"
#include "ur_dashboard_msgs/SafetyMode.h"
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/Marker.h>
#include <jsoncpp/json/json.h>

#include <limits>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <thread>
#include <functional>
#include <bitset>

#include "simod_rolls_detection/utils_rot.h"
#include "simod_rolls_detection/ur_cobot_utils.h"
#include "simod_rolls_detection/ur_cobot.h"
#include "simod_rolls_detection/filters.h"

#define POSE_DIM 6

class SimodVisionCtrl
{
public:
  SimodVisionCtrl();

  void spinner();

  int getRate() const { return rate_; }

private:
  static constexpr double kROSParamsTimeoutSec_ = 10.;

  enum InitFlagCB
  {
    JOINTS,
    WRENCH,
    SAFETY_MODE,
    ROBOT_MODE
  };

  // State machine usata nel .cpp
  enum class MissionState
  {
    MOVE_TO_HOMING,
    MOVE_TO_PALLET_VIEW,
    DETECT_PALLET,
    LOAD_AND_SAVE_BOX_POSE,
    COMPUTE_CLOSE_VIEW,
    MOVE_TO_CLOSE_VIEW,
    DETECT_CLOSE_LINE,
    RETURN_HOMING_AND_DONE,
    DONE
  };

  ros::NodeHandle nh_;

  // Topic-related
  std::vector<ros::Subscriber> joints_state_sub_;
  std::vector<ros::Subscriber> safety_mode_sub_;
  std::vector<ros::Subscriber> robot_mode_sub_;
  std::vector<ros::Publisher> joints_vel_pub_;

  // Debugging
  std::vector<ros::Publisher> joints_vel_monitor_pub_;
  std::vector<ros::Publisher> desired_pose_pub_;
  std::vector<ros::Publisher> measured_pose_pub_;
  ros::Publisher status_pub_;

  std::vector<std::bitset<4>> init_flags_;
  ros::Time t0_;
  double deltaT_;
  ros::Time t0Abs_;
  ros::Time motion_start_time_;

  // Time-invariant
  std::string tmp_param;
  std::vector<std::string> wrench_topic_;
  int rate_;
  bool reversed_joints_;

  // Time-varying
  std::vector<Vector6d> joints_position_;
  std::vector<Vector6d> joints_velocity_;
  std::vector<Vector6d> joints_torques_;
  std::vector<Vector6d> wrench_;

  std::vector<uint8_t> safety_mode_;
  std::vector<int8_t> robot_mode_;

  bool isRobotRunning(const int robot_idx) const;
  void sendZeroJointVelCommand();

  bool ur2_shift_start_ = false;

private:
  /* ROS callbacks */
  void jointStatesCallbackUR2(const sensor_msgs::JointState::ConstPtr &joints_state);

  void safetyModeCallbackUR2(const ur_dashboard_msgs::SafetyMode::ConstPtr &safety_mode_msg);

  void robotModeCallbackUR2(const ur_dashboard_msgs::RobotMode::ConstPtr &robot_mode_msg);

  LowPassFilter jointvelFilter;
  URCobot UR2;

  MissionState mission_state_;

  //
  bool loadVisionViewsFromFile(const std::string &json_path,
                               const std::string &robot_id,
                               Eigen::Matrix<double, 6, 1> &homing_world,
                               Eigen::Matrix<double, 6, 1> &pallet_view_world,
                               Eigen::Matrix<double, 6, 1> &close_view_box);

  Eigen::Affine3d applyBoxOffset(const Eigen::Affine3d &T_0_box,
                                 const Eigen::Matrix<double, 6, 1> &off);

  static Eigen::Affine3d vec6ToAffineRPY(const Eigen::Matrix<double, 6, 1> &v);
  Json::Value affineToJson(const Eigen::Affine3d &, const std::string &);

  bool loadBoxPoseFromFile(const std::string &path, Eigen::Affine3d &T);
  bool loadJsonFromFile(const std::string &path, Json::Value &root);
  bool savePoseJson(const Json::Value &j, const std::string &path);
  bool callTrigger(const std::string &srv_name);

  // Pose e offset

  Eigen::Affine3d T_0_homing_, T_0_pallet_view_, T_0_close_view_, T_0_box_;
  Eigen::Matrix<double, 6, 1> homing_world_, pallet_view_world_, close_view_box_;

  // frame e path
  std::string world_frame_id_, tf_base_frame_, camera_frame_id_;
  std::string vision_views_json_, box_pose_json_path_, line_json_save_path_;
  std::string pallet_detect_srv_, closeline_detect_srv_;
  std::string output_dir_, box_pose_out_file_, close_line_out_file_;
};

#endif /* SIMOD_VISION_CTRL_H */
