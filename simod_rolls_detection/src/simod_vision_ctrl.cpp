#include "simod_rolls_detection/simod_vision_ctrl.h"
#include <stddef.h>
#include <cmath>

#include <geometric_shapes/shapes.h>
#include <geometric_shapes/shape_operations.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/CollisionObject.h>
#include <shape_msgs/Mesh.h>
#include <boost/variant.hpp>  
#include <ros/package.h>                      

#include <ima_planner/AddPackEnvMesh.h>

SimodVisionCtrl::SimodVisionCtrl()
    : jointvelFilter(50.0, 0.01), UR2("2", nh_)
{
  // --- Init strutture dati ---
  jointvelFilter.Reset();
  init_flags_.resize(2);
  joints_position_.resize(2);
  joints_velocity_.resize(2);
  joints_torques_.resize(2);

  safety_mode_.resize(2);
  robot_mode_.resize(2);

  init_flags_[UR2.getRobotIdx()].reset();

  // --- Lettura parametri ROS ---
  bool ret = true;

  ret &= nh_.getParam("simod_vision_ctrl/rate", rate_);
  if (!ret)
  {
    ROS_FATAL("Could not retrieve ROS rate parameters.");
    ros::shutdown();
    return;
  }

  //
  ret &= nh_.getParam("simod_vision_ctrl/world_frame_id", world_frame_id_);
  if (!ret)
  {
    ROS_FATAL("Could not retrieve ROS world_frame_id parameters.");
    ros::shutdown();
    return;
  }

  ret &= nh_.getParam("simod_vision_ctrl/tf_base_frame", tf_base_frame_);
  if (!ret)
  {
    ROS_FATAL("Could not retrieve ROS tf_base_frame parameters.");
    ros::shutdown();
    return;
  }
  ret &= nh_.getParam("simod_vision_ctrl/camera_frame_id", camera_frame_id_);

  // file Json & Services
  ret &= nh_.getParam("simod_vision_ctrl/vision_views_json", vision_views_json_);
  ret &= nh_.getParam("simod_vision_ctrl/box_pose_json_path", box_pose_json_path_);
  ret &= nh_.getParam("simod_vision_ctrl/roll_packs_json_path", roll_packs_json_path_);
  ret &= nh_.getParam("simod_vision_ctrl/line_json_save_path", line_json_save_path_);

  ret &= nh_.getParam("simod_vision_ctrl/pallet_detect_srv", pallet_detect_srv_);
  ret &= nh_.getParam("simod_vision_ctrl/rollpacks_detect_srv", rollpacks_detect_srv_);
  ret &= nh_.getParam("simod_vision_ctrl/closeline_detect_srv", closeline_detect_srv_);

  ret &= nh_.getParam("simod_vision_ctrl/output_dir", output_dir_);
  ret &= nh_.getParam("simod_vision_ctrl/box_pose_out_file", box_pose_out_file_);
  ret &= nh_.getParam("simod_vision_ctrl/close_line_out_file", close_line_out_file_);

  ret &= nh_.getParam("simod_vision_ctrl/enable_pack_collision_mesh", enable_pack_collision_mesh_);
  ret &= nh_.getParam("simod_vision_ctrl/pack_mesh_root", pack_mesh_root_);
  ret &= nh_.getParam("simod_vision_ctrl/scenario", scenario_);
  ret &= nh_.getParam("simod_vision_ctrl/pack_id", pack_id_);
  nh_.param("simod_vision_ctrl/pack_mesh_scale", pack_mesh_scale_,0.001);
  nh_.param("simod_vision_ctrl/pack_mesh_use_box_pose", pack_mesh_use_box_pose_, true);


  ret &= nh_.getParam("simod_vision_ctrl/enable_pallet_detection", enable_pallet_detection_);
  ret &= nh_.getParam("simod_vision_ctrl/enable_paddle_insertion_test", enable_paddle_insertion_test_);
  
  if (!ret)
  {
    ROS_FATAL("Could not retrieve ROS parameters.");
    ros::shutdown();
    return;
  }

  // --- leggi viste dal JSON ---
  if (!loadVisionViewsFromFile(vision_views_json_, "2",
                               homing_world_, pallet_view_world_, roll_pack_view_world_, close_view_box_))
  {
    ROS_FATAL("[simod_vision_ctrl] loadVisionViewsFromFile failed: %s", vision_views_json_.c_str());
    ros::shutdown();
    return;
  }

  // --- Subscribers ---
  joints_state_sub_.resize(2);
  safety_mode_sub_.resize(2);
  robot_mode_sub_.resize(2);
  joints_state_sub_[UR2.getRobotIdx()] = nh_.subscribe(
      "arm2/joint_states", 50, &SimodVisionCtrl::jointStatesCallbackUR2, this);
  safety_mode_sub_[UR2.getRobotIdx()] = nh_.subscribe(
      "arm2/ur_hardware_interface/safety_mode", 10, &SimodVisionCtrl::safetyModeCallbackUR2, this);
  robot_mode_sub_[UR2.getRobotIdx()] = nh_.subscribe(
      "arm2/ur_hardware_interface/robot_mode", 10, &SimodVisionCtrl::robotModeCallbackUR2, this);

  // --- Publishers ---
  joints_vel_pub_.resize(2);
  joints_vel_monitor_pub_.resize(2);
  desired_pose_pub_.resize(2);
  measured_pose_pub_.resize(2);
  joints_vel_pub_[UR2.getRobotIdx()] = nh_.advertise<std_msgs::Float64MultiArray>(
      "arm2/joint_group_vel_controller/command", 1, true);
  joints_vel_monitor_pub_[UR2.getRobotIdx()] = nh_.advertise<std_msgs::Float64MultiArray>(
      "arm2/joint_group_vel_controller/monitor", 1, true);
  desired_pose_pub_[UR2.getRobotIdx()] = nh_.advertise<std_msgs::Float64MultiArray>(
      "arm2/desired_pose", 1, true);
  measured_pose_pub_[UR2.getRobotIdx()] = nh_.advertise<std_msgs::Float64MultiArray>(
      "arm2/measured_pose", 1, true);

  T_0_homing_ = vec6ToAffineRPY(homing_world_);
  T_0_pallet_view_ = vec6ToAffineRPY(pallet_view_world_);
  T_0_roll_pack_view_ = vec6ToAffineRPY(roll_pack_view_world_); 

  debug_mesh_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("debug_pack_mesh",1,true);
  
  {
    std::vector<double> v;
    if (nh_.getParam("simod_vision_ctrl/paddle_insertion_offset", v) && v.size() == 6) {
      for (int i = 0; i < 6; ++i) paddle_insert_off_6_[i] = v[i];
    } else {
      ROS_FATAL("[simod_vision_ctrl] Could not retrive Paddle Insertion Offset");
      ros::shutdown();
      return;
    
    }

  }
  

  nh_.param<std::string>("simod_vision_ctrl/add_pack_service",
                         add_pack_srv_name_, std::string("/ima_planner/add_pack_env_mesh"));
  add_pack_client_ = nh_.serviceClient<ima_planner::AddPackEnvMesh>(add_pack_srv_name_);
  // Little waits
  add_pack_client_.waitForExistence(ros::Duration(2.0));


  // Initialize mission to first phase
  mission_state_ = MissionState::MOVE_TO_HOMING;
  UR2.action_state = ActionState::MOVE_TO_TARGET;
}

// ============ helpers ============

//------------------------ PUBLIC FUNCTIONS -------------------------------------------//

void SimodVisionCtrl::spinner()
{
  ros::spinOnce();

  ROS_INFO_ONCE("Ready to spin.");

  // Aggiorna kinematics (z_E) prima di valutare la FSM
  if (init_flags_[UR2.getRobotIdx()].test(JOINTS))
  {
    UR2.updateKinematics(joints_position_[UR2.getRobotIdx()],
                         joints_velocity_[UR2.getRobotIdx()]);
  }

  switch (mission_state_)
  {
  case MissionState::MOVE_TO_HOMING:
  {
    const bool homingUR2 = UR2.reachPose(T_0_homing_, 0.01);
    if (homingUR2 && UR2.action_state == ActionState::DONE)
    {
      UR2.action_state = ActionState::MOVE_TO_TARGET;
 
      if (enable_pallet_detection_){
          mission_state_ = MissionState::MOVE_TO_PALLET_VIEW;
      }
      else {
         mission_state_ = MissionState::MOVE_TO_ROLL_PACK_VIEW;
      }
  
      ROS_INFO("[simod_vision_ctrl] Reached HOMING.");
    }
  }
  break;

  case MissionState::MOVE_TO_PALLET_VIEW:
  {
    const bool palletViewUR2 = UR2.reachPose(T_0_pallet_view_, 0.01);
    if (palletViewUR2 && UR2.action_state == ActionState::DONE)
    {
      UR2.action_state = ActionState::MOVE_TO_TARGET;
      mission_state_ = MissionState::DETECT_PALLET;
      ROS_INFO("[simod_vision_ctrl] Reached PALLET_VIEW.");
    }
  }
  break;

  
  case MissionState::DETECT_PALLET:
  {
    ROS_INFO("[simod_vision_ctrl] DETECT_PALLET: %s", pallet_detect_srv_.c_str());
    const bool ok = callTrigger(pallet_detect_srv_);
    if (!ok)
    {
      ROS_WARN("[simod_vision_ctrl] pallet detection service failed, retrying...");
      break; // resta nello stato e riprova al prossimo spinner()
    }
    mission_state_ = MissionState::LOAD_AND_SAVE_BOX_POSE_FROM_PALLET;
  }
  break;
 
  case MissionState::MOVE_TO_ROLL_PACK_VIEW:
  {
    const bool rollPackViewUR2 = UR2.reachPose(T_0_roll_pack_view_, 0.01);
    if (rollPackViewUR2 && UR2.action_state == ActionState::DONE)
    {
      UR2.action_state = ActionState::MOVE_TO_TARGET;
      mission_state_ = MissionState::DETECT_ROLL_PACKS;
      ROS_INFO("[simod_vision_ctrl] Reached ROLL_PACK_VIEW.");
    }
  }
  break;

  case MissionState::DETECT_ROLL_PACKS:
  {
    ROS_INFO("[simod_vision_ctrl] DETECT_ROLL_PACKS: %s", rollpacks_detect_srv_.c_str());
    const bool ok = callTrigger(rollpacks_detect_srv_);
    if (!ok)
    {
      ROS_WARN("[simod_vision_ctrl] roll-packs detection service failed, retrying...");
      break;
    }
    mission_state_ = MissionState::LOAD_AND_SAVE_BOX_POSE_FROM_PACKS; 
  }
  break;

  case MissionState::LOAD_AND_SAVE_BOX_POSE_FROM_PALLET:
  {
    ROS_INFO("[simod_vision_ctrl] LOAD_AND_SAVE_BOX_POSE_FROM_PALLET from %s",
            box_pose_json_path_.c_str());
    if (!loadBoxPoseFromFile(box_pose_json_path_, T_0_box_)) {
      ROS_ERROR("[simod_vision_ctrl] cannot load box pose (pallet JSON). Stopping.");
      ros::shutdown();
      return;
    }

    Json::Value jbox = affineToJson(T_0_box_, world_frame_id_);
    if (savePoseJson(jbox, box_pose_out_file_))
      ROS_INFO("[simod_vision_ctrl] box pose saved to: %s", box_pose_out_file_.c_str());
    else
      ROS_WARN("[simod_vision_ctrl] failed to save box pose JSON to: %s", box_pose_out_file_.c_str());

    if (enable_pack_collision_mesh_) {
      const Eigen::Affine3d T_world_mesh = T_0_box_;
      publishPackCollisionMesh(T_world_mesh);
    }

    if (enable_paddle_insertion_test_) {
      mission_state_ = MissionState::MOVE_TO_INSERT_START;
    } else {
      mission_state_ = MissionState::COMPUTE_CLOSE_VIEW;
    }
    
  }
  break;

  case MissionState::LOAD_AND_SAVE_BOX_POSE_FROM_PACKS:
  {
    ROS_INFO("[simod_vision_ctrl] LOAD_AND_SAVE_BOX_POSE_FROM_PACKS from %s",
            roll_packs_json_path_.c_str());
    if (!loadBoxPoseFromPacksFile(roll_packs_json_path_, T_0_box_)) {
      ROS_ERROR("[simod_vision_ctrl] cannot load box pose (packs JSON). Stopping.");
      ros::shutdown();
      return;
    }

    Json::Value jbox = affineToJson(T_0_box_, world_frame_id_);
    if (savePoseJson(jbox, box_pose_out_file_))
      ROS_INFO("[simod_vision_ctrl] box pose saved to: %s", box_pose_out_file_.c_str());
    else
      ROS_WARN("[simod_vision_ctrl] failed to save box pose JSON to: %s", box_pose_out_file_.c_str());
    
    if (enable_pack_collision_mesh_) {
      const Eigen::Affine3d T_world_mesh = T_0_box_;
      publishPackCollisionMesh(T_world_mesh);
    }
    
    if (enable_paddle_insertion_test_) {
      mission_state_ = MissionState::MOVE_TO_INSERT_START;
    } else {
      mission_state_ = MissionState::COMPUTE_CLOSE_VIEW;
    }
  }
  break;

  case MissionState::COMPUTE_CLOSE_VIEW:
  {
    ROS_INFO("[simod_vision_ctrl] COMPUTE_CLOSE_VIEW (offset nel frame box)");
    T_0_close_view_ = applyBoxOffset(T_0_box_, close_view_box_);

    //ROS_INFO posa calcolata 
    Eigen::Vector3d t = T_0_close_view_.translation();
    Eigen::Matrix3d R = T_0_close_view_.linear();

    // Converti in RPY (ZYX)
    double yaw = std::atan2(R(1, 0), R(0, 0));
    double pitch = std::asin(-R(2, 0));
    double roll = std::atan2(R(2, 1), R(2, 2));

    ROS_INFO_STREAM_ONCE("[simod_vision_ctrl] Close view pose (world): "
                    << " xyz = [" << t.x() << ", " << t.y() << ", " << t.z() << "]"
                    << " rpy = [" << roll << ", " << pitch << ", " << yaw << "]");
    mission_state_ = MissionState::MOVE_TO_CLOSE_VIEW;
  }
  break;

  case MissionState::MOVE_TO_CLOSE_VIEW:
  {
    const bool closeViewUR2 = UR2.reachPose(T_0_close_view_, 0.01);
    if (closeViewUR2 && UR2.action_state == ActionState::DONE)
    {
      mission_state_ = MissionState::DETECT_CLOSE_LINE;
      ROS_INFO("[simod_vision_ctrl] Reached CLOSE_LINE_VIEW.");
    }
  }
  break;

  case MissionState::DETECT_CLOSE_LINE:
  {
    ROS_INFO("[simod_vision_ctrl] DETECT_CLOSE_LINE: %s", closeline_detect_srv_.c_str());
    const bool ok = callTrigger(closeline_detect_srv_);
    if (!ok)
    {
      ROS_WARN_THROTTLE(2.0, "[simod_vision_ctrl] close-line service failed, retrying...");
      break; // resta nello stato e riprova al prossimo spinner()
    }

    // Leggi JSON con TUTTI i punti (in plate_center)
    Json::Value jin;
    if (!loadJsonFromFile(line_json_save_path_, jin) ||
        !jin.isMember("points") || !jin["points"].isArray() || jin["points"].empty())
    {
      ROS_WARN("[simod_vision_ctrl] no points found in: %s", line_json_save_path_.c_str());
      break;
    }

    // Seleziona il punto con |y| minimo (più vicino alla y di box_0 = 0 in plate_center)
    double best_abs_y = std::numeric_limits<double>::infinity();
    Eigen::Vector3d best_pt_pc(0, 0, 0); // plate_center

    for (const auto &jp : jin["points"])
    {
      if (!jp.isMember("x") || !jp.isMember("y") || !jp.isMember("z"))
        continue;
      const Eigen::Vector3d p_pc(jp["x"].asDouble(), jp["y"].asDouble(), jp["z"].asDouble());
      const double ay = std::abs(p_pc.y());
      if (ay < best_abs_y)
      {
        best_abs_y = ay;
        best_pt_pc = p_pc;
      }
    }

    if (!std::isfinite(best_abs_y))
    {
      ROS_WARN("[simod_vision_ctrl] parsed points but none valid.");
      break;
    }

    ROS_INFO("[simod_vision_ctrl] selected plate_center point: (%.3f, %.3f, %.3f), |y|=%.3f",
             best_pt_pc.x(), best_pt_pc.y(), best_pt_pc.z(), best_abs_y);

    // Salva SOLO il punto selezionato, nel frame "plate_center"
    Json::Value jout;
    jout["frame_id"] = "plate_center"; 
    jout["point"]["x"] = best_pt_pc.x();
    jout["point"]["y"] = best_pt_pc.y();
    jout["point"]["z"] = best_pt_pc.z();

    if (savePoseJson(jout, close_line_out_file_))
      ROS_INFO("[simod_vision_ctrl] close-line selected point saved to: %s", close_line_out_file_.c_str());
    else
      ROS_WARN("[simod_vision_ctrl] failed to save close-line JSON to: %s", close_line_out_file_.c_str());

    UR2.action_state = ActionState::MOVE_TO_TARGET;
    mission_state_ = MissionState::DONE;
  }
  break;

  case MissionState::MOVE_TO_INSERT_START:
  {
    // start pose for insertion = box pose + offset (in box_center)
    T_0_insert_start_ = applyBoxOffset(T_0_box_, paddle_insert_off_6_);
   
    // --- print pose (world) ---
    const Eigen::Vector3d t = T_0_insert_start_.translation();
    const Eigen::Matrix3d R = T_0_insert_start_.linear();
    const double yaw   = std::atan2(R(1,0), R(0,0));
    const double pitch = std::asin(-R(2,0));
    const double roll  = std::atan2(R(2,1), R(2,2));

    ROS_INFO_STREAM_ONCE("[simod_vision_ctrl] Paddle Insert pose (world): "
                    << " xyz = [" << t.x() << ", " << t.y() << ", " << t.z() << "]"
                    << " rpy = [" << roll << ", " << pitch << ", " << yaw << "]");

    const bool ok = UR2.reachPose(T_0_insert_start_, 0.01);
    if (ok && UR2.action_state == ActionState::DONE) {
      UR2.action_state  = ActionState::START_INSERTION; 
      mission_state_    = MissionState::PADDLE_INSERTION;
      ROS_INFO("[simod_vision_ctrl] Reached insert start; starting paddle insertion.");
    }
    break;
  }

  case MissionState::PADDLE_INSERTION:
  {
    const double contact_N = 8.0;
    const double speed     = 0.005;
    const double max_dist  = 0.20;
    UR2.paddleInsertion(contact_N, speed, max_dist);

    if (UR2.action_state == ActionState::DONE) {
      ROS_INFO("[simod_vision_ctrl] Paddle insertion done.");
      UR2.action_state = ActionState::MOVE_TO_TARGET;
      mission_state_   = MissionState::DONE; 
    }
    break;
  }


  case MissionState::RETURN_HOMING_AND_DONE:
  {
    const bool homingUR2 = UR2.reachPose(T_0_homing_, 0.01);
    if (homingUR2 && UR2.action_state == ActionState::DONE)
    {
      UR2.action_state = ActionState::MOVE_TO_TARGET;
      mission_state_ = MissionState::DONE;
      ROS_INFO("[simod_vision_ctrl] Reached HOMING.");
    }
  }
  break;

  case MissionState::DONE:
  {
    ROS_INFO_THROTTLE(2.0, "[simod_vision_ctrl] DONE.");
  }
  break;
  }
}

//------------------------ PRIVATE FUNCTIONS ------------------------------------------//

void SimodVisionCtrl::sendZeroJointVelCommand()
{
  std::vector<double> q_dot_0(JOINTS_NUM, 0.0);

  std_msgs::Float64MultiArray joints_vel_cmd_msg_UR2;
  joints_vel_cmd_msg_UR2.data = q_dot_0;
  const size_t idx = UR2.getRobotIdx();
  if (idx < joints_vel_pub_.size())
    joints_vel_pub_[idx].publish(joints_vel_cmd_msg_UR2);
}



bool SimodVisionCtrl::isRobotRunning(const int robot_idx) const
{
  return robot_mode_[robot_idx] == ur_dashboard_msgs::RobotMode::RUNNING &&
         (safety_mode_[robot_idx] == ur_dashboard_msgs::SafetyMode::NORMAL ||
          safety_mode_[robot_idx] == ur_dashboard_msgs::SafetyMode::REDUCED);
}

static inline bool getOff6Once(const ros::NodeHandle &nh,
                               const std::string &ns,
                               const char *key,
                               Eigen::Matrix<double, 6, 1> &out)
{
  std::vector<double> v;
  if (!nh.getParam(ns + "/" + key, v) || v.size() != 6)
    return false;
  // copia diretta
  for (int i = 0; i < 6; ++i)
    out[i] = v[i];
  return true;
}

// load Json file with poses
bool SimodVisionCtrl::loadVisionViewsFromFile(const std::string &filepath,
                                              const std::string &robot_id,
                                              Eigen::Matrix<double, 6, 1> &homing,
                                              Eigen::Matrix<double, 6, 1> &pallet_view_world,
                                              Eigen::Matrix<double, 6, 1> &roll_pack_view_world,
                                              Eigen::Matrix<double, 6, 1> &close_view_box)
{
  std::ifstream f(filepath);
  if (!f.is_open())
  {
    ROS_ERROR_STREAM("[loadVisionViewsFromFile] Cannot open JSON: " << filepath);
    return false;
  }

  Json::Value root;
  f >> root;

  if (!root.isMember("vision"))
  {
    ROS_ERROR_STREAM("[loadVisionViewsFromFile] Missing 'vision' in " << filepath);
    return false;
  }

  const std::string ur_key = (robot_id == "2") ? "ur2" : ("ur" + robot_id);
  if (!root["vision"].isMember(ur_key))
  {
    ROS_ERROR_STREAM("[loadVisionViewsFromFile] Missing 'vision/" << ur_key << "' in " << filepath);
    return false;
  }

  const auto &ur = root["vision"][ur_key];

  bool ok = true;
  ok &= URCobotUtils::parseVec6RPY(ur["homing"], homing);
  ok &= URCobotUtils::parseVec6RPY(ur["pallet_view"], pallet_view_world);
  ok &= URCobotUtils::parseVec6RPY(ur["roll_pack_view"], roll_pack_view_world);
  ok &= URCobotUtils::parseVec6RPY(ur["close_line_view"], close_view_box);

  if (!ok)
  {
    ROS_ERROR_STREAM("[loadVisionViewsFromFile] One of input pose is invalid in " << filepath);
    return false;
  }
  return true;
}

bool SimodVisionCtrl::loadBoxPoseFromFile(const std::string &path, Eigen::Affine3d &T)
{
  Json::Value root;
  if (!loadJsonFromFile(path, root))
  {
    ROS_ERROR_STREAM("Failed to load JSON box pose from: " << path);
    return false;
  }

  try
  {
    double x = root["translation"]["x"].asDouble();
    double y = root["translation"]["y"].asDouble();
    double z = root["translation"]["z"].asDouble();
    double roll = root["rotation"]["roll"].asDouble();
    double pitch = root["rotation"]["pitch"].asDouble();
    double yaw = root["rotation"]["yaw"].asDouble();

    Eigen::Affine3d Tnew = Eigen::Affine3d::Identity();
    Tnew.translation() = Eigen::Vector3d(x, y, z);

    Eigen::AngleAxisd Rx(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rz(yaw, Eigen::Vector3d::UnitZ());
    Tnew.linear() = (Rz * Ry * Rx).toRotationMatrix();

    T = Tnew;
    return true;
  }
  catch (...)
  {
    ROS_ERROR_STREAM("Invalid JSON format in: " << path);
    return false;
  }
}

bool SimodVisionCtrl::loadBoxPoseFromPacksFile(const std::string &path, Eigen::Affine3d &T)
{
  Json::Value root;
  if (!loadJsonFromFile(path, root))
  {
    ROS_ERROR("[simod_vision_ctrl] loadBoxPoseFromPacksFile: cannot read '%s'", path.c_str());
    return false;
  }

  // NEW FORMAT: { "frame_id": "...", "box_center_0": { position{...}, rpy{...} }, ... } ---
  if (root.isMember("box_center_0"))
  {
    const Json::Value &bc = root["box_center_0"];
    if (!bc.isMember("position") || !bc.isMember("rpy"))
    {
      ROS_ERROR("[simod_vision_ctrl] '%s' has box_center_0 but missing 'position' or 'rpy'", path.c_str());
      return false;
    }

    const Json::Value &p = bc["position"];
    const Json::Value &rpy = bc["rpy"];

    if (!p.isMember("x") || !p.isMember("y") || !p.isMember("z") ||
        !rpy.isMember("roll") || !rpy.isMember("pitch") || !rpy.isMember("yaw"))
    {
      ROS_ERROR("[simod_vision_ctrl] '%s' box_center_0 missing fields x/y/z or roll/pitch/yaw", path.c_str());
      return false;
    }

    const double x = p["x"].asDouble();
    const double y = p["y"].asDouble();
    const double z = p["z"].asDouble();

    const double roll  = rpy["roll"].asDouble();   
    const double pitch = rpy["pitch"].asDouble();  
    const double yaw   = rpy["yaw"].asDouble();   

    Eigen::Affine3d A = Eigen::Affine3d::Identity();
    A.translation() = Eigen::Vector3d(x, y, z);

    // RPY rotation
    Eigen::AngleAxisd Rx(roll , Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rz(yaw  , Eigen::Vector3d::UnitZ());
    A.linear() = (Rz * Ry * Rx).toRotationMatrix();  // Z * Y * X for intrinsic RPY

    T = A;
    return true;
  }

  // --- LEGACY FORMAT: { "packs": [ { position{...}, orientation{w,x,y,z}, ... } ] } ---
  if (root.isMember("packs") && root["packs"].isArray() && root["packs"].size() > 0)
  {
    const Json::Value &p0 = root["packs"][0];

    if (!p0.isMember("position") || !p0.isMember("orientation"))
    {
      ROS_ERROR("[simod_vision_ctrl] '%s' packs[0] missing 'position' or 'orientation'", path.c_str());
      return false;
    }

    const Json::Value &pos = p0["position"];
    const Json::Value &ori = p0["orientation"];
    if (!pos.isMember("x") || !pos.isMember("y") || !pos.isMember("z") ||
        !ori.isMember("w") || !ori.isMember("x") || !ori.isMember("y") || !ori.isMember("z"))
    {
      ROS_ERROR("[simod_vision_ctrl] '%s' packs[0] missing fields x/y/z or w/x/y/z", path.c_str());
      return false;
    }

    const double x = pos["x"].asDouble();
    const double y = pos["y"].asDouble();
    const double z = pos["z"].asDouble();
    const double wq = ori["w"].asDouble();
    const double xq = ori["x"].asDouble();
    const double yq = ori["y"].asDouble();
    const double zq = ori["z"].asDouble();

    Eigen::Quaterniond q(wq, xq, yq, zq);
    q.normalize();

    Eigen::Affine3d A = Eigen::Affine3d::Identity();
    A.translation() = Eigen::Vector3d(x, y, z);
    A.linear() = q.toRotationMatrix();

    T = A;
    return true;
  }

  ROS_ERROR("[simod_vision_ctrl] loadBoxPoseFromPacksFile: unsupported JSON in '%s' (no box_center_0 and no packs[]).", path.c_str());
  return false;
}



bool SimodVisionCtrl::loadJsonFromFile(const std::string &path, Json::Value &root)
{
  std::ifstream ifs(path, std::ifstream::in);
  if (!ifs.is_open())
  {
    ROS_ERROR_STREAM("Cannot open JSON file: " << path);
    return false;
  }
  Json::CharReaderBuilder rbuilder;
  std::string errs;
  bool ok = Json::parseFromStream(rbuilder, ifs, &root, &errs);
  if (!ok)
  {
    ROS_ERROR_STREAM("Failed parsing JSON file " << path << ": " << errs);
  }
  return ok;
}

// apply pose offset to box center(detect by pallet_detection)
Eigen::Affine3d SimodVisionCtrl::applyBoxOffset(const Eigen::Affine3d &T_world_box,
                                                const Eigen::Matrix<double, 6, 1> &off_box_6)
{
  Eigen::Affine3d T_offset = Eigen::Affine3d::Identity();
  T_offset.translation() = Eigen::Vector3d(off_box_6[0], off_box_6[1], off_box_6[2]);
  T_offset.linear() = rpy(Eigen::Vector3d(off_box_6[3], off_box_6[4], off_box_6[5]));
  return T_world_box * T_offset;
}

Eigen::Affine3d SimodVisionCtrl::vec6ToAffineRPY(const Eigen::Matrix<double, 6, 1> &v)
{
  Eigen::Affine3d T = Eigen::Affine3d::Identity();
  T.translation() = v.head<3>();
  T.linear() = rpy(v.tail<3>()); // stesso ordine usato nei JSON (roll,pitch,yaw)
  return T;
}

// Converte una Affine3d in JSON con position + rpy (ZYX) e frame_id
Json::Value SimodVisionCtrl::affineToJson(const Eigen::Affine3d &T, const std::string &frame_id)
{
  Json::Value j;
  j["frame_id"] = frame_id;

  const Eigen::Vector3d p = T.translation();
  j["position"]["x"] = p.x();
  j["position"]["y"] = p.y();
  j["position"]["z"] = p.z();

  // Estrai RPY con convenzione ZYX (yaw-pitch-roll)
  const Eigen::Matrix3d &R = T.linear();
  const double yaw = std::atan2(R(1, 0), R(0, 0));
  const double pitch = std::asin(-R(2, 0));
  const double roll = std::atan2(R(2, 1), R(2, 2));

  j["rpy"]["roll"] = roll;
  j["rpy"]["pitch"] = pitch;
  j["rpy"]["yaw"] = yaw;

  return j;
}

bool SimodVisionCtrl::savePoseJson(const Json::Value &j,
                                   const std::string &path)
{
  std::ofstream ofs(path);
  if (!ofs.is_open())
  {
    ROS_ERROR("savePoseJson(JSON): cannot open '%s'", path.c_str());
    return false;
  }
  ofs << j.toStyledString();
  return true;
}

bool SimodVisionCtrl::callTrigger(const std::string &srv_name)
{
  std_srvs::Trigger srv;
  if (!ros::service::waitForService(srv_name, ros::Duration(2.0)))
  {
    ROS_ERROR("[simod_vision_ctrl] service '%s' not available.", srv_name.c_str());
    return false;
  }
  if (!ros::service::call(srv_name, srv))
  {
    ROS_ERROR("[simod_vision_ctrl] call to '%s' failed.", srv_name.c_str());
    return false;
  }
  if (!srv.response.success)
  {
    ROS_WARN("[simod_vision_ctrl] '%s' responded: %s",
             srv_name.c_str(), srv.response.message.c_str());
  }
  return srv.response.success;
}



geometry_msgs::Pose SimodVisionCtrl::eigenAffineToPose(const Eigen::Affine3d& T)
{
  geometry_msgs::Pose p;
  const auto t = T.translation();
  const Eigen::Quaterniond q(T.linear());
  p.position.x = t.x(); p.position.y = t.y(); p.position.z = t.z();
  p.orientation.x = q.x(); p.orientation.y = q.y();
  p.orientation.z = q.z(); p.orientation.w = q.w();
  return p;
}

// void SimodVisionCtrl::publishPackCollisionMesh(const Eigen::Affine3d& T_world_mesh)
// {
//   const std::string mesh_id   = "pack_env_mesh";
//   const std::string mesh_name = "pack" + std::to_string(pack_id_) + "_" + scenario_ + ".stl";

//   // Always resolve to a path *inside* the simod_rolls_detection package
//   const std::string pkg = ros::package::getPath("simod_rolls_detection"); // e.g. /home/.../simod_rolls_detection

//   // Normalize whatever was passed (absolute from $(find), or "simod_rolls_detection/files/input",
//   // or just "files/input") into a clean package-internal relative root like "files/input".
//   std::string root = pack_mesh_root_;

//   // If absolute and starts with the package path, strip that prefix.
//   if (root.rfind(pkg, 0) == 0) {
//     root = root.substr(pkg.size());               // drop /abs/.../simod_rolls_detection
//   }
//   // Drop leading slash if any.
//   if (!root.empty() && root[0] == '/') root.erase(0, 1);

//   // If still prefixed with the package name, strip it.
//   const std::string pkg_prefix = "simod_rolls_detection/";
//   if (root.rfind(pkg_prefix, 0) == 0) {
//     root = root.substr(pkg_prefix.size());        // "files/input"
//   }

//   // Trim trailing slash
//   if (!root.empty() && root.back() == '/') root.pop_back();

//   // Build absolute fs path from package + normalized relative root
//   const std::string abs_path     = pkg + "/" + root + "/" + mesh_name;
//   const std::string resource_uri = "file://" + abs_path;

//   // Load + scale mesh
//   std::unique_ptr<shapes::Mesh> mesh(
//       shapes::createMeshFromResource(resource_uri, Eigen::Vector3d::Constant(pack_mesh_scale_)));
//   if (!mesh) {
//     ROS_ERROR("[simod_vision_ctrl] mesh load failed: %s", resource_uri.c_str());
//     return;
//   }

//   shapes::ShapeMsg shape_msg;
//   shapes::constructMsgFromShape(mesh.get(), shape_msg);
//   const shape_msgs::Mesh* mesh_msg_ptr = boost::get<shape_msgs::Mesh>(&shape_msg);
//   if (!mesh_msg_ptr) {
//     ROS_ERROR("[simod_vision_ctrl] converted shape is not a mesh: %s", resource_uri.c_str());
//     return;
//   }

//   moveit_msgs::CollisionObject co;
//   co.header.frame_id = world_frame_id_;
//   co.id = mesh_id;
//   co.meshes.push_back(*mesh_msg_ptr);

//   geometry_msgs::Pose pose_msg;
//   tf::poseEigenToMsg(T_world_mesh, pose_msg);
//   co.mesh_poses.push_back(pose_msg);
//   co.operation = moveit_msgs::CollisionObject::ADD;

//   moveit::planning_interface::PlanningSceneInterface psi;
//   psi.removeCollisionObjects(std::vector<std::string>{mesh_id});
//   ros::Duration(0.05).sleep();
//   psi.applyCollisionObject(co);

//   ROS_INFO("Published collision mesh '%s' OK: %s  (root='%s', scenario=%s, pack_id=%d, scale=%g, frame=%s)",
//            mesh_id.c_str(), resource_uri.c_str(), root.c_str(), scenario_.c_str(), pack_id_, pack_mesh_scale_, world_frame_id_.c_str());
// }

void SimodVisionCtrl::publishPackCollisionMesh(const Eigen::Affine3d& T_world_mesh)
{
  if (!add_pack_client_.isValid()) {
    ROS_WARN("[simod_vision_ctrl] add_pack_client_ invalid; retrying connection to %s",
             add_pack_srv_name_.c_str());
    add_pack_client_ = nh_.serviceClient<ima_planner::AddPackEnvMesh>(add_pack_srv_name_, /*persistent=*/false);
    add_pack_client_.waitForExistence(ros::Duration(1.0));
  }

  ima_planner::AddPackEnvMesh srv;
  srv.request.object_id = "pack_env_mesh";
  srv.request.pack_id   = pack_id_;
  srv.request.scenario  = scenario_;
  srv.request.root_rel  = (pack_mesh_root_.empty() ? "files/input" : pack_mesh_root_);
  srv.request.scale     = pack_mesh_scale_;
  srv.request.frame_id  = world_frame_id_;

  // Pose from Eigen
  geometry_msgs::Pose pose_msg = eigenAffineToPose(T_world_mesh);
  srv.request.pose = pose_msg;

  // Guard against missing server
  if (!ros::service::waitForService(add_pack_srv_name_, ros::Duration(2.0))) {
    ROS_ERROR("[simod_vision_ctrl] service '%s' not available.", add_pack_srv_name_.c_str());
    return;
  }

  if (!add_pack_client_.call(srv) || !srv.response.ok) {
    ROS_ERROR("[simod_vision_ctrl] AddPackEnvMesh FAILED (service=%s, pack_id=%d, scenario=%s, root=%s, scale=%.3g, frame=%s)",
              add_pack_srv_name_.c_str(), pack_id_, scenario_.c_str(),
              srv.request.root_rel.c_str(), pack_mesh_scale_, world_frame_id_.c_str());
  } else {
    ROS_INFO("[simod_vision_ctrl] Collision mesh added (id=%s, pack_id=%d, scenario=%s, frame=%s)",
             srv.request.object_id.c_str(), pack_id_, scenario_.c_str(), world_frame_id_.c_str());
  }
}



//------------------------ CALLBACKS --------------------------------------------------//

void SimodVisionCtrl::jointStatesCallbackUR2(
    const sensor_msgs::JointState::ConstPtr &joints_state)
{
  joints_position_[UR2.getRobotIdx()] = Vector6d(joints_state->position.data());
  joints_velocity_[UR2.getRobotIdx()] = Vector6d(joints_state->velocity.data());
  joints_torques_[UR2.getRobotIdx()] = Vector6d(joints_state->effort.data());
  if (reversed_joints_)
  {
    std::swap(joints_position_[UR2.getRobotIdx()](0),
              joints_position_[UR2.getRobotIdx()](2));
    std::swap(joints_velocity_[UR2.getRobotIdx()](0),
              joints_velocity_[UR2.getRobotIdx()](2));
    std::swap(joints_torques_[UR2.getRobotIdx()](0),
              joints_torques_[UR2.getRobotIdx()](2));
  }
  init_flags_[UR2.getRobotIdx()].set(JOINTS);
}

void SimodVisionCtrl::safetyModeCallbackUR2(
    const ur_dashboard_msgs::SafetyMode::ConstPtr &safety_mode_msg)
{
  safety_mode_[UR2.getRobotIdx()] = safety_mode_msg->mode;
  init_flags_[UR2.getRobotIdx()].set(SAFETY_MODE);
}

void SimodVisionCtrl::robotModeCallbackUR2(
    const ur_dashboard_msgs::RobotMode::ConstPtr &robot_mode_msg)
{
  robot_mode_[UR2.getRobotIdx()] = robot_mode_msg->mode;
  init_flags_[UR2.getRobotIdx()].set(ROBOT_MODE);
}
