#include "simod_rolls_detection/ur_cobot_utils.h"

URCobotUtils::URCobotUtils() {}

/**
 * @brief Launches the services needed for communicating a desired tcp pose and add/remove collision objects
 */
void URCobotUtils::initPlanning(ros::NodeHandle &nh, const std::string &robot_id)
{
  movep_client = nh.serviceClient<ima_planner::DesiredTcpPose>("/ima_planner/desired_tcp_pose");
  movej_client = nh.serviceClient<ima_planner::DesiredJointsAngles>("/ima_planner/desired_joints_angles");
  // movev è robot-specifico
  movev_client = nh.serviceClient<ima_planner::DesiredTcpVel>(
      "/arm" + robot_id + "/moveit_servo_wrapper/desired_tcp_vel");
}

/**
 * @brief add collision object to the scene, right now it is just for the SIMPLE_BOX type
 */
bool URCobotUtils::addCollisionObject(ros::NodeHandle &nh)
{
  collision_object_client = nh.serviceClient<ima_planner::CollisionObject>("/ima_planner/collision_object");
  collision_object_srv.request.object_id = "simple_box";
  collision_object_srv.request.object_type = SIMPLE_BOX;
  collision_object_srv.request.action = ADD;
  collision_object_srv.request.calc_normals = false;

  if (!collision_object_client.call(collision_object_srv))
  {
    ROS_FATAL("Could not call collision object service");
    ros::shutdown();
    return false;
  }

  if (collision_object_srv.response.ok)
  {
    return true;
  }
  else
  {
    ROS_FATAL("Could not load collision object to the scene");
    ros::shutdown();
    return false;
  }
}

/**
 * @brief remove collision object from the scene, right now it is just for the SIMPLE_BOX type
 */
bool URCobotUtils::removeCollisionObject(ros::NodeHandle &nh)
{
  collision_object_client = nh.serviceClient<ima_planner::CollisionObject>("/ima_planner/collision_object");
  collision_object_srv.request.object_id = "simple_box";
  collision_object_srv.request.object_type = SIMPLE_BOX;
  collision_object_srv.request.action = REMOVE;
  collision_object_srv.request.calc_normals = false;

  if (!collision_object_client.call(collision_object_srv))
  {
    ROS_FATAL("Could not call collision object service");
    ros::shutdown();
    return false;
  }

  if (collision_object_srv.response.ok)
  {
    return true;
  }
  else
  {
    ROS_FATAL("Could not unload collision object to the scene");
    ros::shutdown();
    return false;
  }
}

/**
 * @brief set the payload of the robot
 */
bool URCobotUtils::setPayload(const float mass, const float center_of_gravity[3],
                              std::string robot_id, ros::NodeHandle &nh)
{
  set_payload_client = nh.serviceClient<ur_msgs::SetPayload>("/arm" + robot_id + "/ur_hardware_interface/set_payload");

  set_payload_srv.request.mass = mass;
  set_payload_srv.request.center_of_gravity.x = center_of_gravity[0];
  set_payload_srv.request.center_of_gravity.y = center_of_gravity[1];
  set_payload_srv.request.center_of_gravity.z = center_of_gravity[2];

  if (!set_payload_client.call(set_payload_srv))
  {
    ROS_FATAL("Could not call set payload service");
    ros::shutdown();
    return false;
  }

  if (set_payload_srv.response.success)
  {
    ROS_INFO("arm%s payload set to m=%f cog=%f,%f,%f",
             robot_id.c_str(), mass, center_of_gravity[0], center_of_gravity[1], center_of_gravity[2]);
    return true;
  }
  else
  {
    ROS_FATAL("Could not set tool payload");
    ros::shutdown();
    return false;
  }
}

/**
 * @brief set the speed slider of the robot
 */
bool URCobotUtils::setSpeedSlider(const float fraction, std::string robot_id, ros::NodeHandle &nh)
{
  set_speed_slider_client = nh.serviceClient<ur_msgs::SetSpeedSliderFraction>("/arm" + robot_id + "/ur_hardware_interface/set_speed_slider");
  set_speed_slider_srv.request.speed_slider_fraction = fraction;

  if (!set_speed_slider_client.call(set_speed_slider_srv))
  {
    ROS_FATAL("Could not call set speed slider service");
    ros::shutdown();
    return false;
  }

  if (set_speed_slider_srv.response.success)
  {
    ROS_INFO("arm%s speed slider set to %f", robot_id.c_str(), fraction);
    return true;
  }
  else
  {
    ROS_FATAL("Could not set speed slider value");
    ros::shutdown();
    return false;
  }
}

/**
 * @brief Zeros the force/torque sensor on the UR robot.
 */
bool URCobotUtils::zeroForceTorqueSensor(std::string robot_id, ros::NodeHandle &nh)
{
  zero_ft_sensor_client = nh.serviceClient<std_srvs::Trigger>("/arm" + robot_id + "/ur_hardware_interface/zero_ftsensor");

  if (!zero_ft_sensor_client.call(zero_ft_sensor_srv))
  {
    ROS_FATAL("Could not call zero force torque service");
    ros::shutdown();
    return false;
  }

  if (zero_ft_sensor_srv.response.success)
  {
    ROS_INFO("arm%s force torque sensor has been zeroed", robot_id.c_str());
    return true;
  }
  else
  {
    ROS_FATAL("%s", zero_ft_sensor_srv.response.message.c_str());
    ros::shutdown();
    return false;
  }
}

/* ===================== JSON Waypoints (già presenti) ===================== */

CartesianWaypoints URCobotUtils::readCartesianWaypointsFromJSON(const std::string &json_filepath)
{
  std::ifstream json_file(json_filepath);
  if (!json_file.is_open())
  {
    ROS_INFO("Error opening JSON file: %s", json_filepath.c_str());
    ros::shutdown();
  }

  Json::Reader reader;
  Json::Value root;
  json_file >> root;

  if (!root.isMember("start") || !root.isMember("approach") ||
      !root.isMember("grasping") || !root.isMember("mid") || !root.isMember("departure"))
  {
    ROS_INFO("Invalid JSON format: Missing required keys (start, approach, grasping, mid, departure)");
    ros::shutdown();
  }

  CartesianWaypoints waypoints;

  const std::vector<std::string> key_names = {"start", "mid", "approach", "grasping", "departure"};
  for (const std::string &key_name : key_names)
  {
    if (!root[key_name].isArray() || root[key_name].size() != 6)
    {
      ROS_INFO("Invalid JSON format: %s must be an array of 6 elements", key_name.c_str());
      continue;
    }

    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.translation() << root[key_name][0].asDouble(),
        root[key_name][1].asDouble(),
        root[key_name][2].asDouble();

    Eigen::Vector3d rpy_angles(root[key_name][3].asDouble(),
                               root[key_name][4].asDouble(),
                               root[key_name][5].asDouble());
    T.linear() = rpy(rpy_angles);

    if (key_name == "start")
      waypoints.start = T;
    else if (key_name == "mid")
      waypoints.mid = T;
    else if (key_name == "approach")
      waypoints.approach = T;
    else if (key_name == "grasping")
      waypoints.grasping = T;
    else if (key_name == "departure")
      waypoints.departure = T;
  }

  return waypoints;
}

JointsWaypoints URCobotUtils::readJointsWaypointsFromJSON(const std::string &json_filepath)
{
  std::ifstream json_file(json_filepath);
  if (!json_file.is_open())
  {
    ROS_INFO("Error opening JSON file: %s", json_filepath.c_str());
    ros::shutdown();
  }

  Json::Reader reader;
  Json::Value root;
  json_file >> root;

  if (!root.isMember("start"))
  {
    ROS_INFO("Invalid JSON format: Missing required key 'start'");
    ros::shutdown();
  }

  JointsWaypoints waypoints;
  if (!root["start"].isArray() || root["start"].size() != 6)
  {
    ROS_INFO("Invalid JSON format: start must be an array of 6 elements");
  }
  else
  {
    Vector6d q;
    q << root["start"][0].asDouble(), root["start"][1].asDouble(),
        root["start"][2].asDouble(), root["start"][3].asDouble(),
        root["start"][4].asDouble(), root["start"][5].asDouble();
    waypoints.start = q;
  }
  return waypoints;
}

/* ===================== NUOVE UTILITY ACCESSORIE ===================== */

bool URCobotUtils::parseVec6RPY(const Json::Value& arr, Eigen::Matrix<double,6,1>& out) {
  if (!arr.isArray() || arr.size() != 6) return false;
  for (int i = 0; i < 6; ++i) out[i] = arr[i].asDouble();
  return true;
}

/**
 * @brief Carica la posa del box (position + rpy) da file JSON
 * Struttura attesa:
 * {
 *   "box_pose": {
 *     "position": {"x":..., "y":..., "z":...},
 *     "rpy": {"roll":..., "pitch":..., "yaw":...}
 *   }
 * }
 */
bool URCobotUtils::loadBoxPoseFromFile(const std::string& filepath, Eigen::Affine3d& T_0_box)
{
  std::ifstream f(filepath);
  if (!f.is_open()) {
    ROS_ERROR_STREAM("[loadBoxPoseFromFile] Cannot open JSON: " << filepath);
    return false;
  }
  Json::Value root; f >> root;

  if (root.isMember("items") && root["items"].isArray() && !root["items"].empty()) {
    const auto& item0 = root["items"][0];
    if (item0.isMember("box_pose")) {
      const auto& pose = item0["box_pose"];
      if (pose.isMember("position") && pose.isMember("rpy")) {
        const auto& p = pose["position"];
        const auto& r = pose["rpy"];
        if (p.isMember("x") && p.isMember("y") && p.isMember("z") &&
            r.isMember("roll") && r.isMember("pitch") && r.isMember("yaw")) {
          Eigen::Affine3d T = Eigen::Affine3d::Identity();
          T.translation() = Eigen::Vector3d(p["x"].asDouble(),
                                            p["y"].asDouble(),
                                            p["z"].asDouble());
          Eigen::Vector3d rpy_angles(r["roll"].asDouble(),
                                     r["pitch"].asDouble(),
                                     r["yaw"].asDouble());
          T.linear() = rpy(rpy_angles);  
          T_0_box = T;
          return true;
        }
      }
    }
  }

  // Fallback opzionali (se mai servissero):
  // 1) "box_poses": [ {position,rpy}, ... ]
  if (root.isMember("box_poses") && root["box_poses"].isArray() && !root["box_poses"].empty()) {
    const auto& pose0 = root["box_poses"][0];
    if (pose0.isMember("position") && pose0.isMember("rpy")) {
      const auto& p = pose0["position"];
      const auto& r = pose0["rpy"];
      if (p.isMember("x") && p.isMember("y") && p.isMember("z") &&
          r.isMember("roll") && r.isMember("pitch") && r.isMember("yaw")) {
        Eigen::Affine3d T = Eigen::Affine3d::Identity();
        T.translation() = Eigen::Vector3d(p["x"].asDouble(),
                                          p["y"].asDouble(),
                                          p["z"].asDouble());
        Eigen::Vector3d rpy_angles(r["roll"].asDouble(),
                                   r["pitch"].asDouble(),
                                   r["yaw"].asDouble());
        T.linear() = rpy(rpy_angles);
        T_0_box = T;
        return true;
      }
    }
  }

  ROS_ERROR_STREAM("[loadBoxPoseFromFile] No valid pose found in " << filepath);
  return false;
}


/**
 * @brief Carica gli offset S1 da file JSON.
 * Struttura attesa:
 * {
 *   "offsets_s1": {
 *     "ur1": { "start":[6], "approach":[6], "grasping":[6], "homing":[6] },
 *     "ur2": { ... }
 *   }
 * }
 */
bool URCobotUtils::loadS1OffsetsFromFile(const std::string& filepath,
                                         S1Offsets& ur1, S1Offsets& ur2)
{
  std::ifstream f(filepath);
  if (!f.is_open()) {
    ROS_ERROR_STREAM("[loadS1OffsetsFromFile] Cannot open JSON: " << filepath);
    return false;
  }
  Json::Value root; f >> root;

  if (!root.isMember("offsets_s1") ||
      !root["offsets_s1"].isMember("ur1") ||
      !root["offsets_s1"].isMember("ur2")) {
    ROS_ERROR_STREAM("[loadS1OffsetsFromFile] Missing offsets_s1/ur1/ur2 in " << filepath);
    return false;
  }

  const auto& u1 = root["offsets_s1"]["ur1"];
  const auto& u2 = root["offsets_s1"]["ur2"];

  bool ok = true;
  ok &= URCobotUtils::parseVec6RPY(u1["start"],    ur1.start);
  ok &= URCobotUtils::parseVec6RPY(u1["approach"], ur1.approach);
  ok &= URCobotUtils::parseVec6RPY(u1["grasping"], ur1.grasping);
  ok &= URCobotUtils::parseVec6RPY(u1["homing"],   ur1.homing);

  ok &= URCobotUtils::parseVec6RPY(u2["start"],    ur2.start);
  ok &= URCobotUtils::parseVec6RPY(u2["approach"], ur2.approach);
  ok &= URCobotUtils::parseVec6RPY(u2["grasping"], ur2.grasping);
  ok &= URCobotUtils::parseVec6RPY(u2["homing"],   ur2.homing);

  if (!ok) {
    ROS_ERROR_STREAM("[loadS1OffsetsFromFile] Some vectors are not length-6 in " << filepath);
    return false;
  }
  return true;
}



/**
 * @brief Applica i vari offsets alla posizone del box.
 */

Eigen::Affine3d URCobotUtils::applyBoxOffset(const Eigen::Affine3d& T_0_box,
                                             const Eigen::Matrix<double,6,1>& off)
{
  Eigen::Affine3d T = Eigen::Affine3d::Identity();
  T.translation() = Eigen::Vector3d(off[0], off[1], off[2]);

  Eigen::Vector3d rpy_angles(off[3], off[4], off[5]); // roll, pitch, yaw
  T.linear() = rpy(rpy_angles);                       

  return T_0_box * T;
}


/**
 * @brief Costruisce il piano S1 a partire da box e offsets.
 */
void URCobotUtils::buildS1Waypoints(const Eigen::Affine3d &T_0_box,
                                    const S1Offsets &o1,
                                    const S1Offsets &o2,
                                    S1Waypoints &out)

{
  // UR1
  out.ur1.start = applyBoxOffset(T_0_box, o1.start);
  out.ur1.approach = applyBoxOffset(T_0_box, o1.approach);
  out.ur1.grasping = applyBoxOffset(T_0_box, o1.grasping);
  out.ur1.homing = applyBoxOffset(T_0_box, o1.homing);
  // UR2
  out.ur2.start = applyBoxOffset(T_0_box, o2.start);
  out.ur2.approach = applyBoxOffset(T_0_box, o2.approach);
  out.ur2.grasping = applyBoxOffset(T_0_box, o2.grasping);
  out.ur2.homing = applyBoxOffset(T_0_box, o2.homing);
}

// S2 helpers

/**
 * @brief Carica gli offset S2 da file JSON.
 * Struttura attesa:
 * {
 *   "offsets_s2": {
 *     "ur1": { "paddle_insertion":[6], "side":[6], "homing":[6] },
 *     "ur2": { "side":[6], "homing":[6] }
 *   }
 * }
 */
bool URCobotUtils::loadS2OffsetsFromFile(const std::string& filepath,
                                         S2Offsets& ur1, S2Offsets& ur2)
{
  std::ifstream f(filepath);
  if (!f.is_open()) {
    ROS_ERROR_STREAM("[loadS2OffsetsFromFile] Cannot open JSON: " << filepath);
    return false;
  }
  Json::Value root; f >> root;

  if (!root.isMember("offsets_s2") ||
      !root["offsets_s2"].isMember("ur1") ||
      !root["offsets_s2"].isMember("ur2")) {
    ROS_ERROR_STREAM("[loadS2OffsetsFromFile] Missing offsets_s2/ur1/ur2 in " << filepath);
    return false;
  }

  const auto& u1 = root["offsets_s2"]["ur1"];
  const auto& u2 = root["offsets_s2"]["ur2"];

  bool ok = true;
  ok &= parseVec6RPY(u1["paddle_insertion"], ur1.paddle_insertion);
  ok &= parseVec6RPY(u1["side"],             ur1.side);
  ok &= parseVec6RPY(u1["homing"],           ur1.homing);

  ok &= parseVec6RPY(u2["side"],             ur2.side);
  ok &= parseVec6RPY(u2["homing"],           ur2.homing);

  if (!ok) {
    ROS_ERROR_STREAM("[loadS2OffsetsFromFile] Some vectors are not length-6 in " << filepath);
    return false;
  }
  return true;
}

void URCobotUtils::buildS2Waypoints(const Eigen::Affine3d &T_0_box,
                                    const S2Offsets &ur1_offs,
                                    const S2Offsets &ur2_offs,
                                    S2Waypoints &out)
{
  // UR1
  out.ur1.paddle_insertion = applyBoxOffset(T_0_box, ur1_offs.paddle_insertion);
  out.ur1.side = applyBoxOffset(T_0_box, ur1_offs.side);
  out.ur1.homing = applyBoxOffset(T_0_box, ur1_offs.homing);

  // UR2 (non ha paddle_insertion)
  // Se vuoi tenerla valorizzata comunque, puoi mettere identity o copiare side/homing.
  out.ur2.paddle_insertion = Eigen::Affine3d::Identity();
  out.ur2.side = applyBoxOffset(T_0_box, ur2_offs.side);
  out.ur2.homing = applyBoxOffset(T_0_box, ur2_offs.homing);
}
