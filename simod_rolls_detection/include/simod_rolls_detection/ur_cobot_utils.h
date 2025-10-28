#ifndef UR_COBOT_UTILS_H
#define UR_COBOT_UTILS_H

#include "ros/ros.h"
#include "tf2/convert.h"
#include "tf2_eigen/tf2_eigen.h"
#include "ima_planner/CollisionObject.h"
#include "ima_planner/DesiredTcpPose.h"
#include "ima_planner/DesiredJointsAngles.h"
#include "ima_planner/DesiredTcpVel.h"
#include "ur_msgs/SetPayload.h"
#include "ur_msgs/SetSpeedSliderFraction.h"
#include "std_srvs/Trigger.h"
#include "simod_rolls_detection/utils_rot.h"

#include <fstream>
#include <jsoncpp/json/json.h>

// Eigen tipi aggiuntivi
#include <Eigen/Geometry>
#include <vector>
#include <string>

/* ----------------------- Strutture dati ----------------------- */

struct CartesianWaypoints
{
  Eigen::Affine3d start;
  Eigen::Affine3d mid;
  Eigen::Affine3d approach;
  Eigen::Affine3d departure;
  Eigen::Affine3d grasping;
};

struct JointsWaypoints
{
  Vector6d start;
};

/**
 * @brief Class collecting general utilities for UR robots.
 */
class URCobotUtils
{
public:

  /************** STRUCTURES **************/
  // Offset definiti come vettori [x y z roll pitch yaw] nel frame box.
  // Servono per costruire le pose relative (Affine3d) partendo da box_pose

  // Offsets per lo scenario S1
  struct S1Offsets
  {
    Eigen::Matrix<double, 6, 1> start;
    Eigen::Matrix<double, 6, 1> approach;
    Eigen::Matrix<double, 6, 1> grasping;
    Eigen::Matrix<double, 6, 1> homing;
  };

  // Offsets per lo scenario S2
  struct S2Offsets
  {
    Eigen::Matrix<double, 6, 1> paddle_insertion;
    Eigen::Matrix<double, 6, 1> side;
    Eigen::Matrix<double, 6, 1> homing;
  };

  // Struttura di punti cartesiani per lo scenario S1
  struct S1ArmWaypoints
  {
    Eigen::Affine3d start;
    Eigen::Affine3d approach;
    Eigen::Affine3d grasping;
    Eigen::Affine3d homing;
  };

  // Struttura di punti cartesiani per lo scenario S2
  struct S2ArmWaypoints
  {
    Eigen::Affine3d paddle_insertion;
    Eigen::Affine3d side;
    Eigen::Affine3d homing;
  };

  struct S1Waypoints
  {
    S1ArmWaypoints ur1;
    S1ArmWaypoints ur2;
  };

  struct S2Waypoints
  {
    S2ArmWaypoints ur1;
    S2ArmWaypoints ur2;
  };

  /************** ATTRIBUTES **************/
  ros::ServiceClient movep_client;
  ros::ServiceClient movej_client;
  ros::ServiceClient movev_client;

  /************** CONSTRUCTOR **************/
  URCobotUtils();

  /************** METHODS **************/
  void initPlanning(ros::NodeHandle &nh, const std::string &robot_id);
  bool addCollisionObject(ros::NodeHandle &nh);
  bool removeCollisionObject(ros::NodeHandle &nh);
  bool setPayload(const float mass, const float center_of_gravity[3], std::string robot_id, ros::NodeHandle &nh);
  bool setSpeedSlider(const float fraction, std::string robot_id, ros::NodeHandle &nh);
  bool zeroForceTorqueSensor(std::string robot_id, ros::NodeHandle &nh);

  // Waypoints (già presenti)
  CartesianWaypoints readCartesianWaypointsFromJSON(const std::string &json_filepath);
  JointsWaypoints readJointsWaypointsFromJSON(const std::string &json_filepath);

  /* --------- NUOVE UTILITY “ACCESSORIE” A PATH FILE  --------- */

  // utility per leggere [x y z r p y] da JSON in un vettore di 6 double
  static bool parseVec6RPY(const Json::Value &arr, Eigen::Matrix<double, 6, 1> &out);

  // applica l’offset [x y z r p y] nel frame box
  static Eigen::Affine3d applyBoxOffset(const Eigen::Affine3d &T_0_box,
                                        const Eigen::Matrix<double, 6, 1> &off);

  // offsets e builders
  bool loadBoxPoseFromFile(const std::string &filepath, Eigen::Affine3d &T_0_box);
  static bool loadS1OffsetsFromFile(const std::string &filepath, S1Offsets &ur1, S1Offsets &ur2);
  static bool loadS2OffsetsFromFile(const std::string &filepath, S2Offsets &ur1, S2Offsets &ur2);
  static void buildS1Waypoints(const Eigen::Affine3d &T_0_box,
                               const S1Offsets &o1, const S1Offsets &o2, S1Waypoints &out);
  static void buildS2Waypoints(const Eigen::Affine3d &T_0_box,
                               const S2Offsets &o1, const S2Offsets &o2, S2Waypoints &out);

private:
  /************** ATTRIBUTES **************/
  // services
  ros::ServiceClient collision_object_client;
  ima_planner::CollisionObject collision_object_srv;
  uint SIMPLE_BOX = 19; // TODO: valutare libreria comune con ima_planner
  uint ADD = 0;
  uint REMOVE = 2;

  ros::ServiceClient set_payload_client;
  ur_msgs::SetPayload set_payload_srv;

  ros::ServiceClient set_speed_slider_client;
  ur_msgs::SetSpeedSliderFraction set_speed_slider_srv;

  ros::ServiceClient zero_ft_sensor_client;
  std_srvs::Trigger zero_ft_sensor_srv;
};

#endif /* UR_COBOT_UTILS_H */
