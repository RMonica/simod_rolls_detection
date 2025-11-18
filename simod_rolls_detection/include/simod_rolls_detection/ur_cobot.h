#ifndef UR_COBOT_H
#define UR_COBOT_H

// ROS
#include "ros/ros.h"
#include "tf2/convert.h"
#include "tf2_eigen/tf2_eigen.h"
#include "tf/transform_listener.h"
#include "eigen_conversions/eigen_msg.h"
#include "controller_manager_msgs/SwitchController.h"

// Utilities
#include "simod_rolls_detection/ur_cobot_utils.h"
#include "simod_rolls_detection/utils_rot.h"
#include "ima_planner/DesiredTcpPose.h"
#include "ima_planner/DesiredJointsAngles.h"
#include "ima_planner/DesiredTcpVel.h"

enum ControlMode : uint
{
    CARTESIAN_POSITION = 1,
    CARTESIAN_VELOCITY = 2,
    JOINT_VELOCITY = 3
};

enum class ActionState
{
    MOVE_TO_TARGET,
    WAIT_TARGET,
    START_INSERTION,
    WAIT_INSERTION,
    START_DETACH,
    WAIT_DETACH,
    DONE
};

struct AdmittanceParams {
    // Translational matrices
    Eigen::Matrix3d MEt;  // Translational mass matrix
    Eigen::Matrix3d DEt;  // Translational damping matrix
    Eigen::Matrix3d KEt;  // Translational stiffness matrix

    // Rotational matrices
    Eigen::Matrix3d MEr;  // Rotational mass matrix
    Eigen::Matrix3d DEr;  // Rotational damping matrix
    Eigen::Matrix3d KEr;  // Rotational stiffness matrix

    Vector6d y_P;
    Vector7d y_O;

    double phi;
    double db;
    int counter;
    
};


/**
 * @brief Class for moving UR robots using the planner.
 *
 */
class URCobot
{

public:
    /************** ATTRIBUTES **************/
    ActionState action_state;

    /************** CONSTRUCTOR **************/
    URCobot(const std::string robot_id, ros::NodeHandle &nh);

    /************** METHODS **************/
    void initControllerServices();
    bool switchController(const ControlMode mode);
    void initPlanning();
    bool movep(Eigen::Affine3d pose_E_in_0);
    bool movej(const Vector6d &q, bool reverse);
    bool movev(const Eigen::Vector3d &axis, double vel,
               const std::string &frame,
               double until_dist, double until_contact,
               double retract);
    Eigen::Affine3d to0Frame(Eigen::Affine3d pose_in_B);
    // void updateCurrentPose(const Eigen::Affine3d z_E);
    bool isAtTarget(const Eigen::Affine3d pose_E_in_0, double tol);
   
    bool waitUntilReached(const Eigen::Affine3d &target_pose_in_0, double tol, double timeout_sec);

    bool reachPose(const Eigen::Affine3d &target, double tol = 0.01);
    // Esegue la paddle insertion: porta il TCP in pre-inserzione e poi spinge lungo -Z_tcp
    void paddleInsertion(double contact_N,
                                double speed,
                                double max_dist);
    void detachment(double distance);
    // Kinematics
    void readJointVelLimits();
    void readKinParams();
    Matrix6d calcJacobian(const Vector6d &q);
    bool updateKinematics(const Vector6d &q, const Vector6d &q_dot);
    Vector6d invertKinematics(const Vector6d &twist_E_in_B);

    

   

    // getters and setters
    Vector6d getJointVelocityLimits() const;
    Eigen::Affine3d getT_0_B() const;
    Eigen::Affine3d getT_B_0() const;
    Eigen::Affine3d getCurrentPose() const;
    Vector6d getCurrentTwist() const;
    Vector6d getCurrentWrench() const;
    Vector6d getCurrentInternalWrench() const;
    Matrix6d getCurrentJacobian() const;
    Eigen::Affine3d getDesiredPose() const;
    Vector6d getDesiredTwist() const;
    Vector6d getDesiredAcceleration() const;
    Vector6d getDesiredWrench() const;
    std::string getRobotID() const;
    Eigen::Vector3d getAdmittanceForceError() const;
    int getCounter() const;
    double getAdmittancePhi() const;
    int getRobotIdx() const;
    void setCurrentWrench(const Vector6d wrench);
    void setCurrentInternalWrench(const Vector6d wrench);
    void setDesiredPose(const Eigen::Affine3d &pose_E_des_in_0);
    void setDesiredTwist(const Vector6d &twist_E_des_in_0);
    void setDesiredAcceleration(const Vector6d &acc_E_des_in_0);
    void setDesiredWrench(const Vector6d &wrench);
    void setAdmittanceError(const Vector6d position_error, const Vector7d angular_error);
    void setExternalAdmittanceError(const Vector6d position_error, const Vector7d angular_error);
    void setAdmittanceForceError(const Eigen::Vector3d old_error);
    void setCounter(const int &counter);
    void setAdmittancePhi(const double &phi);
    
private:
    /************** ATTRIBUTES **************/
    std::string robot_id = ""; // 1 for robot 1, 2 for robot 2
    int robot_idx = 0;         // robot_id-1
    URCobotUtils utilities;
    ros::NodeHandle nh;

    double rate_;

    // Services
    ros::ServiceClient switch_controller_client;
    ima_planner::DesiredTcpPose movep_srv;
    ima_planner::DesiredJointsAngles movej_srv;
    ima_planner::DesiredTcpVel movev_srv;
    ros::ServiceClient desired_tcp_vel_client;
   
    // TF
    tf::TransformListener tf_listener_;

    // Kinematics
    Matrix44Nd A;
    Matrix6d X;
    Matrix6d Y;
    Matrix6d J; // End Effector Jacobian in Base frame
    Vector6d joint_velocity_limits;

    Eigen::Affine3d z_E; // End Effector Pose in Base frame
    Vector6d z_E_dot;    // End Effector Twist in Base frame
    Vector6d h;          // End Effector Wrench in EE frame
    Vector6d h_int;      // End Effector Internal Wrench in EE frame

    Eigen::Affine3d z_E_des;     // Desired End Effector Pose in 0 frame
    Vector6d z_E_dot_des;        // Desired End Effector Twist in 0 frame
    Vector6d z_E_ddot_des;       // Desired End Effector Acceleration in 0 frame
    Vector6d h_des;              // Desired End Effector Wrench in EE frame TODO: check in which frame it is defined
    Eigen::Vector3d force_error; // Force error along z axis in EE frame
    int counter;
    double phi;

    std::string planning_group;
    std::string frame_id; // FIXME: check if it is the same of base_frame/EE_frame
    std::string base_frame_id;
    std::string EE_frame_id;
    Eigen::Affine3d T_0_B = Eigen::Affine3d::Identity(); // from B to 0
    Eigen::Affine3d T_B_0 = Eigen::Affine3d::Identity();

    // Control
    AdmittanceParams admittance_params{};
    AdmittanceParams external_admittance_params{};

    Eigen::Affine3d target_approach_pose;
    Eigen::Affine3d target_insertion_pose;
    Eigen::Affine3d target_detach_pose;

};

#endif /* UR_COBOT_H */
