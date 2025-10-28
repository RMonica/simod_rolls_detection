#include "simod_rolls_detection/ur_cobot.h"

URCobot::URCobot(const std::string robot_id, ros::NodeHandle &nh)
{

    if (!(robot_id == "1" || robot_id == "2"))
    {
        ROS_ERROR("Invalid robot_id: %s", robot_id.c_str());
        ros::shutdown();
    }

    this->nh = nh;
    this->robot_id = robot_id;
    this->robot_idx = std::stoi(robot_id) - 1;

    // read rate
    if (!nh.getParam("simod_vision_ctrl/rate", rate_))
    {
        ROS_WARN("Rate parameter not found, using default value 500 Hz");
        rate_ = 500.0;
    }

    // read frame names
    nh.getParam("simod_vision_ctrl/arm" + robot_id + "_base_name", base_frame_id);
    nh.getParam("simod_vision_ctrl/arm" + robot_id + "_EE_name", EE_frame_id);

    // read joint velocity limits
    readJointVelLimits();

    // read configuration parameters A,X,Y
    readKinParams();

    double x_0_B, y_0_B, z_0_B, roll_0_B, pitch_0_B, yaw_0_B;
    // retrieve robot base position in world frame
    // FIXME: change name according to approriate tf and add a corresponding tf. Maybe with tf we can even remove these params and add everything on urdf
    // TODO write a function to address this read
    nh.getParam("simod_vision_ctrl/arm" + robot_id + "_base_in_world/x", x_0_B);
    nh.getParam("simod_vision_ctrl/arm" + robot_id + "_base_in_world/y", y_0_B);
    nh.getParam("simod_vision_ctrl/arm" + robot_id + "_base_in_world/z", z_0_B);
    nh.getParam("simod_vision_ctrl/arm" + robot_id + "_base_in_world/roll", roll_0_B);
    nh.getParam("simod_vision_ctrl/arm" + robot_id + "_base_in_world/pitch", pitch_0_B);
    nh.getParam("simod_vision_ctrl/arm" + robot_id + "_base_in_world/yaw", yaw_0_B);

    // From B to 0
    T_0_B.linear() = (Eigen::AngleAxisd(roll_0_B, Eigen::Vector3d::UnitY()) *
                      Eigen::AngleAxisd(pitch_0_B, Eigen::Vector3d::UnitY()) *
                      Eigen::AngleAxisd(yaw_0_B, Eigen::Vector3d::UnitZ()))
                         .toRotationMatrix();
    T_0_B.translation() << x_0_B, y_0_B, z_0_B;

    // From 0 to B
    T_B_0 = T_0_B.inverse();

    utilities.initPlanning(nh, robot_id);

    // Set speed slider
    float speed_slider_fraction;
    nh.getParam("/simod_vision_ctrl/arm" + robot_id + "_speed_slider", speed_slider_fraction);
    utilities.setSpeedSlider(speed_slider_fraction, robot_id, nh);

    // Set payload parameters
    float tool_mass;
    std::vector<float> tool_cog;
    nh.getParam("/simod_vision_ctrl/arm" + robot_id + "/tool_mass", tool_mass);
    nh.getParam("/simod_vision_ctrl/arm" + robot_id + "/tool_cog", tool_cog);
    utilities.setPayload(tool_mass, tool_cog.data(), robot_id, nh);
    utilities.zeroForceTorqueSensor(robot_id, nh);

    // retrieve planner variables
    nh.getParam("/ima_planner/planning_group" + robot_id, planning_group);
    nh.getParam("/ima_planner/tcp_name" + robot_id, frame_id);

    initControllerServices();
    switchController(CARTESIAN_POSITION);

}

void URCobot::initControllerServices()
{
    switch_controller_client = nh.serviceClient<controller_manager_msgs::SwitchController>("/arm" + robot_id + "/controller_manager/switch_controller");
}

/**
 * @brief choose which ROS controller should be started
 *
 * @param mode CARTESIAN_POSITION, CARTESIAN_VELOCITY, JOINT_VELOCITY
 * @return true
 * @return false
 *
 */
bool URCobot::switchController(const ControlMode mode)
{

    controller_manager_msgs::SwitchController cmd;

    switch (mode)
    {
    case CARTESIAN_POSITION:
        cmd.request.start_controllers.push_back("scaled_pos_joint_traj_controller");
        cmd.request.stop_controllers = {"joint_group_vel_controller",
                                        "vel_joint_traj_controller"};
        ROS_INFO("Switching to cartesian scaled_pos_joint_traj_controller");
        break;

    case CARTESIAN_VELOCITY: // TODO: check if it is actually cartesian velocity
        cmd.request.start_controllers.push_back("vel_joint_traj_controller");
        cmd.request.stop_controllers = {"joint_group_vel_controller",
                                        "scaled_pos_joint_traj_controller"};
        ROS_INFO("Switching to vel_joint_traj_controller");
        break;

    case JOINT_VELOCITY:
        cmd.request.start_controllers.push_back("joint_group_vel_controller");
        cmd.request.stop_controllers = {"vel_joint_traj_controller",
                                        "scaled_pos_joint_traj_controller"};
        ROS_INFO("Switching to joint_group_vel_controller");
        break;
    }

    if (!switch_controller_client.call(cmd))
    {
        // ROS_FATAL("Could not switch to arm%s controller: 'joint_group_vel_controller'.", robot_id);
        ROS_FATAL_STREAM("Could not switch to arm" << robot_id << "controller: 'joint_group_vel_controller'.");
        ros::shutdown();
        return false;
    }

    return true;
}

/**
 * @brief Planned motion that makes use of dynamic planner. Motion command is executed via ros_control (check which topic is responsible of collecting values from planned trajectory)
 * @param pose_E_in_0 desired pose of end effector in 0 frame
 * @return true
 * @return false
 */
bool URCobot::movep(Eigen::Affine3d pose_E_in_0)
{

    Eigen::Affine3d desired_TCP_frame_B = T_B_0 * pose_E_in_0;
    geometry_msgs::Pose desired_TCP_pose = tf2::toMsg(desired_TCP_frame_B);

    // forward request
    movep_srv.request.planning_group = planning_group;
    movep_srv.request.frame_id = frame_id;
    movep_srv.request.pose = desired_TCP_pose;
    movep_srv.request.send = true;

    // abbasso velocità durante il planning
    utilities.setSpeedSlider(0.1, robot_id, nh);

    bool ok = utilities.movep_client.call(movep_srv);
    // ripristino prima di uscire (anche in caso di fallimento)
    // utilities.setSpeedSlider(1.0, robot_id, nh);

    if (!ok)
    {
        ROS_INFO("failed to call desiredTcpPose service");
        return false;
    }
    return true;
}

bool URCobot::movej(const Vector6d &q, bool reverse)
{

    std_msgs::Float64MultiArray joints_angles_msg;
    joints_angles_msg.data.resize(6);

    for (int i = 0; i < 6; ++i)
    {
        joints_angles_msg.data[i] = q(i);
    }

    if (reverse)
    {
        double q_temp = joints_angles_msg.data[0];
        joints_angles_msg.data[0] = joints_angles_msg.data[2];
        joints_angles_msg.data[2] = q_temp;
    }

    // forward request
    movej_srv.request.planning_group = planning_group;
    movej_srv.request.joints_angles = joints_angles_msg;
    movej_srv.request.send = true;

    if (!utilities.movej_client.call(movej_srv))
    {
        ROS_INFO("failed to call desiredJointsAngles service");
        return false;
    }
    return true;
}



/*--------------------ACTIONS----------------------------*/
// TODO: To be moved in application specific file

bool URCobot::reachPose(const Eigen::Affine3d &target, double tol)
{
    if (action_state != ActionState::WAIT_TARGET)
    {
        if (!switchController(CARTESIAN_POSITION))
            return false;
        if (!movep(target))
            return false;
        action_state = ActionState::WAIT_TARGET;
        return false;
    }
    if (isAtTarget(target, tol))
    {
        action_state = ActionState::DONE;
        return true;
    }
    return false;
}


/*--------------------KINEMATICS----------------------------*/

/**
 * @brief Transforms a pose written in base frame to one written in 0 frame
 *
 * @param pose_in_B
 * @return Eigen::Affine3d
 */
Eigen::Affine3d URCobot::to0Frame(Eigen::Affine3d pose_in_B)
{

    return T_0_B * pose_in_B;
}


/**
 * @brief check if robot is already at target pose
 *
 * @param pose_E_in_0 desired EE pose
 * @param tol tolerance
 * @return true
 * @return false
 */
bool URCobot::isAtTarget(const Eigen::Affine3d pose_E_in_0, double tol)
{

    double position_error = (pose_E_in_0.translation() - to0Frame(z_E).translation()).norm();
    Eigen::Matrix3d R_error = pose_E_in_0.rotation().transpose() * to0Frame(z_E).rotation();
    double angle_error = std::acos((R_error.trace() - 1) / 2) * 180 / M_PI;
    // std::cout << "r_E " << z_E.translation() << std::endl;
    // std::cout << "T_0_B " << T_0_B.matrix() << std::endl;
    // std::cout << "position_error: " << position_error << std::endl;
    // std::cout << "angle_error: " << angle_error << std::endl;
    if (position_error < tol && angle_error < 300 * tol)
    { // TODO: maybe better to specify angle tolerance from outside
        return true;
    }
    else
    {
        return false;
    }
}

bool URCobot::waitUntilReached(const Eigen::Affine3d &pose_E_in_0, double tol, double timeout_sec)
{
    ros::Rate rate(100); // 10 Hz
    double elapsed = 0.0;

    while (ros::ok() && !isAtTarget(pose_E_in_0, tol) && elapsed < timeout_sec)
    {
        rate.sleep();
        elapsed += 1.0 / 100.0;
    }

    return elapsed < timeout_sec;
}

/**
 * @brief Reads the joint velocity limits of the UR robot from the ROS parameter server.
 *
 * This function retrieves the maximum velocity limits for each joint of the UR robot from the ROS parameter server.
 * The joint velocity limits are stored in the `joint_velocity_limits` array [rad], which is used to enforce velocity
 * limits during robot motion.
 */
void URCobot::readJointVelLimits()
{
    static const std::vector<std::string> joints = {"shoulder_pan", "shoulder_lift", "elbow_joint",
                                                    "wrist_1", "wrist_2", "wrist_3"};

    std::string param_location = "/simod_vision_ctrl/joint_limits/";

    bool ret = true;
    Vector6d joint_velocity_limits_deg;
    for (uint i = 0; i < JOINTS_NUM; i++)
    {
        ret &= nh.getParam(param_location + joints[i] + "/max_velocity", joint_velocity_limits_deg(i));
        joint_velocity_limits(i) = deg2rad(joint_velocity_limits_deg(i));
    }
    if (!ret)
    {
        ROS_FATAL("Could not retrieve joint limits parameters from ROS params server.");
        ros::shutdown();
    }
}

/**
 * @brief Reads the kinematic parameters of the UR robot from the ROS parameter server.
 *
 * This function retrieves the kinematic parameters of the UR robot, such as the position and orientation of each joint frame,
 * from the ROS parameter server. The parameters are stored in the `A` and `Y` matrices, which are used to compute the
 * Jacobian matrix of the robot.
 *
 * @param nh The ROS node handle to access the parameter server.
 */
void URCobot::readKinParams()
{
    static const std::vector<std::string> frames = {"shoulder", "upper_arm", "forearm",
                                                    "wrist_1", "wrist_2", "wrist_3"};

    std::string kinematics_ = "/simod_vision_ctrl/arm" + robot_id + "/kinematics/";

    bool ret = true;
    Matrix6d kin_params;
    for (uint i = 0; i < JOINTS_NUM; i++)
    {
        ret &= nh.getParam(kinematics_ + frames[i] + "/x", kin_params(0, i));
        ret &= nh.getParam(kinematics_ + frames[i] + "/y", kin_params(1, i));
        ret &= nh.getParam(kinematics_ + frames[i] + "/z", kin_params(2, i));
        ret &= nh.getParam(kinematics_ + frames[i] + "/roll", kin_params(3, i));
        ret &= nh.getParam(kinematics_ + frames[i] + "/pitch", kin_params(4, i));
        ret &= nh.getParam(kinematics_ + frames[i] + "/yaw", kin_params(5, i));
    }
    if (!ret)
    {
        ROS_FATAL("Could not retrieve kinematic parameters from ROS params server.");
        ros::shutdown();
    }

    Matrix3d R = Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();

    for (int i = 0; i < JOINTS_NUM; ++i)
    {
        t += R * kin_params.block<3, 1>(0, i);
        R *= rpy(kin_params.block<3, 1>(3, i));
        A.block<4, 4>(0, 4 * i) = homTransf(R, t);

        Y.col(i) = screw(R.rightCols(1), t);
        X.col(i) = adjointInv(A.block<4, 4>(0, 4 * i)) * Y.col(i);
    }
}

/**
 * Calculates the Jacobian matrix for the UR robot.
 *
 * The Jacobian matrix relates the joint velocities to the end-effector twist (linear and angular velocities).
 * This function computes the Jacobian matrix based on the current joint positions.
 *
 * @param q The current joint positions.
 * @return The Jacobian matrix in B (base) frame.
 */
Matrix6d URCobot::calcJacobian(const Vector6d &q)
{

    Matrix44Nd C;
    Matrix4d Cij;
    Matrix6Nd Ab = Matrix6Nd::Identity();
    Matrix6N6d Xb = Matrix6N6d::Zero();
    Eigen::Affine3d z_E_test;
    Matrix3d R_E;
    Matrix6N6d J_tot;
    Matrix6d J_n;

    for (int i = 0; i < JOINTS_NUM; ++i)
    {
        C.block<4, 4>(0, 4 * i) =
            prodexp(Y, q, i) *
            A.block<4, 4>(0, 4 * i); // Matrices Y and A from robot config

        for (int j = 0; j < JOINTS_NUM; ++j)
            if (j <= i)
            {
                if (j == i)
                {
                    Xb.block<6, 1>(6 * i, j) = X.col(j); // Matrix X from robot config
                }
                else
                {
                    Cij = relConf(C.block<4, 4>(0, 4 * i), C.block<4, 4>(0, 4 * j));
                    Ab.block<6, 6>(6 * i, 6 * j) = adjoint(Cij);
                }
            }
    }

    z_E_test.matrix() = C.topRightCorner<4, 4>(); // FIXME: there seem to be a difference between this and the z_E retrieved by tf
    R_E = z_E_test.rotation();

    J_tot = Ab * Xb;
    J_n = J_tot.bottomRows(6);
    J << R_E * J_n.topRows(3), R_E * J_n.bottomRows(3);

    return J;
}

/**
 * Updates the kinematics of the UR cobot, including the current pose, twist, and Jacobian matrix.
 *
 * This function first looks up the current transform of the end-effector frame relative to the base frame using the TF listener.
 * It then computes the current Jacobian matrix based on the provided joint positions, and uses that to compute the current end-effector twist.
 * NB: The end-effector twist is written in this order (angular, linear)
 *
 * @param q The current joint positions.
 * @param q_dot The current joint velocities.
 * @return True if the kinematics update was successful, false otherwise.
 */
bool URCobot::updateKinematics(const Vector6d &q,
                               const Vector6d &q_dot)
{
    // Obtain latest pose of z_E
    tf::StampedTransform EE_transform_tf;
    try
    {
        tf_listener_.lookupTransform(base_frame_id, EE_frame_id, ros::Time(0),
                                     EE_transform_tf);
    }
    catch (tf::TransformException ex)
    {
        ROS_WARN("%s", ex.what());
        return false;
    }
    geometry_msgs::Transform EE_transform_geom_msg;
    tf::transformTFToMsg(EE_transform_tf, EE_transform_geom_msg);
    tf::transformMsgToEigen(EE_transform_geom_msg, z_E);

    // Obtain latest Jacobian and Twist
    J = calcJacobian(q);
    z_E_dot = J * q_dot;
    return true;
}

/**
 * Inverts the kinematics to compute the joint velocities required to achieve the desired end-effector twist. Please notice that
 * the angular part of the desired twist should be stored before the linear part.
 *
 * @param twist_E_in_B The desired end-effector twist (angular, linear), expressed in B (base) frame.
 * @return The joint velocities required to achieve the desired end-effector twist.
 */
Vector6d URCobot::invertKinematics(const Vector6d &twist_E_in_B)
{
    return J.inverse() * twist_E_in_B;
}

/**
/************ getters and setters ***************/
Vector6d URCobot::getJointVelocityLimits() const
{
    return joint_velocity_limits;
}

Eigen::Affine3d URCobot::getT_0_B() const
{
    return T_0_B;
}

Eigen::Affine3d URCobot::getT_B_0() const
{
    return T_B_0;
}

Eigen::Affine3d URCobot::getCurrentPose() const
{
    return z_E;
}

Vector6d URCobot::getCurrentTwist() const
{
    return z_E_dot;
}

Vector6d URCobot::getCurrentWrench() const
{
    return h;
}

Vector6d URCobot::getCurrentInternalWrench() const
{
    return h_int;
}

void URCobot::setCurrentWrench(const Vector6d wrench)
{
    h = wrench;
}

/**
 * Sets the current internal wrench acting on the robot.
 *
 * @param wrench The current internal wrench acting on the robot, represented in B (base) frame coordinates.
 */
void URCobot::setCurrentInternalWrench(const Vector6d wrench)
{
    h_int = wrench;
}

Eigen::Affine3d URCobot::getDesiredPose() const
{
    return z_E_des;
}

Vector6d URCobot::getDesiredTwist() const
{
    return z_E_dot_des;
}

Vector6d URCobot::getDesiredAcceleration() const
{
    return z_E_ddot_des;
}

Vector6d URCobot::getDesiredWrench() const
{
    return h_des;
}

void URCobot::setDesiredPose(const Eigen::Affine3d &pose_E_in_0)
{
    z_E_des = pose_E_in_0;
}

void URCobot::setDesiredTwist(const Vector6d &twist_E_in_0)
{
    z_E_dot_des = twist_E_in_0;
}

void URCobot::setDesiredAcceleration(const Vector6d &acc_E_in_0)
{
    z_E_ddot_des = acc_E_in_0;
}

void URCobot::setDesiredWrench(const Vector6d &wrench)
{
    h_des = wrench;
}

Matrix6d URCobot::getCurrentJacobian() const
{
    return J;
}


void URCobot::setAdmittanceError(const Vector6d translational_state_error, const Vector7d angular_state_error)
{
    admittance_params.y_P = translational_state_error;
    admittance_params.y_O = angular_state_error;
}

void URCobot::setExternalAdmittanceError(const Vector6d translational_state_error, const Vector7d angular_state_error)
{
    external_admittance_params.y_P = translational_state_error;
    external_admittance_params.y_O = angular_state_error;
}

void URCobot::setAdmittanceForceError(const Eigen::Vector3d old_error)
{
    // admittance_params.previous_force_error_z = old_error;
    force_error = old_error;
}

Eigen::Vector3d URCobot::getAdmittanceForceError() const
{
    return force_error;
}

void URCobot::setCounter(const int &old_counter)
{

    counter = old_counter;
}

int URCobot::getCounter() const
{
    return counter;
}

void URCobot::setAdmittancePhi(const double &old_phi)
{

    phi = old_phi;
}

double URCobot::getAdmittancePhi() const
{
    return phi;
}

std::string URCobot::getRobotID() const
{
    return robot_id;
}

int URCobot::getRobotIdx() const
{
    return robot_idx;
}
