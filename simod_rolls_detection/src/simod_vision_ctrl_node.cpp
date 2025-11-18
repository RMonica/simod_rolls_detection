#include <signal.h>
#include "std_msgs/Float64MultiArray.h"
#include "simod_rolls_detection/simod_vision_ctrl.h"

void sigIntHandler(int /*sig*/)
{
  ros::NodeHandle nh;
  ros::Publisher pub_UR2 =
    nh.advertise<std_msgs::Float64MultiArray>("arm2/joint_group_vel_controller/command", 1);

  std_msgs::Float64MultiArray stop_msg;
  stop_msg.data = {0, 0, 0, 0, 0, 0};

  pub_UR2.publish(stop_msg);
  
  ros::shutdown();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "simod_vision_ctrl");
 
  SimodVisionCtrl controller;

  ros::Rate r(controller.getRate());

  signal(SIGINT, sigIntHandler);

  while (ros::ok())
  {
    controller.spinner();
    r.sleep();
  }

  return 0;
}

