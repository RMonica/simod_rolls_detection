// ROS
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <eigen_conversions/eigen_msg.h>
#include <actionlib/client/simple_action_client.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

// OpenCV
#include <opencv2/opencv.hpp>

// STL
#include <fstream>
#include <stdint.h>
#include <vector>
#include <cmath>
#include <memory>

#include <simod_rolls_detection/DetectCloseLineAction.h>

typedef ros::NodeHandle Node;
typedef sensor_msgs::Image ImageMsg;
typedef sensor_msgs::CameraInfo CameraInfoMsg;

class CloseLineDetectionTest
{
  public:

  typedef actionlib::SimpleActionClient<simod_rolls_detection::DetectCloseLineAction> ActionClient;

  typedef geometry_msgs::TransformStamped TransformStampedMsg;

  template <typename T> static T SQR(const T & t) {return t * t; }

  struct CameraInfo
  {
    float fx, fy;
    float cx, cy;
  };

  CloseLineDetectionTest(std::shared_ptr<Node> nodeptr): m_nodeptr(nodeptr), m_tf_listener(m_tf_buffer)
  {
    m_timer = m_nodeptr->createTimer(ros::Duration(0.0), [this](const ros::TimerEvent &){this->Run(); }, true);

    m_nodeptr->param<std::string>("rgb_filename", m_image_file_name, "");
    m_nodeptr->param<std::string>("camera_info_filename", m_camera_info_file_name, "");
    m_nodeptr->param<std::string>("camera_pose_filename", m_camera_pose_file_name, "");

    m_nodeptr->param<bool>("use_real_camera", m_use_real_camera, false);

    m_nodeptr->param<double>("layer_height", m_layer_height, 0.0f);

    m_nodeptr->param<double>("initial_guess_x", m_initial_guess_x, 0.0);
    m_nodeptr->param<double>("initial_guess_y", m_initial_guess_y, 0.0);
    m_nodeptr->param<double>("initial_guess_window_size_x", m_initial_guess_window_size_x, 2.0);
    m_nodeptr->param<double>("initial_guess_window_size_y", m_initial_guess_window_size_y, 0.2);

    m_nodeptr->param<std::string>("detect_close_line_action", m_detect_close_line_action, "/detect_close_line");
    m_ac.reset(new ActionClient(*nodeptr, m_detect_close_line_action, true));

    m_nodeptr->param<std::string>("rgb_image_topic", m_rgb_image_topic, "rgb_image_topic");
    m_nodeptr->param<std::string>("camera_info_topic", m_camera_info_topic, "camera_info_topic");
    m_rgb_image_pub = nodeptr->advertise<sensor_msgs::Image>(m_rgb_image_topic, 1);
    m_camera_info_pub = nodeptr->advertise<sensor_msgs::CameraInfo>(m_camera_info_topic, 1);

    m_nodeptr->param<std::string>("world_frame_id", m_world_frame_id, "map");
    m_nodeptr->param<std::string>("camera_frame_id", m_camera_frame_id, "camera");

    m_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>();
  }

  void Load(cv::Mat & rgb_image, CameraInfo & camera_info, Eigen::Affine3f & camera_pose)
  {
    {
      ROS_INFO("loading camera_info file %s", m_camera_info_file_name.c_str());
      camera_info = LoadCameraInfo(m_camera_info_file_name);
    }

    rgb_image = cv::imread(m_image_file_name);

    if (!rgb_image.data)
    {
      ROS_FATAL("could not load rgb image: %s", m_image_file_name.c_str());
      std::exit(1);
    }

    if (!MatrixFromFile(m_camera_pose_file_name, camera_pose))
    {
      ROS_FATAL("could not load camera_pose: %s", m_camera_pose_file_name.c_str());
      std::exit(3);
    }
  }

  CameraInfo LoadCameraInfo(const std::string filename)
  {
    CameraInfo result;

    std::ifstream ifile(filename);
    if (!ifile)
    {
      ROS_FATAL("could not find camera_info file: %s", filename.c_str());
      std::exit(1);
    }

    std::string line;
    while (std::getline(ifile, line))
    {
      std::istringstream istr(line);
      std::string field;
      istr >> field;
      if (field == "fx")
        istr >> result.fx;
      else if (field == "fy")
        istr >> result.fy;
      else if (field == "cx")
        istr >> result.cx;
      else if (field == "cy")
        istr >> result.cy;
      else
      {
        ROS_FATAL("invalid line in camera info file: %s", line.c_str());
        std::exit(1);
      }

      if (!istr)
      {
        ROS_FATAL("could not parse line in camera info file: %s", line.c_str());
        std::exit(1);
      }
    }

    return result;
  }

  bool MatrixFromFile(const std::string & filename, Eigen::Affine3f & matrix)
  {
    std::ifstream file(filename);

    for (uint64 i = 0; i < 3; ++i)
      for (uint64 j = 0; j < 4; ++j)
      {
        float v;
        file >> v;
        matrix.matrix()(i, j) = v;
      }

    if (!file)
      return false;
    return true;
  }

  void Run()
  {
    cv::Mat rgb_image;
    Eigen::Affine3f camera_pose;

    ROS_INFO("close_line_detection_test: loading");

    CameraInfo camera_info;
    if (!m_use_real_camera)
      Load(rgb_image, camera_info, camera_pose);

    float layer_z = m_layer_height;

    if (!m_use_real_camera)
    {
      TransformStampedMsg t;
      tf::transformEigenToMsg(camera_pose.cast<double>(), t.transform);

      t.header.stamp = ros::Time::now();
      t.header.frame_id = m_world_frame_id;
      t.child_frame_id = m_camera_frame_id;

      ROS_INFO("close_line_detection_test: sending simulated camera pose to TF.");
      m_tf_broadcaster->sendTransform(t);
    }

    geometry_msgs::TransformStamped transformStamped;
    const double MAX_WAIT = 5.0; // wait at most 5 seconds for the transform
    try
    {
      ROS_INFO("close_line_detection_test: tf: waiting for camera pose between '%s' and '%s'",
               m_world_frame_id.c_str(), m_camera_frame_id.c_str());
      transformStamped = m_tf_buffer.lookupTransform(m_world_frame_id, m_camera_frame_id,
                                                     ros::Time(0), ros::Duration(MAX_WAIT));
      Eigen::Affine3d camera_pose_d;
      tf::transformMsgToEigen(transformStamped.transform, camera_pose_d);
      camera_pose = camera_pose_d.cast<float>();
      ROS_INFO("close_line_detection_test: tf: received camera pose.");
    }
    catch (tf2::TransformException &ex)
    {
      ROS_FATAL("close_line_detection_test: tf: could not find camera pose between '%s' and '%s' within %f seconds: %s",
                m_world_frame_id.c_str(), m_camera_frame_id.c_str(), double(MAX_WAIT), ex.what());
      std::exit(6);
    }

    simod_rolls_detection::DetectCloseLineGoal goal;
    goal.initial_guess_x = m_initial_guess_x;
    goal.initial_guess_y = m_initial_guess_y;
    goal.initial_guess_window_size_x = m_initial_guess_window_size_x;
    goal.initial_guess_window_size_y = m_initial_guess_window_size_y;

    goal.layer_z = layer_z;

    tf::poseEigenToMsg(camera_pose.cast<double>(), goal.camera_pose);

    ROS_INFO("close_line_detection_test: waiting for server");
    m_ac->waitForServer();

    ROS_INFO("close_line_detection_test: sending goal");
    m_ac->sendGoal(goal);

    if (!m_use_real_camera)
    {
      ROS_INFO("close_line_detection_test: sleeping");
      ros::Duration(0.5).sleep();

      ROS_INFO("close_line_detection_test: publishing");
      PublishImage(rgb_image, "bgr8", m_rgb_image_pub);

      sensor_msgs::CameraInfo camera_info_msg;
      camera_info_msg.K[0] = camera_info.fx;
      camera_info_msg.K[4] = camera_info.fy;
      camera_info_msg.K[2] = camera_info.cx;
      camera_info_msg.K[5] = camera_info.cy;
      m_camera_info_pub.publish(camera_info_msg);
    }

    ROS_INFO("close_line_detection_test: waiting for result.");
    m_ac->waitForResult();

    simod_rolls_detection::DetectCloseLineResult result = *(m_ac->getResult());

    const bool success = result.success;
    ROS_INFO("close_line_detection_test: success: %s.", (success ? "TRUE" : "FALSE"));

    ROS_INFO("close_line_detection_test: received: %d points.", int(result.points.size()));

    // publishing to TF for visualization
    int increment = 0;
    for (const geometry_msgs::Point & pt : result.points)
    {
      Eigen::Affine3d point_pose = Eigen::Affine3d::Identity();
      point_pose.translation().x() = pt.x;
      point_pose.translation().y() = pt.y;
      point_pose.translation().z() = pt.z;

      TransformStampedMsg t;
      tf::transformEigenToMsg(point_pose, t.transform);

      t.header.stamp = ros::Time::now();
      t.header.frame_id = m_world_frame_id;
      t.child_frame_id = "line_point_" + std::to_string(increment);
      increment++;

      m_tf_broadcaster->sendTransform(t);
    }

    ROS_INFO("close_line_detection_test: end.");
  }

  void PublishImage(const cv::Mat & image, const std::string & encoding, ros::Publisher & pub)
  {
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    cv_ptr->image = image;
    cv_ptr->encoding = encoding;
    ImageMsg img = *cv_ptr->toImageMsg();
    pub.publish(img);
  }

  private:
  std::shared_ptr<Node> m_nodeptr;

  ros::Publisher m_rgb_image_pub;
  ros::Publisher m_camera_info_pub;
  std::string m_rgb_image_topic;
  std::string m_camera_info_topic;

  bool m_use_real_camera;
  double m_layer_height;

  double m_initial_guess_x;
  double m_initial_guess_y;
  double m_initial_guess_window_size_x;
  double m_initial_guess_window_size_y;

  std::string m_world_frame_id;
  std::string m_camera_frame_id;

  tf2_ros::Buffer m_tf_buffer;
  tf2_ros::TransformListener m_tf_listener;
  std::shared_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;

  std::string m_detect_close_line_action;

  std::shared_ptr<ActionClient> m_ac;

  ros::Timer m_timer;

  std::string m_image_file_name;
  std::string m_camera_info_file_name;
  std::string m_camera_pose_file_name;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "close_line_detection_test");
  std::shared_ptr<Node> nodeptr(new Node("~"));
  ROS_INFO("close_line_detection test started");

  CloseLineDetectionTest pd(nodeptr);
  ros::spin();

	return 0;
}
