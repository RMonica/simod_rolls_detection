// ROS
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <eigen_conversions/eigen_msg.h>
#include <actionlib/client/simple_action_client.h>
#include <tf2_ros/transform_broadcaster.h>

// OpenCV
#include <opencv2/opencv.hpp>

// STL
#include <fstream>
#include <stdint.h>
#include <vector>
#include <cmath>
#include <memory>

#include <simod_rolls_detection/DetectPalletAction.h>

typedef ros::NodeHandle Node;
typedef sensor_msgs::Image ImageMsg;
typedef sensor_msgs::CameraInfo CameraInfoMsg;

class PalletDetectionTest
{
  public:
  typedef std::vector<int> IntVector;
  typedef std::shared_ptr<IntVector> IntVectorPtr;
  typedef std::vector<float> FloatVector;
  typedef std::vector<double> DoubleVector;
  typedef std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > Vector4fVector;
  typedef std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > Vector4dVector;
  typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > Vector3dVector;
  typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > Vector2dVector;
  typedef std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d> > Affine3dVector;

  typedef uint64_t uint64;
  typedef uint32_t uint32;
  typedef int32_t int32;
  typedef int64_t int64;
  typedef uint8_t uint8;
  typedef uint16_t uint16;

  typedef geometry_msgs::TransformStamped TransformStampedMsg;

  typedef actionlib::SimpleActionClient<simod_rolls_detection::DetectPalletAction> ActionClient;

  template <typename T> static T SQR(const T & t) {return t * t; }

  struct BoundingBox
  {
    Eigen::Vector3f center = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    Eigen::Vector3f size = Eigen::Vector3f(2.0f, 4.0f, 3.0f);
    float rotation = 0.0f;
  };

  struct CameraInfo
  {
    float fx, fy;
    float cx, cy;
  };

  PalletDetectionTest(std::shared_ptr<Node> nodeptr): m_nodeptr(nodeptr)
  {
    m_timer = m_nodeptr->createTimer(ros::Duration(0.0), [this](const ros::TimerEvent &){this->Run(); }, true);

    m_nodeptr->param<std::string>("rgb_filename", m_image_file_name, "");
    m_nodeptr->param<std::string>("depth_filename", m_depth_file_name, "");
    m_nodeptr->param<std::string>("camera_info_filename", m_camera_info_file_name, "");
    m_nodeptr->param<std::string>("expected_pallet_filename", m_expected_pallet_file_name, "");
    m_nodeptr->param<std::string>("camera_pose_filename", m_camera_pose_file_name, "");
    m_nodeptr->param<std::string>("initial_guess_filename", m_initial_guess_file_name, "");

    m_nodeptr->param<bool>("use_real_camera", m_use_real_camera, false);

    m_nodeptr->param<std::string>("detect_pallet_action", m_detect_pallet_action, "/detect_pallet");
    m_ac.reset(new ActionClient(*nodeptr, m_detect_pallet_action, true));

    m_nodeptr->param<std::string>("depth_image_topic", m_depth_image_topic, "camera_info_topic");
    m_nodeptr->param<std::string>("rgb_image_topic", m_rgb_image_topic, "rgb_image_topic");
    m_nodeptr->param<std::string>("camera_info_topic", m_camera_info_topic, "camera_info_topic");
    m_rgb_image_pub = nodeptr->advertise<sensor_msgs::Image>(m_rgb_image_topic, 1);
    m_depth_image_pub = nodeptr->advertise<sensor_msgs::Image>(m_depth_image_topic, 1);
    m_camera_info_pub = nodeptr->advertise<sensor_msgs::CameraInfo>(m_camera_info_topic, 1);

    m_nodeptr->param<std::string>("world_frame_id", m_world_frame_id, "map");
    m_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>();
  }

  void Load(cv::Mat & rgb_image, cv::Mat & depth_image, CameraInfo & camera_info,
            Eigen::Affine3f & camera_pose, BoundingBox & initial_guess, float & floor_z)
  {
    {
      ROS_INFO("loading camera_info file %s", m_camera_info_file_name.c_str());
      camera_info = LoadCameraInfo(m_camera_info_file_name);
    }

    rgb_image = cv::imread(m_image_file_name);
    depth_image = cv::imread(m_depth_file_name, cv::IMREAD_ANYDEPTH);

    if (!rgb_image.data)
    {
      ROS_FATAL("could not load rgb image: %s", m_image_file_name.c_str());
      std::exit(1);
    }

    if (!depth_image.data)
    {
      ROS_FATAL("could not load depth image: %s", m_depth_file_name.c_str());
      std::exit(2);
    }

    if (!MatrixFromFile(m_camera_pose_file_name, camera_pose))
    {
      ROS_FATAL("could not load camera_pose: %s", m_camera_pose_file_name.c_str());
      std::exit(3);
    }

    {
      std::ifstream ifile(m_initial_guess_file_name);
      if (!ifile)
      {
        ROS_FATAL("could not load initial_guess: %s", m_initial_guess_file_name.c_str());
        std::exit(4);
      }
      initial_guess = LoadInitialGuess(ifile, floor_z);
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

  BoundingBox LoadInitialGuess(std::istream & istr, float & floor_z) const
  {
    BoundingBox result;
    floor_z = 0.0f;

    std::string line;
    while (std::getline(istr, line))
    {
      if (line.empty())
        continue; // skip empty lines

      std::istringstream iss(line);
      std::string type;
      iss >> type;
      if (type == "center")
      {
        iss >> result.center.x() >> result.center.y() >> result.center.z();
      }
      else if (type == "size")
      {
        iss >> result.size.x() >> result.size.y() >> result.size.z();
      }
      else if (type == "rotation")
      {
        iss >> result.rotation;
      }
      else if (type == "floor")
      {
        iss >> floor_z;
      }
      else
      {
        ROS_ERROR("LoadExpectedPallet: Unknown type: %s", type.c_str());
        continue;
      }

      if (!iss)
      {
        ROS_ERROR("Unable to parse line: %s", line.c_str());
        continue;
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
    cv::Mat depth_image;
    Eigen::Affine3f camera_pose;

    ROS_INFO("pallet_detection_test: loading");

    BoundingBox initial_guess;
    CameraInfo camera_info;
    float floor_z;
    Load(rgb_image, depth_image, camera_info, camera_pose, initial_guess, floor_z);

    simod_rolls_detection::DetectPalletGoal goal;
    goal.initial_guess_center.x = initial_guess.center.x();
    goal.initial_guess_center.y = initial_guess.center.y();
    goal.initial_guess_center.z = initial_guess.center.z();
    goal.initial_guess_size.x = initial_guess.size.x();
    goal.initial_guess_size.y = initial_guess.size.y();
    goal.initial_guess_size.z = initial_guess.size.z();
    goal.initial_guess_rotation = initial_guess.rotation;

    goal.floor_z = floor_z;

    goal.pallet_description_filename = m_expected_pallet_file_name;

    tf::poseEigenToMsg(camera_pose.cast<double>(), goal.camera_pose);

    ROS_INFO("pallet_detection_test: waiting for server");
    m_ac->waitForServer();

    ROS_INFO("pallet_detection_test: sending goal");
    m_ac->sendGoal(goal);

    if (!m_use_real_camera)
    {
      ROS_INFO("pallet_detection_test: sleeping");
      ros::Duration(0.5).sleep();

      ROS_INFO("pallet_detection_test: publishing");
      PublishImage(rgb_image, "bgr8", m_rgb_image_pub);
      PublishImage(depth_image, "16UC1", m_depth_image_pub);

      sensor_msgs::CameraInfo camera_info_msg;
      camera_info_msg.K[0] = camera_info.fx;
      camera_info_msg.K[4] = camera_info.fy;
      camera_info_msg.K[2] = camera_info.cx;
      camera_info_msg.K[5] = camera_info.cy;
      m_camera_info_pub.publish(camera_info_msg);
    }

    ROS_INFO("pallet_detection_test: waiting for result.");
    m_ac->waitForResult();

    simod_rolls_detection::DetectPalletResult result = *(m_ac->getResult());

    const bool success = result.success;
    ROS_INFO("pallet_detection_test: success: %s.", (success ? "TRUE" : "FALSE"));
    ROS_INFO("pallet_detection_test: consensus: %d.", int(result.consensus));

    Eigen::Affine3d pallet_pose;
    tf::poseMsgToEigen(result.pallet_pose, pallet_pose);
    {
      TransformStampedMsg t;
      tf::transformEigenToMsg(pallet_pose, t.transform);

      t.header.stamp = ros::Time::now();
      t.header.frame_id = m_world_frame_id;
      t.child_frame_id = "pallet";

      m_tf_broadcaster->sendTransform(t);
    }

    Affine3dVector box_poses;
    for (const geometry_msgs::Pose & pose_msg : result.box_poses)
    {
      Eigen::Affine3d pose;
      tf::poseMsgToEigen(pose_msg, pose);
      box_poses.push_back(pose);
    }

    ROS_INFO_STREAM("publishing " << box_poses.size() << " boxes.");
    for (uint64 box_i = 0; box_i < box_poses.size(); box_i++)
    {
      const Eigen::Affine3d pose = box_poses[box_i];

      TransformStampedMsg t;
      tf::transformEigenToMsg(pose, t.transform);

      t.header.stamp = ros::Time::now();
      t.header.frame_id = m_world_frame_id;
      t.child_frame_id = "box_" + std::to_string(box_i);

      m_tf_broadcaster->sendTransform(t);
    }

    ROS_INFO("pallet_detection_test: end.");
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
  ros::Publisher m_depth_image_pub;
  ros::Publisher m_camera_info_pub;
  std::string m_rgb_image_topic;
  std::string m_depth_image_topic;
  std::string m_camera_info_topic;

  bool m_use_real_camera;

  std::string m_world_frame_id;
  std::shared_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;

  std::string m_detect_pallet_action;

  std::shared_ptr<ActionClient> m_ac;

  ros::Timer m_timer;

  std::string m_image_file_name;
  std::string m_depth_file_name;
  std::string m_camera_info_file_name;
  std::string m_expected_pallet_file_name;
  std::string m_camera_pose_file_name;
  std::string m_initial_guess_file_name;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pallet_detection_test");
  std::shared_ptr<Node> nodeptr(new Node("~"));
  ROS_INFO("pallet_detection test started");

  PalletDetectionTest pd(nodeptr);
  ros::spin();

	return 0;
}
