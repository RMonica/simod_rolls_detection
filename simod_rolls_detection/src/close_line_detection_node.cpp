// ROS
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <eigen_conversions/eigen_msg.h>
#include <actionlib/server/simple_action_server.h>

// OpenCV
#include <opencv2/opencv.hpp>

// STL
#include <fstream>
#include <stdint.h>
#include <vector>
#include <set>
#include <cmath>
#include <map>
#include <mutex>

#include <simod_rolls_detection/DetectCloseLineAction.h>

#include <close_line_detection.h>

typedef ros::NodeHandle Node;
typedef sensor_msgs::Image ImageMsg;
typedef sensor_msgs::CameraInfo CameraInfoMsg;

class CloseLineDetectionNode
{
  public:
  typedef actionlib::SimpleActionServer<simod_rolls_detection::DetectCloseLineAction> ActionServer;

  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> Vector3fVector;

  typedef std::vector<std::string> StringVector;

  template <typename T> static T SQR(const T & t) {return t * t; }

  CloseLineDetectionNode(std::shared_ptr<Node> nodeptr): m_nodeptr(nodeptr)
  {
    m_nodeptr->param<std::string>("rgb_image_topic", m_rgb_image_topic, "rgb_image_topic");
    m_nodeptr->param<std::string>("camera_info_topic", m_camera_info_topic, "camera_info_topic");

    m_nodeptr->param<std::string>("detect_close_line_action", m_detect_close_line_action, "/detect_close_line");
    m_as.reset(new ActionServer(*nodeptr, m_detect_close_line_action,
      boost::function<void(const simod_rolls_detection::DetectCloseLineGoalConstPtr &)>(
        [this](const simod_rolls_detection::DetectCloseLineGoalConstPtr & goal){this->Run(goal); }),
      false));

    ROS_INFO("close_line_detection_node: subscribing to color topic %s", m_rgb_image_topic.c_str());
    ROS_INFO("close_line_detection_node: subscribing to camera_info topic %s", m_camera_info_topic.c_str());
    m_rgb_image_sub = m_nodeptr->subscribe<sensor_msgs::Image>(m_rgb_image_topic, 1,
      boost::function<void(const sensor_msgs::Image &)>([this](const sensor_msgs::Image & msg){this->RgbImageCallback(msg); }));
    m_camera_info_sub = m_nodeptr->subscribe<sensor_msgs::CameraInfo>(m_camera_info_topic, 1,
      boost::function<void(const sensor_msgs::CameraInfo &)>([this](const sensor_msgs::CameraInfo & msg){
      this->CameraInfoCallback(msg);
    }));

    m_nodeptr->param<int>("discard_first_camera_frames", m_discard_first_camera_frames, 0);

    {
      const StringVector publish_image_names = CloseLineDetection::GetImagePublisherNames();
      for (const std::string & s : publish_image_names)
      {
        const std::string topic_name = s + "_image";
        m_image_publishers[s] = m_nodeptr->advertise<sensor_msgs::Image>(topic_name, 1);
      }
    }

    m_as->start();
  }

  void Run(const simod_rolls_detection::DetectCloseLineGoalConstPtr & goalptr)
  {
    const simod_rolls_detection::DetectCloseLineGoal & goal = *goalptr;

    cv::Mat rgb_image;
    std::shared_ptr<CloseLineDetection::Intrinsics> camera_info;
    ROS_INFO("close_line_detection_node: action start.");
    {
      ros::Rate wait_rate(100);
      std::unique_lock<std::mutex> lock(m_mutex);
      if (m_discard_first_camera_frames)
        ROS_INFO("close_line_detection_node: first %d camera frames will be discarded.",
                 int(m_discard_first_camera_frames));
      for (int i = 0; i < m_discard_first_camera_frames + 1; i++)
      {
        m_last_rgb_image = cv::Mat();
        m_last_camera_info.reset();
        while (m_last_rgb_image.empty() || !m_last_camera_info)
        {
          lock.unlock();
          ROS_INFO_THROTTLE(2.0, "close_line_detection_node: waiting for images...");
          wait_rate.sleep();
          lock.lock();

          if (!ros::ok())
            return;
        }

        if (!ros::ok())
          return;
      }

      rgb_image = m_last_rgb_image;
      camera_info = m_last_camera_info;
    } // lock released here
    ROS_INFO("close_line_detection_node: received images.");

    Eigen::Affine3d camera_pose;
    tf::poseMsgToEigen(goal.camera_pose, camera_pose);

    CloseLineDetection::Config config;

    m_nodeptr->param<int>("random_seed", config.random_seed, config.random_seed);
    m_nodeptr->param<float>("roll_diameter", config.roll_diameter, config.roll_diameter);
    m_nodeptr->param<int>("non_maximum_suppression_window", config.non_maximum_suppression_window,
                          config.non_maximum_suppression_window);
    m_nodeptr->param<float>("min_correlation_threshold", config.min_correlation_threshold, config.min_correlation_threshold);
    m_nodeptr->param<float>("correlation_lines_weight", config.correlation_lines_weight, config.correlation_lines_weight);
    m_nodeptr->param<int>("median_filter_window", config.median_filter_window, config.median_filter_window);
    m_nodeptr->param<float>("line_max_angle", config.line_max_angle, config.line_max_angle); // radians
    m_nodeptr->param<float>("min_virtual_camera_distance", config.min_virtual_camera_distance,
                            config.min_virtual_camera_distance); // meters

    m_nodeptr->param<float>("basic_max_ransac_distance",
                            config.basic_max_ransac_distance, config.basic_max_ransac_distance);
    m_nodeptr->param<int>("basic_gaussian_blur_half_window_size",
                          config.basic_gaussian_blur_half_window_size, config.basic_gaussian_blur_half_window_size);
    m_nodeptr->param<int>("basic_canny_threshold",
                          config.basic_canny_threshold, config.basic_canny_threshold);
    m_nodeptr->param<int>("basic_hough_threshold",
                          config.basic_hough_threshold, config.basic_hough_threshold);
    m_nodeptr->param<bool>("mode_basic", config.mode_basic, config.mode_basic);

    config.initial_guess_window_size_x = goal.initial_guess_window_size_x;
    config.initial_guess_window_size_y = goal.initial_guess_window_size_y;

    CloseLineDetection cld(config,
                           [this](const uint level, const std::string & message) {this->Log(level, message); },
                           [this](const std::string & name, const cv::Mat & image, const std::string & encoding) {
                             this->PublishImage(image, encoding, name);
                           });

    const float initial_guess_x = goal.initial_guess_x;
    const float initial_guess_y = goal.initial_guess_y;

    const float layer_height = goal.layer_z;

    Vector3fVector points = cld.Run(rgb_image, *camera_info, camera_pose.cast<float>(),
                                    layer_height, initial_guess_x, initial_guess_y);

    ROS_INFO_STREAM("close_line_detection_node: success: " << (points.size() ? "TRUE" : "FALSE"));

    simod_rolls_detection::DetectCloseLineResult result;
    result.success = !points.empty();

    for (const Eigen::Vector3f & pt : points)
    {
      geometry_msgs::Point gpt;
      gpt.x = pt.x();
      gpt.y = pt.y();
      gpt.z = pt.z();
      result.points.push_back(gpt);
    }

    ROS_INFO("close_line_detection_node: action end.");
    m_as->setSucceeded(result);
  }

  void RgbImageCallback(const sensor_msgs::Image & msg)
  {
    std::unique_lock<std::mutex> lock(m_mutex);

    try
    {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
      m_last_rgb_image = cv_ptr->image;
    }
    catch (const cv_bridge::Exception & e)
    {
      ROS_ERROR("RgbImageCallback: Could not convert from '%s' to 'bgr8'.", msg.encoding.c_str());
      return;
    }
  }

  void CameraInfoCallback(const sensor_msgs::CameraInfo & msg)
  {
    std::unique_lock<std::mutex> lock(m_mutex);

    const float fx = msg.K[0];
    const float fy = msg.K[4];
    const float cx = msg.K[2];
    const float cy = msg.K[5];

    std::shared_ptr<CloseLineDetection::Intrinsics> ci(new CloseLineDetection::Intrinsics);
    ci->fx = fx;
    ci->fy = fy;
    ci->cx = cx;
    ci->cy = cy;

    m_last_camera_info = ci;
  }

  void Log(const uint level, const std::string & message)
  {
    switch (level)
    {
    case 0:
      ROS_DEBUG("close_line_detection: %s", message.c_str());
      break;
    case 1:
      ROS_INFO("close_line_detection: %s", message.c_str());
      break;
    case 2:
      ROS_WARN("close_line_detection: %s", message.c_str());
      break;
    case 3:
      ROS_ERROR("close_line_detection: %s", message.c_str());
      break;
    case 4:
      ROS_FATAL("close_line_detection: %s", message.c_str());
      break;
    default:
      ROS_ERROR("close_line_detection_node: Invalid logger level %d, message was '%s'", int(level), message.c_str());
    }
  }

  void PublishImage(const cv::Mat &image, const std::string &encoding, const std::string & name)
  {
    if (m_image_publishers.find(name) == m_image_publishers.end())
    {
      ROS_WARN("close_line_detection_node: Could not find image publisher with name %s", name.c_str());
      return;
    }

    PublishImage(image, encoding, m_image_publishers[name]);
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

  ros::Subscriber m_rgb_image_sub;
  ros::Subscriber m_camera_info_sub;

  std::map<std::string, ros::Publisher> m_image_publishers;

  std::shared_ptr<ActionServer> m_as;

  std::string m_rgb_image_topic;
  std::string m_camera_info_topic;
  std::string m_detect_close_line_action;

  cv::Mat m_last_rgb_image;
  std::shared_ptr<CloseLineDetection::Intrinsics> m_last_camera_info;

  ros::Timer m_timer;
  std::mutex m_mutex;

  int m_discard_first_camera_frames;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pallet detection");
  std::shared_ptr<Node> nodeptr(new Node("~"));
  ROS_INFO("pallet_detection node started");

  CloseLineDetectionNode pd(nodeptr);
  ros::spin();

	return 0;
}
