#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <eigen_conversions/eigen_msg.h>

#include <stdint.h>
#include <fstream>
#include <sstream>

class SaveImages
{

public:
  SaveImages(ros::NodeHandle & node): m_node(node)
  {
    node.param<bool>("save_color", m_save_color, true);
    node.param<bool>("save_depth", m_save_depth, true);
    node.param<bool>("save_pose", m_save_pose, false);
    node.param<bool>("save_camera_info", m_save_camera_info, false);

    node.param<std::string>("tf_camera_frame", m_tf_camera_frame, "");
    node.param<std::string>("tf_base_frame", m_tf_base_frame, "");

    node.param<std::string>("save_folder_name", m_save_folder_name, "");
    if (!m_save_folder_name.empty() && m_save_folder_name[m_save_folder_name.size() - 1] != '/')
      m_save_folder_name += "/";
    
    node.param<std::string>("depth_image_topic", m_depth_image_topic, "/aligned_depth_to_color/image_raw");
    node.param<std::string>("color_image_topic", m_rgb_image_topic, "/color/image_raw");
    node.param<std::string>("camera_info_topic", m_camera_info_topic, "/color/camera_info");

    m_tf_buffer = std::make_shared<tf2_ros::Buffer>();
    m_tf_listener = std::make_shared<tf2_ros::TransformListener>(*m_tf_buffer);

    cv::namedWindow("view");

    ROS_INFO("subscribing to depth topic %s", m_depth_image_topic.c_str());
    ROS_INFO("subscribing to color topic %s", m_rgb_image_topic.c_str());
    ROS_INFO("subscribing to camera_info topic %s", m_camera_info_topic.c_str());
    m_depth_sub = m_node.subscribe<sensor_msgs::Image>(m_depth_image_topic, 1,
      boost::function<void(const sensor_msgs::Image &)>([this](const sensor_msgs::Image & msg){this->depthCallback(msg); }));
    m_rgb_sub = m_node.subscribe<sensor_msgs::Image>(m_rgb_image_topic, 1,
      boost::function<void(const sensor_msgs::Image &)>([this](const sensor_msgs::Image & msg){this->rgbimageCallback(msg); }));

    m_camera_info_sub = m_node.subscribe<sensor_msgs::CameraInfo>(m_camera_info_topic, 1,
      boost::function<void(const sensor_msgs::CameraInfo &)>([this](const sensor_msgs::CameraInfo & msg){
      this->cameraInfoCallback(msg);
    }));

    m_timer = m_node.createTimer(ros::Duration(0.1), [this](const ros::TimerEvent &){this->timer_callback(); });
  }

  void timer_callback()
  {
    const int key = cv::waitKey(100);
    if (key == 27) // ESC
      ros::shutdown();
    else if (key >= 0)
    {
      ROS_INFO("Key pressed.");

      is_camera_info_key_pressed = true;
      is_pose_key_pressed = true;
      is_rgbimg_key_pressed = true;
      is_depth_key_pressed = true;
    }

    poseCallback();
  }

  void cameraInfoCallback(const sensor_msgs::CameraInfo & msg)
  {
    if (!is_camera_info_key_pressed)
      return;
    is_camera_info_key_pressed = false;

    if (!m_save_camera_info)
      return;

    std::string file_name = m_save_folder_name + "camerainfo" + std::to_string(camera_info_count) + ".txt";
    ROS_INFO("Writing camera info file: %s", file_name.c_str());
    {
      std::ofstream ofile(file_name);

      const float fx = msg.K[0];
      const float fy = msg.K[4];
      const float cx = msg.K[2];
      const float cy = msg.K[5];

      ofile << "fx " << fx << "\n";
      ofile << "fy " << fy << "\n";
      ofile << "cx " << cx << "\n";
      ofile << "cy " << cy << "\n";

      if (!ofile)
      {
        ROS_ERROR("Could not write file: %s", file_name.c_str());
      }
    }

    camera_info_count++;
  }

  void poseCallback()
  {
    if (!is_pose_key_pressed)
      return;
    is_pose_key_pressed = false;

    if (!m_save_pose)
      return;

    geometry_msgs::TransformStamped t;
    try
    {
      t = m_tf_buffer->lookupTransform(m_tf_base_frame, m_tf_camera_frame, ros::Time(0.0));
    } catch (const tf2::TransformException & ex)
    {
      ROS_ERROR("save_images: could not get transform from %s to %s: %s",
                m_tf_camera_frame.c_str(), m_tf_base_frame.c_str(), ex.what());
      return;
    }

    Eigen::Affine3d m;
    tf::transformMsgToEigen(t.transform, m);
    std::string file_name = m_save_folder_name + "pose" + std::to_string(pose_count) + ".txt";
    std::ofstream ofile(file_name);
    ofile << m.matrix();

    pose_count++;
  }

  //RGB IMAGE
  void rgbimageCallback(const sensor_msgs::Image & msg)
  {
    cv::Mat rgb_image;
    try {
      rgb_image = cv_bridge::toCvCopy(msg, "bgr8")->image;

      cv::imshow("view", rgb_image);
    }
    catch (const cv_bridge::Exception & e) {
      ROS_ERROR("A Could not convert from '%s' to 'bgr8'.", msg.encoding.c_str());
      return;
    }

    if (!is_rgbimg_key_pressed)
      return;
    is_rgbimg_key_pressed = false;

    if (!m_save_color)
      return;

    std::string img_name = m_save_folder_name + "image" + std::to_string(rgbimg_callback_count) + ".png";
    ROS_INFO("Saving color image: %s", img_name.c_str());
    cv::imwrite(img_name, rgb_image);

    rgbimg_callback_count++;
  }

  //DEPTH IMAGE
  void depthCallback(const sensor_msgs::Image & msg)
  {
    if (!is_depth_key_pressed)
      return;
    is_depth_key_pressed = false;

    if (!m_save_depth)
      return;

    std::string numberString = std::to_string(depth_callback_count);
    cv::String img_name = m_save_folder_name + "depth" + numberString + ".png";
    try {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg);
      cv::Mat depth_image=cv_ptr->image;

      cv::imwrite(img_name,depth_image);
    }
    catch (const cv_bridge::Exception & e) {
      ROS_ERROR("B Could not convert from '%s' to 'bgr8'.", msg.encoding.c_str());
    }

    depth_callback_count++;
  }

  private:
  bool m_save_camera_info;
  int camera_info_count = 0;
  bool is_camera_info_key_pressed = false;

  bool m_save_pose;
  int pose_count = 0;
  bool is_pose_key_pressed = false;

  bool m_save_color;
  int rgbimg_callback_count = 0;
  bool is_rgbimg_key_pressed = false;

  bool m_save_pointcloud;
  int pointcloud_callback_count = 0;
  bool is_pointcloud_key_pressed = false;

  bool m_save_depth;
  int depth_callback_count = 0;
  bool is_depth_key_pressed = false;

  ros::Subscriber m_depth_sub;
  ros::Subscriber m_rgb_sub;
  ros::Subscriber m_camera_info_sub;
  ros::Timer m_timer;

  ros::NodeHandle & m_node;

  std::shared_ptr<tf2_ros::TransformListener> m_tf_listener;
  std::shared_ptr<tf2_ros::Buffer> m_tf_buffer;

  std::string m_tf_camera_frame;
  std::string m_tf_base_frame;
  
  std::string m_depth_image_topic;
  std::string m_rgb_image_topic;
  std::string m_camera_info_topic;

  std::string m_save_folder_name;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "save_images");
  ros::NodeHandle nh("~");

  SaveImages si(nh);

  ros::spin();
  cv::destroyAllWindows();

  return 0;
  }
