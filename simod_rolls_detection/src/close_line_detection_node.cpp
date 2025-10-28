// close_line_detection_node.cpp

// ROS
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <eigen_conversions/eigen_msg.h>
#include <actionlib/server/simple_action_server.h>
#include <tf/transform_datatypes.h>
#include <visualization_msgs/Marker.h>  // <-- RViz markers

// TF2
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>

#include <std_srvs/Trigger.h>
#include <jsoncpp/json/json.h>
#include <limits>

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

  template <typename T>
  static T SQR(const T &t) { return t * t; }

  CloseLineDetectionNode(std::shared_ptr<Node> nodeptr) : m_nodeptr(nodeptr)
  {
    // Topics
    m_nodeptr->param<std::string>("rgb_image_topic", m_rgb_image_topic, "rgb_image_topic");
    m_nodeptr->param<std::string>("camera_info_topic", m_camera_info_topic, "camera_info_topic");

    // Frames
    m_nodeptr->param<std::string>("world_frame_id", m_world_frame_id, "plate_center");
    m_nodeptr->param<std::string>("camera_frame_id", m_camera_frame_id, "arm2_camera_oak_d_pro");

    // Action
    m_nodeptr->param<std::string>("detect_close_line_action", m_detect_close_line_action, "/detect_close_line");
    m_as.reset(new ActionServer(*nodeptr, m_detect_close_line_action,
                                boost::function<void(const simod_rolls_detection::DetectCloseLineGoalConstPtr &)>(
                                    [this](const simod_rolls_detection::DetectCloseLineGoalConstPtr &goal)
                                    { this->Run(goal); }),
                                false));

    // Service + output path
    m_nodeptr->param<std::string>("close_line_points_json_path",
                                  m_close_line_points_json_path,
                                  "files/output/close_line_points.json");
    std::string detect_close_line_srv_name;
    m_nodeptr->param<std::string>("close_line_detect_srv",
                                  detect_close_line_srv_name,
                                  "/detect_close_line");
    m_srv_detect_close_line = m_nodeptr->advertiseService(
        detect_close_line_srv_name,
        &CloseLineDetectionNode::onDetectCloseLineSrv, this);

    // Subscribers
    ROS_INFO("close_line_detection_node: subscribing to color topic %s", m_rgb_image_topic.c_str());
    ROS_INFO("close_line_detection_node: subscribing to camera_info topic %s", m_camera_info_topic.c_str());
    m_rgb_image_sub = m_nodeptr->subscribe<sensor_msgs::Image>(m_rgb_image_topic, 1,
                                                               boost::function<void(const sensor_msgs::Image &)>([this](const sensor_msgs::Image &msg)
                                                                                                                 { this->RgbImageCallback(msg); }));
    m_camera_info_sub = m_nodeptr->subscribe<sensor_msgs::CameraInfo>(m_camera_info_topic, 1,
                                                                      boost::function<void(const sensor_msgs::CameraInfo &)>([this](const sensor_msgs::CameraInfo &msg)
                                                                                                                             { this->CameraInfoCallback(msg); }));

    // Prime frame scartate
    m_nodeptr->param<int>("discard_first_camera_frames", m_discard_first_camera_frames, 0);

    // Image publishers from detector
    {
      const StringVector publish_image_names = CloseLineDetection::GetImagePublisherNames();
      for (const std::string &s : publish_image_names)
      {
        const std::string topic_name = s + "_image";
        m_image_publishers[s] = m_nodeptr->advertise<sensor_msgs::Image>(topic_name, 1);
      }
    }

    // RViz markers (latched)
    line_marker_pub_   = m_nodeptr->advertise<visualization_msgs::Marker>("line_markers", 1, true);
    points_marker_pub_ = m_nodeptr->advertise<visualization_msgs::Marker>("line_points",  1, true);

    m_as->start();
  }

  // ===== ACTION =====
  void Run(const simod_rolls_detection::DetectCloseLineGoalConstPtr &goalptr)
  {
    const simod_rolls_detection::DetectCloseLineGoal &goal = *goalptr;

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
    }
    ROS_INFO("close_line_detection_node: received images.");

    // Camera pose dal goal
    Eigen::Affine3d camera_pose;
    tf::poseMsgToEigen(goal.camera_pose, camera_pose);

    // --- Config ---
    CloseLineDetection::Config config;
    m_nodeptr->param<int>("random_seed", config.random_seed, config.random_seed);
    m_nodeptr->param<float>("roll_diameter", config.roll_diameter, config.roll_diameter);
    m_nodeptr->param<int>("non_maximum_suppression_window", config.non_maximum_suppression_window, config.non_maximum_suppression_window);
    m_nodeptr->param<float>("min_correlation_threshold", config.min_correlation_threshold, config.min_correlation_threshold);
    m_nodeptr->param<float>("correlation_lines_weight", config.correlation_lines_weight, config.correlation_lines_weight);
    m_nodeptr->param<int>("median_filter_window", config.median_filter_window, config.median_filter_window);
    m_nodeptr->param<float>("line_max_angle", config.line_max_angle, config.line_max_angle);                                        // radians
    m_nodeptr->param<float>("min_virtual_camera_distance", config.min_virtual_camera_distance, config.min_virtual_camera_distance); // meters
    m_nodeptr->param<float>("basic_max_ransac_distance", config.basic_max_ransac_distance, config.basic_max_ransac_distance);
    m_nodeptr->param<int>("basic_gaussian_blur_half_window_size", config.basic_gaussian_blur_half_window_size, config.basic_gaussian_blur_half_window_size);
    m_nodeptr->param<int>("basic_canny_threshold", config.basic_canny_threshold, config.basic_canny_threshold);
    m_nodeptr->param<int>("basic_hough_threshold", config.basic_hough_threshold, config.basic_hough_threshold);
    m_nodeptr->param<bool>("mode_basic", config.mode_basic, config.mode_basic);

    config.initial_guess_window_size_x = goal.initial_guess_window_size_x;
    config.initial_guess_window_size_y = goal.initial_guess_window_size_y;

    // ===== HARD-CODE ROTATION (+90° attorno a Z) =====
    const double th = M_PI / 2.0; // +90°
    const Eigen::AngleAxisd Rz(th, Eigen::Vector3d::UnitZ());

    // Camera pose nel frame ruotato: T_{w'c} = R^{-1} * T_{wc}
    Eigen::Affine3d camera_pose_rot = Eigen::Affine3d::Identity();
    camera_pose_rot.linear() = Rz.inverse().toRotationMatrix() * camera_pose.linear();
    camera_pose_rot.translation() = Rz.inverse().toRotationMatrix() * camera_pose.translation();

    // Guess rotati
    const Eigen::Vector3d g_world(goal.initial_guess_x, goal.initial_guess_y, 0.0);
    const Eigen::Vector3d g_rot = Rz.inverse().toRotationMatrix() * g_world;
    const float igx_rot = static_cast<float>(g_rot.x());
    const float igy_rot = static_cast<float>(g_rot.y());

    // Detector
    CloseLineDetection cld(
        config,
        [this](const uint level, const std::string &message)
        { this->Log(level, message); },
        [this](const std::string &name, const cv::Mat &image, const std::string &encoding)
        { this->PublishImage(image, encoding, name); });

    // layer_z è quota → non si ruota
    Vector3fVector points_rot = cld.Run(rgb_image, *camera_info, camera_pose_rot.cast<float>(),
                                        goal.layer_z, igx_rot, igy_rot);

    // Riporta i punti nel frame world originale: p = R * p'
    Vector3fVector points;
    points.reserve(points_rot.size());
    for (const auto &p_r : points_rot)
    {
      const Eigen::Vector3d pr(p_r.x(), p_r.y(), p_r.z());
      const Eigen::Vector3d pw = Rz.toRotationMatrix() * pr;
      points.emplace_back(static_cast<float>(pw.x()),
                          static_cast<float>(pw.y()),
                          static_cast<float>(pw.z()));
    }

    ROS_INFO_STREAM("close_line_detection_node: success: " << (points.empty() ? "FALSE" : "TRUE"));

    // Pubblica markers in RViz
    PublishLineMarkers(points);

    simod_rolls_detection::DetectCloseLineResult result;
    result.success = !points.empty();
    for (const Eigen::Vector3f &pt : points)
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

  // ===== SERVICE: replica Run ma prende camera_pose dal TF =====
  bool onDetectCloseLineSrv(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res)
  {
    // 1) Attendi immagini (come Run)
    cv::Mat rgb_image;
    std::shared_ptr<CloseLineDetection::Intrinsics> camera_info;
    ROS_INFO("close_line_detection_node: service start.");
    {
      ros::Rate wait_rate(100);
      std::unique_lock<std::mutex> lock(m_mutex);
      if (m_discard_first_camera_frames)
        ROS_INFO("close_line_detection_node: first %d camera frames will be discarded.", int(m_discard_first_camera_frames));
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
          if (!ros::ok()) { res.success=false; res.message="shutdown"; return true; }
        }
        if (!ros::ok()) { res.success=false; res.message="shutdown"; return true; }
      }
      rgb_image = m_last_rgb_image;
      camera_info = m_last_camera_info;
    }
    ROS_INFO("close_line_detection_node: received images.");

    // 2) Camera pose dal TF (world <- camera)
    Eigen::Affine3d camera_pose = Eigen::Affine3d::Identity();
    try
    {
      const geometry_msgs::TransformStamped T =
          tf_buffer_.lookupTransform(m_world_frame_id, m_camera_frame_id, ros::Time(0), ros::Duration(0.5));
      camera_pose = Eigen::Affine3d(tf2::transformToEigen(T).matrix());
      const Eigen::Vector3d t = camera_pose.translation();
      const Eigen::Vector3d z_axis = camera_pose.linear().col(2);
      const double z_to_world_up = std::acos(std::max(-1.0, std::min(1.0, z_axis.dot(Eigen::Vector3d::UnitZ())))) * 180.0 / M_PI;
      ROS_INFO("[close_line_detection_node] T_wc: t=[%.3f %.3f %.3f], angle(camZ vs worldZ)= %.1f deg",
               t.x(), t.y(), t.z(), z_to_world_up);
    }
    catch (const tf2::TransformException &ex)
    {
      res.success = false;
      res.message = std::string("TF lookup failed: ") + ex.what();
      return true;
    }

    // 3) Config (come Run) + goal-equivalenti presi da param
    CloseLineDetection::Config config;
    m_nodeptr->param<int>("random_seed", config.random_seed, config.random_seed);
    m_nodeptr->param<float>("roll_diameter", config.roll_diameter, config.roll_diameter);
    m_nodeptr->param<int>("non_maximum_suppression_window", config.non_maximum_suppression_window, config.non_maximum_suppression_window);
    m_nodeptr->param<float>("min_correlation_threshold", config.min_correlation_threshold, config.min_correlation_threshold);
    m_nodeptr->param<float>("correlation_lines_weight", config.correlation_lines_weight, config.correlation_lines_weight);
    m_nodeptr->param<int>("median_filter_window", config.median_filter_window, config.median_filter_window);
    m_nodeptr->param<float>("line_max_angle", config.line_max_angle, config.line_max_angle);
    m_nodeptr->param<float>("min_virtual_camera_distance", config.min_virtual_camera_distance, config.min_virtual_camera_distance);
    m_nodeptr->param<float>("basic_max_ransac_distance", config.basic_max_ransac_distance, config.basic_max_ransac_distance);
    m_nodeptr->param<int>("basic_gaussian_blur_half_window_size", config.basic_gaussian_blur_half_window_size, config.basic_gaussian_blur_half_window_size);
    m_nodeptr->param<int>("basic_canny_threshold", config.basic_canny_threshold, config.basic_canny_threshold);
    m_nodeptr->param<int>("basic_hough_threshold", config.basic_hough_threshold, config.basic_hough_threshold);
    m_nodeptr->param<bool>("mode_basic", config.mode_basic, config.mode_basic);

    // --- finestre di ricerca: leggi come double, poi cast a float ---
    double ig_wx_d = 0.05; // default sensato
    double ig_wy_d = 0.30; // default sensato
    m_nodeptr->param("initial_guess_window_size_x", ig_wx_d, ig_wx_d);
    m_nodeptr->param("initial_guess_window_size_y", ig_wy_d, ig_wy_d);
    config.initial_guess_window_size_x = static_cast<float>(ig_wx_d);
    config.initial_guess_window_size_y = static_cast<float>(ig_wy_d);

    // --- initial_guess + layer_z (accetta anche "layer_height") ---
    float initial_guess_x = 0.0f, initial_guess_y = 0.0f, layer_height = 0.0f;
    bool ok = true;
    ok &= m_nodeptr->getParam("initial_guess_x", initial_guess_x);
    ok &= m_nodeptr->getParam("initial_guess_y", initial_guess_y);
    if (!m_nodeptr->getParam("layer_z", layer_height))
      ok &= m_nodeptr->getParam("layer_height", layer_height);

    if (!ok)
    {
      res.success = false;
      res.message = "params 'initial_guess_x', 'initial_guess_y', or 'layer_z/layer_height' missing";
      return true;
    }

    ROS_INFO("[close_line_detection_node] params: x0=%.3f y0=%.3f z=%.3f wx=%.3f wy=%.3f",
             initial_guess_x, initial_guess_y, layer_height,
             config.initial_guess_window_size_x, config.initial_guess_window_size_y);

    // ===== HARD-CODE ROTATION (+90° attorno a Z) =====
    const double th = M_PI / 2.0; // +90°
    const Eigen::AngleAxisd Rz(th, Eigen::Vector3d::UnitZ());

    // Pose ruotata: T_{w'c} = R^{-1} * T_{wc}
    Eigen::Affine3d camera_pose_rot = Eigen::Affine3d::Identity();
    camera_pose_rot.linear() = Rz.inverse().toRotationMatrix() * camera_pose.linear();
    camera_pose_rot.translation() = Rz.inverse().toRotationMatrix() * camera_pose.translation();

    // Guess ruotati
    const Eigen::Vector3d g_world(initial_guess_x, initial_guess_y, 0.0);
    const Eigen::Vector3d g_rot = Rz.inverse().toRotationMatrix() * g_world;
    const float igx_rot = static_cast<float>(g_rot.x());
    const float igy_rot = static_cast<float>(g_rot.y());

    // 4) Detector + Run
    CloseLineDetection cld(
        config,
        [this](const uint level, const std::string &msg){ this->Log(level, msg); },
        [this](const std::string &name, const cv::Mat &im, const std::string &enc){ this->PublishImage(im, enc, name); });

    Vector3fVector points_rot = cld.Run(
        rgb_image, *camera_info, camera_pose_rot.cast<float>(),
        layer_height, igx_rot, igy_rot);

    // Riporta i punti nel frame world originale
    Vector3fVector points;
    points.reserve(points_rot.size());
    for (const auto &p_r : points_rot)
    {
      const Eigen::Vector3d pr(p_r.x(), p_r.y(), p_r.z());
      const Eigen::Vector3d pw = Rz.toRotationMatrix() * pr;
      points.emplace_back(static_cast<float>(pw.x()),
                          static_cast<float>(pw.y()),
                          static_cast<float>(pw.z()));
    }

    ROS_INFO_STREAM("close_line_detection_node: success: " << (points.empty() ? "FALSE" : "TRUE"));

    if (points.empty())
    {
      res.success = false;
      res.message = "close line detection failed or no points";
      return true;
    }

    // Pubblica markers in RViz
    PublishLineMarkers(points);

    // 5) Salva JSON
    Json::Value j;
    j["frame_id"] = m_world_frame_id;
    Json::Value arr(Json::arrayValue);
    for (const auto &p : points)
    {
      Json::Value jp; jp["x"] = p.x(); jp["y"] = p.y(); jp["z"] = p.z();
      arr.append(jp);
    }
    j["points"] = arr;

    std::ofstream ofs(m_close_line_points_json_path);
    if (!ofs.is_open())
    {
      ROS_ERROR("close_line: cannot open '%s'", m_close_line_points_json_path.c_str());
      res.success = false;
      res.message = "failed to write JSON: " + m_close_line_points_json_path;
      return true;
    }
    ofs << j.toStyledString();

    res.success = true;
    res.message = "close line OK, points saved to " + m_close_line_points_json_path;
    return true;
  }

  // ===== Callbacks =====
  void RgbImageCallback(const sensor_msgs::Image &msg)
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    try
    {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
      m_last_rgb_image = cv_ptr->image;
    }
    catch (const cv_bridge::Exception &e)
    {
      ROS_ERROR("RgbImageCallback: Could not convert from '%s' to 'bgr8'.", msg.encoding.c_str());
      return;
    }
  }

  void CameraInfoCallback(const sensor_msgs::CameraInfo &msg)
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    const float fx = msg.K[0];
    const float fy = msg.K[4];
    const float cx = msg.K[2];
    const float cy = msg.K[5];

    std::shared_ptr<CloseLineDetection::Intrinsics> ci(new CloseLineDetection::Intrinsics);
    ci->fx = fx; ci->fy = fy; ci->cx = cx; ci->cy = cy;
    m_last_camera_info = ci;
  }

  // ===== Logging & Publishing =====
  void Log(const uint level, const std::string &message)
  {
    switch (level)
    {
    case 0: ROS_DEBUG("close_line_detection: %s", message.c_str()); break;
    case 1: ROS_INFO ("close_line_detection: %s", message.c_str()); break;
    case 2: ROS_WARN ("close_line_detection: %s", message.c_str()); break;
    case 3: ROS_ERROR("close_line_detection: %s", message.c_str()); break;
    case 4: ROS_FATAL("close_line_detection: %s", message.c_str()); break;
    default: ROS_ERROR("close_line_detection_node: Invalid logger level %d, message was '%s'", int(level), message.c_str());
    }
  }

  void PublishImage(const cv::Mat &image, const std::string &encoding, const std::string &name)
  {
    if (m_image_publishers.find(name) == m_image_publishers.end())
    {
      ROS_WARN("close_line_detection_node: Could not find image publisher with name %s", name.c_str());
      return;
    }
    PublishImage(image, encoding, m_image_publishers[name]);
  }

  void PublishImage(const cv::Mat &image, const std::string &encoding, ros::Publisher &pub)
  {
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    cv_ptr->image = image;
    cv_ptr->encoding = encoding;
    ImageMsg img = *cv_ptr->toImageMsg();
    pub.publish(img);
  }

  // ===== RViz Markers =====
  void PublishLineMarkers(const Vector3fVector& pts)
  {
    if (pts.empty()) return;

    // LINE_STRIP (verde)
    visualization_msgs::Marker line;
    line.header.frame_id = m_world_frame_id;
    line.header.stamp    = ros::Time::now();
    line.ns = "close_line";
    line.id = 0;
    line.type = visualization_msgs::Marker::LINE_STRIP;
    line.action = visualization_msgs::Marker::ADD;
    line.pose.orientation.w = 1.0;
    line.scale.x = 0.01; // spessore
    line.color.r = 0.0f; line.color.g = 1.0f; line.color.b = 0.0f; line.color.a = 1.0f;
    line.lifetime = ros::Duration(0.0);

    for (const auto& p : pts)
    {
      geometry_msgs::Point gp; gp.x = p.x(); gp.y = p.y(); gp.z = p.z();
      line.points.push_back(gp);
    }

    // SPHERE_LIST (rossi)
    visualization_msgs::Marker dots = line;
    dots.id   = 1;
    dots.type = visualization_msgs::Marker::SPHERE_LIST;
    dots.scale.x = dots.scale.y = dots.scale.z = 0.02;
    dots.color.r = 1.0f; dots.color.g = 0.0f; dots.color.b = 0.0f; dots.color.a = 1.0f;
    dots.points.clear();
    for (const auto& p : pts)
    {
      geometry_msgs::Point gp; gp.x = p.x(); gp.y = p.y(); gp.z = p.z();
      dots.points.push_back(gp);
    }

    line_marker_pub_.publish(line);
    points_marker_pub_.publish(dots);
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

  int m_discard_first_camera_frames{0};

  ros::ServiceServer m_srv_detect_close_line;
  std::string m_world_frame_id;
  std::string m_camera_frame_id;
  std::string m_close_line_points_json_path;

  // TF2 buffer + listener (ordine dichiarazione IMPORTANTE)
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_{tf_buffer_};

  // RViz marker publishers
  ros::Publisher line_marker_pub_;
  ros::Publisher points_marker_pub_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "close_line_detection_node");
  std::shared_ptr<Node> nodeptr(new Node("~"));
  ROS_INFO("close_line_detection node started");

  CloseLineDetectionNode pd(nodeptr);

  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
