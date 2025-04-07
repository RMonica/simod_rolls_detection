// ROS
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/MarkerArray.h>
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

// PCL
#include <pcl/common/colors.h>
#include <pcl/point_cloud.h>

#include <simod_rolls_detection/DetectPalletAction.h>

#include <pallet_detection.h>

typedef ros::NodeHandle Node;
typedef sensor_msgs::Image ImageMsg;
typedef sensor_msgs::CameraInfo CameraInfoMsg;
typedef sensor_msgs::PointCloud2 PointCloud2Msg;
typedef visualization_msgs::Marker MarkerMsg;
typedef visualization_msgs::MarkerArray MarkerArrayMsg;

class PalletDetectionNode
{
  public:
  typedef pcl::PointCloud<pcl::PointXYZRGB> PointXYZRGBCloud;
  typedef std::shared_ptr<PointXYZRGBCloud> PointXYZRGBCloudPtr;
  typedef std::vector<int> IntVector;
  typedef std::shared_ptr<IntVector> IntVectorPtr;
  typedef std::vector<float> FloatVector;
  typedef std::vector<double> DoubleVector;
  typedef std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > Vector4fVector;
  typedef std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > Vector4dVector;
  typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > Vector3dVector;
  typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > Vector2dVector;
  typedef std::set<int> IntSet;
  typedef std::pair<Eigen::Vector3f, Eigen::Vector3f> Vector3fPair;
  typedef std::vector<Vector3fPair> Vector3fPairVector;

  typedef uint64_t uint64;
  typedef uint32_t uint32;
  typedef int32_t int32;
  typedef int64_t int64;
  typedef uint8_t uint8;
  typedef uint16_t uint16;
  typedef std::vector<uint64> Uint64Vector;
  typedef std::pair<uint64, uint64> Uint64Pair;
  typedef std::vector<Uint64Pair> Uint64PairVector;
  typedef std::set<Uint64Pair> Uint64PairSet;

  typedef PalletDetection::ExpectedElementType ExpectedElementType;
  typedef PalletDetection::ExpectedPallet ExpectedPallet;
  typedef PalletDetection::ExpectedElement ExpectedElement;

  typedef std::vector<visualization_msgs::Marker> MarkerVector;

  typedef actionlib::SimpleActionServer<simod_rolls_detection::DetectPalletAction> ActionServer;

  template <typename T> static T SQR(const T & t) {return t * t; }

  Eigen::Vector3f StrToVector3f(const std::string & str)
  {
    Eigen::Vector3f result;
    std::istringstream istr(str);
    istr >> result.x() >> result.y() >> result.z();
    if (!istr)
      throw std::string("Could not parse string to Vector3d: " + str);
    return result;
  }

  PalletDetectionNode(std::shared_ptr<Node> nodeptr): m_nodeptr(nodeptr)
  {
    PalletDetection::Config config;

    m_nodeptr->param<std::string>("depth_image_topic", m_depth_image_topic, "camera_info_topic");
    m_nodeptr->param<std::string>("rgb_image_topic", m_rgb_image_topic, "rgb_image_topic");
    m_nodeptr->param<std::string>("camera_info_topic", m_camera_info_topic, "camera_info_topic");

    m_nodeptr->param<std::string>("detect_pallet_action", m_detect_pallet_action, "/detect_pallet");
    m_as.reset(new ActionServer(*nodeptr, m_detect_pallet_action,
      boost::function<void(const simod_rolls_detection::DetectPalletGoalConstPtr &)>(
        [this](const simod_rolls_detection::DetectPalletGoalConstPtr & goal){this->Run(goal); }),
      false));

    ROS_INFO("pallet_detection_node: subscribing to depth topic %s", m_depth_image_topic.c_str());
    ROS_INFO("pallet_detection_node: subscribing to color topic %s", m_rgb_image_topic.c_str());
    ROS_INFO("pallet_detection_node: subscribing to camera_info topic %s", m_camera_info_topic.c_str());
    m_depth_image_sub = m_nodeptr->subscribe<sensor_msgs::Image>(m_depth_image_topic, 1,
      boost::function<void(const sensor_msgs::Image &)>([this](const sensor_msgs::Image & msg){this->DepthImageCallback(msg); }));
    m_rgb_image_sub = m_nodeptr->subscribe<sensor_msgs::Image>(m_rgb_image_topic, 1,
      boost::function<void(const sensor_msgs::Image &)>([this](const sensor_msgs::Image & msg){this->RgbImageCallback(msg); }));
    m_camera_info_sub = m_nodeptr->subscribe<sensor_msgs::CameraInfo>(m_camera_info_topic, 1,
      boost::function<void(const sensor_msgs::CameraInfo &)>([this](const sensor_msgs::CameraInfo & msg){
      this->CameraInfoCallback(msg);
    }));

    m_point_cloud_pub = nodeptr->advertise<PointCloud2Msg>("point_cloud", 1);
    m_input_point_cloud_pub = nodeptr->advertise<PointCloud2Msg>("input_point_cloud", 1);
    m_valid_points_cloud_pub = nodeptr->advertise<PointCloud2Msg>("valid_points_cloud", 1);
    m_cluster_image_pub = nodeptr->advertise<ImageMsg>("cluster_image", 1);
    m_edge_image_pub = nodeptr->advertise<ImageMsg>("edge_image", 1);
    m_plane_image_pub = nodeptr->advertise<ImageMsg>("plane_image", 1);
    m_depth_image_pub = nodeptr->advertise<ImageMsg>("depth_image", 1);

    m_markers_pub = nodeptr->advertise<MarkerArrayMsg>("markers", 1);

    m_nodeptr->param<std::string>("world_frame_id", m_world_frame_id, "map");

    m_nodeptr->param<int>("depth_hough_threshold", config.depth_hough_threshold, 50);
    m_nodeptr->param<int>("depth_hough_min_length", config.depth_hough_min_length, 100);
    m_nodeptr->param<int>("depth_hough_max_gap", config.depth_hough_max_gap, 50);

    m_nodeptr->param<double>("min_plane_camera_distance", config.min_plane_camera_distance, 0.5);

    m_nodeptr->param<double>("vertical_line_angle_tolerance", config.vertical_line_angle_tolerance, M_PI / 10.0f);

    m_nodeptr->param("ransac_plane_angle_tolerance", config.ransac_plane_angle_tolerance, 5.0 / 180.0 * M_PI);
    m_nodeptr->param("ransac_plane_distance_tolerance", config.ransac_plane_distance_tolerance, 0.025); // used to fit plane
    m_nodeptr->param("ransac_plane_inliers_tolerance", config.ransac_plane_inliers_tolerance, 0.05);    // used to clear inliers around plane

    m_nodeptr->param<double>("plane_camera_max_angle", config.plane_camera_max_angle, 70.0 / 180.0 * M_PI);

    m_nodeptr->param<float>("plane_edge_discontinuity_dist_th", config.plane_edge_discontinuity_dist_th, 0.05f);
    m_nodeptr->param<float>("plane_edge_discontinuity_angle_th", config.plane_edge_discontinuity_angle_th, 20.0f * M_PI / 180.0f);

    m_nodeptr->param<double>("depth_image_max_discontinuity_th", config.depth_image_max_discontinuity_th, 0.05);
    m_nodeptr->param<double>("depth_image_max_vertical_angle", config.depth_image_max_vertical_angle, 20.0 / 180.0 * M_PI);
    m_nodeptr->param<int>("depth_image_normal_window", config.depth_image_normal_window, 2);
    m_nodeptr->param<int>("depth_image_closing_window", config.depth_image_closing_window, 15);

    m_nodeptr->param<double>("min_cluster_points_at_1m", config.min_cluster_points_at_1m, 50000);
    m_nodeptr->param<int>("min_cluster_points", config.min_cluster_points, 10000);

    m_nodeptr->param<double>("pillars_merge_threshold", config.pillars_merge_threshold, 0.025);

    m_nodeptr->param<double>("planes_similarity_max_angle", config.planes_similarity_max_angle, 20.0f * M_PI / 180.0f);
    m_nodeptr->param<double>("planes_similarity_max_distance", config.planes_similarity_max_distance, 0.1f);
    m_nodeptr->param<double>("points_similarity_max_distance", config.points_similarity_max_distance, 0.1f);

    m_nodeptr->param<double>("max_pose_correction_distance", config.max_pose_correction_distance, 2.0f);
    m_nodeptr->param<double>("max_pose_correction_angle", config.max_pose_correction_angle, M_PI / 2.0f);

    m_nodeptr->param<int>("plane_ransac_iterations", config.plane_ransac_iterations, 2000);
    m_nodeptr->param<double>("plane_ransac_max_error", config.plane_ransac_max_error, 0.1);

    m_nodeptr->param<int>("random_seed", config.random_seed, std::random_device()());

    m_pallet_detection.reset(new PalletDetection(config));

    m_pallet_detection->SetLogFunction([this](const uint level, const std::string & message) {this->Log(level, message); });
    m_pallet_detection->SetPublishImageFunction([this](const cv::Mat & image, const std::string & encoding, const std::string & name) {
      this->PublishImage(image, encoding, name);
    });
    m_pallet_detection->SetPublishPalletFunction([this](const ExpectedPallet & real_pallet, const ExpectedPallet & loaded_pallet,
                                                        const ExpectedPallet & estimated_pallet, const ExpectedPallet & estimated_refined_pallet) {
      this->PublishPallet(real_pallet, loaded_pallet, estimated_pallet, estimated_refined_pallet);
    });
    m_pallet_detection->SetPublishCloudFunction([this](const PointXYZRGBCloud & cloud, const std::string & name) {
      this->PublishCloud(cloud, name);
    });

    m_as->start();
  }

  void Run(const simod_rolls_detection::DetectPalletGoalConstPtr & goalptr)
  {
    const simod_rolls_detection::DetectPalletGoal & goal = *goalptr;

    cv::Mat rgb_image;
    cv::Mat depth_image;
    std::shared_ptr<PalletDetection::CameraInfo> camera_info;
    ROS_INFO("pallet_detection_node: action start.");
    {
      ros::Rate wait_rate(100);
      std::unique_lock<std::mutex> lock(m_mutex);
      m_last_rgb_image = cv::Mat();
      m_last_depth_image = cv::Mat();
      m_last_camera_info.reset();
      while (m_last_rgb_image.empty() || m_last_depth_image.empty() || !m_last_camera_info)
      {
        lock.unlock();
        ROS_INFO_THROTTLE(2.0, "pallet_detection_node: waiting for images...");
        wait_rate.sleep();
        lock.lock();

        if (!ros::ok())
          return;
      }


      rgb_image = m_last_rgb_image;
      depth_image = m_last_depth_image;
      camera_info = m_last_camera_info;
    } // lock released here
    ROS_INFO("pallet_detection_node: received images.");

    Eigen::Affine3d camera_pose;
    tf::poseMsgToEigen(goal.camera_pose, camera_pose);

    PalletDetection::BoundingBox initial_guess;
    initial_guess.center.x() = goal.initial_guess_center.x;
    initial_guess.center.y() = goal.initial_guess_center.y;
    initial_guess.center.z() = goal.initial_guess_center.z;
    initial_guess.size.x() = goal.initial_guess_size.x;
    initial_guess.size.y() = goal.initial_guess_size.y;
    initial_guess.size.z() = goal.initial_guess_size.z;
    initial_guess.rotation = goal.initial_guess_rotation;

    const float floor_z = goal.floor_z;
    const std::string expected_pallet_file_name = goal.pallet_description_filename;

    PalletDetection::DetectionResult detection_result =
      m_pallet_detection->Detect(rgb_image,
                                 depth_image,
                                 *camera_info,
                                 camera_pose.cast<float>(),
                                 floor_z,
                                 initial_guess,
                                 expected_pallet_file_name
                                 );

    ROS_INFO_STREAM("success: " << (detection_result.success ? "TRUE" : "FALSE"));
    ROS_INFO_STREAM("final pose: " << detection_result.pose.transpose());

    simod_rolls_detection::DetectPalletResult result;
    result.success = detection_result.success;
    result.consensus = detection_result.consensus;
    {
      Eigen::Affine3d pose = Eigen::Affine3d::Identity();
      pose.translation().head<2>() = detection_result.pose.head<2>();
      pose.translation().z() = floor_z;
      pose.linear() = Eigen::AngleAxisd(detection_result.pose.z(), Eigen::Vector3d::UnitZ()).matrix();
      tf::poseEigenToMsg(pose, result.pallet_pose);
    }
    for (uint64 box_i = 0; box_i < detection_result.boxes.size(); box_i++)
    {
      geometry_msgs::Pose box_pose;
      tf::poseEigenToMsg(detection_result.boxes[box_i], box_pose);
      result.box_poses.push_back(box_pose);
    }

    ROS_INFO("pallet_detection_node: action end.");
    m_as->setSucceeded(result);
  }

  void DepthImageCallback(const sensor_msgs::Image & msg)
  {
    std::unique_lock<std::mutex> lock(m_mutex);

    try
    {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, "16UC1");
      m_last_depth_image = cv_ptr->image;
    }
    catch (const cv_bridge::Exception & e)
    {
      ROS_ERROR("DepthImageCallback: Could not convert from '%s' to '16UC1'.", msg.encoding.c_str());
      return;
    }
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

    std::shared_ptr<PalletDetection::CameraInfo> ci(new PalletDetection::CameraInfo);
    ci->fx = fx;
    ci->fy = fy;
    ci->cx = cx;
    ci->cy = cy;

    m_last_camera_info = ci;
  }

  MarkerVector PalletToVisualizationMarkers(const ExpectedPallet & pallet,
                                            const Eigen::Vector3f & base_color,
                                            const std::string & ns,
                                            uint64 & marker_id)
  {
    MarkerVector result;

    for (const ExpectedElement & elem : pallet)
    {
      if (elem.type == ExpectedElementType::PILLAR)
      {
        const Eigen::Vector4d & pillar = elem.pillar;

        float pillar_lightness = 0.0f;
        if (elem.pillar_left_plane_id != uint64(-1))
          pillar_lightness += 0.4f;
        if (elem.pillar_right_plane_id != uint64(-1))
          pillar_lightness -= 0.4f;

        MarkerMsg marker;
        marker.header.frame_id = m_world_frame_id;
        marker.action = marker.ADD;
        marker.type = marker.CYLINDER;
        marker.id = marker_id++;
        marker.ns = ns;

        marker.scale.x = 0.05;
        marker.scale.y = 0.05;
        marker.scale.z = pillar.w() - pillar.z();

        marker.pose.position.x = pillar.x();
        marker.pose.position.y = pillar.y();
        marker.pose.position.z = (pillar.w() + pillar.z()) / 2.0;

        marker.color.r = std::min(1.0f, std::max(0.0f, base_color.x() + pillar_lightness));
        marker.color.g = std::min(1.0f, std::max(0.0f, base_color.y() + pillar_lightness));
        marker.color.b = std::min(1.0f, std::max(0.0f, base_color.z() + pillar_lightness));
        marker.color.a = 1.0;

        result.push_back(marker);
      }

      if (elem.type == ExpectedElementType::PLANE)
      {
        const Eigen::Vector4d & plane = elem.plane;
        const Eigen::Vector2d & plane_z = elem.plane_z;

        const Eigen::Vector3d normal = plane.head<3>();
        const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
        const Eigen::Vector3d right = up.cross(normal);
        Eigen::Matrix3d rotation;
        rotation.col(0) = right;
        rotation.col(1) = up;
        rotation.col(2) = normal;

        const Eigen::Quaterniond q(rotation);

        Eigen::Vector3d pt;
        if (std::isnan(elem.plane_point.x()))
          pt = -normal * plane.w();
        else
        {
          const Eigen::Vector3d sp = elem.plane_point;
          const float dist = normal.dot(sp) + plane.w();
          pt = sp - normal * dist;
        }

        MarkerMsg marker;
        marker.header.frame_id = m_world_frame_id;
        marker.action = marker.ADD;
        marker.type = marker.CUBE;
        marker.id = marker_id++;
        marker.ns = ns;

        marker.scale.x = 1.0f;
        marker.scale.y = (plane_z.y() - plane_z.x());
        marker.scale.z = 0.01f;

        marker.pose.position.x = pt.x();
        marker.pose.position.y = pt.y();
        marker.pose.position.z = (plane_z.y() + plane_z.x()) / 2.0;
        marker.pose.orientation.x = q.x();
        marker.pose.orientation.y = q.y();
        marker.pose.orientation.z = q.z();
        marker.pose.orientation.w = q.w();

        marker.color.r = base_color.x();
        marker.color.g = base_color.y();
        marker.color.b = base_color.z();
        marker.color.a = 0.5;

        result.push_back(marker);
      }

      if (elem.type == ExpectedElementType::BOX)
      {
        const Eigen::Vector4d & box = elem.box;
        const Eigen::Vector3d & box_size = elem.box_size;

        const double rotation_z = box.w();
        const Eigen::Matrix3d rot_mat = Eigen::AngleAxisd(rotation_z, Eigen::Vector3d::UnitZ()).matrix();
        const Eigen::Quaterniond q(rot_mat);

        MarkerMsg marker;
        marker.header.frame_id = m_world_frame_id;
        marker.action = marker.ADD;
        marker.type = marker.CUBE;
        marker.id = marker_id++;
        marker.ns = ns;

        marker.scale.x = box_size.x();
        marker.scale.y = box_size.y();
        marker.scale.z = box_size.z();

        marker.pose.position.x = box.x();
        marker.pose.position.y = box.y();
        marker.pose.position.z = box.z();
        marker.pose.orientation.x = q.x();
        marker.pose.orientation.y = q.y();
        marker.pose.orientation.z = q.z();
        marker.pose.orientation.w = q.w();

        marker.color.r = base_color.x();
        marker.color.g = base_color.y();
        marker.color.b = base_color.z();
        marker.color.a = 0.5;

        result.push_back(marker);
      }
    }

    return result;
  }

  void PublishImage(const cv::Mat &image, const std::string &encoding, const std::string & name)
  {
    if (name == "cluster_image")
      PublishImage(image, encoding, m_cluster_image_pub);
    else if (name == "edge_image")
      PublishImage(image, encoding, m_edge_image_pub);
    else if (name == "plane_image")
      PublishImage(image, encoding, m_plane_image_pub);
    else if (name == "depth_image")
      PublishImage(image, encoding, m_depth_image_pub);
    else
      ROS_WARN("pallet_detection_node: Could not find image publisher with name %s", name.c_str());
  }

  void PublishCloud(const PointXYZRGBCloud & cloud, const std::string & name)
  {
    if (name == "input_cloud")
      PublishCloud(cloud, m_input_point_cloud_pub);
    else if (name == "cloud")
      PublishCloud(cloud, m_point_cloud_pub);
    else if (name == "valid_points_cloud")
      PublishCloud(cloud, m_valid_points_cloud_pub);
    else
      ROS_WARN("pallet_detection_node: Could not find point cloud publisher with name %s", name.c_str());
  }

  void PublishPallet(const ExpectedPallet & real_pallet, const ExpectedPallet & loaded_pallet,
                     const ExpectedPallet & estimated_pallet, const ExpectedPallet & estimated_refined_pallet)
  {
    ROS_INFO("Publishing pallet.");
    uint64 marker_id = 0;
    MarkerArrayMsg marker_array;
    {
      MarkerMsg marker;
      marker.id = -1;
      marker.action = marker.DELETEALL;
      marker_array.markers.push_back(marker);
    }

    const MarkerVector real_markers =
      PalletToVisualizationMarkers(real_pallet, Eigen::Vector3f(0.0f, 0.0f, 1.0f), "real", marker_id);
    const MarkerVector expected_markers =
      PalletToVisualizationMarkers(loaded_pallet, Eigen::Vector3f(1.0f, 1.0f, 0.0f), "loaded", marker_id);
    const MarkerVector estimated_markers =
      PalletToVisualizationMarkers(estimated_pallet, Eigen::Vector3f(0.0f, 1.0f, 0.0f), "estimated", marker_id);
    const MarkerVector estimated_refined_markers =
      PalletToVisualizationMarkers(estimated_refined_pallet, Eigen::Vector3f(1.0f, 0.0f, 0.0f), "estimated_refined", marker_id);
    marker_array.markers.insert(marker_array.markers.end(), real_markers.begin(), real_markers.end());
    marker_array.markers.insert(marker_array.markers.end(), expected_markers.begin(), expected_markers.end());
    marker_array.markers.insert(marker_array.markers.end(), estimated_markers.begin(), estimated_markers.end());
    marker_array.markers.insert(marker_array.markers.end(), estimated_refined_markers.begin(), estimated_refined_markers.end());

    m_markers_pub.publish(marker_array);
  }

  void Log(const uint level, const std::string & message)
  {
    switch (level)
    {
    case 0:
      ROS_DEBUG("pallet_detection_node: %s", message.c_str());
      break;
    case 1:
      ROS_INFO("pallet_detection_node: %s", message.c_str());
      break;
    case 2:
      ROS_WARN("pallet_detection_node: %s", message.c_str());
      break;
    case 3:
      ROS_ERROR("pallet_detection_node: %s", message.c_str());
      break;
    case 4:
      ROS_FATAL("pallet_detection_node: %s", message.c_str());
      break;
    default:
      ROS_ERROR("pallet_detection_node: Invalid logger level %d, message was %s", int(level), message.c_str());
    }
  }

  void PublishCloud(const pcl::PointCloud<pcl::PointXYZRGB> & cloud, ros::Publisher & pub)
  {
    PointCloud2Msg output;
    pcl::toROSMsg(cloud, output);
    output.header.frame_id = m_world_frame_id;
    output.header.stamp = ros::Time::now();
    pub.publish(output);
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
  ros::Publisher m_input_point_cloud_pub;
  ros::Publisher m_point_cloud_pub;
  ros::Publisher m_valid_points_cloud_pub;
  ros::Publisher m_cluster_image_pub;
  ros::Publisher m_plane_image_pub;
  ros::Publisher m_edge_image_pub;
  ros::Publisher m_depth_image_pub;

  ros::Publisher m_markers_pub;

  ros::Subscriber m_rgb_image_sub;
  ros::Subscriber m_depth_image_sub;
  ros::Subscriber m_camera_info_sub;

  std::string m_world_frame_id;

  std::shared_ptr<ActionServer> m_as;

  std::string m_depth_image_topic;
  std::string m_rgb_image_topic;
  std::string m_camera_info_topic;
  std::string m_detect_pallet_action;

  cv::Mat m_last_rgb_image;
  cv::Mat m_last_depth_image;
  std::shared_ptr<PalletDetection::CameraInfo> m_last_camera_info;

  ros::Timer m_timer;
  std::mutex m_mutex;

  PalletDetection::Ptr m_pallet_detection;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pallet detection");
  std::shared_ptr<Node> nodeptr(new Node("~"));
  ROS_INFO("pallet_detection node started");

  PalletDetectionNode pd(nodeptr);
  ros::spin();

	return 0;
}
