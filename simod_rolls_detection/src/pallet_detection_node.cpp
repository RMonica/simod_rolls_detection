// ROS
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <eigen_conversions/eigen_msg.h>

// OpenCV
#include <opencv2/opencv.hpp>

// STL
#include <fstream>
#include <stdint.h>
#include <vector>
#include <set>
#include <cmath>
#include <map>

// PCL
#include <pcl/common/colors.h>
#include <pcl/point_cloud.h>

#include <pallet_detection.h>

typedef ros::NodeHandle Node;
typedef sensor_msgs::PointCloud2 PointCloud2Msg;
typedef sensor_msgs::Image ImageMsg;
typedef sensor_msgs::CameraInfo CameraInfoMsg;
typedef visualization_msgs::Marker MarkerMsg;
typedef visualization_msgs::MarkerArray MarkerArrayMsg;
typedef geometry_msgs::TransformStamped TransformStampedMsg;

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

    m_point_cloud_pub = nodeptr->advertise<PointCloud2Msg>("point_cloud", 1);
    m_input_point_cloud_pub = nodeptr->advertise<PointCloud2Msg>("input_point_cloud", 1);
    m_valid_points_cloud_pub = nodeptr->advertise<PointCloud2Msg>("valid_points_cloud", 1);
    m_cluster_image_pub = nodeptr->advertise<ImageMsg>("cluster_image", 1);
    m_edge_image_pub = nodeptr->advertise<ImageMsg>("edge_image", 1);
    m_plane_image_pub = nodeptr->advertise<ImageMsg>("plane_image", 1);

    m_nodeptr->param<std::string>("world_frame_id", m_world_frame_id, "map");
    m_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>();

    m_markers_pub = nodeptr->advertise<MarkerArrayMsg>("markers", 1);

    m_timer = m_nodeptr->createTimer(ros::Duration(0.0), [this](const ros::TimerEvent &){this->Run(); }, true);

    m_nodeptr->param<std::string>("rgb_filename", m_image_file_name, "");
    m_nodeptr->param<std::string>("depth_filename", m_depth_file_name, "");
    m_nodeptr->param<std::string>("camera_info_filename", m_camera_info_file_name, "");
    m_nodeptr->param<std::string>("expected_pallet_filename", m_expected_pallet_file_name, "");
    m_nodeptr->param<std::string>("camera_pose_filename", m_camera_pose_file_name, "");
    m_nodeptr->param<std::string>("initial_guess_filename", m_initial_guess_file_name, "");

    m_nodeptr->param<int>("depth_hough_threshold", config.depth_hough_threshold, 50);
    m_nodeptr->param<int>("depth_hough_min_length", config.depth_hough_min_length, 100);
    m_nodeptr->param<int>("depth_hough_max_gap", config.depth_hough_max_gap, 50);

    m_nodeptr->param<double>("min_plane_camera_distance", config.min_plane_camera_distance, 0.5);

    m_nodeptr->param<double>("vertical_line_angle_tolerance", config.vertical_line_angle_tolerance, M_PI / 10.0f);

    m_nodeptr->param("ransac_plane_angle_tolerance", config.ransac_plane_angle_tolerance, 5.0 / 180.0 * M_PI);
    m_nodeptr->param("ransac_plane_distance_tolerance", config.ransac_plane_distance_tolerance, 0.025); // used to fit plane
    m_nodeptr->param("ransac_plane_inliers_tolerance", config.ransac_plane_inliers_tolerance, 0.05);    // used to clear inliers around plane

    m_nodeptr->param<double>("plane_camera_max_angle", config.plane_camera_max_angle, 70.0 / 180.0 * M_PI);

    config.plane_edge_discontinuity_dist_th = m_nodeptr->param<float>("plane_edge_discontinuity_dist_th", 0.05f);
    config.plane_edge_discontinuity_angle_th = m_nodeptr->param<float>("plane_edge_discontinuity_angle_th", 20.0f * M_PI / 180.0f);

    config.depth_image_max_discontinuity_th = m_nodeptr->param<double>("depth_image_max_discontinuity_th", 0.05);
    config.depth_image_max_vertical_angle = m_nodeptr->param<double>("depth_image_max_vertical_angle", 20.0 / 180.0 * M_PI);
    config.depth_image_normal_window = m_nodeptr->param<int>("depth_image_normal_window", 2);
    config.depth_image_closing_window = m_nodeptr->param<int>("depth_image_closing_window", 15);

    config.min_cluster_points_at_1m = m_nodeptr->param<double>("min_cluster_points_at_1m", 50000);
    config.min_cluster_points = m_nodeptr->param<int>("min_cluster_points", 10000);

    config.pillars_merge_threshold = m_nodeptr->param<double>("pillars_merge_threshold", 0.025);

    config.planes_similarity_max_angle = m_nodeptr->param<double>("planes_similarity_max_angle", 20.0f * M_PI / 180.0f);
    config.planes_similarity_max_distance = m_nodeptr->param<double>("planes_similarity_max_distance", 0.1f);
    config.points_similarity_max_distance = m_nodeptr->param<double>("points_similarity_max_distance", 0.1f);

    config.max_pose_correction_distance = m_nodeptr->param<double>("max_pose_correction_distance", 2.0f);
    config.max_pose_correction_angle = m_nodeptr->param<double>("max_pose_correction_angle", M_PI / 2.0f);

    config.plane_ransac_iterations = m_nodeptr->param<int>("plane_ransac_iterations", 2000);
    config.plane_ransac_max_error = m_nodeptr->param<double>("plane_ransac_max_error", 0.1);

    config.random_seed = m_nodeptr->param<int>("random_seed", std::random_device()());

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
  }

  void Load(cv::Mat & rgb_image, cv::Mat & depth_image, PalletDetection::CameraInfo & camera_info,
            Eigen::Affine3f & camera_pose, PalletDetection::BoundingBox & initial_guess, float & floor_z)
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

  PalletDetection::CameraInfo LoadCameraInfo(const std::string filename)
  {
    PalletDetection::CameraInfo result;

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


  PalletDetection::BoundingBox LoadInitialGuess(std::istream & istr, float & floor_z) const
  {
    PalletDetection::BoundingBox result;
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

        marker.color.r = base_color.x();
        marker.color.g = base_color.y();
        marker.color.b = base_color.z();
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

  void PublishImage(const cv::Mat &image, const std::string &encoding, const std::string & name)
  {
    if (name == "cluster_image")
      PublishImage(image, encoding, m_cluster_image_pub);
    else if (name == "edge_image")
      PublishImage(image, encoding, m_edge_image_pub);
    else if (name == "plane_image")
      PublishImage(image, encoding, m_plane_image_pub);
    else
      ROS_ERROR("Could not find image publisher with name %s", name.c_str());
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
      ROS_ERROR("Could not find point cloud publisher with name %s", name.c_str());
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
      ROS_DEBUG("%s", message.c_str());
      break;
    case 1:
      ROS_INFO("%s", message.c_str());
      break;
    case 2:
      ROS_WARN("%s", message.c_str());
      break;
    case 3:
      ROS_ERROR("%s", message.c_str());
      break;
    case 4:
      ROS_FATAL("%s", message.c_str());
      break;
    default:
      ROS_ERROR("Invalid logger level %d, message was %s", int(level), message.c_str());
    }
  }

  void Run()
  {
    cv::Mat rgb_image;
    cv::Mat depth_image;
    Eigen::Affine3f camera_pose;

    ROS_INFO("loading");

    PalletDetection::BoundingBox initial_guess;
    PalletDetection::CameraInfo camera_info;
    float floor_z;
    Load(rgb_image, depth_image, camera_info, camera_pose, initial_guess, floor_z);

    PalletDetection::DetectionResult detection_result =
      m_pallet_detection->Detect(rgb_image,
                                 depth_image,
                                 camera_info,
                                 camera_pose,
                                 floor_z,
                                 initial_guess,
                                 m_expected_pallet_file_name);

    ROS_INFO_STREAM("success: " << (detection_result.success ? "TRUE" : "FALSE"));
    ROS_INFO_STREAM("final pose: " << detection_result.pose.transpose());

    ROS_INFO_STREAM("publishing " << detection_result.boxes.size() << " boxes.");
    for (uint64 box_i = 0; box_i < detection_result.boxes.size(); box_i++)
    {
      const Eigen::Affine3d box = detection_result.boxes[box_i];

      TransformStampedMsg t;
      tf::transformEigenToMsg(box, t.transform);

      t.header.stamp = ros::Time::now();
      t.header.frame_id = m_world_frame_id;
      t.child_frame_id = "box_" + std::to_string(box_i);

      m_tf_broadcaster->sendTransform(t);
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

  ros::Publisher m_markers_pub;

  std::string m_world_frame_id;
  std::shared_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;

  ros::Timer m_timer;

  std::string m_image_file_name;
  std::string m_depth_file_name;
  std::string m_camera_info_file_name;
  std::string m_expected_pallet_file_name;
  std::string m_camera_pose_file_name;
  std::string m_initial_guess_file_name;

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
