#ifndef PALLET_DETECTION_H
#define PALLET_DETECTION_H

// Eigen
#include <Eigen/Dense>

// STL
#include <string>
#include <stdint.h>
#include <memory>

// OpenCV
#include <opencv2/opencv.hpp>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "pallet_ransac.h"
#include "pallet_from_image.h"

class PalletDetection
{
  public:
  typedef PalletRansac::ExpectedElementType ExpectedElementType;
  typedef PalletRansac::ExpectedPallet ExpectedPallet;
  typedef PalletRansac::ExpectedElement ExpectedElement;

  typedef std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > Vector4fVector;
  typedef std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > Vector4dVector;
  typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > Vector3dVector;
  typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > Vector2dVector;
  typedef std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d> > Affine3dVector;

  typedef pcl::PointCloud<pcl::PointXYZRGB> PointXYZRGBCloud;
  typedef PointXYZRGBCloud::Ptr PointXYZRGBCloudPtr;

  typedef unsigned int uint;
  typedef uint64_t uint64;

  typedef std::vector<bool> BoolVector;
  typedef std::vector<int> IntVector;
#if PCL_VERSION >= 101200
  typedef std::shared_ptr<IntVector> IntVectorPtr;
#else
  typedef boost::shared_ptr<IntVector> IntVectorPtr;
#endif

  typedef std::function<void(const uint level, const std::string & s)> LogFunction;
  typedef std::function<void(const cv::Mat & image, const std::string & encoding, const std::string & name)> PublishImageFunction;
  typedef std::function<void(const PointXYZRGBCloud & cloud, const std::string & name)> PublishCloudFunction;
  typedef std::function<void(const ExpectedPallet & real_pallet,
                             const ExpectedPallet & loaded_pallet,
                             const ExpectedPallet & estimated_pallet,
                             const ExpectedPallet & estimated_refined_pallet)> PublishPalletFunction;

  typedef PalletFromImage::CameraInfo CameraInfo;

  typedef std::shared_ptr<PalletDetection> Ptr;

  struct Config
  {
    int depth_hough_threshold = 200;
    int depth_hough_min_length = 100;
    int depth_hough_max_gap = 50;

    double min_plane_camera_distance = 0.5;
    double vertical_line_angle_tolerance = M_PI / 10.0f;

    double ransac_plane_angle_tolerance = 5.0 / 180.0 * M_PI;
    double ransac_plane_distance_tolerance = 0.025;
    double ransac_plane_inliers_tolerance = 0.05;

    double plane_camera_max_angle = 60.0 / 180.0 * M_PI;

    float plane_edge_discontinuity_dist_th = 0.05f;
    float plane_edge_discontinuity_angle_th = 20.0f * M_PI / 180.0f;

    double depth_image_max_discontinuity_th = 0.1;
    double depth_image_max_vertical_angle = 20.0f * M_PI / 180.0f;
    int depth_image_normal_window = 2;
    int depth_image_closing_window = 10;

    double min_cluster_points_at_1m = 100000;
    int min_cluster_points = 30000;

    double pillars_merge_threshold = 0.05;

    double planes_similarity_max_angle = 10.0f / 180.0f * M_PI;
    double planes_similarity_max_distance = 0.1f;
    double points_similarity_max_distance = 0.1f;

    double max_pose_correction_distance = 2.0f;
    double max_pose_correction_angle = M_PI / 2.0f;

    int plane_ransac_iterations = 2000;
    double plane_ransac_max_error = 0.1;

    int random_seed = std::random_device()();
  };

  struct BoundingBox
  {
    Eigen::Vector3f center = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    Eigen::Vector3f size = Eigen::Vector3f(2.0f, 4.0f, 3.0f);
    float rotation = 0.0f;
  };

  struct DetectionResult
  {
    bool success;
    Eigen::Vector3d pose; // x, y, angle
    Affine3dVector boxes; // list of poses
  };

  PalletDetection(const Config & config);

  DetectionResult Detect(const cv::Mat & rgb_image,     // CV_8UC3, bgr
                         const cv::Mat & depth_image,   // CV_16UC1, millimeters
                         const CameraInfo & camera_info,
                         const Eigen::Affine3f & camera_pose,
                         const float floor_height,
                         const BoundingBox & initial_guess,
                         const std::string & pallet_description_filename) const;

  void SetLogFunction(const LogFunction & fun) {m_log = fun; }
  void SetPublishImageFunction(const PublishImageFunction & fun) {m_publish_image = fun; }
  void SetPublishCloudFunction(const PublishCloudFunction & fun) {m_publish_cloud = fun; }
  void SetPublishPalletFunction(const PublishPalletFunction & fun) {m_publish_pallet = fun; }

  private:
  PointXYZRGBCloud ImageToCloud(const cv::Mat & rgb_image, const cv::Mat & depth_image,
                                const CameraInfo & camera_info_msg) const;

  ExpectedPallet LoadExpectedPallet(std::istream & istr) const;

  PointXYZRGBCloudPtr TransformPointCloudZUp(const PointXYZRGBCloud & cloud,
                                             const Eigen::Affine3f & camera_pose,
                                             const float floor_height,
                                             const BoundingBox & bounding_box,
                                             IntVector & valid_indices) const;

  IntVectorPtr FilterCloudByVerticalAngle(const PointXYZRGBCloud & cloud, const IntVector & valid_indices) const;

  cv::Mat FilterDepthImage(const cv::Mat & depth_image, const float threshold) const;

  cv::Mat DepthToFloat(const cv::Mat & depth_image) const;

  int m_depth_hough_threshold;
  double m_depth_hough_min_length;
  double m_depth_hough_max_gap;
  double m_plane_camera_max_angle;
  double m_vertical_line_angle_tolerance;
  double m_min_plane_camera_distance;

  double m_depth_image_max_discontinuity_th;
  double m_depth_image_max_vertical_angle;
  int m_depth_image_normal_window;
  int m_depth_image_closing_window;

  float m_plane_edge_discontinuity_angle_th;
  float m_plane_edge_discontinuity_dist_th;

  double m_pillars_merge_threshold;

  double m_min_cluster_points_at_1m;
  size_t m_min_cluster_points;

  float m_ransac_plane_distance_tolerance;
  float m_ransac_plane_angle_tolerance;
  float m_ransac_plane_inliers_tolerance;

  double m_planes_similarity_max_angle;
  double m_planes_similarity_max_distance;
  double m_points_similarity_max_distance;
  uint64 m_plane_ransac_iterations;
  double m_plane_ransac_max_error;

  double m_max_pose_correction_distance;
  double m_max_pose_correction_angle;

  uint64 m_random_seed;

  LogFunction m_log = [](const uint, const std::string &){};
  PublishImageFunction m_publish_image = [](const cv::Mat &, const std::string &, const std::string &){};
  PublishCloudFunction m_publish_cloud = [](const PointXYZRGBCloud &, const std::string &){};
  PublishPalletFunction m_publish_pallet = [](const ExpectedPallet &, const ExpectedPallet &, const ExpectedPallet &, const ExpectedPallet &){};
};

#endif // PALLET_DETECTION_H
