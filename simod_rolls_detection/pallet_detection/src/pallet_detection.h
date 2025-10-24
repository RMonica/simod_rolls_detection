#ifndef PALLET_DETECTION_H
#define PALLET_DETECTION_H

// Eigen
#include <Eigen/Dense>

// STL
#include <string>
#include <stdint.h>
#include <memory>
#include <sstream>

// OpenCV
#include <opencv2/opencv.hpp>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "pallet_ransac.h"
#include "pallet_from_image.h"
#include "expected_pallet.h"
#include "boxes_to_pallet_description.h"

class PalletDetection
{
  public:
  using ExpectedElement = pallet_detection::ExpectedElement;
  using ExpectedPallet = pallet_detection::ExpectedPallet;
  using ExpectedElementVector = pallet_detection::ExpectedElementVector;
  using ExpectedElementType = pallet_detection::ExpectedElementType;

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

  typedef PalletFromImage::CorrTemplateVector CorrTemplateVector;
  typedef PalletFromImage::CorrTemplate CorrTemplate;

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

    float th_scan_distance_window = 0.05f;
    uint64 th_scan_counter_threshold = 500;
    float th_scan_threshold_enter = 5000;
    float th_scan_threshold_exit = 2500;

    PalletFromImage::CorrTemplateVector correlation_templates;
    uint64 correlation_multiresolution_count = 5;
    float correlation_multiresolution_step = 0.8f;
    float correlation_rescale = 0.5f;
    float correlation_threshold = 0.05f;

    int random_seed = std::random_device()();

    bool auto_generate_plane_pillars = true;
    double auto_generate_plane_pillars_viewpoint_x = -1.0f;
    double auto_generate_plane_pillars_viewpoint_y = 0.0f;

    std::string ToString() const
    {
      std::ostringstream ostr;
      ostr << "depth_hough_threshold " << depth_hough_threshold << "\n";
      ostr << "depth_hough_min_length " << depth_hough_min_length << "\n";
      ostr << "depth_hough_max_gap " << depth_hough_max_gap << "\n";

      ostr << "min_plane_camera_distance " << min_plane_camera_distance << "\n";
      ostr << "vertical_line_angle_tolerance " << vertical_line_angle_tolerance << "\n";

      ostr << "ransac_plane_angle_tolerance " << ransac_plane_angle_tolerance << "\n";
      ostr << "ransac_plane_distance_tolerance " << ransac_plane_distance_tolerance << "\n";
      ostr << "ransac_plane_inliers_tolerance " << ransac_plane_inliers_tolerance << "\n";

      ostr << "plane_camera_max_angle " << plane_camera_max_angle << "\n";

      ostr << "plane_edge_discontinuity_dist_th " << plane_edge_discontinuity_dist_th << "\n";
      ostr << "plane_edge_discontinuity_angle_th " << plane_edge_discontinuity_angle_th << "\n";

      ostr << "depth_image_max_discontinuity_th " << depth_image_max_discontinuity_th << "\n";
      ostr << "depth_image_max_vertical_angle "<< depth_image_max_vertical_angle << "\n";
      ostr << "depth_image_normal_window " << depth_image_normal_window << "\n";
      ostr << "depth_image_closing_window " << depth_image_closing_window << "\n";

      ostr << "min_cluster_points_at_1m " << min_cluster_points_at_1m << "\n";
      ostr << "min_cluster_points " << min_cluster_points << "\n";

      ostr << "pillars_merge_threshold " << pillars_merge_threshold << "\n";

      ostr << "planes_similarity_max_angle " << planes_similarity_max_angle << "\n";
      ostr << "planes_similarity_max_distance " << planes_similarity_max_distance << "\n";
      ostr << "points_similarity_max_distance " << points_similarity_max_distance << "\n";

      ostr << "max_pose_correction_distance " << max_pose_correction_distance << "\n";
      ostr << "max_pose_correction_angle " << max_pose_correction_angle << "\n";

      ostr << "plane_ransac_iterations " << plane_ransac_iterations << "\n";
      ostr << "plane_ransac_max_error " << plane_ransac_max_error << "\n";

      ostr << "th_scan_distance_window " << th_scan_distance_window << "\n";
      ostr << "th_scan_counter_threshold " << th_scan_counter_threshold << "\n";
      ostr << "th_scan_threshold_enter " << th_scan_threshold_enter << "\n";
      ostr << "th_scan_threshold_exit " << th_scan_threshold_exit << "\n";

      PalletFromImage::CorrTemplateVector correlation_templates;
      ostr << "correlation_multiresolution_count " << correlation_multiresolution_count << "\n";
      ostr << "correlation_multiresolution_step " << correlation_multiresolution_step << "\n";
      ostr << "correlation_rescale " << correlation_rescale << "\n";
      ostr << "correlation_threshold " << correlation_threshold << "\n";

      ostr << "random_seed " << random_seed << "\n";

      ostr << "auto_generate_plane_pillars " << auto_generate_plane_pillars << "\n";

      return ostr.str();
    }
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
    uint64 consensus;
    Eigen::Vector3d pose; // x, y, angle
    Affine3dVector boxes; // list of poses
    ExpectedPallet boxes_only_list;
    std::vector<Eigen::Vector4f> cam_positions;
    std::vector<ColoredSegment> cam_segments;
    
    float time_prefilter = 0.0f;
    float time_pallet_from_image = 0.0f;
    float time_load_expected_pallet = 0.0f;
    float time_ransac = 0.0f;
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

  Config m_config;

  LogFunction m_log = [](const uint, const std::string &){};
  PublishImageFunction m_publish_image = [](const cv::Mat &, const std::string &, const std::string &){};
  PublishCloudFunction m_publish_cloud = [](const PointXYZRGBCloud &, const std::string &){};
  PublishPalletFunction m_publish_pallet = [](const ExpectedPallet &, const ExpectedPallet &, const ExpectedPallet &, const ExpectedPallet &){};
};

#endif // PALLET_DETECTION_H
