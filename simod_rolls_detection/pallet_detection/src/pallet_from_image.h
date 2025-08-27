#ifndef PALLET_FROM_IMAGE_H
#define PALLET_FROM_IMAGE_H

#include "pallet_ransac.h"

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <vector>
#include <set>
#include <map>
#include <stdint.h>
#include <functional>

#include <opencv2/opencv.hpp>

// PCL
#include <pcl/common/colors.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "expected_pallet.h"

class PalletFromImage
{
  public:

  using ExpectedElement = pallet_detection::ExpectedElement;
  using ExpectedPallet = pallet_detection::ExpectedPallet;
  using ExpectedElementVector = pallet_detection::ExpectedElementVector;
  using ExpectedElementType = pallet_detection::ExpectedElementType;

  enum class PillarType
  {
    NONE   = 0b00,
    LEFT   = 0b01,
    RIGHT  = 0b10,
    CENTER = 0b11,
  };
  typedef std::vector<PillarType> PillarTypeVector;
  typedef std::vector<PillarTypeVector> PillarTypeVectorVector;

  typedef pcl::PointCloud<pcl::PointXYZRGB> PointXYZRGBCloud;
  typedef PointXYZRGBCloud::Ptr PointXYZRGBCloudPtr;
  typedef std::vector<int> IntVector;
#if PCL_VERSION >= 101200
  typedef std::shared_ptr<IntVector> IntVectorPtr;
#else
  typedef boost::shared_ptr<IntVector> IntVectorPtr;
#endif
  
  typedef std::vector<float> FloatVector;
  typedef std::vector<double> DoubleVector;
  typedef std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > Vector4fVector;
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;
  typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > Vector2fVector;
  typedef std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > Vector4dVector;
  typedef std::pair<Eigen::Vector3f, Eigen::Vector3f> Vector3fPair;
  typedef std::vector<Vector3fPair> Vector3fPairVector;

  struct CorrTemplate
  {
    cv::Mat image;
    float depth;
  };

  typedef std::vector<CorrTemplate> CorrTemplateVector;

  typedef int32_t int32;
  typedef uint64_t uint64;
  typedef int64_t int64;
  typedef uint8_t uint8;

  typedef std::vector<uint64> Uint64Vector;
  typedef std::vector<IntVector> IntVectorVector;
  typedef std::set<uint64> Uint64Set;
  typedef std::vector<std::vector<uint64>> Uint64VectorVector;

  typedef std::function<void(const uint level, const std::string & s)> LogFunction;

  struct CameraInfo
  {
    float fx, fy;
    float cx, cy;
  };

  PalletFromImage(const LogFunction & log,
                  const uint64 min_cluster_points,
                  const uint64 min_cluster_points_at_1m,
                  const float plane_camera_max_angle,
                  const float min_plane_camera_distance,
                  const float depth_hough_threshold,
                  const float depth_hough_min_length,
                  const float depth_hough_max_gap,
                  const float vertical_line_angle_tolerance,
                  const float pillars_merge_threshold,
                  const float ransac_plane_angle_tolerance,
                  const float ransac_plane_distance_tolerance,
                  const float ransac_plane_inliers_tolerance,
                  const float plane_edge_discontinuity_angle_th,
                  const float plane_edge_discontinuity_dist_th,
                  const float th_scan_distance_window,
                  const uint64 th_scan_counter_threshold,
                  const float th_scan_threshold_enter,
                  const float th_scan_threshold_exit,
                  const CorrTemplateVector correlation_templates,
                  const float correlation_multiresolution_count,
                  const float correlation_multiresolution_step,
                  const float correlation_rescale,
                  const float correlation_threshold);

  Eigen::Vector4f RansacFitPlane(const PointXYZRGBCloudPtr cloud, const IntVectorPtr remaining_indices, IntVector & inliers);
  Eigen::Vector4f InliersFitPlane(const PointXYZRGBCloudPtr cloud, const Eigen::Vector4f &prev_coeff, const IntVectorPtr inliers);
  Eigen::Vector4f EnsureVerticalPlane(const Eigen::Vector4f & coeffs, const Eigen::Affine3f & camera_pose,
                                      const Eigen::Vector3f &center);
  IntVectorPtr FindPlaneInliers(const PointXYZRGBCloudPtr cloud, const IntVectorPtr remaining_indices,
                                const Eigen::Vector4f & plane_coefficients);
  FloatVector FindPlaneDistance(const PointXYZRGBCloudPtr cloud, const IntVectorPtr remaining_indices,
                                const Eigen::Vector4f & plane_coefficients);
  IntVectorPtr IntVectorSetDifference(const IntVector & a, const IntVector & b);
  cv::Mat CloseImage(const cv::Mat & image, const size_t closing_size) const;
  cv::Mat DilateImage(const cv::Mat & image, const size_t dilate_size) const;
  template <typename T>
  cv::Mat SimpleEdgeImage(const cv::Mat & image);

  // int32 image
  // float depth_image
  cv::Mat PlaneEdgeImage(const cv::Mat & image, const cv::Mat & depth_image,
                         const Vector4fVector & found_planes, const int32 relevant_plane) const;
  IntVectorPtr FilterInliersBySmallClusters(const IntVector & inliers,
                                            const size_t width, const size_t height,
                                            const uint64 min_cluster_size);
  // filter similar pillars based on resolution
  Vector4dVector GridFilterPillars(const Vector4dVector pillars, const double resolution,
                                   const Uint64Vector pillar_parent_plane,
                                   const PillarTypeVector pillar_type,
                                   Uint64VectorVector & pillar_parent_planes,
                                   PillarTypeVectorVector & pillar_types);

  cv::Mat DoCorrelation(const cv::Mat & rgb_image, const cv::Mat & depth_image);

  void FindPillarsEdgeImage(const Vector4fVector & found_planes,
                            const Uint64Vector & found_plane_indices,
                            const cv::Mat &cluster_image,
                            const cv::Mat & depth_image,
                            const Eigen::Affine3f &camera_pose,
                            const CameraInfo &camera_info,
                            Vector4dVector & found_pillars,
                            Uint64Vector & pillar_parent_plane,
                            PillarTypeVector &pillar_type,
                            cv::Mat &debug_edge_image) const;

  void FindPillarsThresholdScan(const Vector4fVector & found_planes,
                                const Uint64Vector &found_plane_indices,
                                const IntVectorVector &found_plane_inliers,
                                const cv::Mat & depth_image,
                                const PointXYZRGBCloudPtr &z_up_cloud,
                                Vector4dVector & found_pillars,
                                Uint64Vector & pillar_parent_plane) const;

  void Run(const cv::Mat &rgb_image, const cv::Mat &depth_image,
           const PointXYZRGBCloudPtr &z_up_cloud, const IntVectorPtr remaining_indices_ptr,
           const Eigen::Affine3f &camera_pose, const CameraInfo & camera_info_msg,
           ExpectedPallet & pallet);

  const cv::Mat & GetLastEdgeImage() const {return m_last_edge_image; }
  const cv::Mat & GetLastClusterImage() const {return m_last_cluster_image; }
  const cv::Mat & GetLastCorrelationImage() const {return m_last_correlation_image; }
  const cv::Mat & GetLastBoolCorrelationImage() const {return m_last_bool_correlation_image; }

  template <typename T>
  cv::Mat FillHoles(const cv::Mat & image, const T & hole_value, const uint64 iterations = 1) const;

  cv::Mat ResizeImage(const cv::Mat & image, const float fraction, const int interpolation) const;
  cv::Mat ResizeImage(const cv::Mat & image, const int target_x, const int target_y, const int interpolation) const;

  private:
  uint64 m_min_cluster_points;
  uint64 m_min_cluster_points_at_1m;
  float m_plane_camera_max_angle;
  float m_min_plane_camera_distance;

  float m_depth_hough_threshold;
  float m_depth_hough_min_length;
  float m_depth_hough_max_gap;

  float m_vertical_line_angle_tolerance;
  float m_pillars_merge_threshold;

  float m_ransac_plane_angle_tolerance;
  float m_ransac_plane_distance_tolerance;
  float m_ransac_plane_inliers_tolerance;

  float m_plane_edge_discontinuity_angle_th;
  float m_plane_edge_discontinuity_dist_th;

  float m_th_scan_distance_window;
  uint64 m_th_scan_counter_threshold;
  float m_th_scan_threshold_enter;
  float m_th_scan_threshold_exit;

  CorrTemplateVector m_correlation_templates;
  float m_correlation_rescale;
  float m_correlation_threshold;
  float m_correlation_multiresolution_step;
  uint64 m_correlation_multiresolution_count;

  cv::Mat m_last_edge_image;
  cv::Mat m_last_cluster_image;
  cv::Mat m_last_correlation_image;
  cv::Mat m_last_bool_correlation_image;

  LogFunction m_log = [](const uint, const std::string &){};
};

#endif // PALLET_FROM_IMAGE_H
