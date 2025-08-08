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
#include <pcl/features/normal_3d.h>

#include "pallet_detection_solver.h"
#include "pallet_ransac.h"
#include "pallet_from_image.h"
#include "pallet_detection.h"
#include "boxes_to_pallet_description.h"

typedef pcl::PointCloud<pcl::PointXYZRGB> PointXYZRGBCloud;
typedef PointXYZRGBCloud::Ptr PointXYZRGBCloudPtr;
typedef PalletDetection::IntVector IntVector;
typedef PalletDetection::IntVectorPtr IntVectorPtr;
typedef std::vector<float> FloatVector;
typedef std::vector<double> DoubleVector;
typedef std::set<int> IntSet;
typedef std::pair<Eigen::Vector3f, Eigen::Vector3f> Vector3fPair;
typedef std::vector<Vector3fPair> Vector3fPairVector;

typedef uint32_t uint32;
typedef int32_t int32;
typedef int64_t int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef std::vector<uint64> Uint64Vector;
typedef std::pair<uint64, uint64> Uint64Pair;
typedef std::vector<Uint64Pair> Uint64PairVector;
typedef std::set<Uint64Pair> Uint64PairSet;

template <typename T> static T SQR(const T & t) {return t * t; }

template <typename ET> inline
  std::string e_to_string(const ET & et)
{
  std::ostringstream ostr;
  ostr << et;
  return ostr.str();
}

PalletDetection::PalletDetection(const Config & config)
{
  m_config = config;

  m_depth_hough_threshold = config.depth_hough_threshold;
  m_depth_hough_min_length = config.depth_hough_min_length;
  m_depth_hough_max_gap = config.depth_hough_max_gap;

  m_min_plane_camera_distance = config.min_plane_camera_distance;
  m_vertical_line_angle_tolerance = config.vertical_line_angle_tolerance;

  m_ransac_plane_angle_tolerance =  config.ransac_plane_angle_tolerance;
  m_ransac_plane_distance_tolerance =  config.ransac_plane_distance_tolerance; // used to fit plane
  m_ransac_plane_inliers_tolerance =  config.ransac_plane_inliers_tolerance;   // used to clear inliers around plane

  m_plane_camera_max_angle = config.plane_camera_max_angle;

  m_plane_edge_discontinuity_dist_th = config.plane_edge_discontinuity_dist_th;
  m_plane_edge_discontinuity_angle_th = config.plane_edge_discontinuity_angle_th;

  m_depth_image_max_discontinuity_th = config.depth_image_max_discontinuity_th;
  m_depth_image_max_vertical_angle = config.depth_image_max_vertical_angle;
  m_depth_image_normal_window = config.depth_image_normal_window;
  m_depth_image_closing_window = config.depth_image_closing_window;

  m_min_cluster_points_at_1m = config.min_cluster_points_at_1m;
  m_min_cluster_points = config.min_cluster_points;

  m_pillars_merge_threshold = config.pillars_merge_threshold;

  m_planes_similarity_max_angle = config.planes_similarity_max_angle;
  m_planes_similarity_max_distance = config.planes_similarity_max_distance;
  m_points_similarity_max_distance = config.points_similarity_max_distance;

  m_max_pose_correction_distance = config.max_pose_correction_distance;
  m_max_pose_correction_angle = config.max_pose_correction_angle;

  m_plane_ransac_iterations = config.plane_ransac_iterations;
  m_plane_ransac_max_error = config.plane_ransac_max_error;

  m_random_seed = config.random_seed;
}

PalletDetection::PointXYZRGBCloud PalletDetection::ImageToCloud(const cv::Mat & rgb_image, const cv::Mat & depth_image,
                                                                const CameraInfo & camera_info_msg) const
{
  const size_t width = rgb_image.cols;
  const size_t height = rgb_image.rows;

  double fx = camera_info_msg.fx;
  double fy = camera_info_msg.fy;
  double cx = camera_info_msg.cx;
  double cy = camera_info_msg.cy;

  PointXYZRGBCloud cloud;
  cloud.points.resize(height * width);
  cloud.width = width;
  cloud.height = height;
  cloud.is_dense = false;

  for (size_t y = 0; y < height; y++)
    for (size_t x = 0; x < width; x++)
    {
      pcl::PointXYZRGB & pt = cloud[x + y * width];
      pt.z = depth_image.at<float>(y, x);
      pt.x = (x - cx + 0.5f) * pt.z / fx;
      pt.y = (y - cy + 0.5f) * pt.z / fy;

      //pt.a = 1.0;
      pt.r = rgb_image.at<cv::Vec3b>(y, x)[2];
      pt.g = rgb_image.at<cv::Vec3b>(y, x)[1];
      pt.b = rgb_image.at<cv::Vec3b>(y, x)[0];
    }

  return cloud;
}

PalletDetection::ExpectedPallet PalletDetection::LoadExpectedPallet(std::istream & istr) const
{
  ExpectedPallet result;

  std::string line;
  while (std::getline(istr, line))
  {
    if (line.empty())
      continue; // skip empty lines

    if (line[0] == '#')
      continue; // skip comments

    ExpectedElement new_element;

    std::istringstream iss(line);
    std::string type;
    iss >> type;
    if (type == "pillar")
    {
      new_element.type = ExpectedElementType::PILLAR;
      iss >> new_element.pillar.x() >> new_element.pillar.y() >> new_element.pillar.z() >> new_element.pillar.w();

      if (!iss)
      {
        m_log(3, "Unable to parse line: " + line);
        continue;
      }

      std::string cmd;
      while (iss >> cmd)
      {
        if (cmd == "left" || cmd == "right")
        {
          std::string plane_name;
          if (iss >> plane_name)
          {
            if (cmd == "left")
              new_element.pillar_left_plane_name = plane_name;
            else
              new_element.pillar_right_plane_name = plane_name;
          }
          else
          {
            m_log(3, "Expected plane id after '" + cmd + "' at line '" + line + "'");
          }
        }
        else if (cmd == "name")
        {
          std::string name;
          if (iss >> name)
            new_element.name = name;
          else
            m_log(3, "Expected name after '" + cmd + "' at line '" + line + "'");
        }
        else
        {
          m_log(3, "Invalid subcommand '" + cmd + "' at line '" + line + "'");
        }
      }
    }
    else if (type == "plane")
    {
      new_element.type = ExpectedElementType::PLANE;
      iss >> new_element.plane.x() >> new_element.plane.y() >> new_element.plane.z() >> new_element.plane.w();
      iss >> new_element.plane_z.x() >> new_element.plane_z.y();

      const Eigen::Vector4d plane = new_element.plane;
      new_element.plane_point = -plane.head<3>() * plane.w(); // get any point on plane
      new_element.plane_point.z() = (new_element.plane_z.x() + new_element.plane_z.y()) / 2.0; // override z with input z

      if (!iss)
      {
        m_log(3, "Unable to parse line: " + line);
        continue;
      }

      std::string cmd;
      while (iss >> cmd)
      {
        if (cmd == "name")
        {
          std::string name;
          if (iss >> name)
            new_element.name = name;
          else
            m_log(3, "Expected name after '" + cmd + "' at line '" + line + "'");
        }
        else
        {
          m_log(3, "Invalid subcommand '" + cmd + "' at line '" + line + "'");
        }
      }
    }
    else if (type == "box")
    {
      new_element.type = ExpectedElementType::BOX;
      iss >> new_element.box_size.x() >> new_element.box_size.y() >> new_element.box_size.z();
      iss >> new_element.box.x() >> new_element.box.y() >> new_element.box.z() >> new_element.box.w();

      if (!iss)
      {
        m_log(3, "Unable to parse line: " + line);
        continue;
      }

      std::string cmd;
      while (iss >> cmd)
      {
        if (cmd == "name")
        {
          std::string name;
          if (iss >> name)
            new_element.name = name;
          else
            m_log(3, "Expected name after '" + cmd + "' at line '" + line + "'");
        }
        else
        {
          m_log(3, "Invalid subcommand '" + cmd + "' at line '" + line + "'");
        }
      }
    }
    else if (type == "guess")
    {
      // do nothing
      continue;
    }
    else
    {
      m_log(3, "LoadExpectedPallet: Unknown type: " + type);
      continue;
    }

    result.push_back(new_element);
  }

  const std::string upd_plane_ids = pallet_detection::UpdatePlaneIds(result);
  if (upd_plane_ids != "")
    m_log(3, "Error while matching plane ids in expected pallet: " + upd_plane_ids);

  return result;
}

PointXYZRGBCloudPtr PalletDetection::TransformPointCloudZUp(const PointXYZRGBCloud & cloud,
                                                            const Eigen::Affine3f & camera_pose,
                                                            const float floor_height,
                                                            const BoundingBox & bounding_box,
                                                            IntVector & valid_indices) const
{
  PointXYZRGBCloudPtr z_up_cloud(new PointXYZRGBCloud(cloud));

  for (size_t i = 0; i < cloud.size(); i++)
  {
    const pcl::PointXYZRGB & pt = cloud[i];

    const float nan = std::numeric_limits<float>::quiet_NaN();

    Eigen::Vector3f ept(pt.x, pt.y, pt.z);
    ept = camera_pose * ept;
    pcl::PointXYZRGB npt = pt;
    npt.x = ept.x();
    npt.y = ept.y();
    npt.z = ept.z();

    if (pt.z == 0.0)
      npt.x = npt.y = npt.z = nan;

    (*z_up_cloud)[i] = npt;

    // transform in bbox coordinates
    ept = Eigen::AngleAxisf(-bounding_box.rotation, Eigen::Vector3f(0.0f, 0.0f, 1.0f)) * (ept - bounding_box.center);
    if ((ept.array().abs() > (bounding_box.size.array() / 2.0)).any())
      continue; // out of bounding box
    if (pt.z == 0.0)
      continue;
    if (npt.z < floor_height)
      continue;

    valid_indices.push_back(i);
  }

  return z_up_cloud;
}

cv::Mat PalletDetection::FilterDepthImage(const cv::Mat & depth_image, const float threshold) const
{
  const int64 width = depth_image.cols;
  const int64 height = depth_image.rows;

  cv::Mat result = depth_image.clone();

  const int64 WINDOW = 1;
  for (int64 y = 0; y < height; y++)
    for (int64 x = 0; x < width; x++)
    {
      bool any_similar = false;
      const float v = depth_image.at<float>(y, x);
      for (int64 dy = -WINDOW; dy <= WINDOW; dy++)
        for (int64 dx = -WINDOW; dx <= WINDOW; dx++)
        {
          if (!dx && !dy)
            continue; // do not check self
          if (SQR(dx) + SQR(dy) > WINDOW)
            continue; // circular neighborhood
          const int64 nx = x + dx;
          const int64 ny = y + dy;
          if (nx < 0 || ny < 0)
            continue;
          if (nx >= width || ny >= height)
            continue;
          const float nv = depth_image.at<float>(ny, nx);
          if (std::abs(v - nv) < threshold)
            any_similar = true;
        }
      if (!any_similar)
        result.at<float>(y, x) = 0.0f;
    }

  return result;
}

cv::Mat PalletDetection::DepthToFloat(const cv::Mat & depth_image) const
{
  if (depth_image.type() == CV_32FC1)
    return depth_image;

  cv::Mat image_out;
  depth_image.convertTo(image_out, CV_32FC1, 1.0f / 1000.0f);
  return image_out;
}

IntVectorPtr PalletDetection::FilterCloudByVerticalAngle(const PointXYZRGBCloud & cloud, const IntVector & valid_indices) const
{
  m_log(1, "Filter Cloud By Vertical Angle start, with " + std::to_string(valid_indices.size()) + " indices.");
  IntVectorPtr result(new IntVector);

  const int WINDOW = m_depth_image_normal_window;

  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;

  BoolVector is_in_input(cloud.size(), false);
  for (uint64 i = 0; i < valid_indices.size(); i++)
  {
    const int idx = valid_indices[i];
    is_in_input[idx] = true;
  }

  BoolVector is_in_result(cloud.size(), false);

  for (uint64 i = 0; i < valid_indices.size(); i++)
  {
    const int idx = valid_indices[i];
    const int x = int(idx) % cloud.width;
    const int y = int(idx) / cloud.width;

    const pcl::PointXYZRGB & ipi = cloud[idx];
    const Eigen::Vector3f eipi(ipi.x, ipi.y, ipi.z);

    if (eipi.array().isNaN().any())
      continue;

    pcl::Indices indices;
    indices.reserve(WINDOW * WINDOW);
    for (int dy = -WINDOW; dy <= WINDOW; dy++)
      for (int dx = -WINDOW; dx <= WINDOW; dx++)
      {
        if (SQR(dx) + SQR(dy) > SQR(WINDOW))
          continue;

        const int nx = x + dx;
        const int ny = y + dy;

        if (nx < 0 || nx >= int(cloud.width))
          continue;
        if (ny < 0 || ny >= int(cloud.height))
          continue;

        const uint64 ni = nx + ny * cloud.width;

        const pcl::PointXYZRGB & npi = cloud[ni];
        const Eigen::Vector3f enpi(npi.x, npi.y, npi.z);
        if (enpi.array().isNaN().any())
          continue;

        indices.push_back(ni);
      }

    if (int(indices.size()) < WINDOW)
    {
      result->push_back(idx);
      continue;
    }

    float nx, ny, nz, c;
    ne.computePointNormal(cloud, indices, nx, ny, nz, c);
    const Eigen::Vector3f enormal(nx, ny, nz);
    if (enormal.array().isNaN().any())
      continue;

    const Eigen::Vector3f vertical_axis = Eigen::Vector3f::UnitZ();
    const float angle = std::acos(std::abs(enormal.normalized().dot(vertical_axis)));

    if (angle > m_depth_image_max_vertical_angle)
    {
      result->push_back(idx);
      is_in_result[idx] = true;
    }
  }

  IntVectorPtr result_dilated(new IntVector);

  const int WINDOW2 = m_depth_image_closing_window;

  for (uint64 i = 0; i < valid_indices.size(); i++)
  {
    const int idx = valid_indices[i];
    if (is_in_result[idx]) // already in result
    {
      result_dilated->push_back(idx);
      continue;
    }

    if (!is_in_input[idx])
      continue; // not in input

    {
      const pcl::PointXYZRGB & pi = cloud[idx];
      const Eigen::Vector3f epi(pi.x, pi.y, pi.z);
      if (epi.array().isNaN().any())
        continue; // NAN
    }

    const int x = int(idx) % cloud.width;
    const int y = int(idx) / cloud.width;

    bool found_valid = false;

    for (int dy = -WINDOW2; dy <= WINDOW2; dy++)
      for (int dx = -WINDOW2; dx <= WINDOW2; dx++)
      {
        if (SQR(dx) + SQR(dy) > SQR(WINDOW2))
          continue;

        const int nx = x + dx;
        const int ny = y + dy;

        if (nx < 0 || nx >= int(cloud.width))
          continue;
        if (ny < 0 || ny >= int(cloud.height))
          continue;

        const uint64 ni = nx + ny * cloud.width;

        if (is_in_result[ni])
          found_valid = true;
      }

    if (found_valid)
      result_dilated->push_back(idx);
  }

  m_log(1, "Filter Cloud By Vertical Angle end, with " + std::to_string(result_dilated->size()) + " indices.");

  return result_dilated;
}

PalletDetection::DetectionResult PalletDetection::Detect(const cv::Mat & rgb_image,        // CV_8UC3, bgr
                                                         const cv::Mat & depth_image_in,   // CV_16UC1, millimeters
                                                         const CameraInfo & camera_info,
                                                         const Eigen::Affine3f & camera_pose,
                                                         const float floor_height,
                                                         const BoundingBox & initial_guess,
                                                         const std::string & pallet_description_filename) const
{
  DetectionResult result;
  result.success = false;

  srand(m_random_seed);

  cv::Mat depth_image = depth_image_in;
  depth_image = DepthToFloat(depth_image);
  depth_image = FilterDepthImage(depth_image, m_depth_image_max_discontinuity_th);
  m_publish_image(depth_image, "32FC1", "depth_image");

  const uint64 width = rgb_image.cols;
  const uint64 height = rgb_image.rows;

  m_log(1, "image_to_cloud");
  const PointXYZRGBCloud cloud = ImageToCloud(rgb_image, depth_image, camera_info);

  m_log(1, "publish_cloud");
  m_publish_cloud(cloud, "input_cloud");

  BoundingBox world_bounding_box = initial_guess;

  IntVectorPtr valid_indices_ptr(new IntVector);
  const PointXYZRGBCloudPtr z_up_cloud = TransformPointCloudZUp(cloud, camera_pose,
                                                                floor_height,
                                                                world_bounding_box,
                                                                *valid_indices_ptr);
  valid_indices_ptr = FilterCloudByVerticalAngle(*z_up_cloud, *valid_indices_ptr);

  {
    PointXYZRGBCloud debug_cloud;
    for (const int i : *valid_indices_ptr)
      debug_cloud.push_back((*z_up_cloud)[i]);
    debug_cloud.height = 1;
    debug_cloud.width = debug_cloud.size();
    m_publish_cloud(debug_cloud, "valid_points_cloud");
  }

  PalletFromImage pfi(m_log, m_min_cluster_points, m_min_cluster_points_at_1m,
                      m_plane_camera_max_angle, m_min_plane_camera_distance, m_depth_hough_threshold,
                      m_depth_hough_min_length, m_depth_hough_max_gap, m_vertical_line_angle_tolerance,
                      m_pillars_merge_threshold, m_ransac_plane_angle_tolerance,
                      m_ransac_plane_distance_tolerance, m_ransac_plane_inliers_tolerance,
                      m_plane_edge_discontinuity_angle_th, m_plane_edge_discontinuity_dist_th,
                      m_config.th_scan_distance_window,
                      m_config.th_scan_counter_threshold,
                      m_config.th_scan_threshold_enter,
                      m_config.th_scan_threshold_exit,
                      m_config.correlation_templates,
                      m_config.correlation_multiresolution_count,
                      m_config.correlation_multiresolution_step,
                      m_config.correlation_rescale,
                      m_config.correlation_threshold);
  ExpectedPallet real_pallet;
  pfi.Run(rgb_image, depth_image, z_up_cloud, valid_indices_ptr, camera_pose, camera_info, real_pallet);

  // Expected pallet RANSAC
  ExpectedPallet loaded_pallet;
  ExpectedPallet estimated_pallet;
  ExpectedPallet estimated_refined_pallet;

  {
    m_log(1, "Loading expected pallet file " + pallet_description_filename);
    std::ifstream ifile(pallet_description_filename);
    if (!ifile)
    {
      m_log(3, "Could not find expected pallet file.");
      return result;
    }
    Eigen::Vector3d initial_guess(world_bounding_box.center.x(), world_bounding_box.center.y(), world_bounding_box.rotation);

    ExpectedPallet expected_pallet = LoadExpectedPallet(ifile);

    if (m_config.auto_generate_plane_pillars)
    {
      m_log(1, "Generating planes and pillars from boxes.");
      ExpectedPallet expected_pallet_boxes_only;
      for (const ExpectedElement & e : expected_pallet)
        if (e.type == ExpectedElementType::BOX)
          expected_pallet_boxes_only.push_back(e);

      m_log(1, "  There are " + std::to_string(expected_pallet_boxes_only.size()) + " boxes.");

      const Eigen::Vector3f viewpoint = Eigen::Vector3f(m_config.auto_generate_plane_pillars_viewpoint_x,
                                                        m_config.auto_generate_plane_pillars_viewpoint_y, 0.0f);
      BoxesToPalletDescription btpd;
      const ExpectedPallet planes_and_pillars = btpd.Run(expected_pallet_boxes_only, viewpoint);

      m_log(1, "  Generated " + std::to_string(planes_and_pillars.size()) + " planes and pillars.");

      expected_pallet.clear();
      expected_pallet.insert(expected_pallet.end(), planes_and_pillars.begin(), planes_and_pillars.end());
      expected_pallet.insert(expected_pallet.end(), expected_pallet_boxes_only.begin(), expected_pallet_boxes_only.end());
      const std::string res = pallet_detection::UpdatePlaneIds(expected_pallet);
      if (res != "")
        m_log(3, "  Plane IDs consistency check error: " + res);
    }

    PalletRansac pallet_ransac(m_log,
                               m_plane_ransac_max_error,
                               m_plane_ransac_iterations,
                               m_max_pose_correction_distance,
                               m_max_pose_correction_angle,
                               m_planes_similarity_max_angle,
                               m_planes_similarity_max_distance,
                               m_points_similarity_max_distance,
                               m_random_seed);

    expected_pallet = pallet_ransac.TransformPallet(expected_pallet, Eigen::Vector3d::Zero(), floor_height);

    Uint64PairVector consensus;
    Eigen::Vector3d pose;
    Eigen::Vector3d refined_pose;
    pallet_ransac.Run(expected_pallet,
                      real_pallet,
                      initial_guess,
                      pose,
                      refined_pose,
                      consensus);

    m_log(1, "Best consensus is " + std::to_string(consensus.size()));
    m_log(1, "Best pose is " + e_to_string(pose.transpose()));
    m_log(1, "Best refined pose is " + e_to_string(refined_pose.transpose()));

    {
      for (const Uint64Pair & corresp : consensus)
      {
        const uint64 exp_elem_i = corresp.second;
        const ExpectedElementType type = expected_pallet[exp_elem_i].type;

        if (type == ExpectedElementType::PILLAR)
        {
          m_log(1, "pillar consensus");
        }

        if (type == ExpectedElementType::PLANE)
        {
          m_log(1, "plane consensus");
        }
      }
    }

    result.pose = refined_pose;
    result.success = consensus.size() >= 2; // need at least two elements to estimate pose
    result.consensus = consensus.size();

    loaded_pallet = pallet_ransac.TransformPallet(expected_pallet, initial_guess);
    estimated_refined_pallet = pallet_ransac.TransformPallet(expected_pallet, refined_pose);
    estimated_pallet = pallet_ransac.TransformPallet(expected_pallet, pose);

    for (const Uint64Pair & corresp : consensus)
    {
      const Eigen::Vector3d plane_point = real_pallet[corresp.first].plane_point;
      estimated_refined_pallet[corresp.second].plane_point = plane_point;
      estimated_pallet[corresp.second].plane_point = plane_point;
    }
  }

  for (ExpectedElement e : estimated_refined_pallet)
  {
    if (e.type != ExpectedElementType::BOX)
      continue;

    const Eigen::Vector4d box = e.box;
    Eigen::Affine3d pose;
    pose.translation() = box.head<3>();
    pose.linear() = Eigen::AngleAxisd(box.w(), Eigen::Vector3d::UnitZ()).matrix();
    result.boxes.push_back(pose);
  }

  // pallet visualization
  m_publish_pallet(real_pallet, loaded_pallet, estimated_pallet, estimated_refined_pallet);

  // visualization
  {
    const cv::Mat debug_edge_image = pfi.GetLastEdgeImage();
    const cv::Mat cluster_image = pfi.GetLastClusterImage();
    const cv::Mat correlation_image = pfi.GetLastCorrelationImage();
    const cv::Mat bool_correlation_image = pfi.GetLastBoolCorrelationImage().clone();


    if (!debug_edge_image.empty())
      m_publish_image(debug_edge_image, "rgb8", "edge_image");
    if (!correlation_image.empty())
      m_publish_image(correlation_image, "mono8", "correlation_image");
    if (!bool_correlation_image.empty())
    {
      cv::Mat m = bool_correlation_image * 255;
      m_publish_image(m, "mono8", "bool_correlation_image");
    }

    // renumbering
    std::vector<uint64> pixel_counter;
    for (size_t y = 0; y < height; y++)
      for (size_t x = 0; x < width; x++)
      {
        const int32_t v = cluster_image.at<int32_t>(y, x);
        if (pixel_counter.size() <= uint64(v))
          pixel_counter.resize(v + 1, 0);
        pixel_counter[v]++;
      }

    std::vector<uint64> assoc(pixel_counter.size(), 0);
    size_t counter = 0;
    for (size_t i = 0; i < pixel_counter.size(); i++)
    {
      assoc[i] = counter;
      if (pixel_counter[i])
        counter++;
    }

    cv::Mat plane_image = cluster_image.clone();
    for (size_t y = 0; y < height; y++)
      for (size_t x = 0; x < width; x++)
      {
        const int32_t v = cluster_image.at<int32_t>(y, x);
        plane_image.at<int32_t>(y, x) = assoc[v];
      }

    cv::Mat plane_color_image = cv::Mat(height, width, CV_8UC3);
    for (size_t y = 0; y < height; y++)
      for (size_t x = 0; x < width; x++)
      {
        const int32_t v = plane_image.at<int32_t>(y, x);
        if (v == 0)
          plane_color_image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
        else
        {
          const pcl::RGB color = pcl::GlasbeyLUT::at(v % 256);
          plane_color_image.at<cv::Vec3b>(y, x) = cv::Vec3b(color.r, color.g, color.b);
        }
      }

    m_publish_image(plane_color_image, "rgb8", "plane_image");

    PointXYZRGBCloudPtr cloud_with_inliers(new PointXYZRGBCloud(*z_up_cloud));
    cv::Mat cluster_color_image = cv::Mat(height, width, CV_8UC3);
    for (size_t y = 0; y < height; y++)
      for (size_t x = 0; x < width; x++)
      {
        const int32_t v = cluster_image.at<int32_t>(y, x);
        if (v == 0)
          cluster_color_image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
        else
        {
          const pcl::RGB color = pcl::GlasbeyLUT::at(v % 256);
          cluster_color_image.at<cv::Vec3b>(y, x) = cv::Vec3b(color.r, color.g, color.b);
          pcl::PointXYZRGB & pt = (*cloud_with_inliers)[y * width + x];
          pt.r = color.r;
          pt.g = color.g;
          pt.b = color.b;
        }
      }

    m_publish_image(cluster_color_image, "rgb8", "cluster_image");

    m_log(1, "Publishing cloud.");

    m_publish_cloud(*cloud_with_inliers, "cloud");
  }

  return result;
}
