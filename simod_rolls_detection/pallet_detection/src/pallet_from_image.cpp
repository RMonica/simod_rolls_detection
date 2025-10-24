#include "pallet_from_image.h"

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_parallel_plane.h>
#include <pcl/common/colors.h>
#include <pcl/search/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>

#include <sstream>
#include <fstream>
#include <deque>

template <typename T>
T SQR(const T & t) {return t * t; }

typedef pcl::SampleConsensusModelParallelPlane<pcl::PointXYZRGB> RANSAC_PP_Model;
typedef pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> RANSAC_P_Model;

template <typename ET> inline
std::string e_to_string(const ET & et)
{
  std::ostringstream ostr;
  ostr << et;
  return ostr.str();
}

PalletFromImage::PalletFromImage(const LogFunction & log,
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
                                 const float correlation_threshold)
{
  m_min_cluster_points = min_cluster_points;
  m_min_cluster_points_at_1m = min_cluster_points_at_1m;
  m_plane_camera_max_angle = plane_camera_max_angle;
  m_min_plane_camera_distance = min_plane_camera_distance;

  m_depth_hough_threshold = depth_hough_threshold;
  m_depth_hough_min_length = depth_hough_min_length;
  m_depth_hough_max_gap = depth_hough_max_gap;

  m_vertical_line_angle_tolerance = vertical_line_angle_tolerance;
  m_pillars_merge_threshold = pillars_merge_threshold;

  m_ransac_plane_angle_tolerance = ransac_plane_angle_tolerance;
  m_ransac_plane_distance_tolerance = ransac_plane_distance_tolerance;
  m_ransac_plane_inliers_tolerance = ransac_plane_inliers_tolerance;

  m_plane_edge_discontinuity_angle_th = plane_edge_discontinuity_angle_th;
  m_plane_edge_discontinuity_dist_th = plane_edge_discontinuity_dist_th;

  m_th_scan_distance_window = th_scan_distance_window;
  m_th_scan_counter_threshold = th_scan_counter_threshold;
  m_th_scan_threshold_enter = th_scan_threshold_enter;
  m_th_scan_threshold_exit = th_scan_threshold_exit;

  m_correlation_templates = correlation_templates;
  m_correlation_multiresolution_count = correlation_multiresolution_count;
  m_correlation_multiresolution_step = correlation_multiresolution_step;
  m_correlation_rescale = correlation_rescale;
  m_correlation_threshold = correlation_threshold;

  m_log = log;
}

cv::Mat FindClusterDistanceFunction(const cv::Mat & binary_image)
{
  const int width = binary_image.cols;
  const int height = binary_image.rows;

  const float nan = std::numeric_limits<float>::quiet_NaN();

  cv::Mat result(height, width, CV_32FC1);
  result = nan;

  cv::Mat queued(height, width, CV_8UC1);
  queued = uint8_t(false);

  std::deque<cv::Vec2i> frontier;

  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++)
    {
      if (binary_image.at<uint8_t>(y, x))
      {
        frontier.push_back(cv::Vec2i(x, y));
        result.at<float>(y, x) = 0.0f;
        queued.at<uint8_t>(y, x) = true;
      }
    }

  while (!frontier.empty())
  {
    const cv::Vec2i pt = frontier.front();
    const int y = pt[1];
    const int x = pt[0];

    frontier.pop_front();
    queued.at<uint8_t>(y, x) = false;
    const float v = result.at<float>(y, x);

    for (int dy = -1; dy <= 1; dy++)
      for (int dx = -1; dx <= 1; dx++)
      {
        if (!dx && !dy)
          continue;

        const float dist = std::sqrt(SQR(dx) + SQR(dy));

        const int nx = x + dx;
        const int ny = y + dy;

        if (nx < 0 || nx >= width)
          continue;
        if (ny < 0 || ny >= height)
          continue;

        const float nv = result.at<float>(ny, nx);
        if (std::isnan(nv) || nv > v + dist)
        {
          result.at<float>(ny, nx) = v + dist;
          if (!queued.at<uint8_t>(ny, nx))
          {
            frontier.push_back(cv::Vec2i(nx, ny));
            queued.at<uint8_t>(ny, nx) = true;
          }
        }
      }
  }

  return result;
}

void PalletFromImage::FindPillarsEdgeImage(const Vector4fVector & found_planes,
                                           const Uint64Vector & found_plane_indices,
                                           const cv::Mat & cluster_image,
                                           const cv::Mat & depth_image,
                                           const Eigen::Affine3f & camera_pose,
                                           const CameraInfo &camera_info,
                                           Vector4dVector & found_pillars,
                                           Uint64Vector & pillar_parent_plane,
                                           PillarTypeVector & pillar_type,
                                           cv::Mat & debug_edge_image,
                                           cv::Mat & debug_sdf_image) const
{
  const size_t height = cluster_image.rows;
  const size_t width = cluster_image.cols;

  debug_edge_image = cv::Mat(height, width, CV_8UC1, uint8(0));
  std::vector<std::vector<cv::Vec4i> > debug_found_lines(found_planes.size());

  debug_sdf_image = cv::Mat(height, width, CV_8UC1, uint8(255));

  for (size_t plane_i = 0; plane_i < found_planes.size(); plane_i++)
  {
    const Eigen::Vector4f & plane_coefficients = found_planes[plane_i];
    const Eigen::Vector3f plane_normal = plane_coefficients.head<3>();
    const float plane_w = plane_coefficients.w();

    cv::Mat this_plane_cluster_image = (cluster_image == int32(plane_i + 1));
    cv::Mat this_plane_cluster_image_closed = CloseImage(this_plane_cluster_image, 10);
    cv::Mat this_plane_distance_function = FindClusterDistanceFunction(this_plane_cluster_image_closed);
    cv::Mat this_plane_neg_distance_function = -FindClusterDistanceFunction(255 - this_plane_cluster_image_closed);
    this_plane_distance_function += this_plane_neg_distance_function;

    cv::Mat this_plane_distance_function_view;
    this_plane_distance_function.convertTo(this_plane_distance_function_view, CV_8UC1, 1, 128);

    debug_sdf_image = cv::min(debug_sdf_image, this_plane_distance_function_view);

    cv::Mat local_edge_image = PlaneEdgeImage(cluster_image, depth_image, found_planes, plane_i);
    local_edge_image = DilateImage(local_edge_image, 1);

    debug_edge_image = local_edge_image + debug_edge_image;

    std::vector<cv::Vec4i> linesP;
    cv::HoughLinesP(local_edge_image, linesP, 5, CV_PI/180, m_depth_hough_threshold,
                    m_depth_hough_min_length, m_depth_hough_max_gap);

    debug_found_lines[plane_i] = linesP;

    for (cv::Vec4i l : linesP)
    { 
      // rays in camera coordinates
      const Eigen::Vector3f cl_start((l[0] - camera_info.cx) / camera_info.fx, (l[1] - camera_info.cy) / camera_info.fy, 1.0f);
      const Eigen::Vector3f cl_end((l[2] - camera_info.cx) / camera_info.fx, (l[3] - camera_info.cy) / camera_info.fy, 1.0f);
      const Eigen::Vector3f co_start = Eigen::Vector3f::Zero();
      const Eigen::Vector3f co_end = Eigen::Vector3f::Zero();

      // rays in world coordinates
      const Eigen::Vector3f l_start = camera_pose.linear() * cl_start;
      const Eigen::Vector3f l_end = camera_pose.linear() * cl_end;
      const Eigen::Vector3f o_start = camera_pose * co_start;
      const Eigen::Vector3f o_end = camera_pose * co_end;

      // check inclination in world coordinates
      if (std::atan2((l_start - l_end).head<2>().norm(), std::abs(l_start.z() - l_end.z())) > m_vertical_line_angle_tolerance)
        continue;

      float total_gradient = 0.0f;
      {
        cv::LineIterator line(this_plane_distance_function, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]));

        for (uint64 i = 0; i < uint64(line.count); i++)
        {
          const cv::Point p = line.pos();
          const int x = p.x;
          const int y = p.y;

          if (x < 1 || x >= int(width - 1))
            continue;
          if (y < 1 || y >= int(height - 1))
            continue;

          const float dx = this_plane_distance_function.at<float>(y, x + 1) - this_plane_distance_function.at<float>(y, x - 1);
          const float dy = this_plane_distance_function.at<float>(y + 1, x) - this_plane_distance_function.at<float>(y - 1, x);
          const Eigen::Vector3f local_gradient = Eigen::Vector3f(dx, dy, 0.0f);

          const Eigen::Vector3f world_gradient = camera_pose.linear() * local_gradient;

          const Eigen::Vector3f up = Eigen::Vector3f::UnitZ();
          const Eigen::Vector3f right = up.cross(plane_normal);

          const float gradient_on_right = world_gradient.dot(right);
          total_gradient += gradient_on_right;
          line++;
        }

        if (line.count)
          total_gradient /= float(line.count);
      }

      // intersect both lines with plane
      const float t_start = (-plane_w - o_start.dot(plane_normal)) /
                            l_start.dot(plane_normal);
      const float t_end = (-plane_w - o_end.dot(plane_normal)) /
                          l_end.dot(plane_normal);
      const Eigen::Vector3f start = o_start + l_start * t_start;
      const Eigen::Vector3f end = o_end + l_end * t_end;

      if (std::abs(total_gradient) < 0.1)
        continue; // middle pillar
      const PillarType ptype = (total_gradient > 0) ? PillarType::RIGHT : PillarType::LEFT;

      const Eigen::Vector4d pillar((start.x() + end.x()) / 2.0, (start.y() + end.y()) / 2.0,
                                   std::min(start.z(), end.z()), std::max(start.z(), end.z()));
      found_pillars.push_back(pillar);
      pillar_parent_plane.push_back(found_plane_indices[plane_i]);
      pillar_type.push_back(ptype);
    }
  }

  cv::cvtColor(debug_edge_image, debug_edge_image, cv::COLOR_GRAY2RGB);

  for (size_t plane_i = 0; plane_i < found_planes.size(); plane_i++)
  {
    const pcl::RGB color = pcl::GlasbeyLUT::at(plane_i % 256);
    for (cv::Vec4i l : debug_found_lines[plane_i])
      cv::line(debug_edge_image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(color.r, color.g, color.b), 3, cv::LINE_AA);
  }
}

void PalletFromImage::FindPillarsThresholdScan(const Vector4fVector & found_planes,
                                               const Uint64Vector & found_plane_indices,
                                               const IntVectorVector & found_plane_inliers,
                                               const cv::Mat & depth_image,
                                               const PointXYZRGBCloudPtr & z_up_cloud,
                                               Vector4dVector & found_pillars,
                                               Uint64Vector & pillar_parent_plane) const
{
  found_pillars.clear();
  pillar_parent_plane.clear();

  const float nan = std::numeric_limits<float>::quiet_NaN();

  const size_t width = depth_image.cols;

  for (size_t plane_i = 0; plane_i < found_planes.size(); plane_i++)
  {
    const Eigen::Vector4f & plane_coefficients = found_planes[plane_i];
    const Eigen::Vector3f & plane_center = -plane_coefficients.head<3>() * plane_coefficients.w();
    const IntVector & inliers = found_plane_inliers[plane_i];

    const Eigen::Vector3f vertical = Eigen::Vector3f::UnitZ();
    const Eigen::Vector3f plane_normal = plane_coefficients.head<3>();
    const Eigen::Vector3f along_plane = vertical.cross(plane_normal);

    struct DistanceWithZ
    {
      float d;
      float z;
      float w;
    };

    std::vector<DistanceWithZ> distances;
    for (const int inl : inliers)
    {
      const size_t x = size_t(inl) % width;
      const size_t y = size_t(inl) / width;
      const float depth = depth_image.at<float>(y, x);

      const pcl::PointXYZRGB & ppt = (*z_up_cloud)[inl];
      const Eigen::Vector3f ept(ppt.x, ppt.y, ppt.z);

      Eigen::Vector3f proj = (ept - plane_center) - (ept - plane_center).dot(plane_normal) * plane_normal;
      proj.z() = 0.0f;
      DistanceWithZ dist_along_plane;
      dist_along_plane.d = proj.dot(along_plane);
      dist_along_plane.z = ept.z();
      dist_along_plane.w = SQR(depth);
      distances.push_back(dist_along_plane);
    }

    std::sort(distances.begin(), distances.end(),
              [](const DistanceWithZ & a, const DistanceWithZ & b) -> bool {return a.d < b.d; });

    bool is_entered = false;
    int64 last_point_in_window = 0;
    float total_weight = 0.0;
    for (int64 i = 0; i < int64(distances.size()); i++)
    {
      // update last point in window
      total_weight += distances[i].w;
      while (distances[last_point_in_window].d < distances[i].d - m_th_scan_distance_window)
      {
        total_weight -= distances[last_point_in_window].w;
        last_point_in_window++;
      }
      const uint64 counter = i - last_point_in_window + 1;

      if ((!is_entered && counter > m_th_scan_counter_threshold && total_weight > m_th_scan_threshold_enter) ||
          (is_entered && (total_weight < m_th_scan_threshold_exit || i+1 == int64(distances.size())))) // always exit at last point
      {
        // compute average in window
        float avg_distance = 0.0;
        float min_z = nan;
        float max_z = nan;
        for (int64 ni = last_point_in_window; ni <= i; ni++)
        {
          avg_distance += distances[ni].d;
          if (std::isnan(min_z) || min_z > distances[ni].z)
            min_z = distances[ni].z;
          if (std::isnan(max_z) || max_z < distances[ni].z)
            max_z = distances[ni].z;
        }
        avg_distance /= counter;

        const Eigen::Vector3f ept = avg_distance * along_plane + plane_center;

        const Eigen::Vector4d pillar(ept.x(), ept.y(), min_z, max_z);
        found_pillars.push_back(pillar);
        pillar_parent_plane.push_back(found_plane_indices[plane_i]);

        is_entered = !is_entered;
      }
    }
  }
}

cv::Mat PalletFromImage::DoCorrelation(const cv::Mat & rgb_image, const cv::Mat & depth_img)
{
  if (m_correlation_templates.empty())
  {
    m_log(1, "DoCorrelation: no correlation templates provided, skipping.");
    cv::Mat result(rgb_image.rows, rgb_image.cols, CV_8UC1);
    result = uint8(1);
    return result;
  }

  m_log(1, "DoCorrelation start.");

  const int orig_width = rgb_image.cols;
  const int orig_height = rgb_image.rows;

  const float my_rescale = m_correlation_rescale;

  cv::Mat image = rgb_image.clone();
  cv::resize(image, image, cv::Size(), my_rescale, my_rescale, cv::INTER_NEAREST);
  cv::Mat depth_image = depth_img.clone();
  cv::resize(depth_image, depth_image, cv::Size(), my_rescale, my_rescale, cv::INTER_NEAREST);

  const int width = image.cols;
  const int height = image.rows;

  cv::Mat max_result(height, width, CV_32FC1);
  max_result = 0.0f;
  cv::Mat max_resize_pow(height, width, CV_32FC1);
  max_resize_pow = 0.0f;
  cv::Mat max_templ(height, width, CV_32SC1);
  max_templ = -1;

  cv::Mat image_channels[3];
  cv::split(image, image_channels);
  for (uint64 c = 0; c < 3; c++)
  {
    cv::Canny(image_channels[c], image_channels[c], 100, 200, 3);

    const int dilation_size = 3;
    cv::blur(image_channels[c], image_channels[c], cv::Size(dilation_size, dilation_size));
  }

  for (uint64 corr_templ_i = 0; corr_templ_i < m_correlation_templates.size(); corr_templ_i++)
  {
    const CorrTemplate & corr_templ = m_correlation_templates[corr_templ_i];

    cv::Mat templ = corr_templ.image;
    const float template_depth = corr_templ.depth;

    cv::resize(templ, templ, cv::Size(), my_rescale, my_rescale, cv::INTER_NEAREST);

    const float resize_fraction = m_correlation_multiresolution_step;
    const int64 resize_min = 0;
    const int64 resize_max = m_correlation_multiresolution_count;
    for (int64 resize_i = resize_min; resize_i < resize_max; resize_i++)
    {
      cv::Mat resized_templ;
      const float resize_pow = std::pow<float>(resize_fraction, resize_i);
      const float resized_template_depth = template_depth / resize_pow;
      cv::resize(templ, resized_templ, cv::Size(), resize_pow, resize_pow, cv::INTER_NEAREST);

      cv::Mat templ_channels[3];
      cv::split(resized_templ, templ_channels);

      for (uint64 c = 0; c < 3; c++)
      {
        cv::Canny(templ_channels[c], templ_channels[c], 100, 200, 3);

        const int dilation_size = 3;
        cv::blur(templ_channels[c], templ_channels[c], cv::Size(dilation_size, dilation_size));
      }

      const int result_cols = width - resized_templ.cols + 1;
      const int result_rows = height - resized_templ.rows + 1;
      cv::Mat result(result_rows, result_cols, CV_32FC1);
      result = 0.0f;

      for (uint64 c = 0; c < 3; c++)
      {
        cv::Mat r;
        cv::matchTemplate(image_channels[c], templ_channels[c], r, cv::TM_SQDIFF_NORMED);
        result += r / 3.0f;
      }
      result = 1.0f - result;

      for (int y = 0; y < result_rows; y++)
        for (int x = 0; x < result_cols; x++)
        {
          const int gx = x + resized_templ.cols / 2;
          const int gy = y + resized_templ.rows / 2;

          const float g_depth = depth_image.at<float>(gy, gx);
          if (g_depth == 0.0f)
            continue;
          if (g_depth < resized_template_depth * resize_fraction || g_depth > resized_template_depth / resize_fraction)
            continue;

          const float v = result.at<float>(y, x);

          if (v > max_result.at<float>(gy, gx))
          {
            max_result.at<float>(gy, gx) = v;
            max_resize_pow.at<float>(gy, gx) = resize_pow;
            max_templ.at<int32>(gy, gx) = corr_templ_i;
          }
        }
    }
  }

  m_log(1, "DoCorrelation expand.");

  {
    const float my_further_rescale = 0.5f;
    cv::resize(max_result, max_result, cv::Size(), my_further_rescale, my_further_rescale, cv::INTER_NEAREST);
    cv::resize(max_resize_pow, max_resize_pow, cv::Size(), my_further_rescale, my_further_rescale, cv::INTER_NEAREST);
    cv::resize(max_templ, max_templ, cv::Size(), my_further_rescale, my_further_rescale, cv::INTER_NEAREST);
    cv::Mat prev_max_result = max_result.clone();

    cv::Mat rescaled_depth_image = depth_image;
    cv::resize(depth_image, rescaled_depth_image, cv::Size(), my_further_rescale, my_further_rescale, cv::INTER_NEAREST);

    const int rescaled_width = prev_max_result.cols;
    const int rescaled_height = prev_max_result.rows;

    for (int y = 0; y < rescaled_height; y++)
      for (int x = 0; x < rescaled_width; x++)
      {
        const float v = prev_max_result.at<float>(y, x);
        const float p = max_resize_pow.at<float>(y, x);
        const int32 corr_templ_i = max_templ.at<int32>(y, x);
        if (corr_templ_i < 0)
          continue;
        const CorrTemplate & corr_templ = m_correlation_templates[corr_templ_i];
        const int window_width = int(p * corr_templ.image.cols * my_rescale * my_further_rescale);
        const int window_height = int(p * corr_templ.image.rows * my_rescale * my_further_rescale);

        for (int dy = 0; dy < window_height; dy++)
          for (int dx = 0; dx < window_width; dx++)
          {
            const int nx = dx + x - window_width/2;
            const int ny = dy + y - window_height/2;

            const float g_depth = rescaled_depth_image.at<float>(ny, nx);
            if (g_depth == 0.0f)
              continue;

            if (max_result.at<float>(ny, nx) < v)
              max_result.at<float>(ny, nx) = v;
          }
      }
  }

  cv::resize(max_result, max_result, cv::Size(orig_width, orig_height), 0.0f, 0.0f, cv::INTER_LINEAR);

  m_log(1, "DoCorrelation end.");

  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;
  cv::minMaxLoc(max_result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

  const float corr_th = 0.05f;

  cv::Mat grayscale_result;
  max_result.convertTo(grayscale_result, CV_8UC1, 255);

  cv::Mat bool_max_result = max_result > corr_th;

  m_last_correlation_image = grayscale_result;
  m_last_bool_correlation_image = bool_max_result;

  return bool_max_result;
}

Eigen::Vector4f PalletFromImage::EnsureVerticalPlane(const Eigen::Vector4f & coeffs, const Eigen::Affine3f & camera_pose,
                                                     const Eigen::Vector3f & center)
{
  Eigen::Vector4f coefficients = coeffs;

  // make sure plane is vertical
  {
    const float norm = coefficients.head<2>().norm();
    if (norm > 0.0001f)
      coefficients = coefficients / norm;
    coefficients.z() = 0.0;
    coefficients.w() = -coefficients.head<3>().dot(center);
  }

  // ensure plane normal points towards the camera
  {
    const Eigen::Vector3f plane_normal_camera = camera_pose.linear().transpose() * coefficients.head<3>().normalized();
    const float dot = plane_normal_camera.dot(Eigen::Vector3f::UnitZ());
    if (dot > 0.0f)
      coefficients = -coefficients;
  }

  return coefficients;
}

void PalletFromImage::Run(const cv::Mat & rgb_image, const cv::Mat & depth_image,
                          const PointXYZRGBCloudPtr & z_up_cloud, const IntVectorPtr valid_indices_ptr_in,
                          const Eigen::Affine3f & camera_pose, const CameraInfo &camera_info,
                          ExpectedPallet & pallet)
{
  const size_t MIN_CLUSTER_SIZE = m_min_cluster_points;

  m_log(1, "Cloud size is " + std::to_string(z_up_cloud->size()));

  const float nan = std::numeric_limits<float>::quiet_NaN();

  const size_t width = rgb_image.cols;
  const size_t height = rgb_image.rows;

  IntVectorPtr valid_indices_ptr = valid_indices_ptr_in;

  {
    const cv::Mat correlation = DoCorrelation(rgb_image, depth_image);

    const IntVectorPtr old_valid_indices_ptr = valid_indices_ptr;
    valid_indices_ptr.reset(new IntVector);

    for (const int index : *old_valid_indices_ptr)
    {
      const size_t x = size_t(index) % width;
      const size_t y = size_t(index) / width;

      if (!correlation.at<uint8>(y, x))
        continue;

      valid_indices_ptr->push_back(index);
    }
  }

  IntVectorPtr remaining_indices_ptr = valid_indices_ptr;

  //std::vector<IntVector> clusters;
  cv::Mat cluster_image = cv::Mat(height, width, CV_32SC1);
  cluster_image = int32_t(0);
  cv::Mat dist_buffer = cv::Mat(height, width, CV_32FC1);
  dist_buffer = 1000000.0f;

  size_t cluster_counter = 1;
  Vector4fVector found_planes;
  Vector3fVector found_plane_centers;
  Vector2fVector found_plane_z;
  IntVectorVector found_plane_inliers;
  while (true)
  {
    if (remaining_indices_ptr->size() < m_min_cluster_points)
      break;
    m_log(1, "Ransac fit plane on " + std::to_string(remaining_indices_ptr->size()) + " remaining indices.");
    IntVectorPtr inliers(new IntVector);
    Eigen::Vector4f coefficients = RansacFitPlane(z_up_cloud, remaining_indices_ptr, *inliers);
    inliers = FindPlaneInliers(z_up_cloud, remaining_indices_ptr, coefficients);

    m_log(1, "Inliers size: " + std::to_string(inliers->size()));
    if (inliers->size() < m_min_cluster_points)
      break;

    remaining_indices_ptr = IntVectorSetDifference(*remaining_indices_ptr, *inliers);

    float weighted_point_count = 0.0;
    for (const int index : *inliers)
    {
      const size_t x = size_t(index) % width;
      const size_t y = size_t(index) / width;
      const float depth = depth_image.at<float>(y, x);
      if (depth == 0.0f)
        continue;
      const float weight = SQR(depth);
      weighted_point_count += weight;
    }
    m_log(1, "Weighted inliers size at 1 m: " + std::to_string(weighted_point_count));
    if (weighted_point_count < m_min_cluster_points_at_1m)
      continue;

    {
      const Eigen::Vector3f plane_normal_camera = camera_pose.linear().transpose() * coefficients.head<3>().normalized();
      const float dot = plane_normal_camera.dot(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
      const float cos_angle = std::abs(dot);
      if (cos_angle < std::cos(m_plane_camera_max_angle))
        continue;
      if (dot > 0.0f)
        coefficients = -coefficients; // ensure coefficients point towards the camera
    }

    {
      const Eigen::Vector3f camera_position = camera_pose.translation();
      const Eigen::Vector3f plane_normal = coefficients.head<3>().normalized();
      const float plane_camera_distance = std::abs(camera_position.dot(plane_normal) + coefficients.w());

      if (plane_camera_distance < m_min_plane_camera_distance)
        continue;
    }

    m_log(1, "Plane coefficients: " + e_to_string(coefficients.transpose()));

    IntVectorPtr relaxed_inliers = FindPlaneInliers(z_up_cloud, valid_indices_ptr, coefficients);
    relaxed_inliers = FilterInliersBySmallClusters(*relaxed_inliers, width, height, MIN_CLUSTER_SIZE);

    {
      float weighted_point_count = 0.0;
      for (const int index : *relaxed_inliers)
      {
        const size_t x = size_t(index) % width;
        const size_t y = size_t(index) / width;
        const float depth = depth_image.at<float>(y, x);
        if (depth == 0.0f)
          continue;
        const float weight = SQR(depth);
        weighted_point_count += weight;
      }
      if (weighted_point_count < m_min_cluster_points_at_1m)
        continue;
    }

    {
      uint64 counter = 0;
      Eigen::Vector3f center = Eigen::Vector3f::Zero();
      double min_z = nan;
      double max_z = nan;
      const Eigen::Vector3f axis_z = Eigen::Vector3f::UnitZ();
      for (const int index : *relaxed_inliers)
      {
        const size_t x = size_t(index) % width;
        const size_t y = size_t(index) / width;
        const float depth = depth_image.at<float>(y, x);
        if (depth == 0.0f)
          continue;
        pcl::PointXYZRGB pt = (*z_up_cloud)[index];
        const Eigen::Vector3f ept(pt.x, pt.y, pt.z);
        center += ept;
        counter++;

        {
          const float z = ept.dot(axis_z);
          if (std::isnan(min_z) || min_z > z)
            min_z = z;
          if (std::isnan(max_z) || max_z < z)
            max_z = z;
        }
      }
      if (counter)
        center /= counter;
      found_plane_centers.push_back(center);

      // recompute plane coefficients based on new inliers
      coefficients = InliersFitPlane(z_up_cloud, coefficients, relaxed_inliers);
      coefficients = EnsureVerticalPlane(coefficients, camera_pose, center);

      found_plane_z.push_back(Eigen::Vector2f(min_z, max_z));
      found_plane_inliers.push_back(*relaxed_inliers);
    }
    found_planes.push_back(coefficients);

    const FloatVector distance_inliers = FindPlaneDistance(z_up_cloud, relaxed_inliers, coefficients);

    for (size_t i = 0; i < relaxed_inliers->size(); i++)
    {
      int index = (*relaxed_inliers)[i];
      const size_t x = size_t(index) % width;
      const size_t y = size_t(index) / width;
      if (distance_inliers[i] < dist_buffer.at<float>(y, x))
      {
        cluster_image.at<int32_t>(y, x) = cluster_counter;
        dist_buffer.at<float>(y, x) = distance_inliers[i];
      }
    }

    cluster_counter++;
  }

  PalletRansac::ExpectedPallet real_pallet;
  Uint64Vector found_plane_indices(found_planes.size());
  for (uint64 plane_i = 0; plane_i < found_planes.size(); plane_i++)
  {
    const Eigen::Vector4f & plane = found_planes[plane_i];
    const Eigen::Vector3f & plane_center = found_plane_centers[plane_i];
    const Eigen::Vector2f & plane_z = found_plane_z[plane_i];
    ExpectedElement elem;
    elem.type = ExpectedElementType::PLANE;
    elem.plane = plane.cast<double>();
    elem.plane_point = plane_center.cast<double>();
    elem.plane_z = plane_z.cast<double>();
    real_pallet.push_back(elem);
    found_plane_indices[plane_i] = real_pallet.size() - 1;
  }

  m_last_cluster_image = cluster_image;

  Vector4dVector found_pillars;
  Uint64Vector found_pillars_parent_plane;
  PillarTypeVector found_pillars_type;
  // extract edges
  FindPillarsEdgeImage(found_planes, found_plane_indices, cluster_image, depth_image,
                       camera_pose, camera_info, found_pillars, found_pillars_parent_plane, found_pillars_type,
                       m_last_edge_image, m_last_sdf_image);

  // FindPillarsThresholdScan(found_planes, found_plane_indices, found_plane_inliers,
  //                          depth_image, z_up_cloud, found_pillars, pillar_parent_plane);

  Uint64VectorVector pillars_parent_plane;
  PillarTypeVectorVector pillars_type;
  {
    m_log(1, "Merging " + std::to_string(found_pillars.size()) + " pillars...");
    found_pillars = GridFilterPillars(found_pillars, m_pillars_merge_threshold, found_pillars_parent_plane, found_pillars_type,
                                      pillars_parent_plane, pillars_type);
    m_log(1, "Remaining " + std::to_string(found_pillars.size()) + " pillars after merging.");
  }

  for (uint64 pillar_i = 0; pillar_i < found_pillars.size(); pillar_i++)
  {
    const Eigen::Vector4d & pillar = found_pillars[pillar_i];
    ExpectedElement elem;
    elem.type = ExpectedElementType::PILLAR;
    elem.pillar = pillar;

    for (uint64 i = 0; i < pillars_parent_plane[pillar_i].size(); i++)
    {
      if (pillars_type[pillar_i][i] == PillarType::LEFT)
        elem.pillar_left_plane_id = pillars_parent_plane[pillar_i][i];
      else if (pillars_type[pillar_i][i] == PillarType::RIGHT)
        elem.pillar_right_plane_id = pillars_parent_plane[pillar_i][i];
    }
    real_pallet.push_back(elem);
  }
  pallet = real_pallet;
}

Eigen::Vector4f PalletFromImage::RansacFitPlane(const PointXYZRGBCloudPtr cloud, const IntVectorPtr remaining_indices, IntVector & inliers)
{
  RANSAC_PP_Model::Ptr pp_model(new RANSAC_PP_Model(cloud));
  pp_model->setAxis(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
  pp_model->setEpsAngle(m_ransac_plane_angle_tolerance); // 5 degrees
  pp_model->setIndices(remaining_indices);

  Eigen::VectorXf coefficients;
  inliers.clear();
  pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac(pp_model);
  ransac.setDistanceThreshold(m_ransac_plane_distance_tolerance);
  ransac.computeModel();
  ransac.getInliers(inliers);
  ransac.getModelCoefficients(coefficients);

  return Eigen::Vector4f(coefficients);
}

PalletFromImage::IntVectorPtr PalletFromImage::FindPlaneInliers(const PointXYZRGBCloudPtr cloud, const IntVectorPtr remaining_indices,
                                                                const Eigen::Vector4f & plane_coefficients)
{
  RANSAC_P_Model::Ptr p_model(new RANSAC_P_Model(cloud));
  p_model->setIndices(remaining_indices);

  Eigen::VectorXf coeff = plane_coefficients;
  IntVectorPtr inliers(new IntVector);
  p_model->selectWithinDistance(coeff, m_ransac_plane_inliers_tolerance, *inliers);
  return inliers;
}

PalletFromImage::FloatVector PalletFromImage::FindPlaneDistance(const PointXYZRGBCloudPtr cloud, const IntVectorPtr remaining_indices,
                                                                const Eigen::Vector4f & plane_coefficients)
{
  RANSAC_P_Model::Ptr p_model(new RANSAC_P_Model(cloud));
  p_model->setIndices(remaining_indices);

  Eigen::VectorXf coeff = plane_coefficients;
  DoubleVector distances;
  p_model->getDistancesToModel(coeff, distances);
  FloatVector result(distances.size());
  for (size_t i = 0; i < distances.size(); i++)
    result[i] = distances[i];
  return result;
}

Eigen::Vector4f PalletFromImage::InliersFitPlane(const PointXYZRGBCloudPtr cloud,
                                                 const Eigen::Vector4f & prev_coeff,
                                                 const IntVectorPtr inliers)
{
  RANSAC_PP_Model::Ptr pp_model(new RANSAC_PP_Model(cloud));

  pp_model->setInputCloud(cloud);
  pp_model->setAxis(Eigen::Vector3f::UnitZ());
  pp_model->setEpsAngle(0.0);

  pcl::Indices samples = *inliers;
  Eigen::VectorXf coeff;
  Eigen::VectorXf prev_c = prev_coeff;
  pp_model->optimizeModelCoefficients(samples, prev_c, coeff);

  return Eigen::Vector4f(coeff);
}

PalletFromImage::IntVectorPtr PalletFromImage::IntVectorSetDifference(const IntVector & a, const IntVector & b)
{
  IntVectorPtr result_ptr(new IntVector);
  result_ptr->resize(a.size());
  IntVector::iterator it;
  it = std::set_difference(a.begin(), a.end(), b.begin(), b.end(), result_ptr->begin());
  result_ptr->resize(it - result_ptr->begin());
  return result_ptr;
}

cv::Mat PalletFromImage::CloseImage(const cv::Mat & image, const size_t closing_size) const
{
  cv::Mat closed_img;
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                              cv::Size(2 * closing_size + 1, 2 * closing_size + 1),
                                              cv::Point(closing_size, closing_size));
  cv::morphologyEx(image, closed_img, cv::MORPH_CLOSE, element);
  return closed_img;
}

cv::Mat PalletFromImage::DilateImage(const cv::Mat & image, const size_t dilate_size) const
{
  cv::Mat dilated_img;
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                              cv::Size(2 * dilate_size + 1, 2 * dilate_size + 1),
                                              cv::Point(dilate_size, dilate_size));
  cv::morphologyEx(image, dilated_img, cv::MORPH_DILATE, element);
  return dilated_img;
}

cv::Mat PalletFromImage::ResizeImage(const cv::Mat & image, const float fraction, const int interpolation) const
{
  cv::Mat result;
  cv::resize(image, result, cv::Size(), fraction, fraction, interpolation);
  return result;
}

cv::Mat PalletFromImage::ResizeImage(const cv::Mat & image, const int target_x, const int target_y, const int interpolation) const
{
  cv::Mat result;
  cv::resize(image, result, cv::Size(target_x, target_y), 0.0f, 0.0f, interpolation);
  return result;
}

template <typename T>
cv::Mat PalletFromImage::SimpleEdgeImage(const cv::Mat & image)
{
  const size_t res_y = image.rows;
  const size_t res_x = image.cols;
  cv::Mat edge_image = cv::Mat(res_y, res_x, CV_8UC1);
  edge_image = uint8(0);
  for (size_t y = 0; y < res_y; y++)
    for (size_t x = 0; x < res_x; x++)
    {
      const T v = image.at<T>(y, x);
      for (int64 dy = -1; dy <= 1; dy++)
        for (int64 dx = -1; dx <= 1; dx++)
        {
          if (dx && dy)
            continue; // 4-neighborhood

          const int64 nx = int64(x) + dx;
          const int64 ny = int64(y) + dy;
          if (nx < 0 || ny < 0)
            continue;
          if (uint64(nx) >= res_x || uint64(ny) >= res_y)
            continue;

          const T nv = image.at<T>(ny, nx);
          if (nv != v)
            edge_image.at<uint8>(y, x) = uint8(255);
        }
    }
  return edge_image;
}

template <typename T>
cv::Mat PalletFromImage::FillHoles(const cv::Mat & image, const T & hole_value, const uint64 iterations) const
{
  const int width = image.cols;
  const int height = image.rows;

  cv::Mat result = image;

  for (uint64 iter = 0; iter < iterations; iter++)
  {
    cv::Mat img = result;
    result = img.clone();

    for (int y = 0; y < height; y++)
      for (int x = 0; x < width; x++)
      {
        if (result.at<T>(y, x) != hole_value) // not hole
          continue;

        bool found = false;

        for (int dy = -1; dy <= 1 && !found; dy++)
          for (int dx = -1; dx <= 1 && !found; dx++)
          {
            const int nx = x + dx;
            const int ny = y + dy;

            if (nx < 0 || nx >= width)
              continue;
            if (ny < 0 || ny >= height)
              continue;

            const T nv = result.at<T>(ny, nx);
            if (nv != hole_value)
            {
              result.at<T>(y, x) = nv;
              found = true;
            }
          }
      }
  }

  return result;
}

template
cv::Mat PalletFromImage::FillHoles<float>(const cv::Mat & image, const float & hole_value, const uint64 iterations) const;
template
cv::Mat PalletFromImage::FillHoles<int32_t>(const cv::Mat & image, const int32_t & hole_value, const uint64 iterations) const;

// int32 filled_plane_image
// float depth_image
cv::Mat PalletFromImage::PlaneEdgeImage(const cv::Mat & plane_image, const cv::Mat & depth_image, const Vector4fVector & found_planes,
                                        const int32 relevant_plane) const
{
  const size_t res_y = plane_image.rows;
  const size_t res_x = plane_image.cols;
  cv::Mat edge_image = cv::Mat(res_y, res_x, CV_8UC1);
  edge_image = uint8(0);
  for (size_t y = 0; y < res_y; y++)
    for (size_t x = 0; x < res_x; x++)
    {
      const int32 v = plane_image.at<int32>(y, x);
      if (v != relevant_plane + 1)
        continue;

      for (int64 dy = -1; dy <= 1; dy++)
        for (int64 dx = -1; dx <= 1; dx++)
        {
          if (dx && dy)
            continue; // 4-neighborhood

          const int64 nx = int64(x) + dx;
          const int64 ny = int64(y) + dy;
          if (nx < 0 || ny < 0)
            continue;
          if (uint64(nx) >= res_x || uint64(ny) >= res_y)
            continue;

          const int32 nv = plane_image.at<int32>(ny, nx);

          if (nv != v)
          {
            bool diff_found = false;
            if (nv == 0) // one is background
              diff_found = true;
            else // neither is background: compare planes
            {
              const Eigen::Vector3f plane_normal = found_planes[v - 1].head<3>();
              const Eigen::Vector3f nplane_normal = found_planes[nv - 1].head<3>();

              if (plane_normal.dot(nplane_normal) < std::cos(m_plane_edge_discontinuity_angle_th))
                diff_found = true;
              const float d = depth_image.at<float>(y, x);
              const float nd = depth_image.at<float>(ny, nx);
              if (nd == 0.0f || nd > d + m_plane_edge_discontinuity_dist_th)
                diff_found = true;
            }

            if (diff_found)
              edge_image.at<uint8>(y, x) = uint8(255);
          }
        }
    }
  return edge_image;
}

PalletFromImage::IntVectorPtr PalletFromImage::FilterInliersBySmallClusters(const IntVector & inliers,
                                          const size_t width, const size_t height,
                                          const uint64 min_cluster_size)
{
  cv::Mat binary_cluster = cv::Mat(height, width, CV_8UC1);
  binary_cluster = 0;
  for (const int index : inliers)
  {
    const size_t x = size_t(index) % width;
    const size_t y = size_t(index) / width;

    binary_cluster.at<uint8>(y, x) = 255;
  }

  cv::Mat local_cluster;
  const int num_labels = cv::connectedComponents(binary_cluster, local_cluster, 4, CV_32S);
  std::vector<uint64> pixel_counter(num_labels, 0);
  for (size_t y = 0; y < height; y++)
    for (size_t x = 0; x < width; x++)
    {
      const int32_t v = local_cluster.at<int32_t>(y, x);
      pixel_counter[v]++;
    }

  IntVectorPtr result(new IntVector);
  for (size_t y = 0; y < height; y++)
    for (size_t x = 0; x < width; x++)
    {
      const int32_t v = local_cluster.at<int32_t>(y, x);
      if (v == 0) // background
        continue;
      if (pixel_counter[v] < min_cluster_size)
        continue;

      int index = x + y * width;
      result->push_back(index);
    }

  return result;
}

// filter similar pillars based on resolution
PalletFromImage::Vector4dVector PalletFromImage::GridFilterPillars(const Vector4dVector pillars, const double resolution,
                                                                   const Uint64Vector pillar_parent_plane,
                                                                   const PillarTypeVector pillar_type,
                                                                   Uint64VectorVector & pillar_parent_planes,
                                                                   PillarTypeVectorVector & pillar_types)
{
  struct PillarWeight
  {
    Eigen::Vector4d pillar;
    uint64 weight;
    Uint64Vector parent_planes;
    PillarTypeVector pillar_types;
  };

  typedef std::pair<int64, int64> Coordinates;
  typedef std::pair<Coordinates, PillarWeight> CellPair;
  typedef std::map<Coordinates, PillarWeight> CellMap;

  CellMap map;
  for (uint64 pillar_i = 0; pillar_i < pillars.size(); pillar_i++)
  {
    const Eigen::Vector4d pillar = pillars[pillar_i];

    Eigen::Vector2d ecoords = (pillar.head<2>() / resolution).array().round();
    Coordinates coords(ecoords.x(), ecoords.y());
    if (map.find(coords) == map.end())
    {
      PillarWeight pw;
      pw.pillar = pillar;
      pw.weight = 1;
      pw.parent_planes.push_back(pillar_parent_plane[pillar_i]);
      pw.pillar_types.push_back(pillar_type[pillar_i]);
      map.insert(CellPair(coords, pw));
    }
    else
    {
      PillarWeight & pillar_weight = map[coords];
      pillar_weight.pillar.x() += pillar.x();
      pillar_weight.pillar.y() += pillar.y();
      pillar_weight.pillar.z() = std::min(pillar.z(), pillar_weight.pillar.z());
      pillar_weight.pillar.w() = std::max(pillar.w(), pillar_weight.pillar.w());

      const Uint64Vector::iterator maybe_parent_plane = std::find(pillar_weight.parent_planes.begin(),
                                                                  pillar_weight.parent_planes.end(),
                                                                  pillar_parent_plane[pillar_i]);
      if (maybe_parent_plane == pillar_weight.parent_planes.end())
      {
        pillar_weight.parent_planes.push_back(pillar_parent_plane[pillar_i]);
        pillar_weight.pillar_types.push_back(pillar_type[pillar_i]);
      }
      else
      {
        const uint64 id = maybe_parent_plane - pillar_weight.parent_planes.begin();
        pillar_weight.pillar_types[id] = PillarType(uint64(pillar_weight.pillar_types[id]) | uint64(pillar_type[pillar_i]));
      }
      pillar_weight.weight += 1;
    }

  }

  Vector4dVector pillars_out;
  pillar_parent_planes.clear();
  for (const CellPair cell : map)
  {
    const PillarWeight & pillar_weight = cell.second;
    Eigen::Vector4d pillar = pillar_weight.pillar;
    pillar.x() /= double(pillar_weight.weight);
    pillar.y() /= double(pillar_weight.weight);

    pillars_out.push_back(pillar);
    pillar_parent_planes.push_back(pillar_weight.parent_planes);
    pillar_types.push_back(pillar_weight.pillar_types);
  }
  return pillars_out;
}

