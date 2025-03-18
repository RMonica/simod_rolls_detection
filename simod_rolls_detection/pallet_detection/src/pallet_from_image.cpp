#include "pallet_from_image.h"

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_parallel_plane.h>
#include <pcl/common/colors.h>
#include <pcl/search/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>

#include <sstream>

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
                                 const float plane_edge_discontinuity_dist_th)
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

  m_log = log;
}

void PalletFromImage::Run(const cv::Mat & rgb_image, const cv::Mat & depth_image,
                          const PointXYZRGBCloudPtr & z_up_cloud, const IntVectorPtr valid_indices_ptr,
                          const Eigen::Affine3f & camera_pose, const CameraInfo &camera_info,
                          ExpectedPallet & pallet)
{
  const size_t MIN_CLUSTER_SIZE = m_min_cluster_points;

  IntVectorPtr remaining_indices_ptr = valid_indices_ptr;

  m_log(1, "Cloud size is " + std::to_string(z_up_cloud->size()));

  const float nan = std::numeric_limits<float>::quiet_NaN();

  const size_t width = rgb_image.cols;
  const size_t height = rgb_image.rows;

  const double camera_info_fx = camera_info.fx;
  const double camera_info_fy = camera_info.fy;
  const double camera_info_cx = camera_info.cx;
  const double camera_info_cy = camera_info.cy;

  //std::vector<IntVector> clusters;
  cv::Mat cluster_image = cv::Mat(height, width, CV_32SC1);
  cluster_image = int32_t(0);
  cv::Mat dist_buffer = cv::Mat(height, width, CV_32FC1);
  dist_buffer = 1000000.0f;

  size_t cluster_counter = 1;
  Vector4fVector found_planes;
  Vector3fVector found_plane_centers;
  Vector2fVector found_plane_z;
  while (true)
  {
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

    // make sure plane is vertical
    {
      const float norm = coefficients.head<2>().norm();
      coefficients = coefficients / norm;
      coefficients.z() = 0.0;
    }

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
      found_plane_z.push_back(Eigen::Vector2f(min_z, max_z));
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

  Vector3fPairVector found_lines;
  Vector4dVector found_pillars;
  // extract edges
  {
    cv::Mat debug_edge_image = cv::Mat(height, width, CV_8UC1, uint8(0));
    std::vector<std::vector<cv::Vec4i> > debug_found_lines(found_planes.size());

    for (size_t plane_i = 0; plane_i < found_planes.size(); plane_i++)
    {
      //cv::Mat local_edge_image = (cluster_image == int32(plane_i + 1));
      cv::Mat local_edge_image = PlaneEdgeImage(cluster_image, depth_image, found_planes, plane_i);
      local_edge_image = DilateImage(local_edge_image, 1);

      debug_edge_image = local_edge_image + debug_edge_image;

      std::vector<cv::Vec4i> linesP; // will hold the results of the detection
      cv::HoughLinesP(local_edge_image, linesP, 5, CV_PI/180, m_depth_hough_threshold,
                      m_depth_hough_min_length, m_depth_hough_max_gap); // runs the actual detection

      debug_found_lines[plane_i] = linesP;

      for (cv::Vec4i l : linesP)
      {
        // rays in camera coordinates
        const Eigen::Vector3f cl_start((l[0] - camera_info_cx) / camera_info_fx, (l[1] - camera_info_cy) / camera_info_fy, 1.0f);
        const Eigen::Vector3f cl_end((l[2] - camera_info_cx) / camera_info_fx, (l[3] - camera_info_cy) / camera_info_fy, 1.0f);
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

        const Eigen::Vector4f & plane_coefficients = found_planes[plane_i];

        // intersect both lines with plane
        const float t_start = (-plane_coefficients.w() - o_start.dot(plane_coefficients.head<3>())) /
                              l_start.dot(plane_coefficients.head<3>());
        const float t_end = (-plane_coefficients.w() - o_end.dot(plane_coefficients.head<3>())) /
                            l_end.dot(plane_coefficients.head<3>());
        const Eigen::Vector3f start = o_start + l_start * t_start;
        const Eigen::Vector3f end = o_end + l_end * t_end;

        found_lines.push_back(Vector3fPair(start, end));
        const Eigen::Vector4d pillar((start.x() + end.x()) / 2.0, (start.y() + end.y()) / 2.0,
                                     std::min(start.z(), end.z()), std::max(start.z(), end.z()));
        found_pillars.push_back(pillar);
      }
    }

    cv::cvtColor(debug_edge_image, debug_edge_image, cv::COLOR_GRAY2RGB);

    for (size_t plane_i = 0; plane_i < found_planes.size(); plane_i++)
    {
      const pcl::RGB color = pcl::GlasbeyLUT::at(plane_i % 256);
      for (cv::Vec4i l : debug_found_lines[plane_i])
        cv::line(debug_edge_image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(color.r, color.g, color.b), 3, cv::LINE_AA);
    }

    m_last_edge_image = debug_edge_image;
    m_last_cluster_image = cluster_image;
  }

  {
    m_log(1, "Merging " + std::to_string(found_pillars.size()) + " pillars...");
    found_pillars = GridFilterPillars(found_pillars, m_pillars_merge_threshold);
    m_log(1, "Remaining " + std::to_string(found_pillars.size()) + " pillars after merging.");
  }

  PalletRansac::ExpectedPallet real_pallet;
  for (const Eigen::Vector4d & pillar : found_pillars)
  {
    ExpectedElement elem;
    elem.type = ExpectedElementType::PILLAR;
    elem.pillar = pillar;
    real_pallet.push_back(elem);
  }
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

PalletFromImage::IntVectorPtr PalletFromImage::IntVectorSetDifference(const IntVector & a, const IntVector & b)
{
  IntVectorPtr result_ptr(new IntVector);
  result_ptr->resize(a.size());
  IntVector::iterator it;
  it = std::set_difference(a.begin(), a.end(), b.begin(), b.end(), result_ptr->begin());
  result_ptr->resize(it - result_ptr->begin());
  return result_ptr;
}

cv::Mat PalletFromImage::CloseImage(const cv::Mat & image, const size_t closing_size)
{
  cv::Mat closed_img;
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                              cv::Size(2 * closing_size + 1, 2 * closing_size + 1),
                                              cv::Point(closing_size, closing_size));
  cv::morphologyEx(image, closed_img, cv::MORPH_CLOSE, element);
  return closed_img;
}

cv::Mat PalletFromImage::DilateImage(const cv::Mat & image, const size_t dilate_size)
{
  cv::Mat dilated_img;
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                              cv::Size(2 * dilate_size + 1, 2 * dilate_size + 1),
                                              cv::Point(dilate_size, dilate_size));
  cv::morphologyEx(image, dilated_img, cv::MORPH_DILATE, element);
  return dilated_img;
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

// int32 image
// float depth_image
cv::Mat PalletFromImage::PlaneEdgeImage(const cv::Mat & image, const cv::Mat & depth_image, const Vector4fVector & found_planes,
                                        const int32 relevant_plane)
{
  const size_t res_y = image.rows;
  const size_t res_x = image.cols;
  cv::Mat edge_image = cv::Mat(res_y, res_x, CV_8UC1);
  edge_image = uint8(0);
  for (size_t y = 0; y < res_y; y++)
    for (size_t x = 0; x < res_x; x++)
    {
      const int32 v = image.at<int32>(y, x);
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

          const int32 nv = image.at<int32>(ny, nx);

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
PalletFromImage::Vector4dVector PalletFromImage::GridFilterPillars(const Vector4dVector pillars, const double resolution)
{
  typedef std::pair<Eigen::Vector4d, uint64> PillarWeightPair;
  typedef std::pair<int64, int64> Coordinates;
  typedef std::pair<Coordinates, PillarWeightPair> CellPair;
  typedef std::map<Coordinates, PillarWeightPair> CellMap;

  CellMap map;
  for (const Eigen::Vector4d & pillar : pillars)
  {
    Eigen::Vector2d ecoords = (pillar.head<2>() / resolution).array().round();
    Coordinates coords(ecoords.x(), ecoords.y());
    if (map.find(coords) == map.end())
    {
      map.insert(CellPair(coords, PillarWeightPair(pillar, 1)));
    }
    else
    {
      PillarWeightPair & pillar_weight = map[coords];
      pillar_weight.first.x() += pillar.x();
      pillar_weight.first.y() += pillar.y();
      pillar_weight.first.z() = std::min(pillar.z(), pillar_weight.first.z());
      pillar_weight.first.w() = std::max(pillar.w(), pillar_weight.first.w());
      pillar_weight.second += 1;
    }

  }

  Vector4dVector pillars_out;
  for (const CellPair cell : map)
  {
    const PillarWeightPair & pillar_weight = cell.second;
    Eigen::Vector4d pillar = pillar_weight.first;
    pillar.x() /= double(pillar_weight.second);
    pillar.y() /= double(pillar_weight.second);

    pillars_out.push_back(pillar);
  }
  return pillars_out;
}

