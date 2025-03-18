#include "pallet_detection_solver.h"

#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <cmath>
#include <vector>

PalletDetectionPoseEstimation::PalletDetectionPoseEstimation()
{
  m_cost_threshold = std::numeric_limits<double>::quiet_NaN();
}

PalletDetectionPoseEstimation::PalletDetectionPoseEstimation(const double cost_threshold)
{
  m_cost_threshold = cost_threshold;
}

template <class T> T SQR(const T & t) {return t * t; }

class LineDistanceCostFn
{
  public:
  LineDistanceCostFn(const Eigen::Vector3d &line_coeffs, const Eigen::Vector2d &c_point)
      : line_coeffs_(line_coeffs), c_point_(c_point) {}

  template <typename T>
  bool operator()(const T *const x, T *e) const
  {
    // normalize line_coeff
    Eigen::Vector3d normalized_line_coeffs;
    double d = sqrt(line_coeffs_.x()*line_coeffs_.x() + line_coeffs_.y()*line_coeffs_.y());
    normalized_line_coeffs.x() = line_coeffs_.x()/d;
    normalized_line_coeffs.y() = line_coeffs_.y()/d;
    normalized_line_coeffs.z() = line_coeffs_.z()/d;

    e[0] = normalized_line_coeffs.x() * (c_point_.x() * cos(x[2]) - c_point_.y() * sin(x[2]) + x[0]) +
           normalized_line_coeffs.y() * (c_point_.x() * sin(x[2]) + c_point_.y() * cos(x[2]) + x[1]) +
           normalized_line_coeffs.z();
    return true;
  }

  private:
  const Eigen::Vector3d line_coeffs_;
  const Eigen::Vector2d c_point_;
};

class PlanesDistanceCostFn
{
  public:
  PlanesDistanceCostFn(const Eigen::Vector3d & plane, const Eigen::Vector3d & ideal_plane, const double weight)
      : m_plane(plane), m_ideal_plane(ideal_plane), m_weight(weight) {}

  template <typename T>
  bool operator()(const T *const x, T *e) const
  {
    T nx(m_plane.x());
    T ny(m_plane.y());
    T d(m_plane.z());
    T nxp = nx * cos(x[2]) - ny * sin(x[2]);
    T nyp = nx * sin(x[2]) + ny * cos(x[2]);
    T px = -nx * d;
    T py = -ny * d;
    T dp = -(nxp * (px * cos(x[2]) - py * sin(x[2]) + x[0]) + nyp * (px * sin(x[2]) + py * cos(x[2]) + x[1]));

    const double SQRT2 = std::sqrt(2.0); // divide by SQRT2 so that x, y count as one

    e[0] = (nxp - m_ideal_plane.x()) * m_weight / SQRT2;
    e[1] = (nyp - m_ideal_plane.y()) * m_weight / SQRT2;
    e[2] = (dp - m_ideal_plane.z()) * m_weight;

    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
  const Eigen::Vector3d m_plane;
  const Eigen::Vector3d m_ideal_plane;
  const double m_weight;
};

class PointsDistanceCostFn
{
  public:
  PointsDistanceCostFn(const Eigen::Vector2d & point, const Eigen::Vector2d & ideal_point, const double weight)
      : m_point(point), m_ideal_point(ideal_point), m_weight(weight) {}

  template <typename T>
  bool operator()(const T *const x, T *e) const
  {
    T nx(m_point.x());
    T ny(m_point.y());
    T nxp = nx * cos(x[2]) - ny * sin(x[2]) + x[0];
    T nyp = nx * sin(x[2]) + ny * cos(x[2]) + x[1];

    e[0] = (nxp - m_ideal_point.x()) * m_weight;
    e[1] = (nyp - m_ideal_point.y()) * m_weight;

    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
  const Eigen::Vector2d m_point;
  const Eigen::Vector2d m_ideal_point;
  const double m_weight;
};

class SimilarToInitialGuessCostFn
{
  public:
  SimilarToInitialGuessCostFn(const Eigen::Vector3d & initial_guess, const Eigen::Vector3d & weights)
      : m_weights(weights), m_initial_guess(initial_guess) {}

  template <typename T>
  bool operator()(const T *const x, T *e) const
  {
    e[0] = (x[0] - m_initial_guess[0]) * m_weights[0];
    e[1] = (x[1] - m_initial_guess[1]) * m_weights[1];
    e[2] = (x[2] - m_initial_guess[2]) * m_weights[2];
    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
  const Eigen::Vector3d m_weights;
  const Eigen::Vector3d m_initial_guess;
};

Eigen::Vector3d PalletDetectionPoseEstimation::estimate_pose(const PlaneCorrVector & planes, const PointCorrVector & points,
                                                             const Eigen::Vector3d & initial_guess)
{

  ceres::Problem problem;

  const Eigen::Vector3d similarity_to_initial_guess_weights = Eigen::Vector3d::Ones() * 0.0001;

  // Configure solver options
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 10000;
  options.logging_type = ceres::SILENT;

  // Initial guess
  double err[3] = {initial_guess.x(), initial_guess.y(), initial_guess.z()};

  // build planes
  for (size_t i = 0; i < planes.size(); i++)
  {
    const Eigen::Vector3d plane = planes[i].plane;
    const Eigen::Vector3d ideal_plane = planes[i].ideal_plane;
    const float weight = planes[i].weight;

    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<PlanesDistanceCostFn, 3, 3>(new PlanesDistanceCostFn(plane, ideal_plane, weight)),
                             nullptr,
                             err);
  }

  for (size_t i = 0; i < points.size(); i++)
  {
    const Eigen::Vector2d point = points[i].point;
    const Eigen::Vector2d ideal_point = points[i].ideal_point;
    const float weight = points[i].weight;

    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<PointsDistanceCostFn, 2, 3>(new PointsDistanceCostFn(point, ideal_point, weight)),
                             nullptr,
                             err);
  }

  problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<SimilarToInitialGuessCostFn, 3, 3>(
      new SimilarToInitialGuessCostFn(initial_guess, similarity_to_initial_guess_weights)), nullptr, err);

  // Solve the problem
  ceres::Solver::Summary summary;
  try
  {
    ceres::Solve(options, &problem, &summary);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Exception caught: " << e.what() << std::endl;
  }
  catch (...)
  {
    std::cerr << "Unknown exception caught" << std::endl;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return Eigen::Vector3d(nan, nan, nan);
  }

  //std::cout << summary.FullReport() << std::endl;
  const bool has_converged = (summary.termination_type == ceres::TerminationType::CONVERGENCE);
  if (!m_quiet)
  {
    std::cout << "has converged: " << (has_converged ? "TRUE" : "FALSE") << std::endl;
    std::cout << "final_cost: " << summary.final_cost << std::endl;
  }

  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (!has_converged)
    return Eigen::Vector3d(nan, nan, nan);

  if (!std::isnan(m_cost_threshold) && summary.final_cost > m_cost_threshold)
    return Eigen::Vector3d(nan, nan, nan);

  Eigen::Vector3d result(err[0], err[1], err[2]);
  return result;
}

void PalletDetectionPoseEstimation::test()
{
  {
    PlaneCorrVector planes;
    PointCorrVector points;
    points.push_back(PointCorr(Eigen::Vector2d(0.0, 1.0), Eigen::Vector2d(1.0, -2.0)));
    points.push_back(PointCorr(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(0.0, -3.0)));
    Eigen::Vector3d result = estimate_pose(planes, points);
    std::cout << "Test 1: expected 0 -2 -pi/2 got "  << result.transpose() << std::endl;
  }

  {
    PlaneCorrVector planes;
    PointCorrVector points;
    planes.push_back(PlaneCorr(Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 1.0, -2.0)));
    planes.push_back(PlaneCorr(Eigen::Vector3d(0.0, -1.0, 0.0), Eigen::Vector3d(1.0, 0.0, -1.0)));
    Eigen::Vector3d result = estimate_pose(planes, points);
    std::cout << "Test 2: expected 1 2 pi/2 got "  << result.transpose() << std::endl;
  }

  {
    PlaneCorrVector planes;
    PointCorrVector points;
    points.push_back(PointCorr(Eigen::Vector2d(0.0, 1.0), Eigen::Vector2d(0.0,  2.0)));
    points.push_back(PointCorr(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(1.0,  3.0)));
    planes.push_back(PlaneCorr(Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 1.0, -2.0)));
    planes.push_back(PlaneCorr(Eigen::Vector3d(0.0, -1.0, 0.0), Eigen::Vector3d(1.0, 0.0, -1.0)));
    Eigen::Vector3d result = estimate_pose(planes, points);
    std::cout << "Test 3: expected 1 2 pi/2 got "  << result.transpose() << std::endl;
  }

  {
    PlaneCorrVector planes;
    PointCorrVector points;
    points.push_back(PointCorr(Eigen::Vector2d(0.0, 1.0), Eigen::Vector2d(0.0,  2.1)));
    points.push_back(PointCorr(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(0.9,  3.0)));
    planes.push_back(PlaneCorr(Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 1.0, -2.1)));
    planes.push_back(PlaneCorr(Eigen::Vector3d(0.0, -1.0, 0.0), Eigen::Vector3d(1.0, 0.0, -1.0)));
    Eigen::Vector3d result = estimate_pose(planes, points);
    std::cout << "Test 4: expected 1 2 pi/2 got "  << result.transpose() << std::endl;
  }
}
