#include "roll_pack_pose_model.h"

#include <ceres/ceres.h>

template <class T> T SQR(const T & t) {return t * t; }

static const double my_nan = std::numeric_limits<double>::quiet_NaN();

typedef RollPackPoseEstimator::MyRollPackPoseModelPtr MyRollPackPoseModelPtr;

template <typename T>
void RollPackPoseModel<T>::SetParameterBounds(void * void_problem,
                                              double * x,
                                              const int offset)
{
  ceres::Problem & problem = *(ceres::Problem *)void_problem;

  problem.SetParameterLowerBound(x, offset + 3, -M_PI); // rz
  problem.SetParameterUpperBound(x, offset + 3, M_PI); // rz
}

class PointDistanceCostFn
{
  public:
  PointDistanceCostFn(const Eigen::Affine3d & inv_camera_pose,
                      const Eigen::Matrix3d & camera_matrix,
                      const Eigen::Vector3d & observed_point,
                      const Eigen::Vector3d & reference_point,
                      const double pt_wgt, const double depth_wgt)
         : m_reference_point(reference_point), m_observed_point(observed_point),
          m_inv_camera_pose(inv_camera_pose), m_camera_matrix(camera_matrix),
          m_pt_wgt(pt_wgt), m_depth_wgt(depth_wgt) {}

  static constexpr int ON = 3;

  template <typename T>
  bool operator()(const T * const x, T * e) const
  {
    RollPackPoseModel<T> def_model(x, 0);

    T px(m_reference_point.x());
    T py(m_reference_point.y());
    T pz(m_reference_point.z());
    T nx(0.0);
    T ny(0.0);
    T nz(0.0);
    def_model.ApplyToPoint(m_inv_camera_pose, m_camera_matrix, px, py, pz, nx, ny, nz);

    e[0] = (nx - double(m_observed_point.x())) * m_pt_wgt;
    e[1] = (ny - double(m_observed_point.y())) * m_pt_wgt;
    if (!std::isnan(m_observed_point.z()))
      e[2] = (nz - double(m_observed_point.z())) * m_depth_wgt;
    else
      e[2] = T(0.0);
    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
  const Eigen::Vector3d m_reference_point;
  const Eigen::Vector3d m_observed_point;
  const Eigen::Affine3d & m_inv_camera_pose;
  const Eigen::Matrix3d & m_camera_matrix;
  const double m_pt_wgt;
  const double m_depth_wgt;
};

RollPackPoseEstimator::RollPackPoseEstimator(const double px_weight, const double depth_weight,
                                             const Eigen::Affine3d & camera_pose,
                                             const double fx, const double fy, const double cx, const double cy):
    m_px_weight(px_weight), m_depth_weight(depth_weight)
{
  m_camera_pose = camera_pose;
  m_inv_camera_pose = camera_pose.inverse();
  m_camera_matrix = Eigen::Matrix3d::Identity();
  m_camera_matrix(0, 0) = fx;
  m_camera_matrix(1, 1) = fy;
  m_camera_matrix(0, 2) = cx;
  m_camera_matrix(1, 2) = cy;
}

MyRollPackPoseModelPtr RollPackPoseEstimator::Estimate(const Vector3dVector & observed_points,
                                                       const Vector3dVector & reference_points,
                                                       double & final_cost) const
{
  final_cost = std::numeric_limits<double>::quiet_NaN();

  ceres::Problem problem;

  // Configure solver options
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  //options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 1000;
  options.logging_type = ceres::SILENT;

  // Initial guess
  constexpr int NUM_PARAMS = MyRollPackPoseModel::GetNumParams();
  double params[NUM_PARAMS];

  const int num_points = reference_points.size();
  const int num_initial_points = observed_points.size();
  if (num_points != num_initial_points)
  {
    std::cerr << "Points size mismatch! Current points " << num_points << " <> initial points " << num_initial_points << std::endl;
    return MyRollPackPoseModelPtr();
  }

  { // initial guess
    // find a point in front of the camera as initial guess
    const Eigen::Vector3d camera_forward = Eigen::Vector3d::UnitZ() * 1.5f;
    const Eigen::Vector3d world_forward = m_camera_pose * camera_forward;

    MyRollPackPoseModel model(world_forward, 0.0);
    model.Serialize(params, 0);
  }

  const double pt_weight = m_px_weight / reference_points.size();
  const double depth_weight = m_depth_weight / reference_points.size();

  for (int i = 0; i < num_points; i++)
  {
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<PointDistanceCostFn,
                                                             PointDistanceCostFn::ON, NUM_PARAMS>(
                             new PointDistanceCostFn(m_inv_camera_pose,
                                                     m_camera_matrix,
                                                     observed_points[i], reference_points[i],
                                                     pt_weight, depth_weight)),
                             nullptr,
                             params);
  }

  MyRollPackPoseModel::SetParameterBounds((void *)&problem,
                                          params, 0);

  // Solve the problem
  ceres::Solver::Summary summary;
  try
  {
    ceres::Solve(options, &problem, &summary);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    return MyRollPackPoseModelPtr();
  }
  catch (...)
  {
    std::cerr << "Unknown exception caught" << std::endl;
    return MyRollPackPoseModelPtr();
  }

  //std::cout << summary.FullReport() << std::endl;
  const bool has_converged = (summary.termination_type == ceres::TerminationType::CONVERGENCE);

  if (!m_quiet)
  {
    std::cout << "has converged: " << (has_converged ? "TRUE" : "FALSE") << std::endl;
    std::cout << "final_cost: " << summary.final_cost << std::endl;
  }

  if (has_converged)
    final_cost = summary.final_cost;

  if (!has_converged)
    return MyRollPackPoseModelPtr();

  MyRollPackPoseModelPtr result(new MyRollPackPoseModel(params, 0));

  return result;
}
