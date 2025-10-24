#include "roll_pack_detection_model.h"

#include <ceres/ceres.h>

template <class T> T SQR(const T & t) {return t * t; }

static const double my_nan = std::numeric_limits<double>::quiet_NaN();

template <int COUNT, typename T>
void RollPackDetectionModel<COUNT, T>::SetParameterBounds(void * void_problem,
                                                          const double template_image_width,
                                                          const double template_image_height,
                                                          double * x,
                                                          const int offset)
{
  (void)void_problem; // avoid warning
  (void)template_image_width;
  (void)template_image_height;
  (void)x;
  (void)offset;
  // ceres::Problem & problem = *(ceres::Problem *)void_problem;

  // const double max_acceptable_deformation_x =  template_image_width / COUNT;
  // const double min_acceptable_deformation_x = -(template_image_width / COUNT) / 4.0f;
  // const double max_acceptable_deformation_y =  (template_image_height / COUNT) / 4.0f;
  // const double min_acceptable_deformation_y = -(template_image_height / COUNT) / 4.0f;

  // // t
  // for (int i = 0; i < COUNT - 1; i++)
  // {
  //   // tx
  //   problem.SetParameterLowerBound(x, offset + HOMOGRAPHY_SIZE + i, min_acceptable_deformation_x);
  //   problem.SetParameterUpperBound(x, offset + HOMOGRAPHY_SIZE + i, max_acceptable_deformation_x);
  //   // ty
  //   problem.SetParameterLowerBound(x, offset + HOMOGRAPHY_SIZE + (COUNT - 1) + i, min_acceptable_deformation_y);
  //   problem.SetParameterUpperBound(x, offset + HOMOGRAPHY_SIZE + (COUNT - 1) + i, max_acceptable_deformation_y);
  // }
}

template <int COUNT>
class PointDistanceCostFn
{
  public:
  PointDistanceCostFn(const int element_n, const Eigen::Vector2d observed_point,
                      const Eigen::Vector2d & reference_point, const double this_wgt)
      : m_element_n(element_n), m_reference_point(reference_point),
          m_observed_point(observed_point), m_this_wgt(this_wgt){}

  static constexpr int ON = 2;

  template <typename T>
  bool operator()(const T * const x, T * e) const
  {
    RollPackDetectionModel<COUNT, T> def_model(x, 0);

    T px(m_reference_point.x());
    T py(m_reference_point.y());
    T nx(0.0);
    T ny(0.0);
    def_model.ApplyToPoint(m_element_n, px, py, nx, ny);

    e[0] = (nx - double(m_observed_point.x())) * m_this_wgt;
    e[1] = (ny - double(m_observed_point.y())) * m_this_wgt;
    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
  const int m_element_n;
  const Eigen::Vector2d m_reference_point;
  const Eigen::Vector2d m_observed_point;
  const double m_this_wgt;
};

template <int COUNT>
class SimilarToZeroTranslCostFn
{
  public:
  typedef RollPackDetectionModel<COUNT, double> MyRollPackDetectionModel;

  static constexpr int ON = 2;

  SimilarToZeroTranslCostFn(const int i1, const double weight_x, const double weight_y)
      : m_weight_x(weight_x), m_weight_y(weight_y), m_i1(i1)
  {}

  template <typename T>
  bool operator()(const T * const x, T * e) const
  {
    RollPackDetectionModel<COUNT, T> def_model(x, 0);
    e[0] = def_model.translations_x[m_i1] * m_weight_x;
    e[1] = def_model.translations_y[m_i1] * m_weight_y;

    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
  const double m_weight_x;
  const double m_weight_y;
  const int m_i1;
};

template <int COUNT>
int RollPackDetectionEstimator<COUNT>::PointToElementN(const Eigen::Vector2d &pt) const
{
  return PointToElementN(pt.x(), pt.y());
}

template <int COUNT>
int RollPackDetectionEstimator<COUNT>::PointToElementN(const double x, const double y) const
{
  return RollPackDetectionModel<COUNT, double>::PointToElementN(m_template_size_x, m_template_size_y, x, y);
}

template <int COUNT> typename RollPackDetectionEstimator<COUNT>::MyRollPackDetectionModelPtr
RollPackDetectionEstimator<COUNT>::Estimate(const HomographyArray & initial_homography,
                                            const Vector2dVector & reference_points,
                                            const Vector2dVector & observed_points,
                                            double & final_cost) const
{
  // initial guess
  MyRollPackDetectionModel model(initial_homography);
  return Estimate(model, reference_points, observed_points, final_cost);
}


template <int COUNT> typename RollPackDetectionEstimator<COUNT>::MyRollPackDetectionModelPtr
RollPackDetectionEstimator<COUNT>::Estimate(const MyRollPackDetectionModel & initial_model,
                                            const Vector2dVector & reference_points,
                                            const Vector2dVector & observed_points,
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
  constexpr int NUM_PARAMS = MyRollPackDetectionModel::GetNumParams();
  double params[NUM_PARAMS];

  const int num_points = reference_points.size();
  const int num_initial_points = observed_points.size();
  if (num_points != num_initial_points)
  {
    std::cerr << "Points size mismatch! Current points " << num_points << " <> initial points " << num_initial_points << std::endl;
    return MyRollPackDetectionModelPtr();
  }

  // initial guess
  initial_model.Serialize(params, 0);

  const double translation_weight_x = m_translation_px_weight_x / (COUNT - 1);
  const double translation_weight_y = m_translation_px_weight_y / (COUNT - 1);
  for (int x = 0; x < COUNT - 1; x++)
  {
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SimilarToZeroTranslCostFn<COUNT>,
                                                             SimilarToZeroTranslCostFn<COUNT>::ON, NUM_PARAMS>(
                             new SimilarToZeroTranslCostFn<COUNT>(x, translation_weight_x, translation_weight_y)),
                             nullptr,
                             params);
  }

  for (int i = 0; i < num_points; i++)
  {
    const double pt_weight = 1.0 / num_points;
    const int element_n = PointToElementN(reference_points[i]);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<PointDistanceCostFn<COUNT>,
                                                             PointDistanceCostFn<COUNT>::ON, NUM_PARAMS>(
                             new PointDistanceCostFn<COUNT>(element_n, observed_points[i], reference_points[i],
                                                            pt_weight)),
                             new ceres::HuberLoss(m_max_error_for_huber_loss * pt_weight),
                             params);
  }

  MyRollPackDetectionModel::SetParameterBounds((void *)&problem,
                                               m_template_size_x, m_template_size_y,
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
    return MyRollPackDetectionModelPtr();
  }
  catch (...)
  {
    std::cerr << "Unknown exception caught" << std::endl;
    return MyRollPackDetectionModelPtr();
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
    return MyRollPackDetectionModelPtr();

  MyRollPackDetectionModelPtr result(new MyRollPackDetectionModel(params, 0));

  return result;
}

// explicit instantiation
template class RollPackDetectionEstimator<7>;
template class RollPackDetectionEstimator<1>;
template class RollPackDetectionEstimator<2>;
