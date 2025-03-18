#ifndef PALLET_DETECTION_SOLVER_H
#define PALLET_DETECTION_SOLVER_H

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <vector>

class PalletDetectionPoseEstimation
{
  public:
  struct PlaneCorr
  {
    Eigen::Vector3d plane;
    Eigen::Vector3d ideal_plane;
    double weight;

    PlaneCorr(const Eigen::Vector3d & plane, const Eigen::Vector3d & ideal_plane, const double weight = 1.0)
        : plane(plane), ideal_plane(ideal_plane), weight(weight) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct PointCorr
  {
    Eigen::Vector2d point;
    Eigen::Vector2d ideal_point;
    double weight;

    PointCorr(const Eigen::Vector2d & point, const Eigen::Vector2d & ideal_point, const double weight = 1.0)
        : point(point), ideal_point(ideal_point), weight(weight) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  typedef std::vector<PlaneCorr, Eigen::aligned_allocator<PlaneCorr> > PlaneCorrVector;
  typedef std::vector<PointCorr, Eigen::aligned_allocator<PointCorr> > PointCorrVector;

  PalletDetectionPoseEstimation(const double cost_threshold);
  PalletDetectionPoseEstimation();

  Eigen::Vector3d estimate_pose(const PlaneCorrVector & planes, const PointCorrVector & points,
                                const Eigen::Vector3d & initial_guess = Eigen::Vector3d::Zero());

  void SetQuiet(const bool quiet) {m_quiet = quiet; }

  void test();

  private:
  double m_cost_threshold;
  bool m_quiet = false;
};

#endif // PALLET_DETECTION_SOLVER_H
