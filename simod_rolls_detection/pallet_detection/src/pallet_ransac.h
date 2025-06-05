#ifndef PALLET_RANSAC_H
#define PALLET_RANSAC_H

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <stdint.h>
#include <vector>
#include <set>
#include <cmath>
#include <random>

#include "expected_pallet.h"

class PalletRansac
{
  public:
  typedef uint64_t uint64;
  typedef std::vector<uint64> Uint64Vector;
  typedef std::pair<uint64, uint64> Uint64Pair;
  typedef std::vector<Uint64Pair> Uint64PairVector;
  typedef std::set<Uint64Pair> Uint64PairSet;

  typedef std::function<void(const uint level, const std::string & s)> LogFunction;

  using ExpectedElement = pallet_detection::ExpectedElement;
  using ExpectedPallet = pallet_detection::ExpectedPallet;
  using ExpectedElementVector = pallet_detection::ExpectedElementVector;
  using ExpectedElementType = pallet_detection::ExpectedElementType;

  PalletRansac(const LogFunction & log,
               const double pallet_convergence_threshold,
               const uint64 iterations,
               const double max_pose_correction_distance,
               const double max_pose_correction_angle,
               const double planes_similarity_max_angle,
               const double planes_similarity_max_distance,
               const double point_similarity_max_distance,
               const uint64 random_seed);

  Eigen::Vector4d TransformPlane(const Eigen::Vector4d & plane, const Eigen::Affine3d & transform);
  Eigen::Vector4d TransformPlane(const Eigen::Vector4d & plane, const Eigen::Vector3d & transform2d, const double ztransl);
  Eigen::Vector4d TransformPillar(const Eigen::Vector4d & pillar, const Eigen::Affine3d & transform);
  Eigen::Vector4d TransformPillar(const Eigen::Vector4d & pillar, const Eigen::Vector3d & transform2d, const double ztransl);
  Eigen::Vector4d TransformBox(const Eigen::Vector4d & box, const Eigen::Affine3d & transform, const double angle);
  Eigen::Vector4d TransformBox(const Eigen::Vector4d & box, const Eigen::Vector3d & transform2d, const double ztransl);
  ExpectedElement TransformElement(const ExpectedElement & ee, const Eigen::Vector3d & transform2d, const double ztransl);
  ExpectedPallet TransformPallet(const ExpectedPallet & ep, const Eigen::Vector3d & transform2d, const double ztransl = 0.0);

  double NormalDistance(const Eigen::Vector3d & n1, const Eigen::Vector3d & n2, const double max_angle) const;
  double PlanesDistance(const Eigen::Vector4d & p1, const Eigen::Vector4d & p2, const double max_angle, const double max_distance) const;

  Eigen::Vector3d CombinePoses(const Eigen::Vector3d & a, const Eigen::Vector3d & b);

  bool CheckNewCorrCompatibility(const ExpectedPallet & expected_pallet,
                                 const ExpectedPallet & real_pallet,
                                 const Uint64PairVector & available_corresp,
                                 const Uint64PairSet & conflict_corresp,
                                 const Uint64Vector & already_selected_corrs,
                                 const uint64 maybe_new_corr,
                                 const bool check_stability) const;

  void Run(const ExpectedPallet & expected_pallet,
           const ExpectedPallet & real_pallet,
           const Eigen::Vector3d & initial_guess,
           Eigen::Vector3d & pose,
           Eigen::Vector3d & refined_pose,
           Uint64PairVector &consensus);

  private:
  double m_pallet_convergence_threshold;
  uint64 m_iterations;

  double m_max_pose_correction_distance;
  double m_max_pose_correction_angle;
  double m_planes_similarity_max_angle;
  double m_planes_similarity_max_distance;
  double m_point_similarity_max_distance;

  LogFunction m_log = [](const uint, const std::string &){};

  std::mt19937 m_random_gen;
};

#endif // PALLET_RANSAC_H
