#ifndef PALLET_RANSAC_H
#define PALLET_RANSAC_H

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <stdint.h>
#include <vector>
#include <set>
#include <cmath>
#include <random>

class PalletRansac
{
  public:
  typedef uint64_t uint64;
  typedef std::vector<uint64> Uint64Vector;
  typedef std::pair<uint64, uint64> Uint64Pair;
  typedef std::vector<Uint64Pair> Uint64PairVector;
  typedef std::set<Uint64Pair> Uint64PairSet;

  typedef std::function<void(const uint level, const std::string & s)> LogFunction;

  PalletRansac(const LogFunction & log,
               const double pallet_convergence_threshold,
               const uint64 iterations,
               const double max_pose_correction_distance,
               const double max_pose_correction_angle,
               const double planes_similarity_max_angle,
               const double planes_similarity_max_distance,
               const double point_similarity_max_distance,
               const uint64 random_seed);

  enum class ExpectedElementType
  {
    PILLAR,
    PLANE,
    BOX,
  };

  enum class PillarType
  {
    NONE   = 0b00,
    LEFT   = 0b01,
    RIGHT  = 0b10,
    CENTER = 0b11,
  };
  typedef std::vector<PillarType> PillarTypeVector;

  struct PillarPlaneRelation
  {
    PillarType pillar_type = PillarType::NONE;
    uint64 parent_plane_id = 0;

    PillarPlaneRelation() {}
    PillarPlaneRelation(const uint64 plane, const PillarType type): pillar_type(type), parent_plane_id(plane) {}
  };
  typedef std::vector<PillarPlaneRelation> PillarPlaneRelationVector;

  struct ExpectedElement
  {
    ExpectedElementType type = ExpectedElementType(0);
    std::string name;
    Eigen::Vector4d pillar = Eigen::Vector4d::Zero(); // x, y, min_z, max_z
    uint64 pillar_left_plane_id = uint64(-1); // if not -1, this is a left pillar for plane element plane_id
    uint64 pillar_right_plane_id = uint64(-1); // same, but for right pillar
    Eigen::Vector4d plane = Eigen::Vector4d::Zero();  // x, y, z, d
    Eigen::Vector2d plane_z = Eigen::Vector2d(0.0, 1.0); // min_z, max_z
    Eigen::Vector3d plane_point = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()); // one significant point of this plane

    Eigen::Vector3d box_size = Eigen::Vector3d::Zero();
    Eigen::Vector4d box = Eigen::Vector4d::Zero(); // x, y, z, rotation

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
  typedef std::vector<ExpectedElement, Eigen::aligned_allocator<ExpectedElement> > ExpectedElementVector;
  typedef ExpectedElementVector ExpectedPallet;

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
