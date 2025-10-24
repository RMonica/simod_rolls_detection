#ifndef ROLL_PACK_POSE_MODEL_H
#define ROLL_PACK_POSE_MODEL_H

// STL
#include <cmath>
#include <array>
#include <string>
#include <memory>
#include <vector>

// Eigen
#include <Eigen/Dense>

template <typename T>
struct RollPackPoseModel
{
  T tx, ty, tz;
  T rz;

  typedef Eigen::Matrix<T, 3, 1> Vector3T;

  static constexpr int GetNumParams() {return 4; }

  RollPackPoseModel()
  {
    tx = T(0.0);
    ty = T(0.0);
    tz = T(0.0);
    rz = T(0.0);
  }

  RollPackPoseModel(const Eigen::Vector3d & initial_guess_position, const double initial_guess_rotation)
  {
    tx = T(initial_guess_position.x());
    ty = T(initial_guess_position.y());
    tz = T(initial_guess_position.z());
    rz = T(initial_guess_rotation);
  }

  // from serialized array
  RollPackPoseModel(const T * const x, const int offset)
  {
    tx = x[offset + 0];
    ty = x[offset + 1];
    tz = x[offset + 2];
    rz = x[offset + 3];
  }

  void Serialize(T * const x, const int offset) const
  {
    x[offset + 0] = tx;
    x[offset + 1] = ty;
    x[offset + 2] = tz;
    x[offset + 3] = rz;
  }

  void ApplyToPoint(const Eigen::Affine3d & inv_camera_pose,
                    const Eigen::Matrix3d & camera_matrix,
                    const T & ix, const T & iy, const T & iz, T & ox, T & oy, T & oz) const
  {
    Vector3T iv(ix * cos(rz) - iy * sin(rz) + tx, ix * sin(rz) + iy * cos(rz) + ty, iz + tz);
    Vector3T ov = camera_matrix * inv_camera_pose * iv;
    oz = ov.z();
    ox = ov.x() / ov.z();
    oy = ov.y() / ov.z();
  }

  bool IsValid() const
  {
    return true;
  }

  static void SetParameterBounds(/* ceres::Problem */ void * problem,
                                 double *x, const int offset);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class RollPackPoseEstimator
{
  public:
  typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > Vector3dVector;
  typedef RollPackPoseModel<double> MyRollPackPoseModel;
  typedef std::shared_ptr<MyRollPackPoseModel> MyRollPackPoseModelPtr;

  RollPackPoseEstimator(const double px_weight, const double depth_weight,
                        const Eigen::Affine3d & camera_pose,
                        const double fx, const double fy, const double cx, const double cy);

  // NULL if error
  MyRollPackPoseModelPtr Estimate(const Vector3dVector & observed_points,  // u, v, depth; depth may be NAN if depth unavailable
                                  const Vector3dVector & reference_points,
                                  double & final_cost) const;

  private:
  double m_px_weight; // error for one pixel of translation in image space
  double m_depth_weight; // error for depth distance
  Eigen::Affine3d m_inv_camera_pose;
  Eigen::Affine3d m_camera_pose;
  Eigen::Matrix3d m_camera_matrix;

  bool m_quiet = true;
};

#endif // ROLL_PACK_POSE_MODEL_H
