#include "simod_rolls_detection/utils_rot.h"

/**
 * @brief Compute the skew-symmetric matrix of a 3D vector.
 *
 * @param v The input 3D vector.
 * @return The skew-symmetric matrix of the input vector.
 */
Matrix3d skew(const Eigen::Vector3d& v)
{
  Matrix3d A;
  A << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;

  return A;
}

/**
 * @brief Compute the screw vector from a rotation axis and a translation vector.
 *
 * @param e The rotation axis vector.
 * @param p The translation vector.
 * @param h The pitch of the screw.
 * @return The screw vector [omega, v], where omega is the rotation axis and v is the linear velocity.
 */
Vector6d screw(const Eigen::Vector3d& e, const Eigen::Vector3d& p, const double h)
{
  Vector6d Y;
  Y << e, p.cross(e) + h * e;

  return Y;
}

/**
 * @brief Compute a homogeneous transformation matrix from a rotation matrix and a translation vector.
 *
 * @param R The 3x3 rotation matrix.
 * @param t The 3x1 translation vector.
 * @return The 4x4 homogeneous transformation matrix.
 */
Matrix4d homTransf(const Matrix3d& R, const Eigen::Vector3d& t)
{
  Matrix4d T;
  T << R, t, 0, 0, 0, 1;

  return T;
}

Matrix6d augment(const Matrix3d& R)
{
  Matrix6d R_aug;
  R_aug << R, Matrix3d::Zero(), Matrix3d::Zero(), R;

  return R_aug;
}

Matrix6d adjoint(const Matrix4d& C)
{
  Matrix6d AdMatrix;
  Matrix3d R        = C.topLeftCorner<3, 3>();
  Eigen::Vector3d t = C.topRightCorner<3, 1>();

  AdMatrix << R, Matrix3d::Zero(), skew(t) * R, R;
  return AdMatrix;
}

Matrix6d adjointInv(const Matrix4d& C)
{
  Matrix6d AdMatrixInv;
  Matrix3d R        = C.topLeftCorner<3, 3>();
  Eigen::Vector3d t = C.topRightCorner<3, 1>();

  AdMatrixInv << R.transpose(), Matrix3d::Zero(), -R.transpose() * skew(t), R.transpose();
  return AdMatrixInv;
}

Matrix6d adBracket(const Vector6d& V)
{
  Matrix6d LB;
  Eigen::Vector3d omega = V.head<3>();
  Eigen::Vector3d vel   = V.tail<3>();

  LB << skew(omega), Matrix3d::Zero(), skew(vel), skew(omega);
  return LB;
}

Matrix3d SO3exp(const Eigen::Vector3d& e, const double phi)
{
  Matrix3d R;
  R = Matrix3d::Identity() + sin(phi) * skew(e) + (1 - cos(phi)) * skew(e) * skew(e);

  return R;
}

Matrix3d rpy(const Eigen::Vector3d& rpy)
{
  Matrix3d R;
  Eigen::Vector3d ex(1, 0, 0);
  Eigen::Vector3d ey(0, 1, 0);
  Eigen::Vector3d ez(0, 0, 1);

  R = SO3exp(ex, rpy(0)) * SO3exp(ey, rpy(1)) * SO3exp(ez, rpy(2));

  return R;
}

Matrix4d SE3exp(const Vector6d& X, const double phi)
{
  Matrix4d C;
  Matrix3d R;
  Eigen::Vector3d xi  = X.head<3>();
  Eigen::Vector3d eta = X.tail<3>();
  Eigen::Vector3d t;

  R = SO3exp(xi, phi);
  t = (Matrix3d::Identity() - R) * skew(xi) * eta + (xi.dot(eta)) * phi * xi;
  C = homTransf(R, t);

  return C;
}

Matrix4d relConf(const Matrix4d& Ci, const Matrix4d& Cj)
{
  Matrix4d Cij;

  Cij = Ci.inverse() * Cj;

  return Cij;
}

Matrix4d prodexp(const Matrix6d& X, const Vector6d& q, const int n)
{
  Matrix4d P;
  // Double qi;

  P = Matrix4d::Identity();

  for (int i = 0; i <= n; ++i)
  {
    // qi = q(i);
    P = P * SE3exp(X.col(i), q(i));
  }

  return P;
}

Matrix3d vectorToEigenMatrix3d(const std::vector<double> &vector) 
{
  Eigen::Matrix3d matrix;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            matrix(i, j) = vector[i * 3 + j];
        }
    }
    return matrix;
}

double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}

Vector6d swapTwistVector(const Vector6d& vec) {
  Vector6d swapped_vec;
  swapped_vec << vec.tail(3), vec.head(3); 
  return swapped_vec;
}
