#ifndef UTILS_ROT_H
#define UTILS_ROT_H

#include <Eigen/Dense>
#include <math.h>
#include <vector>

#define JOINTS_NUM 6

typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Matrix<double, 12, 1> Vector12d;
typedef Eigen::Matrix<double, 4, 4 * JOINTS_NUM> Matrix44Nd;
typedef Eigen::Matrix<double, 6 * JOINTS_NUM, 6 * JOINTS_NUM> Matrix6Nd;
typedef Eigen::Matrix<double, 6 * JOINTS_NUM, JOINTS_NUM> Matrix6N6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 3, 3> Matrix3d;
typedef Eigen::Matrix<double, 4, 4> Matrix4d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

Matrix3d skew(const Eigen::Vector3d& v);

Vector6d screw(const Eigen::Vector3d& e, const Eigen::Vector3d& p, const double h = 0);

Matrix4d homTransf(const Matrix3d& R, const Eigen::Vector3d& t);

Matrix6d augment(const Matrix3d& R);

Matrix6d adjoint(const Matrix4d& C);

Matrix6d adjointInv(const Matrix4d& C);

Matrix6d adBracket(const Vector6d& V);

Matrix3d SO3exp(const Eigen::Vector3d& e, const double phi);

Matrix3d rpy(const Eigen::Vector3d& rpy);

Matrix4d SE3exp(const Vector6d& X, const double phi);

Matrix4d relConf(const Matrix4d& Ci, const Matrix4d& Cj);

Matrix4d prodexp(const Matrix6d& X, const Vector6d& q, const int n);

Matrix3d eul2Rotm(const Eigen::Vector3d& eul);

Matrix3d vectorToEigenMatrix3d(const std::vector<double> &vector);

Vector6d swapTwistVector(const Vector6d& vec);

double deg2rad(double deg);

#endif // UTILS_ROT_H
