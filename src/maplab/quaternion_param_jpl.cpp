#include "PoseSpline/maplab/quaternion_param_jpl.h"

// #include <maplab-common/pose_types.h>
#include "PoseSpline/maplab/quaternion-math.h"

namespace ceres_error_terms {
namespace {
inline void get_dQuaternionJpl_dTheta(
    const double* q_ptr, double* jacobian_row_major) {
  CHECK_NOTNULL(q_ptr);
  CHECK_NOTNULL(jacobian_row_major);

  const Eigen::Map<const Eigen::Matrix<double, 4, 1>> q(q_ptr);
  Eigen::Matrix<double, 4, 1> temp;
  if (q(3) < 0) {
    temp = -q;
  } else {
    temp = q;
  }
  CHECK_GE(temp(3), 0.);

  Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> jacobian(
      jacobian_row_major);
  jacobian.setZero();
  jacobian(0, 0) = temp(3);
  jacobian(0, 1) = -temp(2);
  jacobian(0, 2) = temp(1);
  jacobian(1, 0) = temp(2);
  jacobian(1, 1) = temp(3);
  jacobian(1, 2) = -temp(0);
  jacobian(2, 0) = -temp(1);
  jacobian(2, 1) = temp(0);
  jacobian(2, 2) = temp(3);
  jacobian(3, 0) = -temp(0);
  jacobian(3, 1) = -temp(1);
  jacobian(3, 2) = -temp(2);
  jacobian *= 0.5;
}
}  // namespace

bool JplQuaternionParameterization::Plus(
    const double* x, const double* delta, double* x_plus_delta) const {
  CHECK_NOTNULL(x);
  CHECK_NOTNULL(delta);
  CHECK_NOTNULL(x_plus_delta);

  Eigen::Map<const Eigen::Vector3d> rot_vector_delta(delta);
  Eigen::Map<const Eigen::Vector4d> q(x);
  Eigen::Map<Eigen::Vector4d> q_product(x_plus_delta);
  CHECK_GE(q(3), 0.);

  double square_norm_delta = rot_vector_delta.squaredNorm();
  // 0.262rad is ~15deg which keeps the small angle approximation error limited
  // to about 1%.
  static constexpr double kSmallAngleApproxThresholdDoubleSquared =
      (0.262 * 2) * (0.262 * 2);

  Eigen::Vector4d q_delta;
  if (square_norm_delta < kSmallAngleApproxThresholdDoubleSquared) {
    // The delta theta norm is below the threshold so we can use the small
    // angle approximation.
    q_delta << 0.5 * rot_vector_delta, 1.0;
    q_delta.normalize();
  } else {
    const double norm_delta_half = 0.5 * sqrt(square_norm_delta);
    q_delta.head<3>() =
        0.5 * rot_vector_delta / norm_delta_half * std::sin(norm_delta_half);
    q_delta(3) = std::cos(norm_delta_half);
  }
  common::positiveQuaternionProductJPL(q_delta, q, q_product);
  CHECK_GE(q_product(3), 0.);

  return true;
}

bool JplQuaternionParameterization::ComputeJacobian(
    const double* quat, double* jacobian_row_major) const {
  CHECK_NOTNULL(quat);
  CHECK_NOTNULL(jacobian_row_major);
  get_dQuaternionJpl_dTheta(quat, jacobian_row_major);
  return true;
}



}  // namespace ceres_error_terms
