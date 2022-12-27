#ifndef CERES_ERROR_TERMS_PARAMETERIZATION_QUATERNION_PARAM_JPL_H_
#define CERES_ERROR_TERMS_PARAMETERIZATION_QUATERNION_PARAM_JPL_H_

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace ceres_error_terms {

class JplQuaternionParameterization : public ceres::LocalParameterization {
 public:
  virtual ~JplQuaternionParameterization() {}
  virtual bool Plus(
      const double* x, const double* delta, double* x_plus_delta) const;
  virtual bool ComputeJacobian(const double* x, double* jacobian) const;
  virtual int GlobalSize() const {
    return 4;
  }
  virtual int LocalSize() const {
    return 3;
  }

  template<typename T>
  static Eigen::Matrix<T,4,4> quatRightComp( const Eigen::Matrix<T,4,1> q )
  {
      // [  q3, -q2,  q1, q0]
      // [  q2,  q3, -q0, q1]
      // [ -q1,  q0,  q3, q2]
      // [ -q0, -q1, -q2, q3]

      Eigen::Matrix<T,4,4> Q;
      Q(0,0) =  q[3]; Q(0,1) = -q[2]; Q(0,2) =  q[1]; Q(0,3) =  q[0];
      Q(1,0) =  q[2]; Q(1,1) =  q[3]; Q(1,2) = -q[0]; Q(1,3) =  q[1];
      Q(2,0) = -q[1]; Q(2,1) =  q[0]; Q(2,2) =  q[3]; Q(2,3) =  q[2];
      Q(3,0) = -q[0]; Q(3,1) = -q[1]; Q(3,2) = -q[2]; Q(3,3) =  q[3];

      return Q;
  }


  template <typename  T>
  static bool liftJacobian(const T* x, T* jacobian) {

      Eigen::Map<Eigen::Matrix<T, 3, 4, Eigen::RowMajor> > J_lift(jacobian);

      Eigen::Matrix<T, 4, 1> q_inv(-x[0],-x[1],-x[2],x[3]);
      Eigen::Matrix<T, 3, 4> Jq_pinv;
      Jq_pinv.setZero();
      Jq_pinv.template topLeftCorner<3,3>() = Eigen::Matrix<T,3,3>::Identity() * T(2.0);
      J_lift = Jq_pinv * quatRightComp(q_inv);

      return true;
  }
  // Additional interface
  bool ComputeLiftJacobian(const double* x, double* jacobian) const {
      liftJacobian<double>(x,jacobian);
      return true;
  }

};



}  // namespace ceres_error_terms

#endif  // CERES_ERROR_TERMS_PARAMETERIZATION_QUATERNION_PARAM_JPL_H_
